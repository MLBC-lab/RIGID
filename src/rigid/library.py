"""Immutable candidate identities and bounded, process-local screening caches."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import DataStructs

from .chem.graph import FEATURE_SCHEMA_VERSION, batch_graphs, graph_from_smiles
from .chem.properties import ChemicalProperties, chemical_properties, morgan_fingerprint
from .chem.standardize import standardize_smiles
from .data.schema import CompoundRecord
from .provenance import atomic_json, sha256_file
from .serialization import read_json

_MIB = 1024 * 1024
_FINGERPRINT_VERSION = "morgan-radius2-bits2048-v1"


def _digest(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(not allow_zero):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


class _ByteLRU:
    def __init__(self, limit: int) -> None:
        _positive_integer(limit, "Cache byte limit", allow_zero=True)
        self.limit = limit
        self.size = 0
        self.hits = self.misses = self.evictions = 0
        self.entries: OrderedDict[Any, tuple[Any, int]] = OrderedDict()

    def get(self, key: Any) -> Any:
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key: Any, value: Any, size: int) -> None:
        if size > self.limit or not self.limit:
            return
        if key in self.entries:
            self.size -= self.entries.pop(key)[1]
        while self.entries and self.size + size > self.limit:
            self.size -= self.entries.popitem(last=False)[1][1]
            self.evictions += 1
        self.entries[key] = value, size
        self.size += size

    def clear(self) -> None:
        self.entries.clear()
        self.size = 0

    def info(self) -> dict[str, int]:
        return {
            "entries": len(self.entries),
            "payload_bytes": self.size,
            "limit_bytes": self.limit,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class PreparedLibrary:
    """Reuse chemistry and model outputs across many target queries.

    Duplicate structures are rejected unless deduplication is explicitly requested.
    In that case, the first source identifier is retained and the remaining IDs
    become aliases. Candidate order is stable and defines ranking tie breaks.

    Cache limits describe retained array/graph/fingerprint payload bytes, not the
    interpreter's object overhead or temporary computation batches. Caches are
    process-local and not designed for concurrent writes from multiple threads.
    """

    def __init__(
        self,
        records: Iterable[CompoundRecord],
        *,
        deduplicate: bool = False,
        cache_bytes: int = 128 * _MIB,
        graph_cache_bytes: int = 64 * _MIB,
        fingerprint_cache_bytes: int = 16 * _MIB,
        property_cache_size: int = 1024,
    ) -> None:
        _positive_integer(property_cache_size, "property_cache_size", allow_zero=True)
        if not isinstance(deduplicate, bool):
            raise ValueError("deduplicate must be a boolean")
        records = tuple(records)
        if not records:
            raise ValueError("Candidates must be nonempty")
        if any(not isinstance(record, CompoundRecord) for record in records):
            raise TypeError("Candidates must be CompoundRecord instances")
        if len({record.compound_id for record in records}) != len(records):
            raise ValueError("Candidate identifiers must be unique")
        groups: dict[str, list[CompoundRecord]] = {}
        aliases: dict[str, str] = {}
        for record in records:
            groups.setdefault(record.canonical_smiles, []).append(record)
            for identifier in (record.compound_id, *record.aliases):
                normalized = identifier.strip().casefold()
                if not normalized:
                    raise ValueError("Candidate aliases must be nonempty")
                if normalized in aliases and aliases[normalized] != record.canonical_smiles:
                    raise ValueError(f"Ambiguous candidate identity or alias: {identifier}")
                aliases[normalized] = record.canonical_smiles
        if len(groups) != len(records) and not deduplicate:
            raise ValueError("Candidate structures must be unique after standardization")
        retained, source_ids = [], []
        for group in groups.values():
            first = group[0]
            identifiers = tuple(record.compound_id for record in group)
            retained.append(
                replace(
                    first,
                    aliases=tuple(
                        sorted(
                            {alias for record in group for alias in record.aliases}
                            | set(identifiers[1:])
                        )
                    ),
                    moa=tuple(sorted({label for record in group for label in record.moa})),
                )
            )
            source_ids.append(identifiers)
        self._records = tuple(retained)
        self._source_ids = tuple(source_ids)
        self._source_sha256: str | None = None
        self._feature_schema = FEATURE_SCHEMA_VERSION
        self._arrays = _ByteLRU(cache_bytes)
        self._graphs = _ByteLRU(graph_cache_bytes)
        self._fingerprints = _ByteLRU(fingerprint_cache_bytes)
        self._property_limit = property_cache_size
        self._properties: OrderedDict[int, ChemicalProperties] = OrderedDict()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[CompoundRecord, ...]:
        return self._records

    @property
    def feature_schema(self) -> str:
        return self._feature_schema

    @property
    def source_sha256(self) -> str | None:
        return self._source_sha256

    @property
    def sha256(self) -> str:
        return _digest(self._identity())

    def _identity(self) -> dict:
        return {
            "schema_version": 1,
            "feature_schema": self.feature_schema,
            "fingerprint_schema": _FINGERPRINT_VERSION,
            "candidates": [
                {
                    "compound_id": record.compound_id,
                    "canonical_smiles": record.canonical_smiles,
                    "aliases": list(record.aliases),
                    "source_ids": list(source_ids),
                }
                for record, source_ids in zip(self.records, self._source_ids, strict=True)
            ],
        }

    def summary(self) -> dict:
        return {
            "candidate_count": len(self),
            "source_record_count": sum(map(len, self._source_ids)),
            "merged_record_count": sum(map(len, self._source_ids)) - len(self),
            "identity_sha256": self.sha256,
            "source_sha256": self.source_sha256,
            "feature_schema": self.feature_schema,
            "fingerprint_schema": _FINGERPRINT_VERSION,
            "cache": self.cache_info(),
        }

    def save_manifest(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError("Library manifest exists; choose a new filename")
        atomic_json(
            path,
            {
                **self._identity(),
                "identity_sha256": self.sha256,
                "source_sha256": self.source_sha256,
            },
            overwrite=False,
        )

    @classmethod
    def from_manifest(cls, path: str | Path, **cache_options: Any) -> PreparedLibrary:
        value = read_json(path)
        fields = {
            "schema_version",
            "feature_schema",
            "fingerprint_schema",
            "candidates",
            "identity_sha256",
            "source_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Invalid candidate library manifest fields")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["feature_schema"] != FEATURE_SCHEMA_VERSION
        ):
            raise ValueError("Library feature/schema version is not supported")
        if value["fingerprint_schema"] != _FINGERPRINT_VERSION:
            raise ValueError("Library fingerprint schema is not supported")
        source_sha256 = value["source_sha256"]
        if source_sha256 is not None and (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError("Invalid library source SHA256")
        if not isinstance(value["candidates"], list):
            raise ValueError("Library candidates must be a list")
        identity = {key: value[key] for key in fields - {"identity_sha256", "source_sha256"}}
        if _digest(identity) != value["identity_sha256"]:
            raise ValueError("Library identity checksum mismatch")
        records, provenance = [], []
        for row in value["candidates"]:
            if not isinstance(row, dict) or set(row) != {
                "compound_id",
                "canonical_smiles",
                "aliases",
                "source_ids",
            }:
                raise ValueError("Invalid candidate identity fields")
            if (
                not isinstance(row["compound_id"], str)
                or not isinstance(row["canonical_smiles"], str)
                or not isinstance(row["aliases"], list)
                or any(not isinstance(alias, str) for alias in row["aliases"])
            ):
                raise ValueError("Candidate identifiers, SMILES, and aliases must be strings")
            ids = row["source_ids"]
            if (
                not isinstance(ids, list)
                or not ids
                or ids[0] != row["compound_id"]
                or any(not isinstance(item, str) or not item.strip() for item in ids)
                or len(set(ids)) != len(ids)
                or set(ids[1:]) - set(row["aliases"])
            ):
                raise ValueError("Invalid candidate source identifier provenance")
            records.append(
                CompoundRecord(
                    row["compound_id"],
                    row["canonical_smiles"],
                    aliases=tuple(row["aliases"]),
                    canonical_smiles=row["canonical_smiles"],
                )
            )
            provenance.append(tuple(ids))
        library = cls(records, **cache_options)
        library._source_ids = tuple(provenance)
        library._source_sha256 = value["source_sha256"]
        if library.sha256 != value["identity_sha256"]:
            raise ValueError("Library identity changed during validation")
        return library

    @classmethod
    def from_csv(
        cls, path: str | Path, *, deduplicate: bool = False, **cache_options: Any
    ) -> PreparedLibrary:
        """Read compound_id/smiles plus optional semicolon-delimited aliases/moa."""
        path = Path(path)
        source_hash = sha256_file(path)
        records = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if not {"compound_id", "smiles"}.issubset(fields) or len(set(fields)) != len(fields):
                raise ValueError("Library CSV requires unique compound_id and smiles columns")
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed library CSV row {line}")
                try:
                    records.append(
                        CompoundRecord(
                            row["compound_id"].strip(),
                            row["smiles"].strip(),
                            aliases=tuple(
                                filter(None, (s.strip() for s in row.get("aliases", "").split(";")))
                            ),
                            moa=tuple(
                                filter(None, (s.strip() for s in row.get("moa", "").split(";")))
                            ),
                        )
                    )
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid library CSV row {line}: {error}") from error
        library = cls(records, deduplicate=deduplicate, **cache_options)
        if sha256_file(path) != source_hash:
            raise ValueError("Library source changed while it was being read")
        library._source_sha256 = source_hash
        return library

    def _indices(self, indices: Sequence[int]) -> tuple[int, ...]:
        result = tuple(indices)
        if not result:
            raise ValueError("A library batch cannot be empty")
        if any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(self) for i in result
        ):
            raise IndexError("Candidate index is outside the library")
        if len(set(result)) != len(result):
            raise ValueError("A library batch cannot contain duplicate indices")
        return result

    def batches(self, batch_size: int = 128) -> Iterable[tuple[int, ...]]:
        _positive_integer(batch_size, "batch_size")
        for start in range(0, len(self), batch_size):
            yield tuple(range(start, min(start + batch_size, len(self))))

    def graph_batch(self, indices: Sequence[int]):
        graphs = []
        for index in self._indices(indices):
            graph = self._graphs.get(index)
            if graph is None:
                graph = graph_from_smiles(self.records[index].canonical_smiles)
                arrays = (
                    graph.atom_features,
                    graph.bond_features,
                    graph.edge_index,
                    graph.reverse_edge,
                )
                for array in arrays:
                    array.flags.writeable = False
                self._graphs.put(index, graph, sum(array.nbytes for array in arrays))
            graphs.append(graph)
        return batch_graphs(graphs)

    def properties(self, index: int) -> ChemicalProperties:
        self._indices((index,))
        if index in self._properties:
            self._properties.move_to_end(index)
            return self._properties[index]
        value = chemical_properties(self.records[index].canonical_smiles)
        if self._property_limit:
            if len(self._properties) == self._property_limit:
                self._properties.popitem(last=False)
            self._properties[index] = value
        return value

    def cached_rows(
        self,
        namespace: str,
        indices: Sequence[int],
        compute: Callable[[tuple[int, ...]], np.ndarray],
    ) -> np.ndarray:
        """Return ordered numeric rows; compute missing rows once per supplied batch.

        The caller owns the namespace contract: include the complete model state,
        preprocessing feature version, and every input other than candidate identity.
        Returned arrays do not expose mutable cache storage.
        """
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("A nonempty cache namespace is required")
        indices = self._indices(indices)
        rows = [self._arrays.get((namespace, index)) for index in indices]
        positions = [position for position, row in enumerate(rows) if row is None]
        if positions:
            missing = tuple(indices[position] for position in positions)
            computed = np.asarray(compute(missing))
            if (
                computed.ndim < 1
                or len(computed) != len(missing)
                or not np.issubdtype(computed.dtype, np.number)
                or np.iscomplexobj(computed)
            ):
                raise ValueError("Cached model output must have one numeric row per candidate")
            if not np.isfinite(computed).all():
                raise ValueError("Cached model output must be finite")
            for position, value in zip(positions, computed, strict=True):
                row = np.array(value, copy=True)
                row.flags.writeable = False
                rows[position] = row
                self._arrays.put((namespace, indices[position]), row, max(row.nbytes, 1))
        return np.stack(rows)

    def _fingerprint(self, smiles: str):
        value = self._fingerprints.get(smiles)
        if value is None:
            value = morgan_fingerprint(smiles)
            self._fingerprints.put(smiles, value, value.GetNumBits() // 8)
        return value

    def training_similarity(
        self,
        reference_smiles: Sequence[str],
        *,
        batch_size: int = 512,
        reference_batch_size: int = 4096,
    ) -> np.ndarray:
        """Exact Morgan maximum similarity without a dense pairwise score matrix."""
        _positive_integer(reference_batch_size, "reference_batch_size")
        reference = tuple(
            dict.fromkeys(standardize_smiles(s).canonical_smiles for s in reference_smiles)
        )
        if not reference:
            raise ValueError("Training reference set must be nonempty")
        namespace = _digest(
            {
                "stage": "training-similarity",
                "reference": reference,
                "fingerprint": _FINGERPRINT_VERSION,
            }
        )

        def compute(indices: tuple[int, ...]) -> np.ndarray:
            fingerprints = [self._fingerprint(self.records[i].canonical_smiles) for i in indices]
            result: np.ndarray = np.zeros(len(indices), dtype=np.float64)
            for start in range(0, len(reference), reference_batch_size):
                other = [
                    self._fingerprint(s) for s in reference[start : start + reference_batch_size]
                ]
                for position, fingerprint in enumerate(fingerprints):
                    result[position] = max(
                        result[position],
                        max(DataStructs.BulkTanimotoSimilarity(fingerprint, other)),
                    )
            return result

        return np.concatenate(
            [self.cached_rows(namespace, indices, compute) for indices in self.batches(batch_size)]
        )

    def clear_cache(self) -> None:
        for cache in (self._arrays, self._graphs, self._fingerprints):
            cache.clear()
        self._properties.clear()

    def cache_info(self) -> dict:
        return {
            "arrays": self._arrays.info(),
            "graphs": self._graphs.info(),
            "fingerprints": self._fingerprints.info(),
            "properties": {"entries": len(self._properties), "limit_entries": self._property_limit},
        }


def model_fingerprint(model: Any) -> str:
    """Hash current tensor contents, not object identity or a mutable filename."""
    import torch

    digest = hashlib.sha256()
    modules = []
    for name, module in model.named_modules():
        attributes = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_")
            and (value is None or isinstance(value, (str, bool, int, float)))
        }
        modules.append(
            {
                "name": name,
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "attributes": attributes,
            }
        )
    digest.update(
        _digest(
            {
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "config": getattr(model, "config", {}),
                "modules": modules,
            }
        ).encode()
    )
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(
            _digest(
                {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            ).encode()
        )
        # Byte views also support tensor dtypes that NumPy cannot represent.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prediction_namespace(
    model: Any, *, stage: str, genes_sha256: str, device: str, context: np.ndarray | None = None
) -> str:
    return _digest(
        {
            "model": model_fingerprint(model),
            "stage": stage,
            "genes": genes_sha256,
            "features": FEATURE_SCHEMA_VERSION,
            "device": str(device),
            "context": None
            if context is None
            else {
                "shape": list(context.shape),
                "dtype": str(context.dtype),
                "bytes": context.tobytes().hex(),
            },
        }
    )
