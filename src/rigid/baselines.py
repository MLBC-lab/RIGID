"""Observed-response retrieval against an explicitly independent reference catalog."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .data import ProfileDataset, TrainStandardizer, audit_leakage, load_dataset
from .data.schema import VALID_SPLITS, CompoundRecord, ProfileRecord
from .evaluation import CandidateAnnotation, TargetAnnotation, evaluate_rankings
from .provenance import dataset_hashes
from .serialization import read_json, require_object


@dataclass(frozen=True)
class ReferenceNeighborConfig:
    reference_splits: tuple[str, ...] = ("train",)
    match_cell: bool = True
    dose_tolerance_um: float = 0.0
    time_tolerance_h: float = 0.0
    aggregation: str = "max"
    chunk_profiles: int = 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_splits, (list, tuple))
            or not self.reference_splits
            or any(
                not isinstance(split, str) or split not in VALID_SPLITS - {"", "excluded"}
                for split in self.reference_splits
            )
            or len(set(self.reference_splits)) != len(self.reference_splits)
        ):
            raise ValueError("reference_splits must contain unique named, active partitions")
        object.__setattr__(self, "reference_splits", tuple(self.reference_splits))
        if type(self.match_cell) is not bool:
            raise ValueError("match_cell must be a boolean")
        for name in ("dose_tolerance_um", "time_tolerance_h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite nonnegative number")
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be a finite nonnegative number") from error
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, value)
        if not isinstance(self.aggregation, str) or self.aggregation not in {"max", "mean"}:
            raise ValueError("aggregation must be max or mean")
        if type(self.chunk_profiles) is not int or self.chunk_profiles < 1:
            raise ValueError("chunk_profiles must be a positive integer")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["reference_splits"] = list(self.reference_splits)
        return value

    @classmethod
    def load(cls, path: str | Path) -> ReferenceNeighborConfig:
        value = require_object(read_json(path, maximum_bytes=65536), context="Reference config")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown reference configuration fields: {sorted(unknown)}")
        return cls(**value)


def _row_hash(row: np.ndarray) -> bytes:
    values = np.array(row, dtype="<f4", copy=True)
    values[values == 0] = 0.0
    return hashlib.sha256(values.tobytes()).digest()


def _require_independent(query: ProfileDataset, reference: ProfileDataset) -> None:
    query_ids = {profile.profile_id for profile in query.profiles}
    overlap = query_ids & {profile.profile_id for profile in reference.profiles}
    if overlap:
        raise ValueError(f"Query/reference profile IDs overlap: {sorted(overlap)[:5]}")
    query_rows = {_row_hash(row) for row in query.expression}
    for profile, row in zip(reference.profiles, reference.expression, strict=True):
        if _row_hash(row) in query_rows:
            raise ValueError(
                f"Reference profile {profile.profile_id} duplicates a query dataset expression vector"
            )
    identities: dict[str, str] = {}
    for compound in (*query.compounds, *reference.compounds):
        for identifier in (compound.compound_id, *compound.aliases):
            key = identifier.strip().casefold()
            if key in identities and identities[key] != compound.canonical_smiles:
                raise ValueError(f"Conflicting compound identity or alias: {identifier}")
            identities[key] = compound.canonical_smiles


def _require_provenance(reference: ProfileDataset) -> None:
    source = reference.provenance
    if source.kind == "unknown":
        raise ValueError("Reference catalog requires explicit synthetic or experimental provenance")
    if (
        not isinstance(source.description, str)
        or not source.description.strip()
        or "source provenance has not been specified" in source.description
    ):
        raise ValueError("Reference catalog requires a meaningful source description")
    if source.kind == "experimental" and not (source.source_urls or source.input_sha256):
        raise ValueError(
            "Experimental reference provenance requires source URLs or input checksums"
        )
    if not isinstance(source.source_urls, (tuple, list)) or any(
        not isinstance(url, str) or not url.strip() for url in source.source_urls
    ):
        raise ValueError("Reference source URLs must be nonempty strings in a sequence")
    if not isinstance(source.input_sha256, dict) or any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for name, digest in source.input_sha256.items()
    ):
        raise ValueError("Reference input checksums must map source names to SHA-256 digests")


def _candidates(
    query: ProfileDataset, candidate_ids: Sequence[str] | None
) -> tuple[list[CompoundRecord], dict[str, str], dict[str, list[str]]]:
    lookup = query.compound_lookup
    if candidate_ids is None:
        declared = list(lookup)
    else:
        if isinstance(candidate_ids, (str, bytes)) or not isinstance(candidate_ids, Sequence):
            raise ValueError("candidate_ids must be a sequence of declared compound identifiers")
        declared = list(candidate_ids)
        if (
            not declared
            or any(not isinstance(cid, str) or cid not in lookup for cid in declared)
            or len(set(declared)) != len(declared)
        ):
            raise ValueError("candidate_ids must be unique identifiers in the query compound table")
    candidates: list[CompoundRecord] = []
    structure_to_id: dict[str, str] = {}
    identity_members: dict[str, list[str]] = {}
    for cid in declared:
        record = lookup[cid]
        if record.canonical_smiles not in structure_to_id:
            structure_to_id[record.canonical_smiles] = cid
            candidates.append(record)
            identity_members[cid] = []
        representative = structure_to_id[record.canonical_smiles]
        identity_members[representative].append(cid)
    return candidates, structure_to_id, identity_members


def _unit_rows(values: np.ndarray, normalizer: TrainStandardizer) -> tuple[np.ndarray, np.ndarray]:
    standardized = (np.asarray(values, dtype=np.float64) - normalizer.mean) / normalizer.scale
    standardized -= standardized.mean(-1, keepdims=True)
    norms = np.linalg.norm(standardized, axis=-1)
    valid = norms > 0
    standardized[valid] /= norms[valid, None]
    return standardized, valid


def _context_matches(
    target: ProfileRecord, response: ProfileRecord, config: ReferenceNeighborConfig
) -> bool:
    return (
        (not config.match_cell or target.cell_id == response.cell_id)
        and abs(target.dose_um - response.dose_um) <= config.dose_tolerance_um
        and abs(target.time_h - response.time_h) <= config.time_tolerance_h
    )


def reference_nearest_neighbor(
    dataset_path: str | Path,
    reference_dataset: str | Path,
    split: str = "test",
    k: int = 50,
    *,
    config: ReferenceNeighborConfig | None = None,
    candidate_ids: Sequence[str] | None = None,
    scaffold_holdout: bool = False,
    acyclic_policy: str = "group",
) -> dict:
    """Rank a declared molecular universe using independent observed responses.

    Reference compounds may overlap held-out source compounds by design; response
    measurements may not overlap query data. This access regime differs from a
    predictor that receives candidate structures alone.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if split not in {"validation", "test", "external"}:
        raise ValueError("Evaluation requires validation, test, or external targets")
    if config is not None and not isinstance(config, ReferenceNeighborConfig):
        raise ValueError("config must be a ReferenceNeighborConfig")
    if not isinstance(acyclic_policy, str) or acyclic_policy not in {
        "group",
        "compound",
        "exclude",
    }:
        raise ValueError("acyclic_policy must be group, compound, or exclude")
    config = config or ReferenceNeighborConfig()
    if Path(dataset_path).resolve() == Path(reference_dataset).resolve():
        raise ValueError("Reference catalog must be a separate dataset")
    query_hashes, reference_hashes = dataset_hashes(dataset_path), dataset_hashes(reference_dataset)
    query, reference = load_dataset(dataset_path), load_dataset(reference_dataset)
    query.genes.require_match(reference.genes)
    audit = audit_leakage(query, scaffold=scaffold_holdout, acyclic_policy=acyclic_policy)
    audit.raise_for_leakage()
    _require_provenance(reference)
    _require_independent(query, reference)
    selected = [i for i, profile in enumerate(query.profiles) if profile.split == split]
    if not selected:
        raise ValueError(f"No targets in split {split}")
    training = [profile for profile in query.profiles if profile.split == "train"]
    normalizer = TrainStandardizer.fit(query.expression, query.split_labels, query.genes)
    candidates, structure_to_id, identity_members = _candidates(query, candidate_ids)
    positions = {record.compound_id: i for i, record in enumerate(candidates)}
    reference_lookup = reference.compound_lookup
    mapped_ids = {
        record.compound_id: structure_to_id[record.canonical_smiles]
        for record in reference.compounds
        if record.canonical_smiles in structure_to_id
    }
    reference_indices = [
        i
        for i, profile in enumerate(reference.profiles)
        if profile.split in config.reference_splits
    ]
    if not reference_indices:
        raise ValueError("No reference profiles in the selected reference_splits")
    valid_indices, excluded_constant, excluded_catalog = [], [], []
    for offset in range(0, len(reference_indices), config.chunk_profiles):
        indices = reference_indices[offset : offset + config.chunk_profiles]
        _, valid = _unit_rows(reference.expression[indices], normalizer)
        for index, keep in zip(indices, valid, strict=True):
            profile = reference.profiles[index]
            if not keep:
                excluded_constant.append(profile.profile_id)
            elif profile.compound_id not in mapped_ids:
                excluded_catalog.append(profile.profile_id)
            else:
                valid_indices.append(index)
    rankings, coverage = {}, {}
    for index in selected:
        target = query.profiles[index]
        unit_target, valid = _unit_rows(query.expression[index : index + 1], normalizer)
        if not valid[0]:
            raise ValueError(f"Target {target.profile_id} has undefined Pearson correlation")
        sums: NDArray[np.float64] = np.zeros(len(candidates), dtype=np.float64)
        maxima: NDArray[np.float64] = np.full(len(candidates), -np.inf, dtype=np.float64)
        counts: NDArray[np.int64] = np.zeros(len(candidates), dtype=np.int64)
        eligible = [
            i for i in valid_indices if _context_matches(target, reference.profiles[i], config)
        ]
        for offset in range(0, len(eligible), config.chunk_profiles):
            indices = eligible[offset : offset + config.chunk_profiles]
            unit_reference, _ = _unit_rows(reference.expression[indices], normalizer)
            correlations = np.clip(unit_reference @ unit_target[0], -1, 1)
            for ref_index, correlation in zip(indices, correlations, strict=True):
                cid = mapped_ids[reference.profiles[ref_index].compound_id]
                position = positions[cid]
                counts[position] += 1
                sums[position] += correlation
                maxima[position] = max(maxima[position], correlation)
        available = np.flatnonzero(counts)
        scores = maxima if config.aggregation == "max" else sums / np.maximum(counts, 1)
        order = available[np.argsort(-scores[available], kind="stable")][:k]
        rankings[target.profile_id] = [
            {
                "rank": rank,
                "candidate_id": candidates[item].compound_id,
                "smiles": candidates[item].canonical_smiles,
                "score": float(scores[item]),
                "reference_profile_count": int(counts[item]),
            }
            for rank, item in enumerate(order, 1)
        ]
        source_structure = query.compound_lookup[target.compound_id].canonical_smiles
        source_id = structure_to_id.get(source_structure)
        coverage[target.profile_id] = {
            "eligible_count": len(available),
            "unscoreable_count": len(candidates) - len(available),
            "candidate_coverage": len(available) / len(candidates) if candidates else 0.0,
            "eligible_reference_profile_count": len(eligible),
            "source_in_declared_catalog": source_id is not None,
            "source_scoreable": source_id is not None and bool(counts[positions[source_id]]),
            "returned": len(order),
        }
    query_lookup = query.compound_lookup
    annotation_labels: dict[str, set[str]] = {}
    annotation_members: dict[str, list[str]] = {}
    for record in query.compounds:
        annotation_labels.setdefault(record.canonical_smiles, set()).update(record.moa)
        annotation_members.setdefault(record.canonical_smiles, []).append(record.compound_id)
    annotations = {
        record.compound_id: CandidateAnnotation(
            moa=frozenset(annotation_labels[record.canonical_smiles]),
            scaffold=record.scaffold,
        )
        for record in candidates
    }
    targets = {
        query.profiles[i].profile_id: TargetAnnotation(
            structure_to_id.get(
                query_lookup[query.profiles[i].compound_id].canonical_smiles,
                query.profiles[i].compound_id,
            ),
            frozenset(
                annotation_labels[query_lookup[query.profiles[i].compound_id].canonical_smiles]
            ),
        )
        for i in selected
    }
    metrics = evaluate_rankings(
        {tid: [row["candidate_id"] for row in rows] for tid, rows in rankings.items()},
        targets,
        annotations,
        k=k,
        training_scaffolds=[query_lookup[profile.compound_id].scaffold for profile in training],
    ).to_dict()
    if query_hashes != dataset_hashes(dataset_path) or reference_hashes != dataset_hashes(
        reference_dataset
    ):
        raise ValueError("Dataset files changed during reference retrieval")
    kinds = {query.provenance.kind, reference.provenance.kind}
    source_kind = (
        "synthetic"
        if "synthetic" in kinds
        else "experimental"
        if kinds == {"experimental"}
        else "unknown"
    )
    return {
        "schema_version": 1,
        "kind": "independent-reference-nearest-neighbor",
        "source_kind": source_kind,
        "query_source_kind": query.provenance.kind,
        "reference_source_kind": reference.provenance.kind,
        "mixed_source_kinds": len(kinds) > 1,
        "reproduces_manuscript": False,
        "split": split,
        "requested_k": k,
        "candidate_count": len(candidates),
        "candidate_pool": "declared query molecular catalog; canonical structure identities",
        "candidate_ids": [record.compound_id for record in candidates],
        "candidate_identity_members": identity_members,
        "annotation_identity_members": {
            record.compound_id: annotation_members[record.canonical_smiles] for record in candidates
        },
        "reference_to_candidate_ids": mapped_ids,
        "reference_compounds_outside_catalog": sorted(set(reference_lookup) - set(mapped_ids)),
        "input_sha256": query_hashes,
        "reference_input_sha256": reference_hashes,
        "gene_manifest": query.genes.to_dict(),
        "query_provenance": query.provenance.to_dict(),
        "reference_provenance": reference.provenance.to_dict(),
        "configuration": config.to_dict(),
        "leakage_audit": audit.to_dict(),
        "normalizer": normalizer.to_dict(),
        "access_regime": "independent observed candidate responses; overlapping compound identities allowed",
        "query_reference_overlap_check": "all profile IDs and exact expression vectors in both datasets",
        "ranking_profile_ids": [reference.profiles[i].profile_id for i in valid_indices],
        "excluded_constant_reference_profile_ids": excluded_constant,
        "excluded_outside_catalog_profile_ids": excluded_catalog,
        "coverage": coverage,
        "rankings": rankings,
        "metrics": metrics,
        "notes": [
            "Candidate structures are declared independently of target identity; missing responses are not rank fillers.",
            "Gene standardization is fitted exclusively to the query dataset training partition.",
            "Reference compound identities may include a held-out query source, but profile IDs and exact expression copies may not overlap.",
            "Exact recovery uses canonical standardized molecular identity; MOA annotations do not enter retrieval scores.",
            "This observed-response access regime is not equivalent to structure-only prediction.",
            "Duplicate checks cannot prove biological independence; source provenance and experimental design must be reviewed.",
        ],
    }
