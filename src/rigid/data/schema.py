"""Validated records and an immutable ordering for target expression features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from numbers import Real
from pathlib import Path

import numpy as np

from rigid.chem.standardize import standardize_smiles

VALID_SPLITS = frozenset({"train", "validation", "calibration", "test", "external", "excluded", ""})
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class GeneManifest:
    gene_ids: tuple[str, ...]
    namespace: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_ids", tuple(self.gene_ids))
        if not self.gene_ids or any(
            not isinstance(gene, str) or not gene.strip() for gene in self.gene_ids
        ):
            raise ValueError("Gene manifest must contain nonempty string identifiers")
        if len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("Gene manifest contains duplicate identifiers")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("Gene namespace must be a nonempty string")

    def __len__(self) -> int:
        return len(self.gene_ids)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"gene_ids": self.gene_ids, "namespace": self.namespace}, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def require_match(self, other: GeneManifest | Iterable[str]) -> None:
        gene_ids = other.gene_ids if isinstance(other, GeneManifest) else tuple(other)
        if self.gene_ids != tuple(gene_ids):
            raise ValueError("Gene columns or gene order do not match the ordered manifest")
        if isinstance(other, GeneManifest) and self.namespace != other.namespace:
            raise ValueError("Gene namespaces do not match")

    def to_dict(self) -> dict:
        return {"gene_ids": list(self.gene_ids), "namespace": self.namespace, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict) -> GeneManifest:
        if not isinstance(value, dict):
            raise ValueError("Gene manifest must be an object")
        if set(value) - {"gene_ids", "namespace", "sha256"}:
            raise ValueError("Gene manifest contains unknown fields")
        gene_ids = value.get("gene_ids")
        if not isinstance(gene_ids, (list, tuple)):
            raise ValueError("Gene manifest gene_ids must be a list of string identifiers")
        manifest = cls(tuple(gene_ids), value.get("namespace", "unspecified"))
        if value.get("sha256", manifest.sha256) != manifest.sha256:
            raise ValueError("Gene manifest checksum does not match its contents")
        return manifest

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GeneManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SourceProvenance:
    kind: str = "unknown"
    description: str = "User-supplied dataset; source provenance has not been specified."
    source_urls: tuple[str, ...] = ()
    input_sha256: dict[str, str] = field(default_factory=dict)
    transformations: tuple[str, ...] = ()
    license: str = "unspecified"

    def __post_init__(self) -> None:
        if self.kind not in {"synthetic", "experimental", "unknown"}:
            raise ValueError("Provenance kind must be synthetic, experimental, or unknown")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SourceProvenance:
        value = dict(value)
        for name in ("source_urls", "transformations"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)


@dataclass(frozen=True)
class CompoundRecord:
    compound_id: str
    smiles: str
    moa: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    canonical_smiles: str = ""
    scaffold: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.compound_id, str) or not self.compound_id.strip():
            raise ValueError("compound_id must be nonempty")
        for name in ("moa", "aliases"):
            values = getattr(self, name)
            if isinstance(values, str):
                raise ValueError(f"{name} must be a sequence of label strings")
            values = tuple(values)
            if any(
                not isinstance(value, str) or not value.strip() or ";" in value for value in values
            ):
                raise ValueError(
                    f"{name} must contain nonempty strings without semicolon delimiters"
                )
            object.__setattr__(self, name, tuple(sorted(set(values))))
        normalized = standardize_smiles(self.smiles)
        if self.canonical_smiles and self.canonical_smiles != normalized.canonical_smiles:
            raise ValueError(
                f"Stored canonical SMILES differs from standardization for {self.compound_id}"
            )
        if self.scaffold and self.scaffold != normalized.scaffold:
            raise ValueError(f"Stored scaffold differs from standardization for {self.compound_id}")
        object.__setattr__(self, "canonical_smiles", normalized.canonical_smiles)
        object.__setattr__(self, "scaffold", normalized.scaffold)


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    compound_id: str
    cell_id: str
    dose_um: float
    time_h: float
    split: str = ""

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.profile_id, self.compound_id, self.cell_id)
        ):
            raise ValueError("Profile, compound, and cell identifiers must be nonempty")
        for name in ("dose_um", "time_h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number")
            object.__setattr__(self, name, float(value))
        if not np.isfinite(self.dose_um) or self.dose_um < 0:
            raise ValueError("dose_um must be finite and nonnegative")
        if not np.isfinite(self.time_h) or self.time_h <= 0:
            raise ValueError("time_h must be finite and positive")
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split {self.split!r}; use train, validation, calibration, test, external, or excluded"
            )


@dataclass(frozen=True)
class ProfileDataset:
    genes: GeneManifest
    compounds: tuple[CompoundRecord, ...]
    profiles: tuple[ProfileRecord, ...]
    expression: np.ndarray
    provenance: SourceProvenance = field(default_factory=SourceProvenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compounds", tuple(self.compounds))
        object.__setattr__(self, "profiles", tuple(self.profiles))
        with np.errstate(over="raise", invalid="raise"):
            try:
                matrix = np.array(self.expression, dtype=np.float32, copy=True)
            except FloatingPointError as error:
                raise ValueError(
                    "Expression values must be finite and representable as float32"
                ) from error
        if matrix.shape != (len(self.profiles), len(self.genes)):
            raise ValueError("Expression shape must match profile count and ordered gene manifest")
        if not np.isfinite(matrix).all():
            raise ValueError("Expression profiles contain NaN or infinite values")
        matrix.setflags(write=False)
        object.__setattr__(self, "expression", matrix)
        compound_ids = [item.compound_id for item in self.compounds]
        profile_ids = [item.profile_id for item in self.profiles]
        if len(set(compound_ids)) != len(compound_ids):
            raise ValueError("Duplicate compound identifiers")
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("Duplicate profile identifiers")
        unknown = sorted(set(profile.compound_id for profile in self.profiles) - set(compound_ids))
        if unknown:
            raise ValueError(f"Profiles reference unknown compound identifiers: {unknown[:5]}")

    @property
    def split_labels(self) -> tuple[str, ...]:
        return tuple(profile.split for profile in self.profiles)

    @property
    def compound_lookup(self) -> dict[str, CompoundRecord]:
        return {compound.compound_id: compound for compound in self.compounds}

    def subset(self, split: str) -> ProfileDataset:
        indices = [i for i, profile in enumerate(self.profiles) if profile.split == split]
        profiles = tuple(self.profiles[i] for i in indices)
        ids = {profile.compound_id for profile in profiles}
        return replace(
            self,
            profiles=profiles,
            expression=self.expression[indices],
            compounds=tuple(compound for compound in self.compounds if compound.compound_id in ids),
        )
