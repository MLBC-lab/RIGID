"""Reproducible, group-preserving holdout and rotating-fold dataset protocols."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .chem.standardize import AcyclicPolicy
from .data import ProfileDataset, audit_leakage, load_dataset, save_dataset
from .data.splits import compound_groups
from .provenance import atomic_json, dataset_hashes, environment_report, sha256_file
from .serialization import read_json, require_object

_PROTECTED = frozenset({"external", "excluded", "calibration"})
_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitProtocol:
    mode: str = "repeated_holdout"
    strategy: str = "scaffold"
    seed: int = 17
    repetitions: int = 1
    folds: int = 1
    fractions: tuple[float, float, float] | None = None
    acyclic_policy: AcyclicPolicy = "group"

    def __post_init__(self) -> None:
        if self.mode not in ("repeated_holdout", "kfold"):
            raise ValueError("mode must be repeated_holdout or kfold")
        if self.strategy not in ("compound", "scaffold"):
            raise ValueError("strategy must be compound or scaffold")
        if self.acyclic_policy not in ("group", "compound", "exclude"):
            raise ValueError("acyclic_policy must be group, compound, or exclude")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("repetitions", "folds"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError(f"{name} must be an integer in [1, 10000]")
        if self.split_count > 10000:
            raise ValueError("A protocol cannot contain more than 10000 splits")
        if self.mode == "kfold":
            if self.folds < 3:
                raise ValueError("kfold requires at least three folds for train/validation/test")
            if self.fractions is not None:
                raise ValueError("kfold determines fractions from folds; omit fractions")
        else:
            if self.folds != 1:
                raise ValueError("repeated_holdout requires folds=1; use repetitions")
            values = (0.7, 0.15, 0.15) if self.fractions is None else self.fractions
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError("fractions must be three positive numbers summing to one")
            normalized = []
            for value in values:
                if type(value) not in (int, float):
                    raise ValueError("fractions must contain finite positive numbers")
                try:
                    number = float(value)
                except OverflowError as error:
                    raise ValueError("fractions must contain finite positive numbers") from error
                if not math.isfinite(number) or not 0 < number < 1:
                    raise ValueError("fractions must contain finite positive numbers below one")
                normalized.append(number)
            if not math.isclose(sum(normalized), 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError("fractions must sum to one")
            object.__setattr__(self, "fractions", tuple(normalized))

    @property
    def split_count(self) -> int:
        return self.repetitions * self.folds

    @classmethod
    def from_dict(cls, value: dict) -> SplitProtocol:
        value = require_object(value, context="split protocol")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown split protocol fields: {sorted(unknown)}")
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path) -> SplitProtocol:
        return cls.from_dict(read_json(path, maximum_bytes=1024 * 1024))

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.fractions is not None:
            value["fractions"] = list(self.fractions)
        return value


@dataclass(frozen=True)
class SplitAssignment:
    split_id: str
    repetition: int
    fold: int
    seed: int
    assignments: dict[str, str]
    group_assignments: dict[str, str]
    profile_counts: dict[str, int]
    group_counts: dict[str, int]

    @property
    def partition_sha256(self) -> str:
        """Identify profile assignments independently of seed and fold labels."""
        payload = json.dumps(self.assignments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {**asdict(self), "partition_sha256": self.partition_sha256}


def _group_plan(dataset: ProfileDataset, protocol: SplitProtocol) -> tuple[dict, dict]:
    groups = compound_groups(
        dataset.compounds,
        include_scaffold=protocol.strategy == "scaffold",
        acyclic_policy=protocol.acyclic_policy,
    )
    counts: dict[str, int] = defaultdict(int)
    reserved: dict[str, set[str]] = defaultdict(set)
    for profile in dataset.profiles:
        group = groups.get(profile.compound_id)
        if group is None or profile.split == "excluded":
            continue
        if profile.split in _PROTECTED:
            reserved[group].add(profile.split)
        else:
            counts[group] += 1
    conflicts = sorted(
        group for group, labels in reserved.items() if group in counts or len(labels) > 1
    )
    if conflicts:
        raise ValueError(
            "Protected calibration/external groups overlap another partition; "
            f"resolve the source assignments before splitting: {conflicts[:5]}"
        )
    required = protocol.folds if protocol.mode == "kfold" else 3
    if len(counts) < required:
        raise ValueError(
            f"Need at least {required} independent eligible groups; found {len(counts)}"
        )
    return groups, dict(counts)


def _allocate(counts: dict[str, int], proportions: tuple[float, ...], seed: int) -> dict[str, int]:
    keys = sorted(counts)
    rng = np.random.default_rng(seed)
    order = [keys[index] for index in rng.permutation(len(keys))]
    order.sort(key=lambda key: -counts[key])
    target = np.asarray(proportions) * sum(counts.values())
    assigned: NDArray[np.int64] = np.zeros(len(proportions), dtype=np.int64)
    result = {}
    for position, key in enumerate(order):
        empty = [index for index, count in enumerate(assigned) if count == 0]
        choices = empty if len(order) - position == len(empty) else range(len(proportions))
        bucket = max(choices, key=lambda index: target[index] - assigned[index])
        assigned[bucket] += counts[key]
        result[key] = bucket
    return result


def _iter_plan(
    dataset: ProfileDataset, protocol: SplitProtocol, groups: dict, counts: dict
) -> Iterator[SplitAssignment]:
    number = 0
    for repetition in range(protocol.repetitions):
        seed = (protocol.seed + repetition) % 2**32
        proportions = (
            tuple([1 / protocol.folds] * protocol.folds)
            if protocol.mode == "kfold"
            else protocol.fractions
        )
        assert proportions is not None
        allocation = _allocate(counts, proportions, seed)
        for fold in range(protocol.folds):
            number += 1
            group_assignments = {
                group: (
                    "test"
                    if bucket == fold
                    else "validation"
                    if bucket == (fold + 1) % protocol.folds
                    else "train"
                )
                if protocol.mode == "kfold"
                else _NAMES[bucket]
                for group, bucket in allocation.items()
            }
            assignments = {
                profile.profile_id: profile.split
                if profile.split in _PROTECTED
                else group_assignments.get(groups.get(profile.compound_id, ""), "excluded")
                for profile in dataset.profiles
            }
            yield SplitAssignment(
                split_id=f"split-{number:04d}",
                repetition=repetition + 1,
                fold=fold + 1,
                seed=seed,
                assignments=assignments,
                group_assignments=group_assignments,
                profile_counts=dict(sorted(Counter(assignments.values()).items())),
                group_counts=dict(sorted(Counter(group_assignments.values()).items())),
            )


def plan_splits(dataset: ProfileDataset, protocol: SplitProtocol) -> Iterator[SplitAssignment]:
    """Yield deterministic assignments without modifying data or fitting models."""
    if not isinstance(protocol, SplitProtocol):
        raise ValueError("protocol must be a SplitProtocol")
    groups, counts = _group_plan(dataset, protocol)
    yield from _iter_plan(dataset, protocol, groups, counts)


def materialize_splits(
    dataset_path: str | Path,
    output: str | Path,
    protocol: SplitProtocol,
    *,
    max_splits: int = 100,
    max_profiles: int | None = None,
) -> dict:
    """Write numbered, immutable datasets and publish a manifest after all succeed.

    The output path must not exist. A failed operation can leave partial datasets;
    without the final manifest the protocol is incomplete and must not be consumed.
    """
    if not isinstance(protocol, SplitProtocol):
        raise ValueError("protocol must be a SplitProtocol")
    if type(max_splits) is not int or not 1 <= max_splits <= 10000:
        raise ValueError("max_splits must be an integer in [1, 10000]")
    if protocol.split_count > max_splits:
        raise ValueError(
            f"Protocol requests {protocol.split_count} splits, exceeding max_splits={max_splits}"
        )
    source = Path(dataset_path).resolve()
    destination = Path(output).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Protocol output must be outside the source dataset directory")
    if destination.exists():
        raise FileExistsError("Protocol output must be a new directory")
    input_hashes = dataset_hashes(source)
    dataset = load_dataset(source, max_profiles=max_profiles)
    if dataset_hashes(source) != input_hashes:
        raise ValueError("Source dataset changed while loading")
    groups, counts = _group_plan(dataset, protocol)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": "split-protocol",
        "configuration": protocol.to_dict(),
        "split_count": protocol.split_count,
        "source_kind": dataset.provenance.kind,
        "gene_manifest_sha256": dataset.genes.sha256,
        "input_sha256": input_hashes,
        "software": environment_report(),
        "preserved_partitions": sorted(_PROTECTED),
        "compound_groups": dict(sorted(groups.items())),
        "eligible_group_profile_counts": dict(sorted(counts.items())),
        "excluded_by_acyclic_policy": sorted(
            profile.profile_id
            for profile in dataset.profiles
            if profile.split not in _PROTECTED and profile.compound_id not in groups
        ),
        "splits": [],
        "notes": [
            "Settings define a prospective protocol; no model is trained by materialization.",
            "Assignment uses identity/scaffold groups and profile counts, not expression or MOA.",
            "Fractions are approximate because groups are indivisible.",
            "K-fold rotates one test and the following validation fold per repetition.",
            "Repeated holdouts may overlap and are not independent experimental observations.",
            "Different seeds may yield identical partitions; inspect unique_assignment_count and duplicate_assignments.",
            "Only a final manifest.json marks the materialization complete.",
        ],
    }
    partition_members: dict[str, list[str]] = defaultdict(list)
    for assignment in _iter_plan(dataset, protocol, groups, counts):
        description = (
            f"split protocol {assignment.split_id}; repetition={assignment.repetition}; "
            f"fold={assignment.fold}; seed={assignment.seed}; strategy={protocol.strategy}"
        )
        partitioned = replace(
            dataset,
            profiles=tuple(
                replace(profile, split=assignment.assignments[profile.profile_id])
                for profile in dataset.profiles
            ),
            provenance=replace(
                dataset.provenance,
                transformations=(*dataset.provenance.transformations, description),
            ),
        )
        audit = audit_leakage(
            partitioned,
            scaffold=protocol.strategy == "scaffold",
            acyclic_policy=protocol.acyclic_policy,
        )
        audit.raise_for_leakage()
        split_path = destination / assignment.split_id
        save_dataset(partitioned, split_path)
        record = {
            **assignment.to_dict(),
            "dataset_sha256": dataset_hashes(split_path),
            "leakage_audit": audit.to_dict(),
        }
        atomic_json(split_path / "assignment.json", record, overwrite=False)
        manifest["splits"].append(
            {
                "split_id": assignment.split_id,
                "repetition": assignment.repetition,
                "fold": assignment.fold,
                "seed": assignment.seed,
                "partition_sha256": assignment.partition_sha256,
                "profile_counts": assignment.profile_counts,
                "group_counts": assignment.group_counts,
                "assignment_sha256": sha256_file(split_path / "assignment.json"),
                "files_sha256": dataset_hashes(split_path),
            }
        )
        partition_members[assignment.partition_sha256].append(assignment.split_id)
    manifest["unique_assignment_count"] = len(partition_members)
    manifest["duplicate_assignments"] = [
        {"partition_sha256": digest, "split_ids": members}
        for digest, members in partition_members.items()
        if len(members) > 1
    ]
    manifest["warnings"] = (
        [
            f"Only {len(partition_members)} distinct profile partitions occur across "
            f"{protocol.split_count} materialized splits. Identical partitions are not "
            "independent holdouts; changing a seed does not by itself change the assignments."
        ]
        if manifest["duplicate_assignments"]
        else []
    )
    atomic_json(destination / "manifest.json", manifest, overwrite=False)
    return manifest
