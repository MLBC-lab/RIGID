"""Group-preserving splits and explicit cross-partition leakage reports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace

import numpy as np

from rigid.chem.standardize import AcyclicPolicy, scaffold_key, standardize_smiles

from .schema import CompoundRecord, ProfileDataset


class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent: dict[str, str] = {identifier: identifier for identifier in ids}

    def find(self, identifier: str) -> str:
        while self.parent[identifier] != identifier:
            self.parent[identifier] = self.parent[self.parent[identifier]]
            identifier = self.parent[identifier]
        return identifier

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def _identity_keys(compound: CompoundRecord) -> dict[str, tuple[str, ...]]:
    normalized = standardize_smiles(compound.smiles)
    return {
        "compound_id": (compound.compound_id,),
        "alias": tuple(
            value.strip().casefold() for value in (compound.compound_id, *compound.aliases)
        ),
        "canonical_structure": (normalized.canonical_smiles,),
        "connectivity": (normalized.connectivity_key,),
    }


def compound_groups(
    compounds: tuple[CompoundRecord, ...],
    *,
    include_scaffold: bool = False,
    acyclic_policy: AcyclicPolicy = "group",
) -> dict[str, str]:
    union = _UnionFind([compound.compound_id for compound in compounds])
    first: dict[tuple[str, str], str] = {}
    excluded: set[str] = set()
    for compound in sorted(compounds, key=lambda item: item.compound_id):
        keys = _identity_keys(compound)
        if include_scaffold:
            key = scaffold_key(compound.smiles, acyclic_policy)
            if key is None:
                excluded.add(compound.compound_id)
            else:
                keys["scaffold"] = (key,)
        for kind, values in keys.items():
            for value in values:
                pair = (kind, value)
                if pair in first:
                    union.union(compound.compound_id, first[pair])
                else:
                    first[pair] = compound.compound_id
    excluded_groups = {union.find(identifier) for identifier in excluded}
    return {
        compound.compound_id: union.find(compound.compound_id)
        for compound in compounds
        if union.find(compound.compound_id) not in excluded_groups
    }


def assign_splits(
    dataset: ProfileDataset,
    *,
    strategy: str = "scaffold",
    seed: int = 17,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    acyclic_policy: AcyclicPolicy = "group",
) -> ProfileDataset:
    """Assign whole identity/scaffold groups; requested fractions are approximate.

    Balancing uses profile counts without reading expression values or MOA labels.
    Existing assignments are replaced. This is intentionally explicit at the API.
    """
    if strategy not in {"compound", "scaffold"}:
        raise ValueError("Split strategy must be compound or scaffold")
    proportions = np.asarray(fractions, dtype=float)
    if (
        proportions.shape != (3,)
        or not np.isfinite(proportions).all()
        or (proportions < 0).any()
        or not np.isclose(proportions.sum(), 1)
    ):
        raise ValueError("Split fractions must be three nonnegative values summing to one")
    groups = compound_groups(
        dataset.compounds, include_scaffold=strategy == "scaffold", acyclic_policy=acyclic_policy
    )
    group_counts: dict[str, int] = defaultdict(int)
    for profile in dataset.profiles:
        if profile.compound_id in groups:
            group_counts[groups[profile.compound_id]] += 1
    active = [index for index, fraction in enumerate(fractions) if fraction > 0]
    if len(group_counts) < len(active):
        raise ValueError(
            f"Need at least {len(active)} independent groups for nonempty requested splits; found {len(group_counts)}"
        )
    rng = np.random.default_rng(seed)
    keys = sorted(group_counts)
    shuffled = [keys[index] for index in rng.permutation(len(keys))]
    shuffled.sort(key=lambda key: -group_counts[key])
    targets = proportions * sum(group_counts.values())
    assigned_counts = np.zeros(3)
    group_splits = {}
    names = ("train", "validation", "test")
    for position, key in enumerate(shuffled):
        empty = [index for index in active if assigned_counts[index] == 0]
        remaining = len(shuffled) - position
        if remaining == len(empty):
            selected = max(empty, key=lambda index: targets[index])
        else:
            selected = max(active, key=lambda index: targets[index] - assigned_counts[index])
        assigned_counts[selected] += group_counts[key]
        group_splits[key] = names[selected]
    profiles = tuple(
        replace(
            profile,
            split=group_splits[groups[profile.compound_id]]
            if profile.compound_id in groups
            else "excluded",
        )
        for profile in dataset.profiles
    )
    description = f"{strategy} split; seed={seed}; fractions={tuple(fractions)}; acyclic_policy={acyclic_policy}"
    return replace(
        dataset,
        profiles=profiles,
        provenance=replace(
            dataset.provenance, transformations=(*dataset.provenance.transformations, description)
        ),
    )


@dataclass(frozen=True)
class LeakageViolation:
    kind: str
    key: str
    splits: tuple[str, ...]
    compound_ids: tuple[str, ...]


@dataclass(frozen=True)
class LeakageReport:
    violations: tuple[LeakageViolation, ...]
    unassigned_profile_ids: tuple[str, ...]
    split_counts: dict[str, int]
    scaffold_checked: bool
    acyclic_policy: str

    @property
    def ok(self) -> bool:
        return not self.violations and not self.unassigned_profile_ids

    def to_dict(self) -> dict:
        return {"ok": self.ok, **asdict(self)}

    def raise_for_leakage(self) -> None:
        if not self.ok:
            raise ValueError(
                f"Leakage audit failed: {len(self.violations)} overlaps and {len(self.unassigned_profile_ids)} unassigned profiles"
            )


def audit_leakage(
    dataset: ProfileDataset, *, scaffold: bool = True, acyclic_policy: AcyclicPolicy = "group"
) -> LeakageReport:
    occurrences: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    compound_lookup = dataset.compound_lookup
    cached = {}
    counts: dict[str, int] = defaultdict(int)
    unassigned = []
    for profile in dataset.profiles:
        counts[profile.split or "unassigned"] += 1
        if not profile.split:
            unassigned.append(profile.profile_id)
            continue
        if profile.split == "excluded":
            continue
        if profile.compound_id not in cached:
            compound = compound_lookup[profile.compound_id]
            keys = _identity_keys(compound)
            if scaffold:
                value = scaffold_key(compound.smiles, acyclic_policy)
                if value is not None:
                    keys["scaffold"] = (value,)
            cached[profile.compound_id] = keys
        for kind, values in cached[profile.compound_id].items():
            for value in values:
                occurrences[(kind, value)][profile.split].add(profile.compound_id)
    violations = []
    for (kind, key), split_compounds in sorted(occurrences.items()):
        if len(split_compounds) > 1:
            violations.append(
                LeakageViolation(
                    kind,
                    key,
                    tuple(sorted(split_compounds)),
                    tuple(sorted(set().union(*split_compounds.values()))),
                )
            )
    return LeakageReport(
        tuple(violations), tuple(unassigned), dict(sorted(counts.items())), scaffold, acyclic_policy
    )
