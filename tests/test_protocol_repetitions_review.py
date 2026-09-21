"""Repeated allocation attempts must disclose identical profile partitions."""

import json
from dataclasses import replace

import numpy as np
import pytest

from rigid.data import CompoundRecord, GeneManifest, ProfileDataset, ProfileRecord, save_dataset
from rigid.protocols import SplitProtocol, materialize_splits, plan_splits


def _unequal_groups():
    compounds = tuple(
        CompoundRecord(str(index), smiles)
        for index, smiles in enumerate(("CC", "CO", "CCC", "CCCC", "CCO", "CCN"))
    )
    profiles = tuple(
        ProfileRecord(f"{index}-{replicate}", compound.compound_id, "cell", 1, 24)
        for index, compound in enumerate(compounds)
        for replicate in range(index + 1)
    )
    return ProfileDataset(
        GeneManifest(("g1", "g2"), "synthetic"),
        compounds,
        profiles,
        np.arange(len(profiles) * 2).reshape(-1, 2),
    )


def test_partition_hash_depends_only_on_profile_assignments():
    assignment = next(plan_splits(_unequal_groups(), SplitProtocol(strategy="compound")))
    relabeled = replace(
        assignment,
        seed=999,
        repetition=3,
        fold=9,
        split_id="another-name",
        assignments=dict(reversed(list(assignment.assignments.items()))),
    )
    assert assignment.partition_sha256 == relabeled.partition_sha256
    assert len(assignment.partition_sha256) == 64
    assert assignment.to_dict()["partition_sha256"] == assignment.partition_sha256
    modified = dict(assignment.assignments)
    key = next(iter(modified))
    modified[key] = "external"
    assert assignment.partition_sha256 != replace(assignment, assignments=modified).partition_sha256


@pytest.mark.parametrize(
    "mode,folds,expected_unique", [("repeated_holdout", 1, 1), ("kfold", 3, 3)]
)
def test_unequal_group_counts_disclose_duplicate_seeded_allocations(
    tmp_path, mode, folds, expected_unique
):
    source, output = tmp_path / "source", tmp_path / "output"
    save_dataset(_unequal_groups(), source)
    protocol = SplitProtocol(strategy="compound", mode=mode, folds=folds, repetitions=3)
    result = materialize_splits(source, output, protocol)
    assert result["split_count"] == 3 * folds
    assert result["unique_assignment_count"] == expected_unique
    assert len(result["duplicate_assignments"]) == expected_unique
    assert result["warnings"] and "not independent holdouts" in result["warnings"][0]
    assert result == json.loads((output / "manifest.json").read_text())
    hashes = {}
    for entry in result["splits"]:
        record = json.loads((output / entry["split_id"] / "assignment.json").read_text())
        assert record["partition_sha256"] == entry["partition_sha256"]
        hashes[entry["split_id"]] = entry["partition_sha256"]
    for duplicate in result["duplicate_assignments"]:
        assert len(duplicate["split_ids"]) == 3
        assert {hashes[split_id] for split_id in duplicate["split_ids"]} == {
            duplicate["partition_sha256"]
        }


def test_single_assignment_has_no_duplicate_warning(tmp_path):
    source = tmp_path / "source"
    save_dataset(_unequal_groups(), source)
    result = materialize_splits(source, tmp_path / "output", SplitProtocol(strategy="compound"))
    assert result["unique_assignment_count"] == 1
    assert result["duplicate_assignments"] == []
    assert result["warnings"] == []
