"""Partition protocols preserve scientific isolation and never overwrite artifacts."""

import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    audit_leakage,
    load_dataset,
    save_dataset,
)
from rigid.protocols import SplitProtocol, materialize_splits, plan_splits
from rigid.provenance import dataset_hashes, sha256_file


def dataset_with_groups():
    smiles = (
        "CCO",
        "CCN",
        "CCC",
        "CCCC",
        "CCCl",
        "CCBr",
        "CCF",
        "CCS",
        "CCCO",
        "CCCN",
        "c1ccccc1",
        "C1CCCCC1",
        "c1ccncc1",
        "C1CCCC1",
        "C1CC1",
        "C1CCC1",
        "C1CCNCC1",
        "C1CCOC1",
    )
    compounds = tuple(CompoundRecord(f"c{i}", value) for i, value in enumerate(smiles))
    profiles = tuple(
        ProfileRecord(f"p{i}", compound.compound_id, "cell", 1, 24)
        for i, compound in enumerate(compounds)
    )
    return ProfileDataset(
        GeneManifest(("g1", "g2")),
        compounds,
        profiles,
        np.arange(len(profiles) * 2).reshape(-1, 2),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "fold"},
        {"strategy": "random"},
        {"acyclic_policy": "ignore"},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 17.0},
        {"repetitions": True},
        {"repetitions": 0},
        {"repetitions": 10001},
        {"folds": False},
        {"folds": 2},
        {"mode": "kfold", "folds": 2},
        {"mode": "kfold", "folds": 3, "fractions": [0.7, 0.15, 0.15]},
        {"mode": "kfold", "folds": 101, "repetitions": 100},
        {"fractions": "0.7,0.15,0.15"},
        {"fractions": [0.8, 0.2]},
        {"fractions": [0.8, 0.2, 0]},
        {"fractions": [False, 0.5, 0.5]},
        {"fractions": [float("nan"), 0.5, 0.5]},
        {"fractions": [float("inf"), 0.5, 0.5]},
        {"fractions": [10**1000, 0.1, 0.1]},
        {"fractions": [0.5, 0.2, 0.2]},
    ],
)
def test_strict_configuration(changes):
    with pytest.raises(ValueError):
        SplitProtocol(**changes)


def test_configuration_roundtrip_and_strict_json(tmp_path):
    protocol = SplitProtocol(fractions=[0.6, 0.2, 0.2], repetitions=3)
    assert protocol.fractions == (0.6, 0.2, 0.2)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol.to_dict()), encoding="utf-8")
    assert SplitProtocol.load(path) == protocol
    assert protocol.split_count == 3
    with pytest.raises(ValueError, match="Unknown"):
        SplitProtocol.from_dict({"seeds": 17})
    with pytest.raises(ValueError, match="object"):
        SplitProtocol.from_dict([])
    path.write_text('{"seed": 17, "seed": 18}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        SplitProtocol.load(path)


def test_holdout_is_deterministic_nonempty_and_seeded():
    dataset = dataset_with_groups()
    protocol = SplitProtocol(strategy="compound", repetitions=3, seed=2**32 - 1)
    plan = list(plan_splits(dataset, protocol))
    assert plan == list(plan_splits(dataset, protocol))
    assert [item.seed for item in plan] == [2**32 - 1, 0, 1]
    assert [item.split_id for item in plan] == ["split-0001", "split-0002", "split-0003"]
    assert len({tuple(sorted(item.assignments.items())) for item in plan}) == 3
    for item in plan:
        assert set(item.profile_counts) == {"train", "validation", "test"}
        assert sum(item.profile_counts.values()) == len(dataset.profiles)
    assert set(dataset.split_labels) == {""}


def test_folds_rotate_every_group_through_test_and_validation():
    protocol = SplitProtocol(mode="kfold", strategy="compound", folds=5, repetitions=2)
    plan = list(plan_splits(dataset_with_groups(), protocol))
    assert len(plan) == 10
    for repetition in (1, 2):
        local = [item for item in plan if item.repetition == repetition]
        assert len({item.seed for item in local}) == 1
        for profile in local[0].assignments:
            labels = Counter(item.assignments[profile] for item in local)
            assert labels == {"train": 3, "validation": 1, "test": 1}


def test_no_expression_moa_or_input_order_dependence():
    dataset = dataset_with_groups()
    changed = replace(
        dataset,
        compounds=tuple(replace(compound, moa=("different",)) for compound in dataset.compounds),
        profiles=tuple(reversed(dataset.profiles)),
        expression=-dataset.expression[::-1],
    )
    protocol = SplitProtocol(strategy="compound", repetitions=2)
    original = list(plan_splits(dataset, protocol))
    altered = list(plan_splits(changed, protocol))
    assert original == altered


def test_aliases_connectivity_and_scaffolds_stay_together():
    dataset = dataset_with_groups()
    additions = (
        CompoundRecord("alias", "CC=O", aliases=("c0",)),
        CompoundRecord("same", "OCC"),
        CompoundRecord("ring", "Cc1ccccc1"),
    )
    extra_profiles = tuple(
        ProfileRecord(f"extra-{compound.compound_id}", compound.compound_id, "cell", 2, 48)
        for compound in additions
    )
    dataset = replace(
        dataset,
        compounds=(*dataset.compounds, *additions),
        profiles=(*dataset.profiles, *extra_profiles),
        expression=np.concatenate((dataset.expression, np.zeros((3, 2)))),
    )
    for assignment in plan_splits(dataset, SplitProtocol(repetitions=4)):
        labels = assignment.assignments
        assert labels["p0"] == labels["extra-alias"] == labels["extra-same"]
        assert labels["p10"] == labels["extra-ring"]


def test_protected_partitions_are_preserved(tmp_path):
    dataset = dataset_with_groups()
    profiles = tuple(
        replace(profile, split={0: "external", 1: "excluded", 2: "calibration"}.get(i, "test"))
        for i, profile in enumerate(dataset.profiles)
    )
    dataset = replace(dataset, profiles=profiles)
    source = tmp_path / "source"
    save_dataset(dataset, source)
    before = dataset_hashes(source)
    manifest = materialize_splits(
        source,
        tmp_path / "protocol",
        SplitProtocol(strategy="compound", repetitions=2),
    )
    for entry in manifest["splits"]:
        destination = tmp_path / "protocol" / entry["split_id"]
        generated = load_dataset(destination)
        assert generated.split_labels[:3] == ("external", "excluded", "calibration")
        np.testing.assert_array_equal(dataset.expression, generated.expression)
        assert generated.genes == dataset.genes
        assert audit_leakage(generated, scaffold=False).ok
        assert sha256_file(destination / "assignment.json") == entry["assignment_sha256"]
        assert dataset_hashes(destination) == entry["files_sha256"]
        assignment = json.loads((destination / "assignment.json").read_text())
        assert assignment["assignments"] == {
            profile.profile_id: profile.split for profile in generated.profiles
        }
    assert dataset_hashes(source) == before
    assert manifest == json.loads((tmp_path / "protocol" / "manifest.json").read_text())


def test_protected_group_conflicts_rejected_before_creation(tmp_path):
    dataset = dataset_with_groups()
    dataset = replace(
        dataset,
        profiles=(replace(dataset.profiles[0], split="external"), *dataset.profiles[1:]),
    )
    source = tmp_path / "source"
    save_dataset(dataset, source)
    # All acyclic molecules are one scaffold group, including the external row.
    with pytest.raises(ValueError, match="Protected"):
        materialize_splits(source, tmp_path / "output", SplitProtocol())
    assert not (tmp_path / "output").exists()


def test_acyclic_exclusion_does_not_change_explicit_protected_labels(tmp_path):
    dataset = dataset_with_groups()
    dataset = replace(
        dataset,
        profiles=(replace(dataset.profiles[0], split="external"), *dataset.profiles[1:]),
    )
    source = tmp_path / "source"
    save_dataset(dataset, source)
    manifest = materialize_splits(
        source, tmp_path / "output", SplitProtocol(acyclic_policy="exclude")
    )
    result = load_dataset(tmp_path / "output" / "split-0001")
    assert result.split_labels[0] == "external"
    assert result.split_labels[1:10] == ("excluded",) * 9
    assert manifest["excluded_by_acyclic_policy"] == sorted(f"p{i}" for i in range(1, 10))


def test_too_few_groups_and_invalid_protocol_rejected():
    with pytest.raises(ValueError, match="independent"):
        list(plan_splits(dataset_with_groups(), SplitProtocol(mode="kfold", folds=20)))
    with pytest.raises(ValueError, match="SplitProtocol"):
        list(plan_splits(dataset_with_groups(), {}))


@pytest.mark.parametrize("budget", [True, 0, 10001, 1.5])
def test_invalid_materialization_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="max_splits"):
        materialize_splits("missing", tmp_path / "out", SplitProtocol(), max_splits=budget)


def test_materialization_budget_and_prohibited_destinations(tmp_path):
    source = tmp_path / "source"
    save_dataset(dataset_with_groups(), source)
    with pytest.raises(ValueError, match="exceeding"):
        materialize_splits(source, tmp_path / "out", SplitProtocol(repetitions=101))
    with pytest.raises(ValueError, match="outside"):
        materialize_splits(source, source / "nested", SplitProtocol())
    with pytest.raises(ValueError, match="outside"):
        materialize_splits(source, source, SplitProtocol())
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="new directory"):
        materialize_splits(source, destination, SplitProtocol())
    assert list(destination.iterdir()) == []


def test_materialization_respects_input_limit_and_protocol_type(tmp_path):
    source = tmp_path / "source"
    save_dataset(dataset_with_groups(), source)
    with pytest.raises(ValueError, match="Profile limit"):
        materialize_splits(source, tmp_path / "out", SplitProtocol(), max_profiles=2)
    with pytest.raises(ValueError, match="SplitProtocol"):
        materialize_splits(source, tmp_path / "out", {})


def test_large_group_allocation_still_keeps_all_partitions_nonempty():
    dataset = dataset_with_groups()
    profiles = (
        tuple(ProfileRecord(f"rep-{i}", "c0", "cell", 1, 24) for i in range(100)) + dataset.profiles
    )
    dataset = replace(dataset, profiles=profiles, expression=np.zeros((len(profiles), 2)))
    assignment = next(plan_splits(dataset, SplitProtocol(strategy="compound")))
    assert all(assignment.profile_counts[name] > 0 for name in ("train", "validation", "test"))


def test_source_mutation_during_load_is_detected(tmp_path, monkeypatch):
    import rigid.protocols as protocols

    source = tmp_path / "source"
    save_dataset(dataset_with_groups(), source)
    original = protocols.load_dataset

    def mutating_load(path, **kwargs):
        dataset = original(path, **kwargs)
        (path / "new-file.txt").write_text("changed", encoding="utf-8")
        return dataset

    monkeypatch.setattr(protocols, "load_dataset", mutating_load)
    with pytest.raises(ValueError, match="changed while loading"):
        materialize_splits(source, tmp_path / "out", SplitProtocol())
    assert not (tmp_path / "out").exists()


def test_failed_materialization_has_no_completion_manifest(tmp_path, monkeypatch):
    import rigid.protocols as protocols

    source = tmp_path / "source"
    save_dataset(dataset_with_groups(), source)
    original = protocols.save_dataset
    calls = 0

    def fail_second(dataset, output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk failure")
        return original(dataset, output)

    monkeypatch.setattr(protocols, "save_dataset", fail_second)
    with pytest.raises(OSError, match="disk failure"):
        materialize_splits(source, tmp_path / "out", SplitProtocol(repetitions=2))
    assert (tmp_path / "out" / "split-0001" / "assignment.json").is_file()
    assert not (tmp_path / "out" / "manifest.json").exists()
