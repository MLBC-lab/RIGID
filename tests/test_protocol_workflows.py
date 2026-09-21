"""Command-to-training checks for configurable experimental partitions."""

import json

import numpy as np
import pytest

from rigid.cli import main
from rigid.config import TrainConfig
from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    load_dataset,
    save_dataset,
)
from rigid.pipeline import screen_dataset
from rigid.protocols import SplitProtocol, materialize_splits
from rigid.provenance import atomic_json
from rigid.training import train


def acyclic_data():
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "CCBr", "CCF", "COC"]
    compounds = tuple(CompoundRecord(f"c{i}", value) for i, value in enumerate(smiles))
    profiles = tuple(
        ProfileRecord(f"p{i}", item.compound_id, "cell", 1, 24) for i, item in enumerate(compounds)
    )
    return ProfileDataset(
        GeneManifest(("g1", "g2", "g3", "g4")),
        compounds,
        profiles,
        np.random.default_rng(3).normal(size=(9, 4)),
    )


def test_cli_fold_protocol_materializes_declared_number(tmp_path, capsys):
    save_dataset(acyclic_data(), tmp_path / "data")
    protocol = SplitProtocol(mode="kfold", strategy="compound", folds=3, repetitions=2)
    atomic_json(tmp_path / "protocol.json", protocol.to_dict())
    assert (
        main(
            [
                "plan-splits",
                str(tmp_path / "data"),
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--output",
                str(tmp_path / "plan"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["manifest"]["configuration"]["folds"] == 3
    assert len(list((tmp_path / "plan").glob("split-*/assignment.json"))) == 6
    assert (
        main(
            [
                "plan-splits",
                str(tmp_path / "data"),
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--output",
                str(tmp_path / "plan"),
            ]
        )
        == 2
    )
    capsys.readouterr()


def test_acyclic_compound_policy_survives_training_and_screening(tmp_path):
    save_dataset(acyclic_data(), tmp_path / "data")
    materialize_splits(
        tmp_path / "data",
        tmp_path / "plan",
        SplitProtocol(strategy="scaffold", acyclic_policy="compound"),
    )
    dataset_path = tmp_path / "plan/split-0001"
    config = TrainConfig(
        epochs=1,
        forward_epochs=1,
        hidden_dim=16,
        depth=2,
        dropout=0,
        candidate_pool_size=4,
        acyclic_policy="compound",
    )
    train(dataset_path, tmp_path / "run", config, scaffold_holdout=True)
    result = screen_dataset(tmp_path / "run/model.pt", dataset_path, tmp_path / "screen.json", k=3)
    assert result["metrics"]["target_count"] > 0
    with pytest.raises(ValueError, match="Leakage"):
        train(
            dataset_path,
            tmp_path / "wrong",
            TrainConfig(epochs=1, forward_epochs=1, hidden_dim=16, acyclic_policy="group"),
            scaffold_holdout=True,
        )


def test_simple_split_exposes_fractions_and_acyclic_policy(tmp_path, capsys):
    save_dataset(acyclic_data(), tmp_path / "data")
    assert (
        main(
            [
                "split",
                str(tmp_path / "data"),
                "--output",
                str(tmp_path / "split"),
                "--strategy",
                "scaffold",
                "--acyclic-policy",
                "compound",
                "--fractions",
                "0.5",
                "0.25",
                "0.25",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assigned = load_dataset(tmp_path / "split")
    assert set(assigned.split_labels) == {"train", "validation", "test"}
    assert (
        main(["validate", str(tmp_path / "split"), "--scaffold", "--acyclic-policy", "compound"])
        == 0
    )
    capsys.readouterr()
