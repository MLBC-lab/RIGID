"""Verify the saved-generator to hybrid-ranking example as an executable workflow."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")

from rigid.config import TrainConfig  # noqa: E402
from rigid.data import (  # noqa: E402
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    SourceProvenance,
    save_dataset,
)
from rigid.experiments import train_generator  # noqa: E402
from rigid.training import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "design_candidates.py"


def example_module():
    spec = importlib.util.spec_from_file_location("rigid_design_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_pool_canonicalizes_deduplicates_and_removes_evaluation_labels():
    example = example_module()
    compounds = [
        CompoundRecord("a", "CCO", ("do-not-use",)),
        CompoundRecord("alias", "OCC", ("also-do-not-use",)),
        CompoundRecord("b", "CCC"),
    ]

    def proposal(smiles, valid=True):
        return {
            "smiles": smiles,
            "valid": valid,
            "terminated": valid,
            "steps": 4,
            "log_probability": -2.0,
            "rejection_reason": None if valid else "max_steps",
        }

    report = {"proposals": [proposal("OCC"), proposal("CN"), proposal("NC"), proposal(None, False)]}
    candidates, origins, accounting = example._hybrid_candidates(compounds, report)
    assert {item.canonical_smiles for item in candidates} == {"CCO", "CCC", "CN"}
    assert all(item.moa == () for item in candidates)
    assert origins["a"] == "library+generated"
    assert accounting["library_duplicate_groups"] == [["a", "alias"]]
    assert accounting["generated_unique_structures"] == 2
    assert accounting["generated_valid_after_standardization"] == 3
    assert accounting["generated_new_to_library"] == 1


@pytest.mark.integration
def test_design_script_trains_loads_and_ranks_with_synthetic_provenance(tmp_path):
    compounds = tuple(
        CompoundRecord(cid, smiles, ("synthetic_label",))
        for cid, smiles in (("a", "CC"), ("b", "CO"), ("c", "CCC"), ("d", "CCCC"))
    )
    splits = {"a": "train", "b": "train", "c": "validation", "d": "test"}
    profiles = tuple(
        ProfileRecord(
            f"{compound.compound_id}_{replicate}",
            compound.compound_id,
            "SYNTH_CELL",
            1,
            24,
            splits[compound.compound_id],
        )
        for compound in compounds
        for replicate in range(2)
    )
    dataset = ProfileDataset(
        GeneManifest(("g1", "g2", "g3", "g4"), "synthetic"),
        compounds,
        profiles,
        np.random.default_rng(88).normal(size=(8, 4)),
        SourceProvenance(kind="synthetic"),
    )
    data_path = tmp_path / "data"
    save_dataset(dataset, data_path)
    config = TrainConfig(
        epochs=1,
        forward_epochs=1,
        hidden_dim=8,
        depth=1,
        dropout=0,
        batch_size=2,
        candidate_pool_size=3,
        num_threads=1,
    )
    train(data_path, tmp_path / "ranker", config)
    train_generator(data_path, tmp_path / "generator", config, max_atoms=6)
    output = tmp_path / "design.json"
    arguments = [
        sys.executable,
        str(EXAMPLE),
        "--dataset",
        str(data_path),
        "--ranker",
        str(tmp_path / "ranker" / "model.pt"),
        "--generator",
        str(tmp_path / "generator" / "model.pt"),
        "--target-id",
        "d_0",
        "--output",
        str(output),
        "--count",
        "4",
        "--top-k",
        "3",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        arguments, capture_output=True, text=True, env=environment, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["source_kind"] == "synthetic" and result["contains_synthetic_data"]
    assert result["provenance"]["source_kinds"] == {
        "input": "synthetic",
        "ranker_training": "synthetic",
        "generator_training": "synthetic",
    }
    assert result["generation"]["attempted"] == 4
    assert len(result["generation"]["proposals"]) == 4
    assert result["candidate_pool"]["hybrid_candidate_count"] >= 4
    assert len(result["rankings"]) == 3
    assert all("origin" in row and "score" in row for row in result["rankings"])
    assert result["reproduces_manuscript"] is False
    original = output.read_bytes()
    repeated = subprocess.run(
        arguments, capture_output=True, text=True, env=environment, timeout=60
    )
    assert repeated.returncode == 2 and "Output exists" in repeated.stderr
    assert output.read_bytes() == original
