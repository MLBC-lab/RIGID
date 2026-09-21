"""End-to-end guarantees for training and held-out screening."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from rigid.checkpoint import load_checkpoint
from rigid.config import TrainConfig
from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileRecord,
    make_synthetic_dataset,
    save_dataset,
)
from rigid.pipeline import RIGIDPipeline, screen_dataset
from rigid.training import train


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    dataset = make_synthetic_dataset(root / "data", n_genes=8)
    config = TrainConfig(
        epochs=1,
        forward_epochs=1,
        hidden_dim=16,
        batch_size=8,
        candidate_pool_size=8,
        depth=2,
        dropout=0,
        num_threads=1,
    )
    train(root / "data", root / "run", config, scaffold_holdout=True)
    return root, dataset, config


@pytest.mark.integration
def test_test_data_cannot_change_trained_parameters_or_validation_losses(trained_bundle):
    root, dataset, config = trained_bundle
    expression = dataset.expression.copy()
    expression[np.array(dataset.split_labels) == "test"] += 1000
    extra_compound = CompoundRecord(
        "ADDED_HELDOUT",
        "C1CCCCCCCC1",
        moa=tuple({label for compound in dataset.compounds for label in compound.moa}),
    )
    extra_profile = ProfileRecord("ADDED_PROFILE", "ADDED_HELDOUT", "UNSEEN_CELL", 50, 72, "test")
    altered = replace(
        dataset,
        compounds=(*dataset.compounds, extra_compound),
        profiles=(extra_profile, *dataset.profiles),
        expression=np.vstack([np.full(len(dataset.genes), 1234), expression]),
    )
    save_dataset(altered, root / "altered-data")
    train(root / "altered-data", root / "altered-run", config, scaffold_holdout=True)
    original = load_checkpoint(root / "run/model.pt")
    changed = load_checkpoint(root / "altered-run/model.pt")
    assert original["normalizer"] == changed["normalizer"]
    assert original["context_cells"] == changed["context_cells"]
    for name, record in original["models"].items():
        for parameter, value in record["state_dict"].items():
            torch.testing.assert_close(
                value, changed["models"][name]["state_dict"][parameter], rtol=0, atol=0
            )
    first_history = json.loads((root / "run/history.json").read_text())
    second_history = json.loads((root / "altered-run/history.json").read_text())
    assert [(entry["train_loss"], entry["validation_loss"]) for entry in first_history] == [
        (entry["train_loss"], entry["validation_loss"]) for entry in second_history
    ]


def test_pipeline_load_and_gene_order_refusal(trained_bundle):
    root, dataset, _ = trained_bundle
    pipeline = RIGIDPipeline.load(root / "run/model.pt")
    index = next(i for i, profile in enumerate(dataset.profiles) if profile.split == "test")
    with pytest.raises(ValueError, match="order"):
        pipeline.rank(
            dataset.expression[index],
            GeneManifest(tuple(reversed(dataset.genes.gene_ids)), dataset.genes.namespace),
            dataset.profiles[index],
            list(dataset.compounds),
            k=5,
        )
    assert pipeline.training_source_kind == "synthetic"


def test_ranking_is_batch_invariant_and_label_independent(trained_bundle):
    root, dataset, _ = trained_bundle
    pipeline = RIGIDPipeline.load(root / "run/model.pt")
    index = next(i for i, profile in enumerate(dataset.profiles) if profile.split == "test")
    original = list(dataset.compounds)
    relabeled = [replace(compound, moa=("irrelevant_annotation",)) for compound in original]
    left = pipeline.rank(
        dataset.expression[index],
        dataset.genes,
        dataset.profiles[index],
        original,
        k=len(original),
        batch_size=7,
    )
    right = pipeline.rank(
        dataset.expression[index],
        dataset.genes,
        dataset.profiles[index],
        relabeled,
        k=len(original),
        batch_size=7,
    )
    assert [row["candidate_id"] for row in left] == [row["candidate_id"] for row in right]
    np.testing.assert_allclose(
        [row["score"] for row in left], [row["score"] for row in right], rtol=1e-5, atol=1e-6
    )
    batched = pipeline.rank(
        dataset.expression[index],
        dataset.genes,
        dataset.profiles[index],
        original,
        k=len(original),
        batch_size=28,
    )
    original_scores = {row["candidate_id"]: row["score"] for row in left}
    np.testing.assert_allclose(
        [original_scores[row["candidate_id"]] for row in batched],
        [row["score"] for row in batched],
        rtol=1e-5,
        atol=1e-6,
    )


def test_screening_records_source_types_and_preserves_existing_artifacts(trained_bundle):
    root, _, config = trained_bundle
    output = root / "screening.json"
    result = screen_dataset(root / "run/model.pt", root / "data", output, k=5)
    assert result["source_kind"] == result["training_source_kind"] == "synthetic"
    assert result["reproduces_manuscript"] is False
    assert result["rankings"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        screen_dataset(root / "run/model.pt", root / "data", output, k=5)
    assert output.read_bytes() == before
    model_before = (root / "run/model.pt").read_bytes()
    with pytest.raises(FileExistsError):
        train(root / "data", root / "run", config)
    assert (root / "run/model.pt").read_bytes() == model_before


def test_external_scaffold_novelty_uses_checkpoint_training_library(trained_bundle):
    root, dataset, _ = trained_bundle
    external = dataset.subset("test")
    observed_train_ids = {
        profile.compound_id for profile in dataset.profiles if profile.split == "train"
    }
    training_compound = next(
        compound
        for compound in dataset.compounds
        if compound.compound_id in observed_train_ids and compound.scaffold
    )
    renamed = replace(training_compound, compound_id="renamed-library-candidate", aliases=())
    external = replace(external, compounds=(*external.compounds, renamed))
    save_dataset(external, root / "external-library")
    result = screen_dataset(
        root / "run/model.pt", root / "external-library", root / "external.json", k=100
    )
    for target in result["metrics"]["per_target"]:
        assert target["new_scaffold"]["numerator"] < target["new_scaffold"]["denominator"]


def test_source_alias_overlap_is_refused(trained_bundle):
    root, dataset, _ = trained_bundle
    external = dataset.subset("test")
    training_id = next(
        profile.compound_id for profile in dataset.profiles if profile.split == "train"
    )
    changed = replace(external.compounds[0], aliases=(training_id,))
    external = replace(external, compounds=(changed, *external.compounds[1:]))
    save_dataset(external, root / "aliased-external")
    with pytest.raises(ValueError, match="identities"):
        screen_dataset(root / "run/model.pt", root / "aliased-external", root / "aliased.json")


def test_corrupt_checkpoint_refused_before_deserialization(trained_bundle):
    root, _, _ = trained_bundle
    corrupted = root / "corrupt.pt"
    corrupted.write_bytes((root / "run/model.pt").read_bytes() + b"corruption")
    corrupted.with_suffix(".pt.sha256.json").write_bytes(
        (root / "run/model.pt.sha256.json").read_bytes()
    )
    with pytest.raises(ValueError, match="checksum"):
        RIGIDPipeline.load(corrupted)


def test_candidate_identity_and_nonfinite_target_rejected(trained_bundle):
    root, dataset, _ = trained_bundle
    pipeline = RIGIDPipeline.load(root / "run/model.pt")
    profile = dataset.profiles[0]
    target = dataset.expression[0]
    with pytest.raises(ValueError, match="identifiers"):
        pipeline.rank(target, dataset.genes, profile, [dataset.compounds[0]] * 2)
    with pytest.raises(ValueError, match="structures"):
        pipeline.rank(
            target,
            dataset.genes,
            profile,
            [dataset.compounds[0], replace(dataset.compounds[0], compound_id="duplicate")],
        )
    with pytest.raises(ValueError, match="finite"):
        pipeline.rank(
            np.full(len(dataset.genes), np.nan), dataset.genes, profile, list(dataset.compounds)
        )
