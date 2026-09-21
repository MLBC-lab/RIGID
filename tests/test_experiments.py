"""Small genuine train/load/sample runs on explicitly synthetic test data."""

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from rigid.checkpoint import load_checkpoint  # noqa: E402
from rigid.config import TrainConfig  # noqa: E402
from rigid.data import (  # noqa: E402
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    SourceProvenance,
    save_dataset,
)
from rigid.experiments import (  # noqa: E402
    generate,
    nearest_neighbor,
    screen_vae,
    train_generator,
    train_vae,
)


def small_dataset(*, unsupported=False):
    entries = [
        ("a", "CC", "train"),
        ("b", "CO", "train"),
        ("c", "CCC", "validation"),
        ("d", "CCCC", "test"),
    ]
    if unsupported:
        entries.extend([("e", "F[C@H](Cl)Br", "train"), ("f", "F[C@H](Cl)I", "validation")])
    compounds = tuple(
        CompoundRecord(cid, smiles, ("synthetic_label",)) for cid, smiles, _ in entries
    )
    profiles = tuple(
        ProfileRecord(f"{cid}_{replicate}", cid, "synthetic_cell", 1, 24, split)
        for cid, _, split in entries
        for replicate in range(2)
    )
    expression = np.random.default_rng(12).normal(size=(len(profiles), 4)).astype(np.float32)
    return ProfileDataset(
        GeneManifest(("g1", "g2", "g3", "g4"), "synthetic"),
        compounds,
        profiles,
        expression,
        SourceProvenance(kind="synthetic"),
    )


def tiny_config():
    return TrainConfig(
        epochs=1,
        forward_epochs=1,
        hidden_dim=8,
        depth=1,
        batch_size=2,
        dropout=0,
        num_threads=1,
        top_k=2,
    )


@pytest.mark.integration
def test_generator_training_exclusions_checkpoint_and_condition_alignment(tmp_path):
    dataset = small_dataset(unsupported=True)
    save_dataset(dataset, tmp_path / "data")
    manifest = train_generator(
        tmp_path / "data", tmp_path / "generator", tiny_config(), max_atoms=8
    )
    assert manifest["test_used_for_training_or_selection"] is False
    assert manifest["exclusions"]["train"]["excluded_profile_ids"] == ["e_0", "e_1"]
    assert manifest["exclusions"]["validation"]["excluded_profile_ids"] == ["f_0", "f_1"]
    assert "test" not in manifest["exclusions"]
    checkpoint = tmp_path / "generator" / "model.pt"
    payload = load_checkpoint(checkpoint)
    assert payload["kind"] == "graph-generator" and payload["schema_version"] == 1
    np.testing.assert_allclose(
        payload["normalizer"]["mean"], dataset.expression[:4].mean(0), atol=1e-7
    )
    first = generate(checkpoint, n=3, seed=18, target=dataset.expression[6], genes=dataset.genes)
    second = generate(checkpoint, n=3, seed=18, target=dataset.expression[6], genes=dataset.genes)
    assert first == second
    assert first["attempted"] == 3 and 0 <= first["validity"] <= 1
    with pytest.raises(ValueError, match="target and ordered genes"):
        generate(checkpoint)
    with pytest.raises(ValueError, match="gene order"):
        generate(checkpoint, target=dataset.expression[6], genes=("g2", "g1", "g3", "g4"))
    with pytest.raises(FileExistsError):
        train_generator(tmp_path / "data", tmp_path / "generator", tiny_config())


@pytest.mark.integration
def test_unconditional_generator_uses_unique_compounds_and_no_test_signal(tmp_path):
    original = small_dataset()
    changed = original.expression.copy()
    changed[[i for i, profile in enumerate(original.profiles) if profile.split == "test"]] += 10000
    changed = replace(original, expression=changed)
    for name, dataset in (("first", original), ("second", changed)):
        save_dataset(dataset, tmp_path / name)
        manifest = train_generator(
            tmp_path / name,
            tmp_path / (name + "_run"),
            tiny_config(),
            conditional=False,
            max_atoms=8,
        )
        assert len(manifest["training_profile_ids"]) == 2
        assert len(manifest["validation_profile_ids"]) == 1
    first = load_checkpoint(tmp_path / "first_run" / "model.pt")
    second = load_checkpoint(tmp_path / "second_run" / "model.pt")
    for key in first["state_dict"]:
        torch.testing.assert_close(
            first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0
        )
    assert generate(tmp_path / "first_run" / "model.pt", n=2)["attempted"] == 2


@pytest.mark.integration
def test_vae_can_train_reload_and_rank_held_out_targets(tmp_path):
    dataset = small_dataset()
    save_dataset(dataset, tmp_path / "data")
    manifest = train_vae(
        tmp_path / "data", tmp_path / "vae", tiny_config(), fingerprint_bits=32, latent_dim=4
    )
    assert manifest["fingerprint_schema"] == {"kind": "Morgan", "radius": 2, "bits": 32}
    result = screen_vae(tmp_path / "vae" / "model.pt", tmp_path / "data", samples=3)
    assert result["metrics"]["target_count"] == 2
    assert result["candidate_count"] == 4
    assert all(len(rows) == 2 for rows in result["rankings"].values())
    assert result == screen_vae(tmp_path / "vae" / "model.pt", tmp_path / "data", samples=3)
    with pytest.raises(ValueError, match="Evaluation requires"):
        screen_vae(tmp_path / "vae" / "model.pt", tmp_path / "data", split="train")
    checksum = tmp_path / "vae" / "model.pt.sha256.json"
    checksum.write_text(json.dumps({"sha256": "wrong"}))
    with pytest.raises(ValueError, match="checksum"):
        screen_vae(tmp_path / "vae" / "model.pt", tmp_path / "data")


def test_nearest_neighbor_catalog_contains_only_observed_training_compounds(tmp_path):
    dataset = small_dataset()
    save_dataset(dataset, tmp_path / "data")
    result = nearest_neighbor(tmp_path / "data", k=2)
    assert result["candidate_count"] == 2
    assert result["ranking_profile_ids"] == ["a_0", "a_1", "b_0", "b_1"]
    for ranking in result["rankings"].values():
        assert {row["candidate_id"] for row in ranking} == {"a", "b"}
    assert result["metrics"]["exact"]["value"] == 0
    assert result["metrics"]["same_moa"]["value"] == 1


def test_nearest_neighbor_does_not_use_validation_responses(tmp_path):
    dataset = small_dataset()
    modified = dataset.expression.copy()
    modified[4:6] *= -1000
    save_dataset(dataset, tmp_path / "first")
    save_dataset(replace(dataset, expression=modified), tmp_path / "second")
    first = nearest_neighbor(tmp_path / "first", k=2)
    second = nearest_neighbor(tmp_path / "second", k=2)
    assert first["rankings"] == second["rankings"]
