"""Scoring-policy arithmetic, compatibility, provenance, and command dispatch."""

import json

import numpy as np
import pytest

from rigid.cli import main
from rigid.config import TrainConfig
from rigid.consensus import RerankingConfig
from rigid.data import make_synthetic_dataset
from rigid.pipeline import RIGIDPipeline, screen_dataset
from rigid.provenance import atomic_json
from rigid.runs import load_screening
from rigid.training import train


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "other"},
        {"consensus_weight": 1},
        {"mode": "consensus"},
        {"compatibility_weight": 0},
        {"compatibility_weight": True},
        {"stress_weight": -1},
        {"novelty_weight": float("inf")},
        {"disagreement_weight": "1"},
        {"compatibility_weight": 10**1000},
        {"unknown": 1},
    ],
)
def test_policy_rejects_ambiguous_or_invalid_settings(values):
    with pytest.raises(ValueError):
        RerankingConfig.from_dict(values)


def test_policy_roundtrip_and_arithmetic(tmp_path):
    policy = RerankingConfig(
        mode="consensus", compatibility_weight=2, consensus_weight=0.5, disagreement_weight=0.25
    )
    atomic_json(tmp_path / "policy.json", policy.to_dict())
    assert RerankingConfig.load(tmp_path / "policy.json") == policy
    resolved = policy.resolve(0.2, 0.1)
    assert resolved.stress_weight == 0.2 and resolved.novelty_weight == 0.1
    result = policy.components([1, 2], [0.8, 0.4], [0.4, 0.6])
    np.testing.assert_allclose(result["combined"], [2.2, 4.2])
    np.testing.assert_allclose(result["disagreement_penalty"], [0.1, 0.05])


@pytest.mark.parametrize(
    "primary,independent",
    [([0.5], [0.2, 0.3]), ([float("nan")], [0]), ([1.01], [0]), ([[0]], [[0]])],
)
def test_policy_rejects_invalid_correlations(primary, independent):
    with pytest.raises(ValueError):
        RerankingConfig().components([1], primary, independent)


def test_policy_detects_score_overflow():
    with pytest.raises(ValueError, match="overflow"):
        RerankingConfig(compatibility_weight=1e308).components([1e308], [0], [0])


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("consensus")
    dataset = make_synthetic_dataset(root / "data", n_genes=8)
    config = TrainConfig(
        epochs=1,
        forward_epochs=1,
        hidden_dim=16,
        depth=2,
        batch_size=8,
        candidate_pool_size=8,
        dropout=0,
        mechanism_candidates=0,
        scaffold_candidates=0,
    )
    train(root / "data", root / "run", config)
    return root, dataset, config


def test_default_policy_preserves_primary_scores(fitted):
    root, dataset, config = fitted
    pipeline = RIGIDPipeline.load(root / "run/model.pt")
    index = next(i for i, item in enumerate(dataset.profiles) if item.split == "test")
    args = (
        dataset.expression[index],
        dataset.genes,
        dataset.profiles[index],
        list(dataset.compounds),
    )
    default = pipeline.rank(*args, k=50)
    explicit = pipeline.rank(*args, k=50, reranking=RerankingConfig())
    assert default == explicit
    for row in default:
        expected = row["compatibility"] + config.novelty_weight * (1 - row["max_training_tanimoto"])
        assert row["score"] == pytest.approx(expected)
        assert row["consensus_bonus"] == row["disagreement_penalty"] == 0
    with pytest.raises(ValueError, match="not both"):
        pipeline.rank(*args, reranking=RerankingConfig(), stress_weight=0.2)


def test_consensus_scores_match_declared_rule_and_batches(fitted):
    root, dataset, _ = fitted
    pipeline = RIGIDPipeline.load(root / "run/model.pt")
    index = next(i for i, item in enumerate(dataset.profiles) if item.split == "test")
    args = (
        dataset.expression[index],
        dataset.genes,
        dataset.profiles[index],
        list(dataset.compounds),
    )
    policy = RerankingConfig(
        mode="consensus",
        compatibility_weight=0,
        consensus_weight=1,
        disagreement_weight=0.25,
        stress_weight=0,
        novelty_weight=0,
    )
    first = pipeline.rank(*args, k=50, batch_size=7, reranking=policy)
    second = pipeline.rank(*args, k=50, batch_size=16, reranking=policy)
    for row in first:
        expected = (
            row["predicted_profile_correlation"] + row["independent_profile_correlation"]
        ) / 2
        expected -= 0.25 * row["validator_disagreement"]
        assert row["score"] == pytest.approx(expected)
        assert row["compatibility_contribution"] == 0
    left, right = ({row["candidate_id"]: row["score"] for row in rows} for rows in (first, second))
    for cid in left:
        assert left[cid] == pytest.approx(right[cid], abs=1e-6)
    assert [row["score"] for row in first] == sorted([row["score"] for row in first], reverse=True)


def test_consensus_screening_records_evaluator_role(fitted):
    root, _, _ = fitted
    policy = RerankingConfig(mode="consensus", consensus_weight=0.2)
    record = screen_dataset(
        root / "run/model.pt", root / "data", root / "consensus.json", reranking=policy
    )
    assert record["independent_predictor_role"] == "scoring_component"
    assert record["reranking"]["consensus_weight"] == 0.2
    assert record["source_kind"] == "synthetic"
    assert record["reproduces_manuscript"] is False
    assert load_screening(root / "consensus.json") == json.loads(json.dumps(record))


def test_consensus_cli_loads_policy(fitted, capsys):
    root, _, _ = fitted
    atomic_json(
        root / "policy.json", RerankingConfig(mode="consensus", consensus_weight=0.2).to_dict()
    )
    assert (
        main(
            [
                "rank",
                str(root / "data"),
                "--checkpoint",
                str(root / "run/model.pt"),
                "--reranking",
                str(root / "policy.json"),
                "--output",
                str(root / "cli.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["metrics"]["target_count"] > 0
    assert load_screening(root / "cli.json")["reranking"]["mode"] == "consensus"


@pytest.mark.parametrize(
    "name,value",
    [
        ("mechanism_candidates", -1),
        ("scaffold_candidates", True),
        ("mechanism_candidates", 1.5),
        ("scaffold_candidates", "2"),
    ],
)
def test_candidate_sampling_settings_are_validated(name, value):
    with pytest.raises(ValueError):
        TrainConfig(**{name: value})


def test_vae_cli_exposes_decoder_settings(tmp_path, monkeypatch, capsys):
    import rigid.experiments

    received = {}

    def fake_train(dataset, output, config, **options):
        received.update(options)
        return {"source_kind": "synthetic"}

    monkeypatch.setattr(rigid.experiments, "train_vae", fake_train)
    assert (
        main(
            [
                "train-vae",
                "data",
                "--output",
                str(tmp_path / "run"),
                "--latent-dim",
                "24",
                "--fingerprint-bits",
                "512",
                "--beta",
                "0.2",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert received["latent_dim"] == 24
    assert received["fingerprint_bits"] == 512
    assert received["beta"] == 0.2
