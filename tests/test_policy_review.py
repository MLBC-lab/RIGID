"""Independent checks of scoring-policy boundaries and evaluator isolation."""

import json

import numpy as np
import pytest
import torch

from rigid.cli import main
from rigid.config import TrainConfig
from rigid.consensus import RerankingConfig
from rigid.data import make_synthetic_dataset
from rigid.pipeline import RIGIDPipeline
from rigid.protocols import SplitProtocol
from rigid.training import train


@pytest.mark.parametrize("mode", [[], {}, ["primary"], {"mode": "primary"}])
def test_json_mode_containers_fail_as_validation_errors(mode):
    with pytest.raises(ValueError):
        RerankingConfig.from_dict({"mode": mode})


@pytest.mark.parametrize("field", ["stress_weight", "novelty_weight"])
def test_inherited_policy_weights_are_validated(field):
    options = {"stress_weight": 0.2, "novelty_weight": 0.1}
    options[field] = True
    with pytest.raises(ValueError):
        RerankingConfig().resolve(**options)


def test_explicit_zero_overrides_checkpoint_defaults():
    policy = RerankingConfig(stress_weight=0, novelty_weight=0).resolve(0.9, 0.8)
    assert policy.stress_weight == policy.novelty_weight == 0


def test_disagreement_penalty_without_mean_bonus_is_explicit_consensus():
    policy = RerankingConfig(mode="consensus", disagreement_weight=2)
    values = policy.components([1, 1], [0.5, 0.5], [0.5, -0.5])
    np.testing.assert_array_equal(values["combined"], [1, -1])
    np.testing.assert_array_equal(values["consensus_bonus"], [0, 0])


def test_policy_checks_final_sum_overflow_even_when_components_are_finite():
    policy = RerankingConfig(mode="consensus", compatibility_weight=1e308, consensus_weight=1e308)
    with pytest.raises(ValueError, match="overflow"):
        policy.components([1], [1], [1])


@pytest.fixture(scope="module")
def pipeline_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("policy-review")
    dataset = make_synthetic_dataset(root / "data", n_genes=8)
    train(
        root / "data",
        root / "run",
        TrainConfig(
            epochs=1,
            forward_epochs=1,
            hidden_dim=16,
            depth=2,
            batch_size=8,
            candidate_pool_size=8,
            dropout=0,
        ),
    )
    return root, dataset


def test_independent_prediction_changes_only_consensus_selection(pipeline_fixture, monkeypatch):
    root, dataset = pipeline_fixture
    pipeline = RIGIDPipeline.load(root / "run/model.pt")
    index = next(i for i, profile in enumerate(dataset.profiles) if profile.split == "test")
    normalized = pipeline.normalizer.transform(dataset.expression[index], dataset.genes)
    response = torch.tensor(normalized[None], dtype=torch.float32)
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
        stress_weight=0,
        novelty_weight=0,
    )
    results = []
    for sign in (1, -1):
        monkeypatch.setattr(
            pipeline.independent,
            "forward",
            lambda graph, context, sign=sign: sign * response.expand(len(context), -1),
        )
        primary = pipeline.rank(*args, k=50)
        consensus = pipeline.rank(*args, k=50, reranking=policy)
        results.append((primary, consensus))
    positive, negative = results
    assert [row["candidate_id"] for row in positive[0]] == [
        row["candidate_id"] for row in negative[0]
    ]
    assert [row["score"] for row in positive[0]] == [row["score"] for row in negative[0]]
    first, second = ({row["candidate_id"]: row["score"] for row in values[1]} for values in results)
    for candidate in first:
        assert first[candidate] - second[candidate] == pytest.approx(1, abs=1e-6)


def test_invalid_policy_json_has_structured_cli_error(pipeline_fixture, tmp_path, capsys):
    root, _ = pipeline_fixture
    policy = tmp_path / "policy.json"
    policy.write_text('{"mode": []}', encoding="utf-8")
    output = tmp_path / "screening.json"
    status = main(
        [
            "--json-errors",
            "rank",
            str(root / "data"),
            "--checkpoint",
            str(root / "run/model.pt"),
            "--reranking",
            str(policy),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert json.loads(captured.err)["error"]["code"] == "invalid_request"
    assert not output.exists()


def test_protocol_cli_budget_is_checked_before_missing_dataset(tmp_path, capsys):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(SplitProtocol(repetitions=4).to_dict()), encoding="utf-8")
    status = main(
        [
            "--json-errors",
            "plan-splits",
            str(tmp_path / "missing-dataset"),
            "--protocol",
            str(path),
            "--max-splits",
            "3",
            "--output",
            str(tmp_path / "output"),
        ]
    )
    error = json.loads(capsys.readouterr().err)["error"]
    assert status == 2
    assert "max_splits=3" in error["message"]
    assert not (tmp_path / "output").exists()
