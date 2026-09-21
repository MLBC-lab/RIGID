import copy
import json

import pytest

from rigid.provenance import atomic_json
from rigid.reporting import render_report
from rigid.runs import compare_screenings, inspect_run, load_screening


def screening():
    row = {
        "rank": 1,
        "candidate_id": "molecule",
        "smiles": "CCO",
        "score": 1.0,
        "predicted_profile_correlation": 0.4,
        "independent_profile_correlation": 0.3,
        "chemical_properties": {"qed": 0.5},
    }
    return {
        "schema_version": 1,
        "source_kind": "synthetic",
        "training_source_kind": "synthetic",
        "split": "test",
        "requested_k": 1,
        "candidate_count": 2,
        "rankings": {"t1": [copy.deepcopy(row)], "t2": [copy.deepcopy(row)]},
        "metrics": {
            "per_target": [
                {"target_id": "t1", "exact": True, "same_moa": True, "profile_family": None},
                {"target_id": "t2", "exact": False, "same_moa": True, "profile_family": None},
            ]
        },
        "checkpoint_sha256": "0" * 64,
        "gene_manifest_sha256": "1" * 64,
        "input_sha256": {"profiles.csv": "2" * 64},
        "notes": ["Synthetic fixture"],
    }


def test_matched_comparison_and_missing_annotations(tmp_path):
    left, right = screening(), screening()
    right["metrics"]["per_target"][1]["exact"] = True
    atomic_json(tmp_path / "left.json", left)
    atomic_json(tmp_path / "right.json", right)
    result = compare_screenings(tmp_path / "left.json", tmp_path / "right.json", n_resamples=25)
    assert result["comparisons"]["exact"]["right_minus_left"]["estimate"] == 0.5
    assert result["comparisons"]["profile_family"]["right_minus_left"] is None
    assert result["comparisons"]["profile_family"]["eligible_targets"] == 0
    assert result["top_k_jaccard_by_target"] == {"t1": 1, "t2": 1}
    assert result == compare_screenings(
        tmp_path / "left.json", tmp_path / "right.json", n_resamples=25
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "external"),
        ("requested_k", 2),
        ("candidate_count", 3),
        ("input_sha256", {}),
        ("gene_manifest_sha256", "3" * 64),
    ],
)
def test_incompatible_comparisons_fail(tmp_path, field, value):
    a, b = screening(), screening()
    b[field] = value
    atomic_json(tmp_path / "a.json", a)
    atomic_json(tmp_path / "b.json", b)
    with pytest.raises(ValueError):
        compare_screenings(tmp_path / "a.json", tmp_path / "b.json")


def test_render_escapes_text_and_rejects_corrupt_numeric_values(tmp_path):
    record = screening()
    record["rankings"]["t1"][0]["candidate_id"] = "<script>alert(1)</script>"
    atomic_json(tmp_path / "in.json", record)
    render_report(tmp_path / "in.json", tmp_path / "out.html")
    html = (tmp_path / "out.html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    record["rankings"]["t1"][0]["score"] = "not a number"
    atomic_json(tmp_path / "bad.json", record)
    with pytest.raises(ValueError, match="score"):
        render_report(tmp_path / "bad.json", tmp_path / "bad.html")
    assert not (tmp_path / "bad.html").exists()


def test_inspect_history_and_explicit_unverified_weights(tmp_path):
    atomic_json(tmp_path / "manifest.json", {"schema_version": 1, "source_kind": "synthetic"})
    atomic_json(
        tmp_path / "history.json",
        [
            {"phase": "ranker", "epoch": 1, "train_loss": 2, "validation_loss": 3},
            {"phase": "ranker", "epoch": 2, "train_loss": 1, "validation_loss": 2},
        ],
    )
    result = inspect_run(tmp_path)
    assert result["phases"]["ranker"]["best_epoch"] == 2
    assert not result["checkpoint"]["integrity_verified"]
    values = json.loads((tmp_path / "history.json").read_text())
    values[1]["epoch"] = 1
    atomic_json(tmp_path / "history.json", values)
    with pytest.raises(ValueError, match="Epoch"):
        inspect_run(tmp_path)


def test_inspect_interrupted_session_without_exported_manifest(tmp_path):
    atomic_json(
        tmp_path / "training-session.json",
        {
            "schema_version": 1,
            "protocol": {"configuration": {"epochs": 3}, "input_sha256": {"genes.json": "1" * 64}},
        },
    )
    atomic_json(
        tmp_path / "optimization/forward/heartbeat.json",
        {"event": "cancelled", "completed_epochs": 1, "timestamp": "2026-09-15T12:00:00Z"},
    )
    result = inspect_run(tmp_path)
    assert result["recorded_state"] == "not_exported"
    assert result["manifest_sha256"] is None
    assert result["phase_progress"]["forward"]["event"] == "cancelled"


def test_screening_duplicate_candidates_and_inconsistent_rates_rejected(tmp_path):
    record = screening()
    record["metrics"]["exact"] = {"numerator": 1, "denominator": 2, "value": 0.9}
    atomic_json(tmp_path / "rate.json", record)
    with pytest.raises(ValueError, match="disagrees"):
        load_screening(tmp_path / "rate.json")
    record = screening()
    record["requested_k"] = 2
    second = copy.deepcopy(record["rankings"]["t1"][0])
    second["rank"] = 2
    record["rankings"]["t1"].append(second)
    atomic_json(tmp_path / "duplicate.json", record)
    with pytest.raises(ValueError, match="unique"):
        load_screening(tmp_path / "duplicate.json")


@pytest.mark.parametrize("defect", ["missing_rate_value", "nonstr_source_kind", "missing_targets"])
def test_screening_residual_contract_cases(tmp_path, defect):
    record = screening()
    if defect == "missing_rate_value":
        record["metrics"]["profile_family"] = {"numerator": 0, "denominator": 0}
    elif defect == "nonstr_source_kind":
        record["source_kind"] = []
    else:
        record["metrics"].pop("per_target")
        record["metrics"]["exact"] = {"numerator": 100, "denominator": 100, "value": 1}
    atomic_json(tmp_path / "bad.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "bad.json")
