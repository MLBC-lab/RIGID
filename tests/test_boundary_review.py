"""Adversarial contracts for externally supplied records and matched comparisons."""

import copy
import json

import pytest

from rigid.cli import main
from rigid.data import GeneManifest
from rigid.features import AuxiliarySpec
from rigid.provenance import atomic_json
from rigid.runs import compare_screenings, inspect_run, load_screening


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"gene_ids": "ab"},
        {"gene_ids": None},
        {"gene_ids": ["a", 2]},
        {"gene_ids": ["a"], "namespace": None},
        {"gene_ids": ["a"], "typo_namespace": "entrez"},
    ],
)
def test_nested_gene_manifest_errors_are_contract_errors(value):
    with pytest.raises(ValueError):
        GeneManifest.from_dict(value)


def test_gene_manifest_roundtrip_and_explicit_order_remain_supported():
    manifest = GeneManifest(("b", "a"), "entrez")
    assert GeneManifest.from_dict(manifest.to_dict()) == manifest
    assert GeneManifest.from_dict({"gene_ids": ("b", "a")}).gene_ids == ("b", "a")


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"programs": None},
        {"programs": []},
        {"programs": {"pathway": "ab"}, "source": "fixture"},
        {"programs": {"pathway": ["a", None]}, "source": "fixture"},
        {"programs": {"pathway": ["a", "a"]}, "source": "fixture"},
        {"programs": {" ": ["a"]}, "source": "fixture"},
        {"programs": {1: ["a"]}, "source": "fixture"},
        {"programs": {"pathway": ["a"]}, "source": None},
        {"programs": {"pathway": ["a"]}, "source": "fixture", "stress_programs": "pathway"},
        {"programs": {"pathway": ["a"]}, "source": "fixture", "stress_programs": None},
        {
            "programs": {"pathway": ["a"]},
            "source": "fixture",
            "stress_programs": ["pathway", "pathway"],
        },
    ],
)
def test_nested_program_schema_rejects_ambiguous_types(value):
    with pytest.raises(ValueError):
        AuxiliarySpec.from_dict(value, GeneManifest(("a", "b")))


def test_program_schema_roundtrip_keeps_gene_order_and_stress_names():
    genes = GeneManifest(("a", "b"))
    value = {
        "programs": {"pathway": ["b", "a"]},
        "source": "fixture",
        "stress_programs": ["pathway"],
    }
    specification = AuxiliarySpec.from_dict(value, genes)
    assert specification.programs == {"pathway": ("b", "a")}
    assert AuxiliarySpec.from_dict(specification.to_dict(), genes) == specification
    assert AuxiliarySpec.from_dict({}, genes).programs == {}


def _screening():
    candidate = {
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
        "rankings": {"t1": [copy.deepcopy(candidate)], "t2": [copy.deepcopy(candidate)]},
        "metrics": {
            "k": 1,
            "target_count": 2,
            "exact": {"numerator": 1, "denominator": 2, "value": 0.5},
            "same_moa": {"numerator": 2, "denominator": 2, "value": 1.0},
            "profile_family": {"numerator": 0, "denominator": 0, "value": None},
            "per_target": [
                {
                    "target_id": "t1",
                    "requested_k": 1,
                    "returned": 1,
                    "exact": True,
                    "same_moa": True,
                    "profile_family": None,
                },
                {
                    "target_id": "t2",
                    "requested_k": 1,
                    "returned": 1,
                    "exact": False,
                    "same_moa": True,
                    "profile_family": None,
                },
            ],
        },
        "checkpoint_sha256": "0" * 64,
        "gene_manifest_sha256": "1" * 64,
        "input_sha256": {"profiles.csv": "2" * 64},
        "notes": ["Synthetic fixture"],
    }


def test_aggregate_counts_cannot_exceed_recorded_targets(tmp_path):
    record = _screening()
    record["metrics"]["exact"] = {"numerator": 100, "denominator": 100, "value": 1.0}
    atomic_json(tmp_path / "screening.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "screening.json")


def test_aggregate_rates_must_match_explicit_target_outcomes(tmp_path):
    record = _screening()
    record["metrics"]["exact"] = {"numerator": 2, "denominator": 2, "value": 1.0}
    atomic_json(tmp_path / "screening.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "screening.json")


def test_missing_annotation_field_is_not_silently_treated_as_unknown(tmp_path):
    record = _screening()
    del record["metrics"]["per_target"][1]["exact"]
    atomic_json(tmp_path / "screening.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "screening.json")


def test_comparison_cannot_silently_drop_a_difficult_target(tmp_path):
    left, right = _screening(), _screening()
    right["metrics"]["per_target"][1]["exact"] = None
    right["metrics"]["exact"] = {"numerator": 1, "denominator": 1, "value": 1.0}
    atomic_json(tmp_path / "left.json", left)
    atomic_json(tmp_path / "right.json", right)
    load_screening(tmp_path / "right.json")
    with pytest.raises(ValueError):
        compare_screenings(tmp_path / "left.json", tmp_path / "right.json", n_resamples=5)


def test_comparison_allows_different_success_with_identical_eligibility(tmp_path):
    left, right = _screening(), _screening()
    right["metrics"]["per_target"][1]["exact"] = True
    right["metrics"]["exact"] = {"numerator": 2, "denominator": 2, "value": 1.0}
    atomic_json(tmp_path / "left.json", left)
    atomic_json(tmp_path / "right.json", right)
    result = compare_screenings(tmp_path / "left.json", tmp_path / "right.json", n_resamples=5)
    assert result["comparisons"]["exact"]["eligible_targets"] == 2
    assert result["comparisons"]["exact"]["right_minus_left"]["estimate"] == 0.5


def test_extreme_integer_is_a_contract_error_not_an_uncaught_overflow(tmp_path):
    record = _screening()
    record["rankings"]["t1"][0]["score"] = 10**400
    atomic_json(tmp_path / "screening.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "screening.json")


def test_inspect_run_rejects_boolean_schema_version(tmp_path):
    atomic_json(tmp_path / "manifest.json", {"schema_version": True, "source_kind": "synthetic"})
    with pytest.raises(ValueError):
        inspect_run(tmp_path)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "gene_manifest_sha256"])
def test_screening_digest_syntax_is_checked(tmp_path, field):
    record = _screening()
    record[field] = "not-a-digest"
    atomic_json(tmp_path / "screening.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "screening.json")


def test_screening_input_digest_syntax_is_checked(tmp_path):
    record = _screening()
    record["input_sha256"] = {"profiles.csv": "not-a-digest"}
    atomic_json(tmp_path / "screening.json", record)
    with pytest.raises(ValueError):
        load_screening(tmp_path / "screening.json")


def test_malformed_generation_manifest_produces_structured_cli_error(tmp_path, capsys):
    pytest.importorskip("torch")
    target = tmp_path / "target.json"
    atomic_json(target, {"gene_manifest": {}, "expression": [1]})
    output = tmp_path / "proposals.json"
    status = main(
        [
            "--json-errors",
            "generate",
            "--checkpoint",
            str(tmp_path / "absent.pt"),
            "--target",
            str(target),
            "--output",
            str(output),
        ]
    )
    assert status == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_request"
    assert not output.exists()
