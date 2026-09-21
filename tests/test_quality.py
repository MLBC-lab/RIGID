"""Quality diagnostics must remain descriptive and use training-only references."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    load_dataset,
    read_gct,
    save_dataset,
)
from rigid.quality import DriftThresholds, QualityThresholds, audit_dataset, compare_to_training


def dataset_fixture():
    compounds = (
        CompoundRecord("train", "CCO", moa=("known",)),
        CompoundRecord("test", "c1ccccc1", moa=("unseen",)),
    )
    profiles = (
        ProfileRecord("train-1", "train", "A", 1, 24, "train"),
        ProfileRecord("train-2", "train", "A", 1, 24, "train"),
        ProfileRecord("test-1", "test", "B", 10, 48, "test"),
        ProfileRecord("test-2", "test", "B", 10, 48, "test"),
    )
    return ProfileDataset(
        GeneManifest(("varying", "constant", "equal")),
        compounds,
        profiles,
        np.array([[0, 5, 0], [2, 5, 2], [10, 7, 0], [12, 7, 2]]),
    )


def test_quality_report_is_json_safe_and_does_not_mutate_input():
    dataset = dataset_fixture()
    before = dataset.expression.copy()
    report = audit_dataset(dataset)
    json.dumps(report.to_dict(), allow_nan=False)
    np.testing.assert_array_equal(dataset.expression, before)
    assert report.counts == {"profiles": 4, "compounds": 2, "genes": 3, "finite_values": 12}
    assert report.splits["train"]["cells"] == {"A": 2}
    assert report.splits["test"]["profiles_with_moa"] == 2
    assert report.replicates["groups"] == 2
    assert "unknown_provenance" in {warning["code"] for warning in report.warnings}


def test_chunked_statistics_match_numpy():
    dataset = dataset_fixture()
    report = audit_dataset(dataset, thresholds=QualityThresholds(chunk_size=1))
    np.testing.assert_allclose(
        [gene["mean"] for gene in report.per_gene], dataset.expression.mean(axis=0)
    )
    np.testing.assert_allclose(
        [gene["standard_deviation"] for gene in report.per_gene],
        dataset.expression.astype(np.float64).std(axis=0),
    )


def test_empty_dataset_has_finite_json_and_explicit_warning():
    dataset = replace(dataset_fixture(), profiles=(), expression=np.empty((0, 3)))
    report = audit_dataset(dataset)
    json.dumps(report.to_dict(), allow_nan=False)
    assert "empty_dataset" in {warning["code"] for warning in report.warnings}
    assert report.per_gene == []


def test_constant_genes_and_profiles_are_not_fabricated_correlations():
    dataset = replace(dataset_fixture(), expression=np.ones((4, 3)))
    report = audit_dataset(dataset)
    assert all(gene["constant"] for gene in report.per_gene)
    assert report.outliers["values"] == 0
    assert all(group["mean_pairwise_pearson"] is None for group in report.replicates["records"])
    assert all(group["informative_pairs"] == 0 for group in report.replicates["records"])
    json.dumps(report.to_dict(), allow_nan=False)


def test_replicate_correlations_and_cross_split_warnings():
    dataset = dataset_fixture()
    profiles = tuple(
        replace(profile, compound_id="train", cell_id="A", dose_um=1, time_h=24)
        for profile in dataset.profiles
    )
    expression = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3], [3, 2, 1]])
    report = audit_dataset(replace(dataset, profiles=profiles, expression=expression))
    group = report.replicates["records"][0]
    expected = np.corrcoef(expression)[np.triu_indices(4, k=1)].mean()
    assert group["mean_pairwise_pearson"] == pytest.approx(expected)
    assert {"replicates_cross_split", "low_replicate_correlation"} <= {
        warning["code"] for warning in report.warnings
    }


def test_outlier_warning_is_configurable_and_bounded():
    dataset = dataset_fixture()
    report = audit_dataset(dataset, thresholds=QualityThresholds(outlier_standard_deviations=0.5))
    assert report.outliers["values"] > 0
    assert report.outliers["profiles"] == 4
    assert len(report.outliers["first_profile_ids"]) <= 20


def test_missing_labels_are_coverage_warnings_not_schema_failures():
    dataset = dataset_fixture()
    compounds = tuple(replace(compound, moa=()) for compound in dataset.compounds)
    report = audit_dataset(replace(dataset, compounds=compounds))
    assert report.splits["train"]["moa_profile_coverage"] == 0
    assert "incomplete_moa_coverage" in {warning["code"] for warning in report.warnings}


def test_drift_training_only_reference_and_context_coverage():
    dataset = dataset_fixture()
    report = compare_to_training(dataset, "test")
    genes = {gene["gene_id"]: gene for gene in report.per_gene}
    assert genes["varying"]["training_mean"] == 1
    assert genes["varying"]["training_standard_deviation"] == 1
    assert genes["varying"]["standardized_mean_shift"] == 10
    assert genes["equal"]["population_stability_index"] == 0
    assert genes["constant"]["standardized_mean_shift"] is None
    assert genes["constant"]["constant_reference_changed"]
    assert report.context["unseen_cell_ids"] == ["B"]
    assert report.context["profiles_with_unseen_cell"] == 2
    assert report.context["dose_um"]["profiles_outside_training_range"] == 2
    assert report.context["unseen_moa_labels"] == ["unseen"]
    json.dumps(report.to_dict(), allow_nan=False)


def test_changing_test_values_never_changes_reference_or_bins():
    dataset = dataset_fixture()
    original = compare_to_training(dataset, "test")
    values = dataset.expression.copy()
    values[2:] *= 100
    changed = compare_to_training(replace(dataset, expression=values), "test")
    assert original.reference == changed.reference
    for before, after in zip(original.per_gene, changed.per_gene, strict=True):
        for key in ("training_mean", "training_standard_deviation", "histogram_interior_edges"):
            assert before[key] == after[key]
    assert (
        changed.per_gene[0]["standardized_mean_shift"]
        > original.per_gene[0]["standardized_mean_shift"]
    )


def test_drift_ignores_other_held_out_partitions():
    dataset = dataset_fixture()
    extra = replace(dataset.profiles[2], profile_id="external-1", split="external")
    expanded = replace(
        dataset,
        profiles=(*dataset.profiles, extra),
        expression=np.vstack((dataset.expression, [1e20, -1e20, 100])),
    )
    assert (
        compare_to_training(dataset, "test").to_dict()
        == compare_to_training(expanded, "test").to_dict()
    )


def test_single_row_and_constant_reference_have_no_infinite_scores():
    dataset = dataset_fixture()
    subset = replace(
        dataset,
        profiles=(dataset.profiles[0], dataset.profiles[2]),
        expression=dataset.expression[[0, 2]],
    )
    report = compare_to_training(subset, "test")
    json.dumps(report.to_dict(), allow_nan=False)
    assert all(gene["standardized_mean_shift"] is None for gene in report.per_gene)
    assert (
        len([warning for warning in report.warnings if warning["code"].startswith("small_")]) == 2
    )


@pytest.mark.parametrize("split", ["train", "", "excluded", "holdout"])
def test_invalid_comparison_splits_fail(split):
    with pytest.raises(ValueError, match="Comparison split"):
        compare_to_training(dataset_fixture(), split)


def test_missing_partition_fails_instead_of_reporting_zero_shift():
    with pytest.raises(ValueError, match="nonempty"):
        compare_to_training(dataset_fixture(), "external")


@pytest.mark.parametrize(
    "thresholds",
    [
        {"constant_tolerance": 0},
        {"outlier_standard_deviations": float("nan")},
        {"minimum_replicate_correlation": 2},
        {"chunk_size": True},
        {"minimum_training_profiles": 0},
    ],
)
def test_quality_thresholds_are_validated(thresholds):
    with pytest.raises(ValueError):
        QualityThresholds(**thresholds)


@pytest.mark.parametrize(
    "thresholds",
    [
        {"standardized_mean_shift": -1},
        {"histogram_pseudocount": 0},
        {"population_stability_index": float("inf")},
        {"histogram_bins": 1},
        {"histogram_bins": 101},
        {"minimum_comparison_profiles": True},
    ],
)
def test_drift_thresholds_are_validated(thresholds):
    with pytest.raises(ValueError):
        DriftThresholds(**thresholds)


def test_save_refuses_nonempty_directory_and_preserves_existing_bytes(tmp_path):
    dataset = dataset_fixture()
    save_dataset(dataset, tmp_path)
    snapshots = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="empty"):
        save_dataset(dataset, tmp_path)
    assert snapshots == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize("chunk_size", [1, 3, 4096])
def test_csv_numeric_chunk_boundaries_roundtrip(tmp_path, chunk_size):
    dataset = dataset_fixture()
    save_dataset(dataset, tmp_path)
    loaded = load_dataset(tmp_path, chunk_size=chunk_size)
    np.testing.assert_array_equal(dataset.expression, loaded.expression)
    assert dataset.profiles == loaded.profiles


def test_csv_profile_limit_is_enforced(tmp_path):
    save_dataset(dataset_fixture(), tmp_path)
    with pytest.raises(ValueError, match="Profile limit"):
        load_dataset(tmp_path, max_profiles=3)


@pytest.mark.parametrize("value", ["nan", "inf", "1e40", "", "not-a-number"])
def test_csv_numeric_errors_report_line_number(tmp_path, value):
    save_dataset(dataset_fixture(), tmp_path)
    path = tmp_path / "profiles.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1][-1] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="line 2"):
        load_dataset(tmp_path, verify_checksums=False)


def test_partial_checksum_manifest_is_not_a_verified_dataset(tmp_path):
    save_dataset(dataset_fixture(), tmp_path)
    path = tmp_path / "dataset.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["files_sha256"]["profiles.csv"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum manifest"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("value", ["1e40", "nan", "inf"])
def test_gct_rejects_float32_overflow_and_nonfinite_input(tmp_path, value):
    path = tmp_path / "input.gct"
    path.write_text(f"#1.2\n1\t1\nName\tDescription\tp1\ng1\tna\t{value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_gct(path, GeneManifest(("g1",)))


def test_missing_gct_header_is_a_contract_error(tmp_path):
    path = tmp_path / "input.gct"
    path.write_text("#1.2\n1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identifiers"):
        read_gct(path, GeneManifest(("g1",)))


def test_delimited_labels_and_boolean_contexts_rejected():
    with pytest.raises(ValueError, match="semicolon"):
        CompoundRecord("a", "CCO", moa=("first;second",))
    with pytest.raises(ValueError, match="real number"):
        ProfileRecord("p", "a", "cell", True, 24)


def test_direct_schema_float32_overflow_rejected():
    with pytest.raises(ValueError, match="float32"):
        replace(dataset_fixture(), expression=np.full((4, 3), 1e100))
