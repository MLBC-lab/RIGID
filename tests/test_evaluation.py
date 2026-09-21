"""Scientific edge cases: absent annotations, paired targets, and held-out fitting."""

import json

import numpy as np
import pytest

from rigid.evaluation import (
    CandidateAnnotation,
    DesignabilityModel,
    Rate,
    SplitConformalRegressor,
    TargetAnnotation,
    TemperatureScaler,
    benjamini_hochberg,
    cohort_robustness,
    compare_target_metrics,
    empirical_coverage,
    evaluate_rankings,
    expected_calibration_error,
    jaccard,
    make_decoy_targets,
    ndcg,
    paired_bootstrap_ci,
    pearson_correlation,
    rerank_candidates,
    run_ablations,
    summarize_families,
    top_gene_ndcg,
)


@pytest.fixture
def annotations():
    targets = {
        "t1": TargetAnnotation("a", {"kinase"}, {"growth"}),
        "t2": TargetAnnotation("b", {"immune"}, {"inflammation"}),
        "t3": TargetAnnotation(None, {"", " "}, set()),
    }
    candidates = {
        "a": CandidateAnnotation({"kinase"}, {"growth"}, "s1", 0.8),
        "b": CandidateAnnotation({"immune"}, {"inflammation"}, "s2", 0.1),
        "c": CandidateAnnotation(),
    }
    return targets, candidates


def test_retrieval_denominators_and_unknown_labels(annotations):
    targets, candidates = annotations
    report = evaluate_rankings(
        {"t1": ["c", "a"], "t2": ["a"], "t3": ["c"]},
        targets,
        candidates,
        k=2,
        training_scaffolds=["s1"],
    )
    assert (
        report.exact.numerator == report.same_moa.numerator == report.profile_family.numerator == 1
    )
    assert (
        report.exact.denominator
        == report.same_moa.denominator
        == report.profile_family.denominator
        == 2
    )
    assert report.per_target[2].same_moa is None
    assert report.per_target[0].stress_high.denominator == 1
    assert report.per_target[0].missing_stress_count == 1
    assert report.per_target[0].new_scaffold.value == 0.0
    assert report.per_target[0].missing_scaffold_count == 1
    assert report.per_target[2].stress_high.value is None
    json.dumps(report.to_dict(), allow_nan=False)


def test_unranked_target_counts_as_failure_not_omission(annotations):
    targets, candidates = annotations
    report = evaluate_rankings({}, targets, candidates)
    assert report.exact.value == 0.0
    assert report.exact.denominator == 2
    assert report.target_count == 3


def test_multilabel_moa_overlap(annotations):
    _, candidates = annotations
    report = evaluate_rankings(
        {"x": ["b"]}, {"x": TargetAnnotation(moa={"kinase", "immune"})}, candidates
    )
    assert report.same_moa.value == 1
    assert report.exact.value is None


@pytest.mark.parametrize(
    "rankings,k,strict",
    [
        ({"t1": ["a", "a"]}, 1, False),
        ({"t1": ["unknown"]}, 1, False),
        ({"other": ["a"]}, 1, False),
        ({"t1": ["a"]}, 2, True),
        ({}, 0, False),
    ],
)
def test_invalid_rankings(annotations, rankings, k, strict):
    targets, candidates = annotations
    with pytest.raises(ValueError):
        evaluate_rankings(rankings, targets, candidates, k=k, strict_k=strict)


def test_median_of_target_maxima_matches_equation(annotations):
    targets, candidates = annotations
    report = evaluate_rankings(
        {"t1": ["a", "b"], "t2": ["a"], "t3": ["c"]},
        targets,
        candidates,
        k=2,
        target_profiles={"t1": [1, 2, 3], "t2": [3, 2, 1], "t3": [1, 1, 1]},
        candidate_profiles={"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 2, 3]},
    )
    assert report.median_target_correlation == pytest.approx(0)
    assert report.correlation_target_count == 2
    assert report.per_target[0].max_correlation == pytest.approx(1)
    assert report.per_target[2].max_correlation is None


def test_pearson_constant_and_scale():
    assert pearson_correlation([1, 1], [2, 3]) is None
    assert pearson_correlation([1], [1]) is None
    assert pearson_correlation([1e300, -1e300, 0], [2, -2, 0]) == pytest.approx(1)
    assert pearson_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


@pytest.mark.parametrize("a,b", [([1, 2], [1]), ([1, float("nan")], [1, 2]), ([[1]], [[1]])])
def test_pearson_rejects_invalid_inputs(a, b):
    with pytest.raises(ValueError):
        pearson_correlation(a, b)


def test_ndcg_ideal_and_truncation():
    assert ndcg([3, 2, 0, 0]) == pytest.approx(1)
    assert ndcg([0, 0]) == 0
    assert ndcg([]) == 0
    assert ndcg([0, 1], 2, ideal_relevance=[1, 0]) == pytest.approx(1 / np.log2(3))
    assert ndcg([0, 0], 2, ideal_relevance=[1, 0, 0]) == 0
    assert ndcg([10000, 9999]) == pytest.approx(1)
    with pytest.raises(ValueError):
        ndcg([-1, 2])
    with pytest.raises(ValueError):
        ndcg([2], ideal_relevance=[1])


def test_top_gene_ndcg_does_not_depend_on_sign_by_definition():
    assert top_gene_ndcg([9, 8, 0], [-9, -8, 0], k=2, target_top_n=2) == pytest.approx(1)
    assert top_gene_ndcg([0, 0, 0], [1, 2, 3], k=2) == 0
    assert top_gene_ndcg([9, 8, 0], [0, 1, 5], k=1, target_top_n=2) == 0


def test_bootstrap_pairing_and_determinism():
    x = np.arange(30, dtype=float)
    interval = paired_bootstrap_ci(x + 0.3, x, n_resamples=200, seed=18)
    assert interval.estimate == pytest.approx(0.3)
    assert interval.lower == pytest.approx(0.3)
    assert interval.upper == pytest.approx(0.3)
    assert interval == paired_bootstrap_ci(x + 0.3, x, n_resamples=200, seed=18)
    assert interval.n_targets == 30
    med = paired_bootstrap_ci([0, 0, 100], [0, 50, 50], statistic="median", n_resamples=50)
    assert med.estimate == -50  # Difference of medians, not median of differences.


def test_compare_aligns_ids_and_eligible_pairs():
    result = compare_target_metrics(
        {"b": 3, "a": 2, "c": None}, {"a": 1, "b": 2, "c": 1}, n_resamples=30
    )
    assert result.n_targets == 2
    assert result.estimate == 1
    with pytest.raises(ValueError):
        compare_target_metrics({"a": 1}, {"b": 1})


@pytest.mark.parametrize("values", [[0.1, -0.1], [float("nan")], [1.01]])
def test_bh_rejects_invalid_p(values):
    with pytest.raises(ValueError):
        benjamini_hochberg(values)


def test_bh_known_values_and_original_order():
    actual = benjamini_hochberg([0.04, 0.01, 0.03, 0.002])
    np.testing.assert_allclose(actual, [0.04, 0.02, 0.04, 0.008])
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.01, 1]), [0.015, 0.015, 1])
    assert benjamini_hochberg([]).size == 0


def test_cohort_jaccard_and_empty_sets():
    assert jaccard([], []) is None
    assert jaccard(["a"], []) == 0
    assert jaccard(["a", "a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    output = cohort_robustness(
        {"c1": ["a", "b"], "c2": ["b", "c"]},
        k=2,
        candidate_moa={"a": ["x"], "b": ["y"], "c": ["y"]},
    )
    assert output["mean_jaccard"] == pytest.approx(1 / 3)
    assert output["pairs"][0]["shared_moa_family_rate"] == 1


def test_ece_includes_probability_one_and_empty_bins():
    result = expected_calibration_error([0, 0.5, 1], [0, 1, 1], n_bins=4)
    assert result["ece"] == pytest.approx(1 / 6)
    assert sum(b["count"] for b in result["bins"]) == 3
    assert result["bins"][-1]["confidence"] == 1
    assert result["bins"][1]["confidence"] is None


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_calibration_refuses_wrong_partition(split):
    with pytest.raises(ValueError, match="calibration"):
        TemperatureScaler().fit([1, -1], [1, 0], split=split, sample_ids=["a", "b"])
    with pytest.raises(ValueError, match="calibration"):
        SplitConformalRegressor().fit([1, -1], [1, 0], split=split, sample_ids=["a", "b"])


def test_temperature_scaling_reduces_overconfidence_and_checks_ids():
    model = TemperatureScaler().fit(
        [8, 8, -8, -8],
        [1, 0, 1, 0],
        split="calibration",
        sample_ids=["a", "b", "c", "d"],
        forbidden_ids=["test"],
    )
    assert model.temperature_ > 1
    probabilities = model.predict_proba([8, -8], sample_ids=["e", "f"])
    assert np.all(np.abs(probabilities - 0.5) < 0.1)
    with pytest.raises(ValueError, match="overlap"):
        model.predict_proba([8], sample_ids=["a"])
    with pytest.raises(ValueError, match="overlaps"):
        TemperatureScaler().fit(
            [1, -1], [1, 0], split="calibration", sample_ids=["a", "b"], forbidden_ids=["b"]
        )


def test_conformal_finite_sample_order_statistic():
    model = SplitConformalRegressor(alpha=0.2).fit(
        np.zeros(9), np.arange(9), split="calibration", sample_ids=[str(i) for i in range(9)]
    )
    assert model.radius_ == 7  # ceil((9+1)*.8) -> eighth ordered residual.
    np.testing.assert_allclose(model.predict([10], sample_ids=["test"]), [[3, 17]])
    coverage = empirical_coverage([[0, 1], [1, 2], [1, 2]], [0, 2, 3])
    assert coverage == {"coverage": 2 / 3, "numerator": 2, "denominator": 3}
    with pytest.raises(ValueError, match="overlap"):
        model.predict([10], sample_ids=["0"])


def test_conformal_small_sample_requires_unbounded_interval():
    model = SplitConformalRegressor(alpha=0.01).fit(
        [0], [1], split="calibration", sample_ids=["cal"]
    )
    assert np.isinf(model.radius_)
    interval = model.predict([100])
    assert empirical_coverage(interval, [-10000])["coverage"] == 1


@pytest.mark.parametrize("model", [TemperatureScaler(), SplitConformalRegressor()])
def test_unfitted_calibrators_refuse_predictions(model):
    with pytest.raises(RuntimeError):
        if isinstance(model, TemperatureScaler):
            model.predict_proba([1])
        else:
            model.predict([1])


def test_designability_uses_frozen_validation_ranges_and_inverts_stress():
    model = DesignabilityModel().fit(
        {"correlation": [0.2, 0.4], "stress_high": [0.1, 0.9]},
        {"correlation": 0.5, "stress_high": 0.5},
        split="validation",
        sample_ids=["v1", "v2"],
        forbidden_ids=["test"],
    )
    np.testing.assert_allclose(
        model.transform({"correlation": [0.4, 0.2, 100], "stress_high": [0.1, 0.9, 0.9]}),
        [1, 0, 0.5],
    )
    assert model.ranges_["correlation"] == (0.2, 0.4)
    assert model.to_dict()["validation_ids"] == ("v1", "v2")
    with pytest.raises(ValueError, match="validation"):
        DesignabilityModel().fit(
            {"stress_high": [0, 1]}, {"stress_high": 1}, split="test", sample_ids=["a", "b"]
        )


def test_designability_constant_component_and_explicit_weights():
    model = DesignabilityModel().fit(
        {"stress_high": [0.5, 0.5]}, {"stress_high": 1}, split="validation", sample_ids=["a", "b"]
    )
    np.testing.assert_allclose(model.transform({"stress_high": [0.1, 0.9]}), [0.5, 0.5])
    with pytest.raises(ValueError, match="sum to one"):
        DesignabilityModel().fit(
            {"stress_high": [0.5]}, {"stress_high": 0.3}, split="validation", sample_ids=["a"]
        )


def test_reranking_decomposition_and_stable_ties():
    ranked = rerank_candidates(["a", "b", "c"], [1, 0.9, 0.9], [1, 0, 0], stress_weight=0.2)
    assert [r.candidate_id for r in ranked] == ["b", "c", "a"]
    assert ranked[-1].score == pytest.approx(0.8)
    assert ranked[-1].stress_penalty == pytest.approx(0.2)
    with pytest.raises(ValueError):
        rerank_candidates(["a"], [1], [0], stress_weight=0.1, novelty_weight=0.2)


def test_decoys_are_seeded_and_leave_input_unchanged():
    profiles = np.arange(24).reshape(3, 8).astype(float)
    original = profiles.copy()
    shuffled = make_decoy_targets(profiles, kind="shuffled", seed=9)
    np.testing.assert_array_equal(shuffled, make_decoy_targets(profiles, kind="shuffled", seed=9))
    np.testing.assert_array_equal(np.sort(shuffled, axis=1), profiles)
    np.testing.assert_array_equal(make_decoy_targets(profiles, kind="sign_flipped"), -profiles)
    assert make_decoy_targets(profiles, kind="hybrid").shape == profiles.shape
    np.testing.assert_array_equal(profiles, original)
    with pytest.raises(ValueError):
        make_decoy_targets(profiles[:1], kind="hybrid")


def test_ablation_callback_receives_disabled_components():
    output = run_ablations(lambda disabled: {"disabled": sorted(disabled)}, ["stress", "program"])
    assert output == {
        "full": {"disabled": []},
        "without_stress": {"disabled": ["stress"]},
        "without_program": {"disabled": ["program"]},
    }


def test_family_diversity_and_annotation_coverage(annotations):
    _, candidates = annotations
    summary = summarize_families(["a", "b", "c"], candidates, training_scaffolds=["s1"])
    assert summary["candidate_count"] == 3
    assert summary["scaffold_annotated_count"] == 2
    assert summary["unique_scaffolds"] == 2
    assert summary["new_scaffold_candidates"] == 1
    assert summary["effective_scaffold_count"] == pytest.approx(2)


def test_no_novelty_claim_without_training_scaffolds(annotations):
    targets, candidates = annotations
    report = evaluate_rankings({"t1": ["a"]}, targets, candidates)
    assert report.per_target[0].new_scaffold.value is None
    assert summarize_families(["a"], candidates)["new_scaffold_candidates"] is None


@pytest.mark.parametrize("counts", [(2, 1), (-1, 3), (1.5, 2)])
def test_invalid_rate_counts(counts):
    with pytest.raises(ValueError):
        Rate(*counts)


def test_nonstring_annotations_rejected():
    with pytest.raises(ValueError, match="strings"):
        CandidateAnnotation(moa={12})


@pytest.mark.parametrize("scale", [1e-250, 1e-50, 1.0, 1e250])
def test_pearson_invariant_to_small_and_large_positive_scales(scale):
    assert pearson_correlation(
        np.array([1, 2, 3]) * scale, np.array([3, 2, 1]) * scale
    ) == pytest.approx(-1)


def test_temperature_extreme_logits_remain_finite_without_overflow():
    with np.errstate(all="raise"):
        model = TemperatureScaler().fit(
            [1e308, -1e308, 1e308, -1e308],
            [1, 0, 0, 1],
            split="calibration",
            sample_ids=["a", "b", "c", "d"],
        )
        assert model.temperature_ > 90
        probabilities = model.predict_proba([1e308, -1e308])
    np.testing.assert_array_equal(probabilities, [1.0, 0.0])


def test_cohort_single_string_moa_is_one_label():
    result = cohort_robustness(
        {"c1": ["a"], "c2": ["b"]},
        candidate_moa={"a": "kinase", "b": "stress"},
    )
    assert result["pairs"][0]["shared_moa_family_rate"] == 0
    assert result["pairs"][0]["shared_moa_denominator"] == 1


def test_cohort_rejects_duplicate_candidates_even_with_one_cohort():
    with pytest.raises(ValueError, match="unique"):
        cohort_robustness({"only": ["a", "a"]})
