"""Run an offline synthetic evaluation with separate fit and test partitions.

Install RIGID first, then run ``python examples/evaluation_protocol.py --seed 42``.
The response vectors and retrieval outcomes are simulated; no paper result, model
checkpoint, disease cohort, or experimental observation is represented here.
"""

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from rigid.evaluation import (
    CandidateAnnotation,
    DesignabilityModel,
    SplitConformalRegressor,
    TemperatureScaler,
    benjamini_hochberg,
    cohort_robustness,
    compare_target_metrics,
    empirical_coverage,
    expected_calibration_error,
    make_decoy_targets,
    pearson_correlation,
    rerank_candidates,
    summarize_families,
)


@dataclass
class SyntheticPartition:
    ids: list[str]
    targets: np.ndarray
    responses: np.ndarray
    predicted_alignment: np.ndarray
    observed_alignment: np.ndarray
    baseline_alignment: np.ndarray
    moa_logits: np.ndarray
    moa_success: np.ndarray
    baseline_moa_success: np.ndarray
    components: dict[str, np.ndarray]


def simulate_partition(
    rng: np.random.Generator, name: str, size: int, n_genes: int = 64
) -> SyntheticPartition:
    targets = rng.normal(size=(size, n_genes))
    latent_alignment = rng.uniform(0.15, 0.75, size=size)
    responses = latent_alignment[:, None] * targets
    responses += np.sqrt(1 - latent_alignment**2)[:, None] * rng.normal(size=targets.shape)
    baseline_latent = np.clip(latent_alignment - 0.03, 0, 1)
    baseline_responses = baseline_latent[:, None] * targets
    baseline_responses += np.sqrt(1 - baseline_latent**2)[:, None] * rng.normal(size=targets.shape)
    observed = np.array(
        [pearson_correlation(t, r) for t, r in zip(targets, responses, strict=True)]
    )
    baseline = np.array(
        [pearson_correlation(t, r) for t, r in zip(targets, baseline_responses, strict=True)]
    )
    probability = 1 / (1 + np.exp(-4 * (latent_alignment - 0.45)))
    moa = rng.binomial(1, probability)
    baseline_moa = rng.binomial(1, np.clip(probability - 0.03, 0, 1))
    components = {
        "correlation": observed,
        "moa_success": moa,
        "profile_family_success": rng.binomial(1, 0.25 + 0.5 * latent_alignment),
        "scaffold_diversity": rng.integers(2, 11, size=size) / 10,
        "stress_high": rng.binomial(10, 0.2 + 0.4 * latent_alignment) / 10,
    }
    return SyntheticPartition(
        ids=[f"{name}-{i:03d}" for i in range(size)],
        targets=targets,
        responses=responses,
        predicted_alignment=np.clip(latent_alignment + rng.normal(0, 0.08, size=size), -1, 1),
        observed_alignment=observed,
        baseline_alignment=baseline,
        moa_logits=12 * (latent_alignment - 0.45),
        moa_success=moa,
        baseline_moa_success=baseline_moa,
        components=components,
    )


def run_protocol(seed: int = 42, n_resamples: int = 1000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    validation = simulate_partition(rng, "validation", 40)
    calibration = simulate_partition(rng, "calibration", 160)
    test = simulate_partition(rng, "test", 64)
    partitions = {"validation": validation.ids, "calibration": calibration.ids, "test": test.ids}
    all_ids = [sample_id for ids in partitions.values() for sample_id in ids]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("Synthetic partition IDs overlap")

    temperature = TemperatureScaler().fit(
        calibration.moa_logits,
        calibration.moa_success,
        split="calibration",
        sample_ids=calibration.ids,
        forbidden_ids=validation.ids + test.ids,
    )
    probabilities = temperature.predict_proba(test.moa_logits, sample_ids=test.ids)
    raw_probabilities = 1 / (1 + np.exp(-test.moa_logits))
    conformal = SplitConformalRegressor(alpha=0.1).fit(
        calibration.predicted_alignment,
        calibration.observed_alignment,
        split="calibration",
        sample_ids=calibration.ids,
        forbidden_ids=validation.ids + test.ids,
    )
    intervals = conformal.predict(test.predicted_alignment, sample_ids=test.ids)

    weights = {
        "correlation": 0.30,
        "moa_success": 0.20,
        "profile_family_success": 0.15,
        "scaffold_diversity": 0.20,
        "stress_high": 0.15,
    }
    designability = DesignabilityModel().fit(
        validation.components,
        weights,
        split="validation",
        sample_ids=validation.ids,
        forbidden_ids=calibration.ids + test.ids,
    )
    designability_scores = designability.transform(test.components)
    moa_contrast = compare_target_metrics(
        dict(zip(test.ids, test.moa_success, strict=True)),
        dict(zip(test.ids, test.baseline_moa_success, strict=True)),
        statistic="mean",
        n_resamples=n_resamples,
        seed=seed,
    )
    correlation_contrast = compare_target_metrics(
        dict(zip(test.ids, test.observed_alignment, strict=True)),
        dict(zip(test.ids, test.baseline_alignment, strict=True)),
        statistic="median",
        n_resamples=n_resamples,
        seed=seed,
    )

    control_alignment = {"unaltered": float(np.mean(test.observed_alignment))}
    for kind in ("shuffled", "sign_flipped", "hybrid"):
        decoys = make_decoy_targets(test.targets, kind=kind, seed=seed)
        correlations = [
            pearson_correlation(target, response)
            for target, response in zip(decoys, test.responses, strict=True)
        ]
        control_alignment[kind] = float(np.mean(correlations))

    candidate_ids = [f"toy-compound-{i}" for i in range(12)]
    metadata = {
        candidate: CandidateAnnotation(
            moa={f"toy-family-{i % 4}"}, scaffold=f"toy-scaffold-{i % 6}", stress=i / 12
        )
        for i, candidate in enumerate(candidate_ids)
    }
    ranked = rerank_candidates(
        candidate_ids,
        scores=rng.uniform(0.4, 0.9, len(candidate_ids)),
        stress_scores=[metadata[c].stress for c in candidate_ids],
        stress_weight=0.2,
        novelty_scores=[float(i % 6 >= 3) for i in range(len(candidate_ids))],
        novelty_weight=0.05,
        k=5,
    )
    top_candidates = [row.candidate_id for row in ranked]
    cohort_rankings = {
        "synthetic-cohort-a": top_candidates,
        "synthetic-cohort-b": candidate_ids[2:7],
        "synthetic-cohort-c": candidate_ids[5:10],
    }
    p_values = np.array([0.002, 0.012, 0.04, 0.12, 0.7])
    return {
        "data_origin": "synthetic",
        "claimed_manuscript_reproduction": False,
        "seed": seed,
        "protocol": {
            "gene_count": test.targets.shape[1],
            "partitions": partitions,
            "partition_counts": {name: len(ids) for name, ids in partitions.items()},
            "prediction_origin": "fixed synthetic simulator; no molecular model trained",
            "retrieval_outcomes": "simulated binary target events, not measured top-50 recovery",
            "response_alignment": "one simulated response vector per target",
        },
        "calibration": {
            "temperature": temperature.temperature_,
            "fit_partition": "calibration",
            "evaluation_partition": "test",
            "before": expected_calibration_error(raw_probabilities, test.moa_success),
            "after": expected_calibration_error(probabilities, test.moa_success),
        },
        "conformal": {
            "nominal_coverage": 1 - conformal.alpha,
            "radius": conformal.radius_,
            "test_coverage": empirical_coverage(intervals, test.observed_alignment),
            "first_three_intervals": intervals[:3].tolist(),
        },
        "paired_comparisons": {
            "simulated_moa_success_difference": asdict(moa_contrast),
            "median_response_correlation_difference": asdict(correlation_contrast),
        },
        "retrospective_designability": {
            "requires_known_target_labels": True,
            "fit": designability.to_dict(),
            "test_mean": float(np.mean(designability_scores)),
            "test_min": float(np.min(designability_scores)),
            "test_max": float(np.max(designability_scores)),
        },
        "negative_control_profile_alignment": control_alignment,
        "reranked_candidates": [asdict(row) for row in ranked],
        "candidate_families": summarize_families(
            top_candidates,
            metadata,
            training_scaffolds=[f"toy-scaffold-{i}" for i in range(3)],
        ),
        "cohort_stability": cohort_robustness(
            cohort_rankings, k=5, candidate_moa={c: tuple(m.moa) for c, m in metadata.items()}
        ),
        "multiple_testing_demo": {
            "hypotheses": [f"toy-hypothesis-{i}" for i in range(len(p_values))],
            "p_values": p_values.tolist(),
            "bh_q_values": benjamini_hochberg(p_values).tolist(),
            "source": "illustrative p-values; no biological enrichment experiment",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run_protocol(args.seed, args.resamples), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
