"""Target-resampled uncertainty and multiple-testing control."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import positive_integer, same_shape, vector
from .metrics import _labels


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_targets: int
    n_resamples: int
    seed: int
    method: str = "paired target percentile bootstrap"


def paired_bootstrap_ci(
    a: ArrayLike,
    b: ArrayLike,
    *,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """CI for statistic(a)-statistic(b), resampling identical target indices."""
    left, right = vector(a, "a"), vector(b, "b")
    same_shape(left, right)
    positive_integer(n_resamples, "n_resamples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    reducer: Callable[[np.ndarray], float]
    if isinstance(statistic, str):
        if statistic not in ("mean", "median"):
            raise ValueError("statistic must be mean, median, or a callable")

        def reduce_default(values: np.ndarray) -> float:
            return float(np.mean(values)) if statistic == "mean" else float(np.median(values))

        reducer = reduce_default
    else:
        reducer = statistic
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        selected = rng.integers(0, len(left), size=len(left))
        samples[i] = reducer(left[selected]) - reducer(right[selected])
    estimate = float(reducer(left) - reducer(right))
    if not np.isfinite(estimate) or not np.all(np.isfinite(samples)):
        raise ValueError("Statistic returned a non-finite result")
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])
    return BootstrapInterval(
        estimate, float(lower), float(upper), confidence, len(left), n_resamples, seed
    )


def compare_target_metrics(
    a: Mapping[str, float | None],
    b: Mapping[str, float | None],
    *,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Align shared, eligible target IDs before a paired comparison."""
    if set(a) != set(b):
        raise ValueError("Comparators must cover exactly the same target identifiers")
    left, right = [], []
    for key in sorted(a):
        value_a, value_b = a[key], b[key]
        if value_a is not None and value_b is not None:
            left.append(value_a)
            right.append(value_b)
    return paired_bootstrap_ci(
        left,
        right,
        statistic=statistic,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def benjamini_hochberg(p_values: ArrayLike) -> np.ndarray:
    """BH-adjusted q-values in original hypothesis order; no silent NaN omission."""
    p = vector(p_values, "p_values", nonempty=False)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    if not p.size:
        return p.copy()
    order = np.argsort(p, kind="stable")
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def jaccard(a: Sequence[str] | set[str], b: Sequence[str] | set[str]) -> float | None:
    """Set Jaccard; two empty candidate sets provide no stability evidence."""
    left, right = set(a), set(b)
    union = left | right
    return len(left & right) / len(union) if union else None


def cohort_robustness(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 50,
    candidate_moa: Mapping[str, Sequence[str] | str | None] | None = None,
) -> dict[str, object]:
    """Pairwise cohort top-k Jaccard and symmetric shared-MOA overlap.

    The shared-family rate uses intersection/minimum family-set size, a declared
    explicit convention; report the denominator alongside the resulting rate.
    """
    positive_integer(k, "k")
    cohorts = sorted(rankings)
    for ranked in rankings.values():
        if len(set(ranked)) != len(ranked):
            raise ValueError("Cohort rankings must contain unique candidates")
    pairs: list[dict[str, object]] = []
    valid: list[float] = []
    for index, first in enumerate(cohorts):
        for second in cohorts[index + 1 :]:
            left, right = list(rankings[first][:k]), list(rankings[second][:k])
            similarity = jaccard(left, right)
            if similarity is not None:
                valid.append(similarity)
            record: dict[str, object] = {
                "cohort_a": first,
                "cohort_b": second,
                "jaccard": similarity,
                "candidate_union": len(set(left) | set(right)),
                "candidate_intersection": len(set(left) & set(right)),
            }
            if candidate_moa is not None:
                families_a = {v for c in left for v in _labels(candidate_moa.get(c))}
                families_b = {v for c in right for v in _labels(candidate_moa.get(c))}
                denominator = min(len(families_a), len(families_b))
                record["shared_moa_family_rate"] = (
                    len(families_a & families_b) / denominator if denominator else None
                )
                record["shared_moa_denominator"] = denominator
            pairs.append(record)
    return {
        "k": k,
        "cohort_count": len(cohorts),
        "pairs": pairs,
        "mean_jaccard": float(np.mean(valid)) if valid else None,
        "eligible_pair_count": len(valid),
    }
