"""Target-level retrieval and expression metrics with explicit denominators."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._validation import positive_integer, same_shape, vector


def _labels(value: frozenset[str] | Sequence[str] | str | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    labels = [value] if isinstance(value, str) else value
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("MOA and profile-family labels must be strings")
    return frozenset(label.strip() for label in labels if label.strip())


@dataclass(frozen=True)
class TargetAnnotation:
    """Known evaluation labels; missing labels never count as successful matches."""

    source_compound: str | None = None
    moa: frozenset[str] = field(default_factory=frozenset)
    profile_families: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_compound", self.source_compound or None)
        object.__setattr__(self, "moa", _labels(self.moa))
        object.__setattr__(self, "profile_families", _labels(self.profile_families))


@dataclass(frozen=True)
class CandidateAnnotation:
    moa: frozenset[str] = field(default_factory=frozenset)
    profile_families: frozenset[str] = field(default_factory=frozenset)
    scaffold: str | None = None
    stress: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "moa", _labels(self.moa))
        object.__setattr__(self, "profile_families", _labels(self.profile_families))
        object.__setattr__(self, "scaffold", self.scaffold or None)
        if self.stress is not None and (not np.isfinite(self.stress) or not 0 <= self.stress <= 1):
            raise ValueError("stress must be in [0, 1] or None")


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if any(not isinstance(x, (int, np.integer)) for x in (self.numerator, self.denominator)):
            raise ValueError("Rate counts must be integers")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("Rate counts must satisfy 0 <= numerator <= denominator")

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict[str, int | float | None]:
        return {**asdict(self), "value": self.value}


@dataclass(frozen=True)
class TargetMetrics:
    target_id: str
    requested_k: int
    returned: int
    exact: bool | None
    same_moa: bool | None
    profile_family: bool | None
    max_correlation: float | None
    finite_profile_count: int
    new_scaffold: Rate
    scaffold_diversity: Rate
    stress_high: Rate
    missing_scaffold_count: int
    missing_stress_count: int


@dataclass(frozen=True)
class EvaluationReport:
    k: int
    target_count: int
    exact: Rate
    same_moa: Rate
    profile_family: Rate
    median_target_correlation: float | None
    correlation_target_count: int
    per_target: tuple[TargetMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("exact", "same_moa", "profile_family"):
            result[name] = getattr(self, name).to_dict()
        for entry, row in zip(result["per_target"], self.per_target, strict=True):
            for name in ("new_scaffold", "scaffold_diversity", "stress_high"):
                entry[name] = getattr(row, name).to_dict()
        return result


def pearson_correlation(target: ArrayLike, response: ArrayLike) -> float | None:
    """Pearson r; a constant profile has undefined correlation (None)."""
    a, b = vector(target, "target"), vector(response, "response")
    same_shape(a, b)
    if a.size < 2:
        return None
    # Scale before centering to avoid overflow for otherwise finite inputs.
    scale_a, scale_b = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    a = a / (scale_a if scale_a else 1.0)
    b = b / (scale_b if scale_b else 1.0)
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 0 else None


def ndcg(
    relevance: ArrayLike, k: int | None = None, *, ideal_relevance: ArrayLike | None = None
) -> float:
    """NDCG with exponential gain; pass full-catalog ideal labels for truncated ranks."""
    rel = vector(relevance, "relevance", nonempty=False)
    ideal = (
        rel
        if ideal_relevance is None
        else vector(ideal_relevance, "ideal_relevance", nonempty=False)
    )
    if np.any(rel < 0) or np.any(ideal < 0):
        raise ValueError("Relevance must be nonnegative")
    if k is not None:
        positive_integer(k, "k")
    depth = len(rel) if k is None else k
    selected, optimum = rel[:depth], np.sort(ideal)[::-1][:depth]
    if not optimum.size or not np.any(optimum):
        return 0.0
    # A common gain scale cancels and prevents overflow for large relevance.
    scale = max(float(np.max(ideal)), float(np.max(rel, initial=0)))

    def gain(values: np.ndarray) -> np.ndarray:
        return np.exp2(values - scale) * (-np.expm1(-np.log(2.0) * values))

    dcg = np.sum(gain(selected) / np.log2(np.arange(len(selected)) + 2))
    idcg = np.sum(gain(optimum) / np.log2(np.arange(len(optimum)) + 2))
    if idcg <= 0 or dcg > idcg * (1 + 1e-12):
        raise ValueError("ideal_relevance does not describe a valid ideal ranking")
    return float(dcg / idcg)


def top_gene_ndcg(
    target: ArrayLike, response: ArrayLike, *, k: int = 50, target_top_n: int = 50
) -> float:
    """Binary relevance for membership in target top-|expression| genes."""
    target_array, response_array = vector(target, "target"), vector(response, "response")
    same_shape(target_array, response_array)
    positive_integer(k, "k")
    positive_integer(target_top_n, "target_top_n")
    labels = np.zeros(len(target_array))
    order = np.argsort(-np.abs(target_array), kind="stable")[:target_top_n]
    # Zero expression does not make a gene relevant merely to fill the cutoff.
    labels[order[np.abs(target_array[order]) > 0]] = 1.0
    predicted_order = np.argsort(-np.abs(response_array), kind="stable")
    return ndcg(labels[predicted_order], k, ideal_relevance=labels)


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    targets: Mapping[str, TargetAnnotation],
    candidates: Mapping[str, CandidateAnnotation],
    *,
    k: int = 50,
    training_scaffolds: Sequence[str] | None = None,
    stress_threshold: float = 0.5,
    target_profiles: Mapping[str, ArrayLike] | None = None,
    candidate_profiles: Mapping[str, ArrayLike] | None = None,
    strict_k: bool = False,
) -> EvaluationReport:
    """Evaluate target success, never treating missing annotations as matching labels.

    Candidate profiles must already match target gene order and experimental context.
    Duplicate candidates are rejected to prevent top-k count inflation.
    """
    positive_integer(k, "k")
    if not np.isfinite(stress_threshold) or not 0 <= stress_threshold <= 1:
        raise ValueError("stress_threshold must be in [0, 1]")
    if set(rankings) - set(targets):
        raise ValueError("Rankings contain unknown target identifiers")
    if (target_profiles is None) != (candidate_profiles is None):
        raise ValueError("Supply both target_profiles and candidate_profiles")
    train_scaffolds = (
        {x for x in training_scaffolds if x} if training_scaffolds is not None else None
    )
    rows: list[TargetMetrics] = []
    for target_id, annotation in targets.items():
        ranked = list(rankings.get(target_id, ()))
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"Duplicate candidates in ranking for {target_id}")
        unknown = set(ranked) - set(candidates)
        if unknown:
            raise ValueError(f"Unknown candidates for {target_id}: {sorted(unknown)[:5]}")
        selected = ranked[:k]
        if strict_k and len(selected) != k:
            raise ValueError(f"Target {target_id} has fewer than k={k} candidates")
        meta = [candidates[c] for c in selected]
        correlations = []
        if (
            target_profiles is not None
            and candidate_profiles is not None
            and target_id in target_profiles
        ):
            for candidate_id in selected:
                if candidate_id in candidate_profiles:
                    corr = pearson_correlation(
                        target_profiles[target_id], candidate_profiles[candidate_id]
                    )
                    if corr is not None:
                        correlations.append(corr)
        scaffolds = [c.scaffold for c in meta if c.scaffold is not None]
        stresses = [c.stress for c in meta if c.stress is not None]
        rows.append(
            TargetMetrics(
                target_id=target_id,
                requested_k=k,
                returned=len(selected),
                exact=annotation.source_compound in selected
                if annotation.source_compound is not None
                else None,
                same_moa=any(annotation.moa & c.moa for c in meta) if annotation.moa else None,
                profile_family=any(annotation.profile_families & c.profile_families for c in meta)
                if annotation.profile_families
                else None,
                max_correlation=max(correlations) if correlations else None,
                finite_profile_count=len(correlations),
                new_scaffold=Rate(sum(s not in train_scaffolds for s in scaffolds), len(scaffolds))
                if train_scaffolds is not None
                else Rate(0, 0),
                scaffold_diversity=Rate(len(set(scaffolds)), len(scaffolds)),
                stress_high=Rate(sum(s >= stress_threshold for s in stresses), len(stresses)),
                missing_scaffold_count=len(meta) - len(scaffolds),
                missing_stress_count=len(meta) - len(stresses),
            )
        )

    def aggregate(name: str) -> Rate:
        eligible = [getattr(row, name) for row in rows if getattr(row, name) is not None]
        return Rate(sum(eligible), len(eligible))

    correlations = [row.max_correlation for row in rows if row.max_correlation is not None]
    return EvaluationReport(
        k=k,
        target_count=len(rows),
        exact=aggregate("exact"),
        same_moa=aggregate("same_moa"),
        profile_family=aggregate("profile_family"),
        median_target_correlation=float(np.median(correlations)) if correlations else None,
        correlation_target_count=len(correlations),
        per_target=tuple(rows),
    )
