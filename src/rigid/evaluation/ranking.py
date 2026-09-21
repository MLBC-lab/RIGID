"""Inspectable reranking and validation-normalized designability."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_fit_partition, positive_integer, same_shape, vector


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    score: float
    compatibility: float
    stress_penalty: float
    novelty_bonus: float


def rerank_candidates(
    candidate_ids: Sequence[str],
    scores: ArrayLike,
    stress_scores: ArrayLike,
    *,
    stress_weight: float,
    novelty_scores: ArrayLike | None = None,
    novelty_weight: float = 0.0,
    k: int | None = None,
) -> list[RankedCandidate]:
    """Additive score decomposition with input-order tie breaking."""
    scores = vector(scores, "scores", nonempty=False)
    stress = vector(stress_scores, "stress_scores", nonempty=False)
    novelty = (
        np.zeros_like(scores)
        if novelty_scores is None
        else vector(novelty_scores, "novelty_scores", nonempty=False)
    )
    same_shape(scores, stress, novelty)
    if len(candidate_ids) != len(scores) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique with one ID per score")
    if np.any((stress < 0) | (stress > 1)) or np.any((novelty < 0) | (novelty > 1)):
        raise ValueError("Stress and novelty scores must lie in [0, 1]")
    if any(not np.isfinite(w) or w < 0 for w in (stress_weight, novelty_weight)):
        raise ValueError("Weights must be finite and nonnegative")
    if novelty_weight > 0 and novelty_scores is None:
        raise ValueError("A positive novelty weight requires novelty_scores")
    if k is not None:
        positive_integer(k, "k")
    penalty, bonus = stress_weight * stress, novelty_weight * novelty
    combined = scores - penalty + bonus
    if not np.all(np.isfinite(combined)):
        raise ValueError("Reranking score overflowed")
    order = np.argsort(-combined, kind="stable")[:k]
    return [
        RankedCandidate(
            candidate_ids[i],
            float(combined[i]),
            float(scores[i]),
            float(penalty[i]),
            float(bonus[i]),
        )
        for i in order
    ]


@dataclass
class DesignabilityModel:
    """Min-max component normalization fit on validation targets only.

    Explicit weights are mandatory: the manuscript does not specify their values.
    Lower-is-better components (normally stress_high) are inverted after scaling.
    """

    weights_: dict[str, float] = field(default_factory=dict, init=False)
    ranges_: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)
    lower_is_better_: tuple[str, ...] = field(default=(), init=False)
    validation_ids_: tuple[str, ...] = field(default=(), init=False)

    def fit(
        self,
        components: Mapping[str, ArrayLike],
        weights: Mapping[str, float],
        *,
        split: str,
        sample_ids: Sequence[str],
        forbidden_ids: Sequence[str] = (),
        lower_is_better: Sequence[str] = ("stress_high",),
    ) -> "DesignabilityModel":
        if not components or set(components) != set(weights):
            raise ValueError("Provide an explicit weight for every component, and no extra weights")
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or not np.isclose(
            sum(weights.values()), 1.0
        ):
            raise ValueError("Nonnegative finite weights must sum to one")
        if set(lower_is_better) - set(components):
            raise ValueError("lower_is_better contains an unknown component")
        arrays = {name: vector(values, name) for name, values in components.items()}
        same_shape(*arrays.values())
        n = len(next(iter(arrays.values())))
        ids = check_fit_partition(split, "validation", sample_ids, n, forbidden_ids)
        self.weights_ = dict(weights)
        self.ranges_ = {name: (float(np.min(a)), float(np.max(a))) for name, a in arrays.items()}
        self.lower_is_better_ = tuple(lower_is_better)
        self.validation_ids_ = ids
        return self

    def transform(self, components: Mapping[str, ArrayLike]) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("Fit designability normalization on the validation fold first")
        if set(components) != set(self.weights_):
            raise ValueError("Component names differ from the fitted normalization")
        arrays = {name: vector(values, name, nonempty=False) for name, values in components.items()}
        same_shape(*arrays.values())
        result = np.zeros_like(next(iter(arrays.values())))
        for name, values in arrays.items():
            lower, upper = self.ranges_[name]
            scaled = (
                np.full_like(values, 0.5)
                if lower == upper
                else np.clip((values - lower) / (upper - lower), 0, 1)
            )
            if name in self.lower_is_better_:
                scaled = 1 - scaled
            result += self.weights_[name] * scaled
        return np.clip(result, 0, 1)

    def to_dict(self) -> dict[str, object]:
        if not self.weights_:
            raise RuntimeError("Designability model has not been fitted")
        return {
            "normalization": "validation min-max; constants mapped to 0.5",
            "weights": self.weights_,
            "ranges": self.ranges_,
            "lower_is_better": self.lower_is_better_,
            "validation_ids": self.validation_ids_,
        }
