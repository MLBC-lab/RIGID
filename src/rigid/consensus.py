"""Explicit scoring policies for primary-model and consensus experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .serialization import read_json, require_object


@dataclass(frozen=True)
class RerankingConfig:
    mode: str = "primary"
    compatibility_weight: float = 1.0
    consensus_weight: float = 0.0
    disagreement_weight: float = 0.0
    stress_weight: float | None = None
    novelty_weight: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"primary", "consensus"}:
            raise ValueError("Reranking mode must be primary or consensus")
        for name in (
            "compatibility_weight",
            "consensus_weight",
            "disagreement_weight",
            "stress_weight",
            "novelty_weight",
        ):
            value = getattr(self, name)
            if value is None and name in {"stress_weight", "novelty_weight"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Weights: {name} must be a finite nonnegative number")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0:
                raise ValueError(f"Weights: {name} must be a finite nonnegative number")
        if self.mode == "primary" and (self.consensus_weight or self.disagreement_weight):
            raise ValueError("Consensus or disagreement weights require consensus mode")
        if self.mode == "consensus" and not (self.consensus_weight or self.disagreement_weight):
            raise ValueError("Consensus mode requires a positive consensus or disagreement weight")
        if not (self.compatibility_weight or self.consensus_weight):
            raise ValueError("At least one compatibility or consensus signal is required")

    @classmethod
    def from_dict(cls, values: dict) -> RerankingConfig:
        values = require_object(values, context="Reranking configuration")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown reranking configuration keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> RerankingConfig:
        return cls.from_dict(read_json(path, maximum_bytes=1024 * 1024))

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve(self, stress_weight: float, novelty_weight: float) -> RerankingConfig:
        values = self.to_dict()
        if self.stress_weight is None:
            values["stress_weight"] = stress_weight
        if self.novelty_weight is None:
            values["novelty_weight"] = novelty_weight
        return self.from_dict(values)

    def components(
        self, compatibility: ArrayLike, primary: ArrayLike, independent: ArrayLike
    ) -> dict[str, np.ndarray]:
        values = [
            np.asarray(value, dtype=np.float64) for value in (compatibility, primary, independent)
        ]
        if any(
            value.ndim != 1 or value.shape != values[0].shape or not np.isfinite(value).all()
            for value in values
        ):
            raise ValueError("Scoring inputs must be aligned finite vectors")
        if any(np.any((value < -1) | (value > 1)) for value in values[1:]):
            raise ValueError("Predictor correlations must lie in [-1, 1]")
        with np.errstate(over="raise", invalid="raise"):
            try:
                contribution = self.compatibility_weight * values[0]
                agreement = self.consensus_weight * (values[1] / 2 + values[2] / 2)
                penalty = self.disagreement_weight * np.abs(values[1] - values[2])
                combined = contribution + agreement - penalty
            except FloatingPointError as error:
                raise ValueError("Consensus score overflowed") from error
        return {
            "compatibility_contribution": contribution,
            "consensus_bonus": agreement,
            "disagreement_penalty": penalty,
            "combined": combined,
        }
