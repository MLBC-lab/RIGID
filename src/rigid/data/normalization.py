"""Feature standardization fitted exclusively to the training partition."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import GeneManifest


@dataclass(frozen=True)
class TrainStandardizer:
    genes: GeneManifest
    mean: np.ndarray
    scale: np.ndarray
    n_training_profiles: int

    def __post_init__(self) -> None:
        mean, scale = np.array(self.mean, dtype=np.float64), np.array(self.scale, dtype=np.float64)
        if mean.shape != (len(self.genes),) or scale.shape != mean.shape:
            raise ValueError("Normalizer dimensions do not match the gene manifest")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Normalizer requires finite means and positive finite scales")
        if self.n_training_profiles < 1:
            raise ValueError("Normalizer must be fitted to at least one training profile")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        expression: np.ndarray,
        split_labels: Sequence[str],
        genes: GeneManifest,
        *,
        minimum_scale: float = 1e-8,
    ) -> TrainStandardizer:
        matrix = _validate(expression, genes)
        if matrix.ndim != 2 or len(split_labels) != len(matrix):
            raise ValueError("Training expression rows and split labels must align")
        if minimum_scale <= 0 or not np.isfinite(minimum_scale):
            raise ValueError("minimum_scale must be positive and finite")
        training = matrix[np.asarray(split_labels) == "train"]
        if not len(training):
            raise ValueError("No training profiles available to fit the normalizer")
        mean, scale = training.mean(axis=0), training.std(axis=0)
        scale = np.where(scale < minimum_scale, 1.0, scale)
        return cls(genes, mean, scale, len(training))

    def transform(self, expression: np.ndarray, genes: GeneManifest) -> np.ndarray:
        self.genes.require_match(genes)
        return ((_validate(expression, genes) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, expression: np.ndarray, genes: GeneManifest) -> np.ndarray:
        self.genes.require_match(genes)
        return (_validate(expression, genes) * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "genes": self.genes.to_dict(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "n_training_profiles": self.n_training_profiles,
            "fit_partition": "train",
        }

    @classmethod
    def from_dict(cls, value: dict) -> TrainStandardizer:
        if value.get("fit_partition") != "train":
            raise ValueError("Normalizer metadata must declare fit_partition=train")
        return cls(
            GeneManifest.from_dict(value["genes"]),
            np.asarray(value["mean"]),
            np.asarray(value["scale"]),
            int(value["n_training_profiles"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TrainStandardizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _validate(expression: np.ndarray, genes: GeneManifest) -> np.ndarray:
    matrix = np.asarray(expression, dtype=np.float64)
    if matrix.ndim not in {1, 2} or matrix.shape[-1] != len(genes):
        raise ValueError("Expression values do not match the gene manifest")
    if not np.isfinite(matrix).all():
        raise ValueError("Expression values must be finite")
    return matrix
