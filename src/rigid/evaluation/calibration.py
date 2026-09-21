"""Calibration fit boundaries and finite-sample split conformal intervals."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_fit_partition, positive_integer, same_shape, vector


def _binary_labels(values: ArrayLike) -> np.ndarray:
    labels = vector(values, "labels")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must contain only 0 or 1")
    return labels


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp = np.exp(logits[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def expected_calibration_error(
    probabilities: ArrayLike, labels: ArrayLike, *, n_bins: int = 10
) -> dict[str, object]:
    """Equal-width binary-event ECE, retaining bin counts and observed frequencies."""
    probabilities = vector(probabilities, "probabilities")
    labels = _binary_labels(labels)
    same_shape(probabilities, labels)
    positive_integer(n_bins, "n_bins")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    assignments = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    bins: list[dict[str, object]] = []
    ece = 0.0
    for i in range(n_bins):
        selected = assignments == i
        count = int(np.sum(selected))
        confidence: float | None
        accuracy: float | None
        if count:
            confidence = float(np.mean(probabilities[selected]))
            accuracy = float(np.mean(labels[selected]))
            ece += count / len(labels) * abs(accuracy - confidence)
        else:
            confidence, accuracy = None, None
        bins.append(
            {
                "lower": i / n_bins,
                "upper": (i + 1) / n_bins,
                "count": count,
                "confidence": confidence,
                "observed_frequency": accuracy,
            }
        )
    return {"ece": float(ece), "n": len(labels), "n_bins": n_bins, "bins": bins}


@dataclass
class TemperatureScaler:
    """Single positive temperature for binary compatibility logits."""

    temperature_: float | None = field(default=None, init=False)
    calibration_ids_: tuple[str, ...] = field(default=(), init=False)

    def fit(
        self,
        logits: ArrayLike,
        labels: ArrayLike,
        *,
        split: str,
        sample_ids: Sequence[str],
        forbidden_ids: Sequence[str] = (),
    ) -> "TemperatureScaler":
        x, y = vector(logits, "logits"), _binary_labels(labels)
        same_shape(x, y)
        ids = check_fit_partition(split, "calibration", sample_ids, len(x), forbidden_ids)
        if len(np.unique(y)) != 2:
            raise ValueError("Temperature fitting requires both outcome classes")

        def objective(log_temperature: float) -> float:
            # Signed softplus avoids subtracting two large, nearly equal terms.
            with np.errstate(over="ignore", under="ignore"):
                signed = (1 - 2 * y) * x / np.exp(log_temperature)
                losses = np.logaddexp(0.0, signed)
                return float(np.sum(losses / len(losses)))

        lower, upper = np.log(0.05), np.log(100.0)
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
        for _ in range(100):
            if objective(c) < objective(d):
                upper, d = d, c
                c = upper - ratio * (upper - lower)
            else:
                lower, c = c, d
                d = lower + ratio * (upper - lower)
        self.temperature_ = float(np.exp((lower + upper) / 2))
        self.calibration_ids_ = ids
        return self

    def predict_proba(
        self, logits: ArrayLike, *, sample_ids: Sequence[str] | None = None
    ) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("Fit on a calibration partition before prediction")
        x = vector(logits, "logits", nonempty=False)
        if sample_ids is not None:
            if len(sample_ids) != len(x):
                raise ValueError("sample_ids length does not match predictions")
            if set(sample_ids).intersection(self.calibration_ids_):
                raise ValueError("Evaluation IDs overlap the calibration partition")
        with np.errstate(over="ignore", under="ignore"):
            return _sigmoid(x / self.temperature_)


@dataclass
class SplitConformalRegressor:
    """Absolute-residual split conformal intervals for scalar target metrics.

    Marginal coverage requires exchangeable calibration and test observations.
    Calibrate once per target, not on dependent candidates as independent samples.
    """

    alpha: float = 0.1
    radius_: float | None = field(default=None, init=False)
    calibration_ids_: tuple[str, ...] = field(default=(), init=False)

    def fit(
        self,
        predictions: ArrayLike,
        observations: ArrayLike,
        *,
        split: str,
        sample_ids: Sequence[str],
        forbidden_ids: Sequence[str] = (),
    ) -> "SplitConformalRegressor":
        pred, obs = vector(predictions, "predictions"), vector(observations, "observations")
        same_shape(pred, obs)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        ids = check_fit_partition(split, "calibration", sample_ids, len(pred), forbidden_ids)
        residuals = np.abs(obs - pred)
        if not np.all(np.isfinite(residuals)):
            raise ValueError("Residual calculation overflowed")
        rank = int(np.ceil((len(pred) + 1) * (1 - self.alpha)))
        self.radius_ = (
            float(np.partition(residuals, rank - 1)[rank - 1])
            if rank <= len(pred)
            else float("inf")
        )
        self.calibration_ids_ = ids
        return self

    def predict(
        self, predictions: ArrayLike, *, sample_ids: Sequence[str] | None = None
    ) -> np.ndarray:
        if self.radius_ is None:
            raise RuntimeError("Fit on a calibration partition before prediction")
        pred = vector(predictions, "predictions", nonempty=False)
        if sample_ids is not None:
            if len(sample_ids) != len(pred):
                raise ValueError("sample_ids length does not match predictions")
            if set(sample_ids).intersection(self.calibration_ids_):
                raise ValueError("Evaluation IDs overlap the calibration partition")
        return np.column_stack((pred - self.radius_, pred + self.radius_))


def empirical_coverage(intervals: ArrayLike, observations: ArrayLike) -> dict[str, float | int]:
    bounds = np.asarray(intervals, dtype=float)
    obs = vector(observations, "observations")
    if bounds.shape != (len(obs), 2) or np.any(np.isnan(bounds)):
        raise ValueError("intervals must have shape (n, 2), without NaN bounds")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Lower interval bounds exceed upper bounds")
    covered = (obs >= bounds[:, 0]) & (obs <= bounds[:, 1])
    return {
        "coverage": float(np.mean(covered)),
        "numerator": int(np.sum(covered)),
        "denominator": len(obs),
    }
