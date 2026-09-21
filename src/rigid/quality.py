"""Descriptive dataset checks and distribution shifts against training-only references."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rigid.data.schema import ProfileDataset

REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class QualityThresholds:
    constant_tolerance: float = 1e-8
    outlier_standard_deviations: float = 6.0
    minimum_replicate_correlation: float = 0.5
    minimum_training_profiles: int = 20
    chunk_size: int = 4096

    def __post_init__(self) -> None:
        _positive(self.constant_tolerance, "constant_tolerance")
        _positive(self.outlier_standard_deviations, "outlier_standard_deviations")
        if not np.isfinite(self.minimum_replicate_correlation) or not (
            -1 <= self.minimum_replicate_correlation <= 1
        ):
            raise ValueError("minimum_replicate_correlation must lie in [-1, 1]")
        _positive_integer(self.minimum_training_profiles, "minimum_training_profiles")
        _positive_integer(self.chunk_size, "chunk_size")


@dataclass(frozen=True)
class DriftThresholds:
    constant_tolerance: float = 1e-8
    standardized_mean_shift: float = 1.0
    population_stability_index: float = 0.2
    histogram_bins: int = 10
    histogram_pseudocount: float = 0.5
    minimum_training_profiles: int = 20
    minimum_comparison_profiles: int = 10

    def __post_init__(self) -> None:
        for name in (
            "constant_tolerance",
            "standardized_mean_shift",
            "population_stability_index",
            "histogram_pseudocount",
        ):
            _positive(getattr(self, name), name)
        for name in ("minimum_training_profiles", "minimum_comparison_profiles"):
            _positive_integer(getattr(self, name), name)
        _positive_integer(self.histogram_bins, "histogram_bins")
        if not 2 <= self.histogram_bins <= 100:
            raise ValueError("histogram_bins must lie in [2, 100]")


@dataclass(frozen=True)
class DatasetQualityReport:
    dataset_kind: str
    gene_manifest_sha256: str
    counts: dict[str, int]
    splits: dict[str, dict[str, Any]]
    per_gene: list[dict[str, Any]]
    replicates: dict[str, Any]
    outliers: dict[str, Any]
    warnings: list[dict[str, Any]]
    thresholds: dict[str, Any]
    schema_version: str = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributionShiftReport:
    dataset_kind: str
    gene_manifest_sha256: str
    reference: dict[str, Any]
    comparison: dict[str, Any]
    per_gene: list[dict[str, Any]]
    context: dict[str, Any]
    warnings: list[dict[str, Any]]
    thresholds: dict[str, Any]
    schema_version: str = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _warning(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _moments(matrix: np.ndarray, chunk_size: int = 4096) -> tuple[np.ndarray, ...]:
    """Merge chunk moments in float64 without a second full-size float64 matrix."""
    mean = np.zeros(matrix.shape[1], dtype=np.float64)
    moment = np.zeros_like(mean)
    minimum = np.full_like(mean, np.inf)
    maximum = np.full_like(mean, -np.inf)
    count = 0
    for start in range(0, len(matrix), chunk_size):
        chunk = np.asarray(matrix[start : start + chunk_size], dtype=np.float64)
        n = len(chunk)
        chunk_mean = chunk.mean(axis=0)
        delta = chunk_mean - mean
        moment += np.sum((chunk - chunk_mean) ** 2, axis=0)
        moment += delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        minimum = np.minimum(minimum, chunk.min(axis=0))
        maximum = np.maximum(maximum, chunk.max(axis=0))
        count += n
    if not count:
        return (np.full_like(mean, np.nan),) * 4
    return mean, np.sqrt(np.maximum(moment / count, 0)), minimum, maximum


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "median": None}
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "median": float(np.median(array)),
    }


def _split_summary(dataset: ProfileDataset, indices: list[int]) -> dict[str, Any]:
    profiles = [dataset.profiles[index] for index in indices]
    lookup = dataset.compound_lookup
    compound_ids = {profile.compound_id for profile in profiles}
    labels = Counter(label for identifier in compound_ids for label in lookup[identifier].moa)
    labeled_profiles = sum(bool(lookup[profile.compound_id].moa) for profile in profiles)
    return {
        "profiles": len(profiles),
        "compounds": len(compound_ids),
        "canonical_structures": len(
            {lookup[identifier].canonical_smiles for identifier in compound_ids}
        ),
        "cells": dict(sorted(Counter(profile.cell_id for profile in profiles).items())),
        "dose_um": _summary([profile.dose_um for profile in profiles]),
        "time_h": _summary([profile.time_h for profile in profiles]),
        "profiles_with_moa": labeled_profiles,
        "moa_profile_coverage": labeled_profiles / len(profiles) if profiles else None,
        "moa_compound_counts": dict(sorted(labels.items())),
    }


def _replicate_summary(
    dataset: ProfileDataset, thresholds: QualityThresholds
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    lookup = dataset.compound_lookup
    for index, profile in enumerate(dataset.profiles):
        key = (
            lookup[profile.compound_id].canonical_smiles,
            profile.cell_id,
            profile.dose_um,
            profile.time_h,
        )
        groups[key].append(index)
    records, warnings = [], []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        values = np.asarray(dataset.expression[indices], dtype=np.float64)
        centered = values - values.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        informative = norms > thresholds.constant_tolerance
        normalized = centered[informative] / norms[informative, None]
        n = len(normalized)
        # Sum of pairwise dot products; avoids allocating a replicate-by-replicate matrix.
        correlation = (
            float(np.clip((np.sum(normalized.sum(axis=0) ** 2) - n) / (n * (n - 1)), -1, 1))
            if n >= 2
            else None
        )
        splits = sorted({dataset.profiles[index].split for index in indices})
        ids = [dataset.profiles[index].profile_id for index in indices]
        records.append(
            {
                "canonical_smiles": key[0],
                "cell_id": key[1],
                "dose_um": key[2],
                "time_h": key[3],
                "profile_ids": ids,
                "splits": splits,
                "count": len(indices),
                "informative_pairs": n * (n - 1) // 2,
                "constant_profiles": int((~informative).sum()),
                "mean_pairwise_pearson": correlation,
            }
        )
        if len(splits) > 1:
            warnings.append(
                _warning(
                    "replicates_cross_split",
                    "Matched-context replicates cross split boundaries.",
                    profile_ids=ids,
                )
            )
        if correlation is not None and correlation < thresholds.minimum_replicate_correlation:
            warnings.append(
                _warning(
                    "low_replicate_correlation",
                    "Matched-context replicate correlation is below the configured threshold.",
                    profile_ids=ids,
                    mean_pairwise_pearson=correlation,
                )
            )
        if n < len(indices):
            warnings.append(
                _warning(
                    "constant_replicate_profiles",
                    "Pearson correlation is undefined for constant expression profiles; these pairs were excluded.",
                    profile_ids=ids,
                    count=len(indices) - n,
                )
            )
    return {
        "grouping": "canonical structure + exact cell_id, dose_um, and time_h",
        "groups": len(records),
        "profiles_in_groups": sum(record["count"] for record in records),
        "records": records,
    }, warnings


def audit_dataset(
    dataset: ProfileDataset, *, thresholds: QualityThresholds | None = None
) -> DatasetQualityReport:
    """Summarize quality concerns without filtering rows or changing the dataset.

    Schema failures are rejected when constructing/loading ``ProfileDataset``.
    These descriptive checks are warnings, not evidence of biological validity.
    """
    thresholds = thresholds or QualityThresholds()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, profile in enumerate(dataset.profiles):
        groups[profile.split].append(index)
    splits = {split: _split_summary(dataset, indices) for split, indices in sorted(groups.items())}
    warnings: list[dict[str, Any]] = []
    if not dataset.profiles:
        warnings.append(_warning("empty_dataset", "No expression profiles are available."))
    if len(groups.get("train", [])) < thresholds.minimum_training_profiles:
        warnings.append(
            _warning(
                "small_training_partition",
                "Training profile count is below the configured diagnostic threshold.",
                count=len(groups.get("train", [])),
            )
        )
    if "" in groups:
        warnings.append(
            _warning(
                "unassigned_profiles",
                "Some profiles have no partition assignment.",
                count=len(groups[""]),
            )
        )
    for split, summary in splits.items():
        if summary["profiles_with_moa"] < summary["profiles"]:
            warnings.append(
                _warning(
                    "incomplete_moa_coverage",
                    "Some profiles lack mechanism labels; label-dependent metrics will have reduced coverage.",
                    split=split,
                    count=summary["profiles"] - summary["profiles_with_moa"],
                )
            )
    if dataset.provenance.kind == "unknown":
        warnings.append(
            _warning("unknown_provenance", "Dataset source type has not been specified.")
        )
    mean, std, minimum, maximum = _moments(dataset.expression, thresholds.chunk_size)
    per_gene = []
    outlier_counts: NDArray[np.int64] = np.zeros(len(dataset.genes), dtype=np.int64)
    outlier_profiles: list[str] = []
    n_outlier_profiles = 0
    constant = std <= thresholds.constant_tolerance
    if dataset.profiles:
        scale = np.where(constant, 1.0, std)
        for start in range(0, len(dataset.profiles), thresholds.chunk_size):
            chunk = np.asarray(
                dataset.expression[start : start + thresholds.chunk_size], dtype=np.float64
            )
            flags = (
                np.abs((chunk - mean) / scale) > thresholds.outlier_standard_deviations
            ) & ~constant
            outlier_counts += flags.sum(axis=0)
            row_flags = np.flatnonzero(flags.any(axis=1))
            n_outlier_profiles += len(row_flags)
            for index in row_flags[: max(0, 20 - len(outlier_profiles))]:
                outlier_profiles.append(dataset.profiles[start + int(index)].profile_id)
        for index, gene in enumerate(dataset.genes.gene_ids):
            per_gene.append(
                {
                    "gene_id": gene,
                    "mean": float(mean[index]),
                    "standard_deviation": float(std[index]),
                    "minimum": float(minimum[index]),
                    "maximum": float(maximum[index]),
                    "constant": bool(constant[index]),
                    "outlying_values": int(outlier_counts[index]),
                }
            )
    if constant.any():
        warnings.append(
            _warning(
                "constant_genes",
                "Some genes have no measurable variance under the configured tolerance.",
                count=int(constant.sum()),
            )
        )
    if n_outlier_profiles:
        warnings.append(
            _warning(
                "extreme_expression_values",
                "Some expression values exceed the configured marginal standard-deviation threshold; inspect source measurements before exclusion.",
                count=n_outlier_profiles,
            )
        )
    replicates, replicate_warnings = _replicate_summary(dataset, thresholds)
    warnings.extend(replicate_warnings)
    return DatasetQualityReport(
        dataset_kind=dataset.provenance.kind,
        gene_manifest_sha256=dataset.genes.sha256,
        counts={
            "profiles": len(dataset.profiles),
            "compounds": len(dataset.compounds),
            "genes": len(dataset.genes),
            "finite_values": int(dataset.expression.size),
        },
        splits=splits,
        per_gene=per_gene,
        replicates=replicates,
        outliers={
            "reference": "entire dataset; descriptive only",
            "profiles": n_outlier_profiles,
            "values": int(outlier_counts.sum()),
            "first_profile_ids": outlier_profiles,
            "profile_id_limit": 20,
        },
        warnings=warnings,
        thresholds=asdict(thresholds),
    )


def _histogram_edges(reference: np.ndarray, thresholds: DriftThresholds) -> np.ndarray:
    if np.ptp(reference) <= thresholds.constant_tolerance:
        center = float(reference.mean())
        return np.array(
            [
                -np.inf,
                center - thresholds.constant_tolerance,
                center + thresholds.constant_tolerance,
                np.inf,
            ]
        )
    interior = np.unique(
        np.quantile(reference, np.linspace(0, 1, thresholds.histogram_bins + 1)[1:-1])
    )
    return np.concatenate(([-np.inf], interior, [np.inf]))


def _psi(
    reference: np.ndarray, comparison: np.ndarray, edges: np.ndarray, pseudocount: float
) -> float:
    expected = np.histogram(reference, bins=edges)[0].astype(np.float64) + pseudocount
    observed = np.histogram(comparison, bins=edges)[0].astype(np.float64) + pseudocount
    expected /= expected.sum()
    observed /= observed.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def compare_to_training(
    dataset: ProfileDataset,
    split: str,
    *,
    thresholds: DriftThresholds | None = None,
) -> DistributionShiftReport:
    """Compare one held-out partition to a reference fitted only on ``train`` rows.

    All statistics are descriptive diagnostics. The returned report is not a
    hypothesis test or a fitted transform for downstream model training.
    """
    if split not in {"validation", "calibration", "test", "external"}:
        raise ValueError("Comparison split must be validation, calibration, test, or external")
    thresholds = thresholds or DriftThresholds()
    reference_indices = [
        i for i, profile in enumerate(dataset.profiles) if profile.split == "train"
    ]
    comparison_indices = [i for i, profile in enumerate(dataset.profiles) if profile.split == split]
    if not reference_indices or not comparison_indices:
        raise ValueError("Distribution shift requires nonempty training and comparison partitions")
    reference = dataset.expression[reference_indices]
    comparison = dataset.expression[comparison_indices]
    reference_mean, reference_std, _, _ = _moments(reference)
    observed_mean, observed_std, _, _ = _moments(comparison)
    per_gene, warnings = [], []
    if len(reference) < thresholds.minimum_training_profiles:
        warnings.append(
            _warning(
                "small_training_partition",
                "Training reference has fewer profiles than the configured diagnostic threshold.",
                count=len(reference),
            )
        )
    if len(comparison) < thresholds.minimum_comparison_profiles:
        warnings.append(
            _warning(
                "small_comparison_partition",
                "Comparison has fewer profiles than the configured diagnostic threshold.",
                count=len(comparison),
            )
        )
    for index, gene in enumerate(dataset.genes.gene_ids):
        train_values = np.asarray(reference[:, index], dtype=np.float64)
        observed_values = np.asarray(comparison[:, index], dtype=np.float64)
        constant = bool(reference_std[index] <= thresholds.constant_tolerance)
        changed_constant = bool(
            constant
            and np.any(
                np.abs(observed_values - reference_mean[index]) > thresholds.constant_tolerance
            )
        )
        standardized_shift = (
            None
            if constant
            else float((observed_mean[index] - reference_mean[index]) / reference_std[index])
        )
        edges = _histogram_edges(train_values, thresholds)
        psi = _psi(train_values, observed_values, edges, thresholds.histogram_pseudocount)
        flagged = (
            changed_constant
            or psi >= thresholds.population_stability_index
            or (
                standardized_shift is not None
                and abs(standardized_shift) >= thresholds.standardized_mean_shift
            )
        )
        per_gene.append(
            {
                "gene_id": gene,
                "training_mean": float(reference_mean[index]),
                "training_standard_deviation": float(reference_std[index]),
                "comparison_mean": float(observed_mean[index]),
                "comparison_standard_deviation": float(observed_std[index]),
                "standardized_mean_shift": standardized_shift,
                "standard_deviation_ratio": None
                if constant
                else float(observed_std[index] / reference_std[index]),
                "population_stability_index": psi,
                "histogram_interior_edges": edges[1:-1].tolist(),
                "training_constant": constant,
                "constant_reference_changed": changed_constant,
                "flagged": bool(flagged),
            }
        )
    flagged_genes = sum(gene["flagged"] for gene in per_gene)
    if flagged_genes:
        warnings.append(
            _warning(
                "expression_distribution_shift",
                "Some genes exceed configured distribution-shift thresholds; this is descriptive, not a test of biological relevance.",
                count=flagged_genes,
            )
        )
    reference_profiles = [dataset.profiles[index] for index in reference_indices]
    observed_profiles = [dataset.profiles[index] for index in comparison_indices]
    train_cells = {profile.cell_id for profile in reference_profiles}
    unseen_cells = sorted({profile.cell_id for profile in observed_profiles} - train_cells)
    context: dict[str, Any] = {
        "unseen_cell_ids": unseen_cells,
        "profiles_with_unseen_cell": sum(
            profile.cell_id not in train_cells for profile in observed_profiles
        ),
    }
    if unseen_cells:
        warnings.append(
            _warning(
                "unseen_cell_context",
                "Comparison contains cell identifiers absent from training.",
                cell_ids=unseen_cells,
            )
        )
    for field in ("dose_um", "time_h"):
        train_values = [getattr(profile, field) for profile in reference_profiles]
        observed_values = [getattr(profile, field) for profile in observed_profiles]
        low, high = min(train_values), max(train_values)
        outside = sum(value < low or value > high for value in observed_values)
        context[field] = {
            "training": _summary(train_values),
            "comparison": _summary(observed_values),
            "profiles_outside_training_range": outside,
        }
        if outside:
            warnings.append(
                _warning(
                    "context_range_shift",
                    "Some comparison contexts are outside the training range.",
                    field=field,
                    count=outside,
                )
            )
    lookup = dataset.compound_lookup
    train_labels = {
        label for profile in reference_profiles for label in lookup[profile.compound_id].moa
    }
    unseen_labels = sorted(
        {label for profile in observed_profiles for label in lookup[profile.compound_id].moa}
        - train_labels
    )
    context["unseen_moa_labels"] = unseen_labels
    if unseen_labels:
        warnings.append(
            _warning(
                "unseen_mechanism_labels",
                "Comparison contains mechanism labels absent from training.",
                labels=unseen_labels,
            )
        )
    return DistributionShiftReport(
        dataset_kind=dataset.provenance.kind,
        gene_manifest_sha256=dataset.genes.sha256,
        reference={
            "partition": "train",
            "profile_count": len(reference),
            "profile_ids": [profile.profile_id for profile in reference_profiles],
            "histogram_fit_partition": "train",
        },
        comparison={
            "partition": split,
            "profile_count": len(comparison),
            "flagged_genes": flagged_genes,
        },
        per_gene=per_gene,
        context=context,
        warnings=warnings,
        thresholds=asdict(thresholds),
    )
