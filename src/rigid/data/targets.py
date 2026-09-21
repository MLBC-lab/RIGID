"""Explicit disease reversal, named program edits, and target interpolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .normalization import _validate
from .schema import GeneManifest


@dataclass(frozen=True)
class ProgramSet:
    genes: GeneManifest
    programs: Mapping[str, tuple[str, ...]]
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Program sets require a source description")
        for name, members in self.programs.items():
            if not name.strip() or not members:
                raise ValueError("Each named program must contain at least one gene")
            unknown = set(members) - set(self.genes.gene_ids)
            if unknown:
                raise ValueError(
                    f"Program {name} has genes missing from the manifest: {sorted(unknown)[:5]}"
                )
            if len(set(members)) != len(members):
                raise ValueError(f"Program {name} contains duplicate genes")

    def indices(self, name: str) -> np.ndarray:
        lookup = {gene: index for index, gene in enumerate(self.genes.gene_ids)}
        return np.asarray([lookup[gene] for gene in self.programs[name]], dtype=int)

    def scores(
        self, expression: np.ndarray, genes: GeneManifest, *, absolute: bool = False
    ) -> np.ndarray:
        self.genes.require_match(genes)
        matrix = _validate(expression, genes)
        if absolute:
            matrix = np.abs(matrix)
        return np.stack(
            [matrix[..., self.indices(name)].mean(axis=-1) for name in self.programs], axis=-1
        )


def reverse_disease_signature(
    disease_minus_control: np.ndarray, genes: GeneManifest, *, strength: float = 1.0
) -> np.ndarray:
    """Negate an already-computed disease-minus-control differential signature.

    This is an algebraic objective, not evidence that a molecule treats disease.
    Apply this in differential-expression space before training normalization.
    """
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Reversal strength must be finite and nonnegative")
    return (-strength * _validate(disease_minus_control, genes)).astype(np.float32)


def edit_programs(
    target: np.ndarray, genes: GeneManifest, programs: ProgramSet, weights: Mapping[str, float]
) -> np.ndarray:
    """Multiply selected genes by program weights; zero removes a program.

    Overlapping program weights multiply. The caller can inspect membership and
    weights directly; no disease or stress gene sets are silently supplied.
    """
    programs.genes.require_match(genes)
    matrix = _validate(target, genes).copy()
    unknown = set(weights) - set(programs.programs)
    if unknown:
        raise ValueError(f"Unknown program names: {sorted(unknown)}")
    scale = np.ones(len(genes))
    for name, weight in weights.items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Program weights must be finite and nonnegative")
        scale[programs.indices(name)] *= weight
    return (matrix * scale).astype(np.float32)


def interpolate_targets(
    left: np.ndarray, right: np.ndarray, genes: GeneManifest, *, alpha: float = 0.5
) -> np.ndarray:
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Interpolation alpha must be between zero and one")
    a, b = _validate(left, genes), _validate(right, genes)
    if a.shape != b.shape:
        raise ValueError("Targets must have identical shapes")
    return ((1 - alpha) * a + alpha * b).astype(np.float32)


def consensus_signature(
    signatures: np.ndarray, genes: GeneManifest, *, method: str = "median"
) -> np.ndarray:
    matrix = _validate(signatures, genes)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("Consensus requires at least one cohort signature")
    if method not in {"mean", "median"}:
        raise ValueError("Consensus method must be mean or median")
    return (np.median(matrix, axis=0) if method == "median" else matrix.mean(axis=0)).astype(
        np.float32
    )
