"""Candidate proposal accounting and bounded score-based retention."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .autoregressive import GeneratedMolecule


@dataclass(frozen=True)
class SelectionReport:
    candidates: tuple[tuple[str, float], ...]
    attempted: int
    valid: int
    unique: int
    accepted: int

    @property
    def validity(self) -> float:
        return self.valid / self.attempted if self.attempted else 0.0

    @property
    def uniqueness(self) -> float:
        return self.unique / self.valid if self.valid else 0.0


def select_candidates(
    proposals: Iterable[GeneratedMolecule],
    scorer: Callable[[str], float],
    accept: Callable[[str], bool] | None = None,
    top_k: int = 50,
) -> SelectionReport:
    """Deduplicate canonical proposals, apply chemistry filters, then rank.

    Scoring or filter errors propagate rather than being mislabeled as invalid
    chemistry. The caller defines novelty, property, and stress constraints.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    attempted = valid = accepted = 0
    unique: set[str] = set()
    retained: list[tuple[str, float]] = []
    for proposal in proposals:
        attempted += 1
        if not proposal.valid or proposal.smiles is None:
            continue
        valid += 1
        if proposal.smiles in unique:
            continue
        unique.add(proposal.smiles)
        if accept is not None and not accept(proposal.smiles):
            continue
        score = float(scorer(proposal.smiles))
        if not math.isfinite(score):
            raise ValueError("candidate scorer returned a nonfinite score")
        retained.append((proposal.smiles, score))
        accepted += 1
    retained.sort(key=lambda item: (-item[1], item[0]))
    return SelectionReport(tuple(retained[:top_k]), attempted, valid, len(unique), accepted)
