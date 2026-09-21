"""Reproducible hard-negative targets and family-level interpretation hooks."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .metrics import CandidateAnnotation


def make_decoy_targets(profiles: ArrayLike, *, kind: str, seed: int = 0) -> np.ndarray:
    """Construct shuffled, sign-flipped, or two-target hybrid profile controls."""
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("profiles must be a nonempty finite targets-by-genes matrix")
    rng = np.random.default_rng(seed)
    if kind == "shuffled":
        return np.stack([row[rng.permutation(values.shape[1])] for row in values])
    if kind == "sign_flipped":
        return -values.copy()
    if kind == "hybrid":
        if len(values) < 2:
            raise ValueError("Hybrid controls require at least two targets")
        partners = (np.arange(len(values)) + rng.integers(1, len(values))) % len(values)
        mask = rng.random(values.shape) < 0.5
        return np.where(mask, values, values[partners])
    raise ValueError("kind must be shuffled, sign_flipped, or hybrid")


def run_ablations(
    evaluator: Callable[[frozenset[str]], Mapping[str, object]],
    components: Sequence[str],
) -> dict[str, Mapping[str, object]]:
    """Evaluate a full run and one-disabled-component runs through a caller hook.

    The callback must retrain when a disabled component affects fitting. This helper
    does not equate post-hoc reranking with a trained-model ablation.
    """
    if not components or len(set(components)) != len(components) or any(not c for c in components):
        raise ValueError("components must be unique nonempty names")
    results = {"full": evaluator(frozenset())}
    results.update({f"without_{name}": evaluator(frozenset([name])) for name in components})
    return results


def summarize_families(
    ranked_candidates: Sequence[str],
    metadata: Mapping[str, CandidateAnnotation],
    *,
    training_scaffolds: Sequence[str] | None = None,
) -> dict[str, object]:
    """Many-to-one scaffold/MOA counts with annotation coverage."""
    if len(set(ranked_candidates)) != len(ranked_candidates):
        raise ValueError("ranked_candidates must be unique")
    if set(ranked_candidates) - set(metadata):
        raise ValueError("Missing candidate metadata")
    annotations = [metadata[c] for c in ranked_candidates]
    scaffolds = Counter(c.scaffold for c in annotations if c.scaffold)
    families = Counter(moa for c in annotations for moa in c.moa)
    annotated = sum(scaffolds.values())
    train = set(training_scaffolds) if training_scaffolds is not None else None
    counts = np.asarray(list(scaffolds.values()), dtype=float)
    fractions = counts / counts.sum() if counts.size else counts
    entropy = -float(np.sum(fractions * np.log(fractions))) if fractions.size else None
    return {
        "candidate_count": len(annotations),
        "scaffold_annotated_count": annotated,
        "moa_annotated_count": sum(bool(c.moa) for c in annotations),
        "unique_scaffolds": len(scaffolds),
        "unique_moa_families": len(families),
        "new_scaffold_candidates": sum(n for s, n in scaffolds.items() if s not in train)
        if train is not None
        else None,
        "scaffold_entropy_nats": entropy,
        "effective_scaffold_count": float(np.exp(entropy)) if entropy is not None else None,
        "scaffolds": dict(sorted(scaffolds.items())),
        "moa_families": dict(sorted(families.items())),
    }
