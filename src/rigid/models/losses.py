"""Ranking and partially observed profile losses."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pairwise_rank_loss(
    positive: Tensor, negative: Tensor, weights: Tensor | None = None, margin: float = 0.0
) -> Tensor:
    """Smooth pairwise ranking loss for aligned positive and negative scores."""
    if positive.shape != negative.shape or positive.numel() == 0:
        raise ValueError("positive and negative scores must have the same nonempty shape")
    values = F.softplus(negative - positive + margin)
    if weights is None:
        return values.mean()
    if weights.shape != values.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite, nonnegative, and match score shape")
    if weights.sum() <= 0:
        raise ValueError("at least one ranking weight must be positive")
    return (weights * values).sum() / weights.sum()


def listwise_rank_loss(scores: Tensor, relevance: Tensor, temperature: float = 1.0) -> Tensor:
    """Cross-entropy against normalized graded relevance within candidate sets."""
    if scores.ndim != 2 or scores.shape != relevance.shape or scores.numel() == 0:
        raise ValueError("scores and relevance must be equally shaped nonempty matrices")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if (
        not torch.isfinite(relevance).all()
        or (relevance < 0).any()
        or (relevance.sum(-1) <= 0).any()
    ):
        raise ValueError(
            "each candidate set needs finite nonnegative relevance and a positive label"
        )
    distribution = relevance / relevance.sum(-1, keepdim=True)
    return -(distribution * F.log_softmax(scores / temperature, dim=-1)).sum(-1).mean()


def masked_profile_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """MSE over observed genes; missing target entries may be represented by NaN."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    observed = torch.isfinite(target)
    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError("mask and target shapes must match")
        observed = observed & mask.bool()
    if not observed.any():
        raise ValueError("at least one target expression value must be observed")
    if not torch.isfinite(prediction[observed]).all():
        raise ValueError("predictions must be finite at observed genes")
    return F.mse_loss(prediction[observed], target[observed])
