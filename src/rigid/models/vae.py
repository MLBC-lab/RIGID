"""A conditional fingerprint VAE for a library-constrained inverse baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoders import TargetEncoder


@dataclass(frozen=True)
class VAEOutput:
    logits: Tensor
    mean: Tensor
    log_variance: Tensor


class ConditionalFingerprintVAE(nn.Module):
    """Predict fingerprint distributions conditioned on a desired profile.

    The decoder has no graph or SMILES output. Candidate library fingerprints
    can be ranked against sampled predictions, preserving chemical validity by
    construction. This explicit baseline choice is not specified in the paper.
    """

    def __init__(
        self,
        profile_dim: int = 978,
        fingerprint_dim: int = 2048,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(fingerprint_dim, hidden_dim, latent_dim) < 1:
            raise ValueError("VAE dimensions must be positive")
        self.fingerprint_dim, self.latent_dim = fingerprint_dim, latent_dim
        self.config = {
            "profile_dim": profile_dim,
            "fingerprint_dim": fingerprint_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "dropout": dropout,
        }
        self.condition_encoder = TargetEncoder(profile_dim, hidden_dim, hidden_dim, dropout)
        self.posterior = nn.Sequential(
            nn.Linear(fingerprint_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fingerprint_dim),
        )

    def forward(self, fingerprints: Tensor, profiles: Tensor, sample: bool = True) -> VAEOutput:
        condition = self.condition_encoder(profiles)
        fingerprints = torch.as_tensor(fingerprints, device=condition.device, dtype=condition.dtype)
        if fingerprints.shape != (len(condition), self.fingerprint_dim):
            raise ValueError("fingerprints must align with target profiles")
        if (
            not torch.isfinite(fingerprints).all()
            or ((fingerprints < 0) | (fingerprints > 1)).any()
        ):
            raise ValueError("fingerprints must contain finite values in [0, 1]")
        mean, log_variance = self.posterior(torch.cat((fingerprints, condition), -1)).chunk(2, -1)
        log_variance = log_variance.clamp(-20, 12)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance) if sample else mean
        return VAEOutput(self.decoder(torch.cat((latent, condition), -1)), mean, log_variance)

    @staticmethod
    def loss(
        output: VAEOutput, fingerprints: Tensor, beta: float = 1.0
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        fingerprints = torch.as_tensor(
            fingerprints, device=output.logits.device, dtype=output.logits.dtype
        )
        reconstruction = (
            F.binary_cross_entropy_with_logits(output.logits, fingerprints, reduction="none")
            .sum(-1)
            .mean()
        )
        kl = (
            -0.5
            * (1 + output.log_variance - output.mean.square() - output.log_variance.exp())
            .sum(-1)
            .mean()
        )
        return reconstruction + beta * kl, {
            "reconstruction": reconstruction.detach(),
            "kl": kl.detach(),
        }

    def sample_fingerprints(
        self, profiles: Tensor, samples: int = 32, generator: torch.Generator | None = None
    ) -> Tensor:
        """Return fingerprint probabilities with shape [targets, samples, bits]."""
        if samples < 1:
            raise ValueError("samples must be positive")
        condition = self.condition_encoder(profiles)
        latent = torch.randn(
            (len(condition), samples, self.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        expanded = condition[:, None].expand(-1, samples, -1)
        return self.decoder(torch.cat((latent, expanded), -1)).sigmoid()

    def rank_library(
        self,
        profiles: Tensor,
        fingerprints: Tensor,
        samples: int = 32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return Monte Carlo mean soft-Tanimoto scores [targets, candidates]."""
        predictions = self.sample_fingerprints(profiles, samples, generator)
        fingerprints = torch.as_tensor(
            fingerprints, device=predictions.device, dtype=predictions.dtype
        )
        if fingerprints.ndim != 2 or fingerprints.shape[1] != self.fingerprint_dim:
            raise ValueError("library fingerprints have the wrong dimension")
        if (
            not torch.isfinite(fingerprints).all()
            or ((fingerprints < 0) | (fingerprints > 1)).any()
        ):
            raise ValueError("library fingerprints must be finite and lie in [0, 1]")
        intersection = torch.einsum("tsf,cf->tsc", predictions, fingerprints)
        union = predictions.sum(-1, keepdim=True) + fingerprints.sum(-1) - intersection
        return (intersection / union.clamp_min(1e-8)).mean(1)
