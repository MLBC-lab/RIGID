"""Dual-encoder target-molecule compatibility ranking."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoders import DMPNNEncoder, TargetEncoder
from .graph import MolecularGraphLike


class RIGIDRanker(nn.Module):
    """Score aligned target/molecule pairs or a library against a shared target.

    Scores are unbounded ranking utilities. A sigmoid of a score is not a
    calibrated probability of inducing the target profile.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        profile_dim: int = 978,
        hidden_dim: int = 128,
        auxiliary_dim: int = 0,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if auxiliary_dim < 0:
            raise ValueError("auxiliary_dim must be nonnegative")
        self.auxiliary_dim = auxiliary_dim
        self.hidden_dim = hidden_dim
        self.config = {
            "atom_dim": atom_dim,
            "bond_dim": bond_dim,
            "profile_dim": profile_dim,
            "hidden_dim": hidden_dim,
            "auxiliary_dim": auxiliary_dim,
            "depth": depth,
            "dropout": dropout,
        }
        self.molecule_encoder = DMPNNEncoder(atom_dim, bond_dim, hidden_dim, depth, dropout)
        self.target_encoder = TargetEncoder(profile_dim, hidden_dim * 2, hidden_dim, dropout)
        self.ranking_head = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1 + auxiliary_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_embeddings(
        self, molecules: Tensor, targets: Tensor, auxiliary: Tensor | None = None
    ) -> Tensor:
        if molecules.ndim != 2 or targets.ndim != 2:
            raise ValueError("embeddings must be matrices")
        if molecules.shape[-1] != self.hidden_dim or targets.shape[-1] != self.hidden_dim:
            raise ValueError("embedding dimensions do not match the ranking head")
        if len(targets) == 1:
            targets = targets.expand(len(molecules), -1)
        if molecules.shape != targets.shape:
            raise ValueError("provide one target per molecule, or one shared target")
        pieces = [
            molecules,
            targets,
            molecules * targets,
            (molecules - targets).abs(),
            F.cosine_similarity(molecules, targets).unsqueeze(-1),
        ]
        if self.auxiliary_dim:
            if auxiliary is None:
                raise ValueError("this ranker requires the configured auxiliary features")
            auxiliary = torch.as_tensor(auxiliary, device=molecules.device, dtype=molecules.dtype)
            if auxiliary.shape != (len(molecules), self.auxiliary_dim):
                raise ValueError("auxiliary features must have shape [molecules, auxiliary_dim]")
            if not torch.isfinite(auxiliary).all():
                raise ValueError("auxiliary features must be finite")
            pieces.append(auxiliary)
        elif auxiliary is not None:
            raise ValueError("auxiliary features supplied to a ranker configured without them")
        return self.ranking_head(torch.cat(pieces, -1)).squeeze(-1)

    def forward(
        self, graph: MolecularGraphLike, profiles: Tensor, auxiliary: Tensor | None = None
    ) -> Tensor:
        return self.score_embeddings(
            self.molecule_encoder(graph), self.target_encoder(profiles), auxiliary
        )

    def score_library(
        self,
        molecules: Tensor,
        targets: Tensor,
        auxiliary: Tensor | None = None,
        chunk_size: int = 1024,
    ) -> Tensor:
        """Score N targets against M encoded molecules with bounded head batches.

        Output is [N, M]; auxiliary features, if configured, are [N, M, A].
        Call under inference_mode() for retrieval or retain gradients for fitting.
        """
        if chunk_size < 1 or molecules.ndim != 2 or targets.ndim != 2:
            raise ValueError("chunk_size must be positive and embeddings must be matrices")
        if len(molecules) == 0 or len(targets) == 0:
            raise ValueError("provide at least one target and molecule")
        if auxiliary is not None and auxiliary.shape != (
            len(targets),
            len(molecules),
            self.auxiliary_dim,
        ):
            raise ValueError(
                "library auxiliary features must have shape [targets, molecules, auxiliary_dim]"
            )
        rows = []
        for index, target in enumerate(targets):
            chunks = []
            for start in range(0, len(molecules), chunk_size):
                stop = start + chunk_size
                features = None if auxiliary is None else auxiliary[index, start:stop]
                chunks.append(self.score_embeddings(molecules[start:stop], target[None], features))
            rows.append(torch.cat(chunks))
        return torch.stack(rows)
