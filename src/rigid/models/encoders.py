"""Molecular and target encoders used by the inverse ranking model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .graph import (
    MolecularGraphLike,
    incoming_messages,
    mean_pool,
    reverse_excluded_messages,
    tensor_graph,
    validate_graph,
)


class DMPNNEncoder(nn.Module):
    """Chemprop-style directed bond messages followed by mean atom pooling.

    This implementation follows the architecture; its feature schema and weights
    are local to RIGID and are not interchangeable with a Chemprop checkpoint.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if min(atom_dim, bond_dim, hidden_dim, depth) < 1:
            raise ValueError("feature sizes, hidden_dim, and depth must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.validate_inputs = validate_inputs
        self.input_projection = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_projection = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: MolecularGraphLike) -> Tensor:
        batch = tensor_graph(graph, self.input_projection.weight.device)
        if self.validate_inputs:
            validate_graph(batch, self.atom_dim, self.bond_dim)
        atoms = batch.atom_features
        src, dst = batch.edge_index
        initial = torch.relu(
            self.input_projection(torch.cat((atoms[src], batch.bond_features), -1))
        )
        messages = initial
        for _ in range(self.depth - 1):
            incoming = reverse_excluded_messages(
                messages, batch.edge_index, batch.reverse_edge, len(atoms)
            )
            messages = self.dropout(torch.relu(initial + self.message_projection(incoming)))
        atom_incoming = incoming_messages(messages, dst, len(atoms))
        states = self.dropout(
            torch.relu(self.atom_projection(torch.cat((atoms, atom_incoming), -1)))
        )
        return mean_pool(states, batch.graph_index, batch.num_graphs)


class TargetEncoder(nn.Module):
    """Encode a vector of ordered, training-normalized expression features."""

    def __init__(
        self,
        profile_dim: int = 978,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(profile_dim, hidden_dim, embedding_dim) < 1:
            raise ValueError("target dimensions must be positive")
        self.profile_dim = profile_dim
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, profiles: Tensor) -> Tensor:
        parameter = next(self.parameters())
        profiles = torch.as_tensor(profiles, dtype=parameter.dtype, device=parameter.device)
        if profiles.ndim == 1:
            profiles = profiles.unsqueeze(0)
        if profiles.ndim != 2 or profiles.shape[-1] != self.profile_dim:
            raise ValueError(f"profiles must have shape [batch, {self.profile_dim}]")
        if not torch.isfinite(profiles).all():
            raise ValueError("profiles must be finite; impute missing values before encoding")
        return self.network(profiles)
