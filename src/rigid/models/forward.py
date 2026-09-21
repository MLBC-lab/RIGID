"""Forward-response baselines for independent validation experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoders import DMPNNEncoder
from .graph import MolecularGraphLike, incoming_messages, tensor_graph, validate_graph


class NeuralProfilePredictor(nn.Module):
    """Learn a molecular-graph to response-vector regression baseline.

    The optional context vector can encode dose, time, and cell identity. This
    is a graph-regression baseline, not the ExPO operator.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        profile_dim: int = 978,
        hidden_dim: int = 128,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if profile_dim < 1 or context_dim < 0:
            raise ValueError("profile_dim must be positive and context_dim nonnegative")
        self.context_dim = context_dim
        self.config = {
            "atom_dim": atom_dim,
            "bond_dim": bond_dim,
            "profile_dim": profile_dim,
            "hidden_dim": hidden_dim,
            "context_dim": context_dim,
            "dropout": dropout,
        }
        self.encoder = DMPNNEncoder(atom_dim, bond_dim, hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, profile_dim),
        )

    @property
    def molecule_encoder(self) -> DMPNNEncoder:
        return self.encoder

    def forward(self, graph: MolecularGraphLike, context: Tensor | None = None) -> Tensor:
        encoded = self.encoder(graph)
        if self.context_dim:
            if context is None:
                raise ValueError("context features are required by this predictor")
            context = torch.as_tensor(context, dtype=encoded.dtype, device=encoded.device)
            if (
                context.shape != (len(encoded), self.context_dim)
                or not torch.isfinite(context).all()
            ):
                raise ValueError("context features have an invalid shape or nonfinite values")
            encoded = torch.cat((encoded, context), -1)
        elif context is not None:
            raise ValueError("context supplied to a predictor configured without context")
        return self.head(encoded)


class GraphTransformerPredictor(nn.Module):
    """Independent atom-message/Transformer response predictor.

    No positional encoding is used: atom enumeration must not alter a molecule's
    prediction. The dense attention stage is quadratic in the largest molecule.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        profile_dim: int = 978,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if (
            min(atom_dim, bond_dim, profile_dim, hidden_dim, heads, layers) < 1
            or hidden_dim % heads
        ):
            raise ValueError("positive dimensions required; hidden_dim must be divisible by heads")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        self.atom_dim, self.bond_dim, self.hidden_dim = atom_dim, bond_dim, hidden_dim
        self.context_dim = context_dim
        self.config = {
            "atom_dim": atom_dim,
            "bond_dim": bond_dim,
            "profile_dim": profile_dim,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "layers": layers,
            "dropout": dropout,
            "context_dim": context_dim,
        }
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.bond_projection = nn.Linear(bond_dim, hidden_dim, bias=False)
        self.message_update = nn.GRUCell(hidden_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            4 * hidden_dim,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden_dim + context_dim, profile_dim)

    def forward(self, graph: MolecularGraphLike, context: Tensor | None = None) -> Tensor:
        batch = tensor_graph(graph, self.atom_projection.weight.device)
        validate_graph(batch, self.atom_dim, self.bond_dim)
        atoms = torch.relu(self.atom_projection(batch.atom_features))
        src, dst = batch.edge_index
        edge_states = self.bond_projection(batch.bond_features)
        for _ in range(2):
            message = incoming_messages(torch.relu(atoms[src] + edge_states), dst, len(atoms))
            atoms = self.message_update(message, atoms)
        counts = torch.bincount(batch.graph_index, minlength=batch.num_graphs)
        width = int(counts.max())
        dense = atoms.new_zeros((batch.num_graphs, width, self.hidden_dim))
        padding = torch.ones((batch.num_graphs, width), dtype=torch.bool, device=atoms.device)
        for index in range(batch.num_graphs):
            local = atoms[batch.graph_index == index]
            dense[index, : len(local)] = local
            padding[index, : len(local)] = False
        transformed = self.transformer(dense, src_key_padding_mask=padding)
        transformed = transformed.masked_fill(padding.unsqueeze(-1), 0)
        pooled = transformed.sum(1) / counts.to(atoms.dtype).unsqueeze(-1)
        if self.context_dim:
            if context is None:
                raise ValueError("context features are required by this predictor")
            context = torch.as_tensor(context, dtype=pooled.dtype, device=pooled.device)
            if (
                context.shape != (len(pooled), self.context_dim)
                or not torch.isfinite(context).all()
            ):
                raise ValueError("context features have an invalid shape or nonfinite values")
            pooled = torch.cat((pooled, context), -1)
        elif context is not None:
            raise ValueError("context supplied to a predictor configured without context")
        return self.head(pooled)
