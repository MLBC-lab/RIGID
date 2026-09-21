"""Tensor conversion and directed molecular graph operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


class MolecularGraphLike(Protocol):
    atom_features: Any
    bond_features: Any
    edge_index: Any
    reverse_edge: Any
    graph_index: Any
    num_graphs: int


@dataclass(frozen=True)
class TensorGraph:
    atom_features: Tensor
    bond_features: Tensor
    edge_index: Tensor
    reverse_edge: Tensor
    graph_index: Tensor
    num_graphs: int


def tensor_graph(graph: MolecularGraphLike, device: torch.device) -> TensorGraph:
    """Accept a NumPy or tensor graph without detaching tensor feature gradients."""
    return TensorGraph(
        torch.as_tensor(graph.atom_features, dtype=torch.float32, device=device),
        torch.as_tensor(graph.bond_features, dtype=torch.float32, device=device),
        torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(graph.reverse_edge, dtype=torch.long, device=device),
        torch.as_tensor(graph.graph_index, dtype=torch.long, device=device),
        int(graph.num_graphs),
    )


def validate_graph(graph: TensorGraph, atom_dim: int, bond_dim: int) -> None:
    """Reject malformed batches before indexed message passing can hide errors."""
    atoms, bonds = graph.atom_features, graph.bond_features
    if atoms.ndim != 2 or atoms.shape[1] != atom_dim or atoms.shape[0] == 0:
        raise ValueError(f"atom_features must have shape [N>0, {atom_dim}]")
    if bonds.ndim != 2 or bonds.shape[1] != bond_dim:
        raise ValueError(f"bond_features must have shape [E, {bond_dim}]")
    n_atoms, n_edges = atoms.shape[0], bonds.shape[0]
    if graph.edge_index.shape != (2, n_edges):
        raise ValueError("edge_index must have shape [2, E]")
    if graph.reverse_edge.shape != (n_edges,):
        raise ValueError("reverse_edge must have shape [E]")
    if graph.graph_index.shape != (n_atoms,) or graph.num_graphs < 1:
        raise ValueError("graph_index must assign every atom to a graph")
    if not torch.isfinite(atoms).all() or not torch.isfinite(bonds).all():
        raise ValueError("molecular features must be finite")
    if (graph.graph_index < 0).any() or (graph.graph_index >= graph.num_graphs).any():
        raise ValueError("graph_index contains an out-of-range graph")
    if (torch.bincount(graph.graph_index, minlength=graph.num_graphs) == 0).any():
        raise ValueError("empty molecular graphs are not supported")
    if n_edges:
        if (graph.edge_index < 0).any() or (graph.edge_index >= n_atoms).any():
            raise ValueError("edge_index contains an out-of-range atom")
        rev = graph.reverse_edge
        if (rev < 0).any() or (rev >= n_edges).any():
            raise ValueError("reverse_edge contains an out-of-range edge")
        order = torch.arange(n_edges, device=rev.device)
        if not torch.equal(rev[rev], order):
            raise ValueError("reverse_edge must be an involution")
        src, dst = graph.edge_index
        if (src == dst).any() or not torch.equal(
            graph.edge_index[:, rev], graph.edge_index.flip(0)
        ):
            raise ValueError("each directed bond needs its opposite directed bond")
        if not torch.equal(graph.graph_index[src], graph.graph_index[dst]):
            raise ValueError("a bond cannot connect molecules in separate graphs")


def incoming_messages(messages: Tensor, destinations: Tensor, n_atoms: int) -> Tensor:
    return messages.new_zeros((n_atoms, messages.shape[-1])).index_add(0, destinations, messages)


def reverse_excluded_messages(
    messages: Tensor, edge_index: Tensor, reverse_edge: Tensor, n_atoms: int
) -> Tensor:
    """For v->w, sum u->v over u != w; the reverse edge must not echo back."""
    src, dst = edge_index
    return incoming_messages(messages, dst, n_atoms)[src] - messages[reverse_edge]


def mean_pool(node_states: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
    total = node_states.new_zeros((num_graphs, node_states.shape[-1]))
    total.index_add_(0, graph_index, node_states)
    counts = torch.bincount(graph_index, minlength=num_graphs).clamp_min(1)
    return total / counts.to(node_states.dtype).unsqueeze(-1)
