"""Molecular graphs with explicit directed bonds and reverse-edge indices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from .standardize import parse_molecule, standardize_smiles

ATOM_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
ATOM_FEATURE_DIM = 46
BOND_FEATURE_DIM = 14
FEATURE_SCHEMA_VERSION = "rigid-atom46-bond14-v1"


def _one_hot(value: Any, choices: Sequence[Any]) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    hybridizations = (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    )
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_NUMBERS)
        + _one_hot(atom.GetTotalDegree(), tuple(range(6)))
        + _one_hot(atom.GetFormalCharge(), (-2, -1, 0, 1, 2))
        + _one_hot(int(atom.GetChiralTag()), (0, 1, 2, 3))
        + _one_hot(atom.GetTotalNumHs(), (0, 1, 2, 3, 4))
        + _one_hot(atom.GetHybridization(), hybridizations)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass() / 100.0]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(
            bond.GetBondType(),
            (
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ),
        )
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(int(bond.GetStereo()), (0, 1, 2, 3, 4, 5))
    )


@dataclass(frozen=True)
class MolecularGraph:
    atom_features: np.ndarray
    bond_features: np.ndarray
    edge_index: np.ndarray
    reverse_edge: np.ndarray
    smiles: str

    def __post_init__(self) -> None:
        if self.atom_features.ndim != 2 or len(self.atom_features) == 0:
            raise ValueError("A molecular graph must contain at least one atom")
        if (
            self.atom_features.shape[1] != ATOM_FEATURE_DIM
            or self.bond_features.ndim != 2
            or self.bond_features.shape[1] != BOND_FEATURE_DIM
        ):
            raise ValueError("Graph features do not match the declared feature schema")
        if not np.isfinite(self.atom_features).all() or not np.isfinite(self.bond_features).all():
            raise ValueError("Graph features must be finite")
        edges = self.bond_features.shape[0]
        if self.edge_index.shape != (2, edges) or self.reverse_edge.shape != (edges,):
            raise ValueError("Directed edge arrays are misaligned")
        if edges and (
            self.edge_index.min() < 0 or self.edge_index.max() >= len(self.atom_features)
        ):
            raise ValueError("Edge endpoint is outside the atom array")
        if edges:
            if self.reverse_edge.min() < 0 or self.reverse_edge.max() >= edges:
                raise ValueError("Reverse-edge index is outside the edge array")
            if not np.array_equal(self.reverse_edge[self.reverse_edge], np.arange(edges)):
                raise ValueError("Reverse-edge mapping must be an involution")
            if not np.array_equal(self.edge_index[:, self.reverse_edge], self.edge_index[::-1]):
                raise ValueError("Reverse edges must have reversed endpoints")


@dataclass(frozen=True)
class GraphBatch:
    atom_features: Any
    bond_features: Any
    edge_index: Any
    reverse_edge: Any
    graph_index: Any
    num_graphs: int
    smiles: tuple[str, ...]

    def to(self, device: str | Any = "cpu") -> GraphBatch:
        """Convert graph arrays to tensors lazily, without importing torch at load."""
        import torch

        return GraphBatch(
            torch.as_tensor(self.atom_features, dtype=torch.float32, device=device),
            torch.as_tensor(self.bond_features, dtype=torch.float32, device=device),
            torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            torch.as_tensor(self.reverse_edge, dtype=torch.long, device=device),
            torch.as_tensor(self.graph_index, dtype=torch.long, device=device),
            self.num_graphs,
            self.smiles,
        )


def graph_from_smiles(smiles: str, *, standardize: bool = True) -> MolecularGraph:
    if standardize:
        smiles = standardize_smiles(smiles).canonical_smiles
    mol = parse_molecule(smiles)
    atoms = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    bonds: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    reverse: list[int] = []
    for bond in mol.GetBonds():
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        index = len(edges)
        edges.extend(((source, target), (target, source)))
        features = bond_features(bond)
        bonds.extend((features, features.copy()))
        reverse.extend((index + 1, index))
    return MolecularGraph(
        atoms,
        np.asarray(bonds, dtype=np.float32).reshape(-1, BOND_FEATURE_DIM),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
        np.asarray(reverse, dtype=np.int64),
        smiles,
    )


def batch_graphs(graphs: Sequence[MolecularGraph]) -> GraphBatch:
    if not graphs:
        raise ValueError("Cannot batch an empty collection of molecular graphs")
    atoms, bonds, edges, reverse, indices = [], [], [], [], []
    atom_offset = edge_offset = 0
    for graph_number, graph in enumerate(graphs):
        atoms.append(graph.atom_features)
        bonds.append(graph.bond_features)
        edges.append(graph.edge_index + atom_offset)
        reverse.append(graph.reverse_edge + edge_offset)
        indices.append(np.full(len(graph.atom_features), graph_number, dtype=np.int64))
        atom_offset += len(graph.atom_features)
        edge_offset += len(graph.bond_features)
    return GraphBatch(
        np.concatenate(atoms),
        np.concatenate(bonds),
        np.concatenate(edges, axis=1),
        np.concatenate(reverse),
        np.concatenate(indices),
        len(graphs),
        tuple(graph.smiles for graph in graphs),
    )


def graph_batch_from_smiles(smiles: Sequence[str]) -> GraphBatch:
    return batch_graphs([graph_from_smiles(value) for value in smiles])


smiles_to_graph = graph_from_smiles
