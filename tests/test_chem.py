"""Chemical identity and graph invariants required by the ranking model."""

from dataclasses import replace

import numpy as np
import pytest

from rigid.chem import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    batch_graphs,
    chemical_properties,
    graph_batch_from_smiles,
    graph_from_smiles,
    maximum_tanimoto,
    scaffold_key,
    standardize_smiles,
)


@pytest.mark.parametrize("smiles", ["", " ", "not_a_smiles", "*", "C1CC"])
def test_invalid_or_empty_molecule_rejected(smiles):
    with pytest.raises(ValueError):
        graph_from_smiles(smiles)


def test_parent_salt_standardization_is_stable():
    parent = standardize_smiles("CC[NH3+].[Cl-]")
    assert parent.canonical_smiles == "CCN"
    assert parent.fragments_removed == 1
    assert standardize_smiles(parent.canonical_smiles).canonical_smiles == parent.canonical_smiles


def test_stereochemistry_retained_but_connectivity_grouped():
    left, right = standardize_smiles("C[C@H](O)C(=O)O"), standardize_smiles("C[C@@H](O)C(=O)O")
    assert left.canonical_smiles != right.canonical_smiles
    assert left.connectivity_key == right.connectivity_key


def test_acyclic_scaffold_policy_is_explicit():
    assert scaffold_key("CCO", "group") == scaffold_key("CCN", "group") == "ACYCLIC"
    assert scaffold_key("CCO", "compound") != scaffold_key("CCN", "compound")
    assert scaffold_key("CCO", "exclude") is None
    assert scaffold_key("Cc1ccccc1") == scaffold_key("Oc1ccccc1")
    with pytest.raises(ValueError):
        scaffold_key("CCO", "silent")


@pytest.mark.parametrize("smiles", ["C", "CCO", "c1ccccc1", "C[C@H](O)C(=O)O"])
def test_directed_graph_reverse_edges(smiles):
    graph = graph_from_smiles(smiles)
    assert graph.atom_features.shape[1] == ATOM_FEATURE_DIM
    assert graph.bond_features.shape[1] == BOND_FEATURE_DIM
    assert graph.edge_index.shape[0] == 2
    np.testing.assert_array_equal(graph.edge_index[:, graph.reverse_edge], graph.edge_index[::-1])
    np.testing.assert_array_equal(
        graph.reverse_edge[graph.reverse_edge], np.arange(len(graph.reverse_edge))
    )


def test_batched_graphs_keep_edges_inside_each_molecule():
    graphs = [graph_from_smiles(smiles) for smiles in ("CCO", "C", "c1ccccc1")]
    batch = batch_graphs(graphs)
    assert batch.num_graphs == 3
    np.testing.assert_array_equal(
        batch.graph_index[batch.edge_index[0]], batch.graph_index[batch.edge_index[1]]
    )
    np.testing.assert_array_equal(batch.edge_index[:, batch.reverse_edge], batch.edge_index[::-1])
    assert set(batch.graph_index) == {0, 1, 2}


def test_empty_batch_rejected_and_single_atom_supported():
    with pytest.raises(ValueError, match="empty"):
        batch_graphs([])
    single = graph_batch_from_smiles(["C"])
    assert single.edge_index.shape == (2, 0)
    assert single.atom_features.shape == (1, ATOM_FEATURE_DIM)


def test_malformed_reverse_mapping_rejected():
    graph = graph_from_smiles("CCO")
    with pytest.raises(ValueError, match="Reverse"):
        replace(graph, reverse_edge=np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError, match="Reverse"):
        replace(graph, reverse_edge=np.full(4, 100, dtype=np.int64))
    with pytest.raises(ValueError, match="at least one"):
        replace(graph, atom_features=np.zeros((0, ATOM_FEATURE_DIM)))


def test_torch_conversion_preserves_graph_indices():
    pytest.importorskip("torch")
    batch = graph_batch_from_smiles(["CCO", "C"])
    converted = batch.to("cpu")
    assert converted.atom_features.device.type == "cpu"
    np.testing.assert_array_equal(converted.reverse_edge.numpy(), batch.reverse_edge)


def test_properties_and_tanimoto_are_chemical_not_score_proxies():
    properties = chemical_properties("CCO")
    assert 45 < properties.molecular_weight < 47
    assert properties.lipinski_violations == 0
    assert 0 <= properties.qed <= 1
    similarity = maximum_tanimoto(["CCO", "c1ccccc1"], ["OCC"])
    assert similarity[0] == pytest.approx(1.0)
    assert 0 <= similarity[1] < 1
    with pytest.raises(ValueError):
        maximum_tanimoto(["CCO"], [])
