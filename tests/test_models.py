"""Architecture invariants independent of any manuscript benchmark numbers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rigid.models import (  # noqa: E402
    ConditionalFingerprintVAE,
    DMPNNEncoder,
    GraphTransformerPredictor,
    NeuralProfilePredictor,
    RIGIDRanker,
    TargetEncoder,
    listwise_rank_loss,
    masked_profile_loss,
    pairwise_rank_loss,
)
from rigid.models.graph import reverse_excluded_messages  # noqa: E402


def graph_fixture(two_graphs=False):
    atoms = torch.tensor([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]])
    if two_graphs:
        atoms = torch.cat((atoms, torch.tensor([[0.6, 0.4]])))
    return SimpleNamespace(
        atom_features=atoms,
        bond_features=torch.tensor([[0.5], [0.5], [1.0], [1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        reverse_edge=torch.tensor([1, 0, 3, 2]),
        graph_index=torch.tensor([0, 0, 0, 1] if two_graphs else [0, 0, 0]),
        num_graphs=2 if two_graphs else 1,
    )


def permute_graph(graph):
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    edges = torch.tensor([3, 1, 0, 2])
    inverse_edges = torch.argsort(edges)
    return SimpleNamespace(
        atom_features=graph.atom_features[permutation],
        bond_features=graph.bond_features[edges],
        edge_index=inverse[graph.edge_index[:, edges]],
        reverse_edge=inverse_edges[graph.reverse_edge[edges]],
        graph_index=graph.graph_index[permutation],
        num_graphs=graph.num_graphs,
    )


def test_message_passing_excludes_reverse_bond():
    graph = graph_fixture()
    messages = torch.arange(1.0, 5.0)[:, None]
    output = reverse_excluded_messages(messages, graph.edge_index, graph.reverse_edge, 3)
    torch.testing.assert_close(output[:, 0], torch.tensor([0.0, 4.0, 1.0, 0.0]))


def test_encoder_is_invariant_to_atom_and_bond_order():
    torch.manual_seed(12)
    encoder = DMPNNEncoder(2, 1, hidden_dim=16, depth=4, dropout=0).eval()
    original = graph_fixture()
    torch.testing.assert_close(
        encoder(original), encoder(permute_graph(original)), atol=1e-6, rtol=1e-6
    )


def test_encoder_handles_isolated_atoms_without_cross_graph_leakage():
    encoder = DMPNNEncoder(2, 1, hidden_dim=16, dropout=0).eval()
    batched = graph_fixture(two_graphs=True)
    full = encoder(batched)
    singleton = SimpleNamespace(
        atom_features=batched.atom_features[3:],
        bond_features=torch.empty(0, 1),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        reverse_edge=torch.empty(0, dtype=torch.long),
        graph_index=torch.tensor([0]),
        num_graphs=1,
    )
    torch.testing.assert_close(full[:1], encoder(graph_fixture()))
    torch.testing.assert_close(full[1:], encoder(singleton))


def test_malformed_reverse_edges_are_rejected():
    graph = graph_fixture()
    graph.reverse_edge = torch.tensor([0, 1, 2, 3])
    with pytest.raises(ValueError, match="opposite directed bond"):
        DMPNNEncoder(2, 1)(graph)


def test_ranker_scores_joint_embeddings_with_auxiliary_and_gradients():
    torch.manual_seed(8)
    ranker = RIGIDRanker(2, 1, profile_dim=5, hidden_dim=16, auxiliary_dim=2, dropout=0)
    graph = graph_fixture(two_graphs=True)
    profiles = torch.randn(2, 5, requires_grad=True)
    scores = ranker(graph, profiles, torch.ones(2, 2))
    assert scores.shape == (2,)
    scores.sum().backward()
    assert profiles.grad.abs().sum() > 0
    for module in (ranker.molecule_encoder, ranker.target_encoder, ranker.ranking_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with pytest.raises(ValueError, match="requires.*auxiliary"):
        ranker(graph, profiles)


def test_chunked_library_scores_equal_pairwise_scoring():
    ranker = RIGIDRanker(2, 1, profile_dim=5, hidden_dim=16, dropout=0).eval()
    molecules = ranker.molecule_encoder(graph_fixture(two_graphs=True))
    targets = ranker.target_encoder(torch.randn(3, 5))
    expected = torch.stack([ranker.score_embeddings(molecules, target[None]) for target in targets])
    torch.testing.assert_close(ranker.score_library(molecules, targets, chunk_size=1), expected)
    assert RIGIDRanker(**ranker.config).config == ranker.config


def test_ranker_can_fit_a_small_pairwise_problem():
    torch.manual_seed(7)
    ranker = RIGIDRanker(2, 1, profile_dim=3, hidden_dim=12, dropout=0)
    graph = graph_fixture(two_graphs=True)
    targets = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    optimizer = torch.optim.Adam(ranker.parameters(), lr=0.015)
    losses = []
    for _ in range(25):
        scores = ranker.score_library(
            ranker.molecule_encoder(graph), ranker.target_encoder(targets)
        )
        loss = listwise_rank_loss(scores, torch.eye(2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.25 * losses[0]


def test_target_encoder_rejects_missing_or_misordered_dimensions():
    encoder = TargetEncoder(profile_dim=5, hidden_dim=8, embedding_dim=4)
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="finite"):
        encoder(torch.full((1, 5), float("nan")))


def test_ranking_loss_direction_and_graded_labels():
    positive, negative = torch.tensor([3.0]), torch.tensor([-2.0])
    assert pairwise_rank_loss(positive, negative) < pairwise_rank_loss(negative, positive)
    assert listwise_rank_loss(torch.tensor([[4.0, -1.0]]), torch.tensor([[1.0, 0.0]])) < 0.01
    with pytest.raises(ValueError, match="positive label"):
        listwise_rank_loss(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="positive"):
        pairwise_rank_loss(positive, negative, weights=torch.zeros(1))


def test_forward_predictors_are_differentiable_and_context_sensitive():
    for predictor in (
        NeuralProfilePredictor(2, 1, profile_dim=5, hidden_dim=16, context_dim=3, dropout=0),
        GraphTransformerPredictor(
            2, 1, profile_dim=5, hidden_dim=16, heads=2, layers=1, context_dim=3, dropout=0
        ),
    ):
        context = torch.randn(2, 3, requires_grad=True)
        response = predictor(graph_fixture(two_graphs=True), context)
        assert response.shape == (2, 5)
        masked_profile_loss(response, torch.randn(2, 5)).backward()
        assert context.grad.abs().sum() > 0
        assert type(predictor)(**predictor.config).config == predictor.config


def test_graph_transformer_is_permutation_invariant():
    predictor = GraphTransformerPredictor(
        2, 1, profile_dim=5, hidden_dim=16, heads=2, layers=1, dropout=0
    ).eval()
    graph = graph_fixture()
    torch.testing.assert_close(
        predictor(graph), predictor(permute_graph(graph)), atol=1e-6, rtol=1e-6
    )


def test_masked_profile_loss_ignores_missing_values():
    prediction = torch.tensor([[2.0, 99.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 2.0]])
    loss = masked_profile_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    assert prediction.grad[0, 1] == 0


def test_conditional_vae_has_finite_gradients_and_seeded_library_ranking():
    torch.manual_seed(4)
    model = ConditionalFingerprintVAE(
        profile_dim=5, fingerprint_dim=8, hidden_dim=16, latent_dim=4, dropout=0
    )
    profiles = torch.randn(2, 5)
    fingerprints = torch.tensor(
        [[1, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0]], dtype=torch.float32
    )
    output = model(fingerprints, profiles)
    loss, components = model.loss(output, fingerprints, beta=0.1)
    loss.backward()
    assert torch.isfinite(loss) and components["kl"] >= 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    first = model.rank_library(
        profiles, fingerprints, samples=3, generator=torch.Generator().manual_seed(2)
    )
    second = model.rank_library(
        profiles, fingerprints, samples=3, generator=torch.Generator().manual_seed(2)
    )
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 2) and ((first >= 0) & (first <= 1)).all()
