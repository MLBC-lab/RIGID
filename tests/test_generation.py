"""Executable chemistry traces, valence masks, sampling, and gradient checks."""

import pytest

torch = pytest.importorskip("torch")
Chem = pytest.importorskip("rdkit.Chem")

from rigid.generation import (  # noqa: E402
    AutoregressiveGraphGenerator,
    GeneratedMolecule,
    GeneratorConfig,
    GraphAction,
    GraphState,
    select_candidates,
    teacher_actions,
)


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)N", "C1CCCCC1", "c1ccccc1O", "CC(Cl)Br", "N#CC"])
def test_teacher_trace_round_trips_molecular_graph(smiles):
    config = GeneratorConfig(hidden_dim=16)
    state = GraphState()
    actions = teacher_actions(smiles, config)
    for action in actions:
        assert action in state.allowed_actions(config)
        state = state.apply(action, config)
    assert state.terminated
    assert state.to_smiles(config) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    if "1" in smiles:
        assert any(action.kind == "connect" for action in actions)


@pytest.mark.parametrize(
    "smiles", ["[NH4+]", "C.C", "F[C@H](Cl)Br", "[13CH4]", "[CH3]", "C/C=C/C", "[Na]"]
)
def test_unsupported_chemistry_is_rejected_explicitly(smiles):
    with pytest.raises(ValueError):
        teacher_actions(smiles, GeneratorConfig())


def test_valence_mask_blocks_hypervalent_carbon_and_duplicate_bonds():
    config = GeneratorConfig(atom_symbols=("C", "F"), max_nodes=8)
    state = GraphState().apply(GraphAction("add", atom_type=0), config)
    for _ in range(4):
        state = state.apply(GraphAction("add", atom_type=1, anchor=0), config)
    actions = state.allowed_actions(config)
    assert actions == (GraphAction("stop"),)
    with pytest.raises(ValueError, match="illegal"):
        state.apply(GraphAction("add", atom_type=0, anchor=0), config)
    assert state.to_smiles(config) == "FC(F)(F)F"


def test_generator_cross_entropy_has_parameter_gradients():
    torch.manual_seed(2)
    model = AutoregressiveGraphGenerator(GeneratorConfig(hidden_dim=16, message_steps=2, dropout=0))
    loss = model.teacher_forcing_loss(["CCO", "C1CC1"])
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    for module in (
        model.atom_embedding,
        model.update,
        model.add_head,
        model.connect_head,
        model.stop_head,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_conditional_generator_uses_target_and_checks_dimensions():
    model = AutoregressiveGraphGenerator(
        GeneratorConfig(hidden_dim=12, condition_dim=5, message_steps=1, dropout=0)
    )
    conditions = torch.randn(1, 5, requires_grad=True)
    loss = model.teacher_forcing_loss(["CC"], conditions)
    loss.backward()
    assert conditions.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires target"):
        model.teacher_forcing_loss(["CC"])


def test_sampling_is_seeded_and_restores_training_mode():
    model = AutoregressiveGraphGenerator(
        GeneratorConfig(hidden_dim=12, max_nodes=4, max_steps=16, message_steps=1)
    )
    model.train()
    first, second = model.sample(4, seed=17), model.sample(4, seed=17)
    assert first == second
    assert model.training
    for item in first:
        assert item.steps <= 16
        if item.valid:
            assert item.terminated and Chem.MolFromSmiles(item.smiles) is not None


def test_nonterminated_sampling_is_counted_as_failure():
    model = AutoregressiveGraphGenerator(GeneratorConfig(hidden_dim=8, max_steps=1))
    proposals = model.sample(2, seed=0)
    assert all(
        not item.valid and not item.terminated and item.rejection_reason for item in proposals
    )


def test_selection_reports_denominators_and_filters():
    proposals = [
        GeneratedMolecule("CC", True, True, 3, -1),
        GeneratedMolecule("CC", True, True, 3, -1),
        GeneratedMolecule("CO", True, True, 3, -1),
        GeneratedMolecule(None, False, False, 3, -1),
    ]
    report = select_candidates(
        proposals, lambda smiles: len(smiles), accept=lambda smiles: smiles != "CO"
    )
    assert (
        report.attempted == 4 and report.valid == 3 and report.unique == 2 and report.accepted == 1
    )
    assert report.validity == 0.75 and report.uniqueness == pytest.approx(2 / 3)
    assert report.candidates == (("CC", 2.0),)
    with pytest.raises(ValueError, match="nonfinite"):
        select_candidates(proposals, lambda _: float("nan"))
