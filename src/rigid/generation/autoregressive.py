"""GraphINVENT-inspired propose-score-rerank candidate proposal model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actions import GeneratorConfig, GraphAction, GraphState, teacher_actions


@dataclass(frozen=True)
class GeneratedMolecule:
    smiles: str | None
    valid: bool
    terminated: bool
    steps: int
    log_probability: float
    rejection_reason: str | None = None


class AutoregressiveGraphGenerator(nn.Module):
    """Predict a normalized distribution over legal add/connect/stop actions.

    Teacher forcing and sampling use the same explicit action masks. Parameters
    are newly initialized; the model is not an upstream GraphINVENT checkpoint.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        super().__init__()
        self.settings = config or GeneratorConfig()
        self.config = asdict(self.settings)
        hidden = self.settings.hidden_dim
        vocabulary = len(self.settings.atom_symbols)
        self.atom_embedding = nn.Embedding(vocabulary, hidden)
        self.bond_embedding = nn.Embedding(4, hidden)
        self.valence_projection = nn.Linear(1, hidden, bias=False)
        self.update = nn.GRUCell(hidden, hidden)
        self.condition_projection = (
            nn.Linear(self.settings.condition_dim, hidden) if self.settings.condition_dim else None
        )
        self.dropout = nn.Dropout(self.settings.dropout)
        self.initial_head = nn.Linear(hidden, vocabulary)
        self.add_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, vocabulary * 3)
        )
        self.connect_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.stop_head = nn.Linear(hidden, 1)

    def _conditions(self, conditions: Tensor | None, count: int) -> Tensor | None:
        device = self.atom_embedding.weight.device
        if self.settings.condition_dim:
            if conditions is None:
                raise ValueError("this generator requires target conditions")
            conditions = torch.as_tensor(conditions, dtype=torch.float32, device=device)
            if conditions.ndim == 1:
                conditions = conditions[None]
            if conditions.shape == (1, self.settings.condition_dim):
                conditions = conditions.expand(count, -1)
            if (
                conditions.shape != (count, self.settings.condition_dim)
                or not torch.isfinite(conditions).all()
            ):
                raise ValueError(
                    "conditions must have shape [molecules, condition_dim] and be finite"
                )
            return conditions
        if conditions is not None:
            raise ValueError("conditions supplied to an unconditional generator")
        return None

    def _encode(self, state: GraphState, condition: Tensor | None) -> tuple[Tensor, Tensor]:
        device = self.atom_embedding.weight.device
        hidden = self.settings.hidden_dim
        if state.atoms:
            atom_indices = torch.tensor(state.atoms, dtype=torch.long, device=device)
            valences = (
                torch.tensor(state.used_valence(), dtype=torch.float32, device=device)[:, None] / 4
            )
            nodes = self.atom_embedding(atom_indices) + self.valence_projection(valences)
            if state.bonds:
                directed = [(a, b, order) for a, b, order in state.bonds] + [
                    (b, a, order) for a, b, order in state.bonds
                ]
                edge = torch.tensor(directed, dtype=torch.long, device=device)
                src, dst, bond_order = edge.unbind(-1)
                bonds = self.bond_embedding(bond_order)
                for _ in range(self.settings.message_steps):
                    aggregate = nodes.new_zeros(nodes.shape).index_add(
                        0, dst, torch.relu(nodes[src] + bonds)
                    )
                    nodes = self.dropout(self.update(aggregate, nodes))
            pooled = nodes.mean(0)
        else:
            nodes = self.atom_embedding.weight.new_zeros((0, hidden))
            pooled = self.atom_embedding.weight.new_zeros(hidden)
        if self.condition_projection is not None:
            if condition is None:
                raise ValueError("a condition is required")
            pooled = pooled + self.condition_projection(condition)
        return nodes, pooled

    def action_logits(
        self, state: GraphState, condition: Tensor | None = None
    ) -> tuple[tuple[GraphAction, ...], Tensor]:
        """Return legal actions and their unnormalized logits in matching order."""
        actions = state.allowed_actions(self.settings)
        if not actions:
            raise ValueError("terminated state has no next action")
        nodes, pooled = self._encode(state, condition)
        if not state.atoms:
            return actions, self.initial_head(pooled)
        repeated = pooled[None].expand(len(nodes), -1)
        add = self.add_head(torch.cat((nodes, repeated), -1)).view(
            len(nodes), len(self.settings.atom_symbols), 3
        )
        stop = self.stop_head(pooled).squeeze(-1)
        connections = [action for action in actions if action.kind == "connect"]
        connection_values: dict[GraphAction, Tensor] = {}
        if connections:
            indices = torch.tensor(
                [(a.anchor, a.partner) for a in connections], device=nodes.device
            )
            left, right = nodes[indices[:, 0]], nodes[indices[:, 1]]
            features = torch.cat(
                (left + right, (left - right).abs(), pooled[None].expand(len(connections), -1)), -1
            )
            logits = self.connect_head(features)
            orders = torch.tensor([a.bond_order - 1 for a in connections], device=nodes.device)
            chosen = logits.gather(1, orders[:, None]).squeeze(-1)
            connection_values = dict(zip(connections, chosen.unbind()))
        values = []
        for action in actions:
            if action.kind == "stop":
                values.append(stop)
            elif action.kind == "add":
                values.append(add[action.anchor, action.atom_type, action.bond_order - 1])
            else:
                values.append(connection_values[action])
        return actions, torch.stack(values)

    def teacher_forcing_loss(
        self, smiles: Sequence[str], conditions: Tensor | None = None
    ) -> Tensor:
        """Mean action cross-entropy across a batch of canonical construction traces."""
        if not smiles:
            raise ValueError("at least one training molecule is required")
        conditions = self._conditions(conditions, len(smiles))
        losses = []
        for index, text in enumerate(smiles):
            condition = None if conditions is None else conditions[index]
            state = GraphState()
            for action in teacher_actions(text, self.settings):
                actions, logits = self.action_logits(state, condition)
                label = torch.tensor([actions.index(action)], device=logits.device)
                losses.append(F.cross_entropy(logits[None], label))
                state = state.apply(action, self.settings)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        conditions: Tensor | None = None,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> list[GeneratedMolecule]:
        """Sample proposals, retaining termination/validity failures in the report."""
        if n < 1 or temperature <= 0:
            raise ValueError("n and temperature must be positive")
        conditions = self._conditions(conditions, n)
        rng = torch.Generator(device=self.atom_embedding.weight.device).manual_seed(seed)
        was_training = self.training
        self.eval()
        results = []
        try:
            for index in range(n):
                condition = None if conditions is None else conditions[index]
                state, log_probability = GraphState(), 0.0
                for step in range(1, self.settings.max_steps + 1):
                    actions, logits = self.action_logits(state, condition)
                    probabilities = (logits / temperature).softmax(-1)
                    selected = int(torch.multinomial(probabilities, 1, generator=rng))
                    log_probability += float(probabilities[selected].clamp_min(1e-30).log())
                    state = state.apply(actions[selected], self.settings)
                    if state.terminated:
                        break
                if not state.terminated:
                    results.append(
                        GeneratedMolecule(
                            None, False, False, step, log_probability, "max_steps reached"
                        )
                    )
                    continue
                try:
                    smiles = state.to_smiles(self.settings)
                    results.append(GeneratedMolecule(smiles, True, True, step, log_probability))
                except ValueError as exc:
                    results.append(
                        GeneratedMolecule(None, False, True, step, log_probability, str(exc))
                    )
        finally:
            self.train(was_training)
        return results
