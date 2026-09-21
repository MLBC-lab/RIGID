"""Explicit graph construction actions with a restricted neutral chemistry domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NEUTRAL_VALENCES = {"C": 4, "N": 3, "O": 2, "F": 1, "P": 3, "S": 2, "Cl": 1, "Br": 1, "I": 1}


@dataclass(frozen=True)
class GeneratorConfig:
    atom_symbols: tuple[str, ...] = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
    hidden_dim: int = 128
    condition_dim: int = 0
    message_steps: int = 3
    max_nodes: int = 48
    max_steps: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        # JSON configuration round trips may restore a list.
        object.__setattr__(self, "atom_symbols", tuple(self.atom_symbols))
        if not self.atom_symbols or len(set(self.atom_symbols)) != len(self.atom_symbols):
            raise ValueError("atom_symbols must be nonempty and unique")
        if any(symbol not in NEUTRAL_VALENCES for symbol in self.atom_symbols):
            raise ValueError("unsupported neutral atom in generator vocabulary")
        if min(self.hidden_dim, self.message_steps, self.max_nodes, self.max_steps) < 1:
            raise ValueError("generator dimensions and limits must be positive")
        if self.condition_dim < 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid condition_dim or dropout")


@dataclass(frozen=True)
class GraphAction:
    kind: Literal["add", "connect", "stop"]
    atom_type: int = -1
    anchor: int = -1
    partner: int = -1
    bond_order: int = 1


@dataclass(frozen=True)
class GraphState:
    """Heavy atoms and undirected integer-order bonds in construction order."""

    atoms: tuple[int, ...] = ()
    bonds: tuple[tuple[int, int, int], ...] = ()
    terminated: bool = False

    def used_valence(self) -> tuple[int, ...]:
        used = [0] * len(self.atoms)
        for start, end, order in self.bonds:
            used[start] += order
            used[end] += order
        return tuple(used)

    def allowed_actions(self, config: GeneratorConfig) -> tuple[GraphAction, ...]:
        """Enumerate only actions within vocabulary, connectivity, and valence limits."""
        if self.terminated:
            return ()
        if not self.atoms:
            return tuple(
                GraphAction("add", atom_type=index) for index in range(len(config.atom_symbols))
            )
        remaining = [
            NEUTRAL_VALENCES[config.atom_symbols[atom]] - used
            for atom, used in zip(self.atoms, self.used_valence())
        ]
        actions = [GraphAction("stop")]
        if len(self.atoms) < config.max_nodes:
            for anchor, capacity in enumerate(remaining):
                for atom_type, symbol in enumerate(config.atom_symbols):
                    for order in range(1, min(capacity, NEUTRAL_VALENCES[symbol], 3) + 1):
                        actions.append(GraphAction("add", atom_type, anchor, bond_order=order))
        existing = {(min(a, b), max(a, b)) for a, b, _ in self.bonds}
        for anchor in range(len(self.atoms)):
            for partner in range(anchor + 1, len(self.atoms)):
                if (anchor, partner) in existing:
                    continue
                for order in range(1, min(remaining[anchor], remaining[partner], 3) + 1):
                    actions.append(
                        GraphAction("connect", anchor=anchor, partner=partner, bond_order=order)
                    )
        return tuple(actions)

    def apply(self, action: GraphAction, config: GeneratorConfig) -> GraphState:
        if action not in self.allowed_actions(config):
            raise ValueError(f"illegal graph action: {action}")
        if action.kind == "stop":
            return GraphState(self.atoms, self.bonds, terminated=True)
        if action.kind == "add":
            bonds = self.bonds
            if self.atoms:
                bonds += ((action.anchor, len(self.atoms), action.bond_order),)
            return GraphState(self.atoms + (action.atom_type,), bonds)
        return GraphState(
            self.atoms, self.bonds + ((action.anchor, action.partner, action.bond_order),)
        )

    def to_smiles(self, config: GeneratorConfig) -> str:
        """Sanitize the completed graph; invalid chemistry raises ValueError."""
        from rdkit import Chem

        if not self.atoms:
            raise ValueError("empty graph is not a molecule")
        mol = Chem.RWMol()
        for atom_type in self.atoms:
            mol.AddAtom(Chem.Atom(config.atom_symbols[atom_type]))
        bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for start, end, order in self.bonds:
            mol.AddBond(start, end, bond_types[order])
        result = mol.GetMol()
        try:
            Chem.SanitizeMol(result)
            smiles = Chem.MolToSmiles(result, canonical=True)
        except (ValueError, RuntimeError) as exc:
            raise ValueError("generated graph failed RDKit sanitization") from exc
        if not smiles or "." in smiles:
            raise ValueError("generator must produce a connected nonempty molecule")
        return smiles


def teacher_actions(smiles: str, config: GeneratorConfig) -> tuple[GraphAction, ...]:
    """Create a deterministic breadth-first trace, including ring closures.

    Charged atoms, isotopes, stereochemistry, radicals, and disconnected salts
    are rejected so training never silently changes the requested chemistry.
    Aromatic molecules use a deterministic Kekule representation.
    """
    from collections import deque

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid training SMILES: {smiles!r}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("generator training requires one connected molecule")
    if mol.GetNumAtoms() > config.max_nodes:
        raise ValueError("training molecule exceeds max_nodes")
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in config.atom_symbols:
            raise ValueError(f"atom {atom.GetSymbol()} is outside the generator vocabulary")
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons():
            raise ValueError(
                "charged atoms, isotopes, and radicals are outside the generator domain"
            )
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            raise ValueError("stereochemistry is outside the generator domain")
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()):
        raise ValueError("bond stereochemistry is outside the generator domain")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    order, parent = [], {0: -1}
    pending = deque([0])
    while pending:
        index = pending.popleft()
        order.append(index)
        for neighbor in sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(index).GetNeighbors()):
            if neighbor not in parent:
                parent[neighbor] = index
                pending.append(neighbor)
    mapping: dict[int, int] = {}
    actions: list[GraphAction] = []
    state = GraphState()
    for index in order:
        atom_type = config.atom_symbols.index(mol.GetAtomWithIdx(index).GetSymbol())
        if index == 0:
            action = GraphAction("add", atom_type)
        else:
            bond_order = mol.GetBondBetweenAtoms(index, parent[index]).GetBondTypeAsDouble()
            if bond_order not in (1.0, 2.0, 3.0):
                raise ValueError("unsupported bond order in training molecule")
            action = GraphAction(
                "add", atom_type, mapping[parent[index]], bond_order=int(bond_order)
            )
        state = state.apply(action, config)
        actions.append(action)
        mapping[index] = len(mapping)
        for bond in mol.GetAtomWithIdx(index).GetBonds():
            neighbor = bond.GetOtherAtomIdx(index)
            if neighbor not in mapping or neighbor == parent[index]:
                continue
            bond_order = bond.GetBondTypeAsDouble()
            if bond_order not in (1.0, 2.0, 3.0):
                raise ValueError("unsupported bond order in training molecule")
            anchor, partner = sorted((mapping[index], mapping[neighbor]))
            action = GraphAction(
                "connect", anchor=anchor, partner=partner, bond_order=int(bond_order)
            )
            state = state.apply(action, config)
            actions.append(action)
    actions.append(GraphAction("stop"))
    if len(actions) > config.max_steps:
        raise ValueError("training trace exceeds max_steps")
    return tuple(actions)
