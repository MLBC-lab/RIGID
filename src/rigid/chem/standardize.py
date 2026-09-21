"""Deterministic parent-compound standardization and chemical identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

AcyclicPolicy = Literal["group", "compound", "exclude"]


@dataclass(frozen=True)
class StandardizedMolecule:
    input_smiles: str
    canonical_smiles: str
    connectivity_key: str
    scaffold: str
    fragments_removed: int


def parse_molecule(smiles: str) -> Chem.Mol:
    """Parse a nonempty, sanitized molecule, rejecting wildcard/query atoms."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("Wildcard atoms are not supported")
    return mol


def standardize_smiles(smiles: str) -> StandardizedMolecule:
    """Clean, select the largest organic parent and neutralize where possible.

    Tautomers and stereoisomers are retained. The connectivity key discards
    stereochemistry only for conservative leakage grouping.
    """
    mol = parse_molecule(smiles)
    n_fragments = len(Chem.GetMolFrags(mol))
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return StandardizedMolecule(smiles, canonical, connectivity, scaffold, n_fragments - 1)


def scaffold_key(smiles: str, acyclic_policy: AcyclicPolicy = "group") -> str | None:
    """Return a split key with an explicit policy for empty Murcko scaffolds.

    ``group`` keeps all acyclic molecules together; ``compound`` groups only
    identical acyclic parents and is less strict; ``exclude`` omits them.
    """
    if acyclic_policy not in {"group", "compound", "exclude"}:
        raise ValueError(f"Unknown acyclic policy: {acyclic_policy}")
    normalized = standardize_smiles(smiles)
    if normalized.scaffold:
        return normalized.scaffold
    if acyclic_policy == "exclude":
        return None
    return "ACYCLIC" if acyclic_policy == "group" else f"ACYCLIC:{normalized.connectivity_key}"
