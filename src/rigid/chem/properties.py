"""Inspectable chemical descriptors and medicinal-chemistry alerts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import QED, Crippen, Descriptors, FilterCatalog, Lipinski, rdFingerprintGenerator

from .standardize import parse_molecule, standardize_smiles


@dataclass(frozen=True)
class ChemicalProperties:
    canonical_smiles: str
    molecular_weight: float
    logp: float
    hydrogen_bond_donors: int
    hydrogen_bond_acceptors: int
    rotatable_bonds: int
    qed: float
    lipinski_violations: int
    pains_alerts: tuple[str, ...]
    brenk_alerts: tuple[str, ...]
    sa_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _alerts(mol, kind) -> tuple[str, ...]:
    parameters = FilterCatalog.FilterCatalogParams()
    parameters.AddCatalog(kind)
    catalog = FilterCatalog.FilterCatalog(parameters)
    return tuple(sorted(entry.GetDescription() for entry in catalog.GetMatches(mol)))


def chemical_properties(smiles: str) -> ChemicalProperties:
    canonical = standardize_smiles(smiles).canonical_smiles
    mol = parse_molecule(canonical)
    weight, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    donors, acceptors = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
    sa_score: float | None
    try:
        from rdkit.Contrib.SA_Score import sascorer

        sa_score = float(sascorer.calculateScore(mol))
    except ImportError:
        sa_score = None
    kinds = FilterCatalog.FilterCatalogParams.FilterCatalogs
    return ChemicalProperties(
        canonical,
        float(weight),
        float(logp),
        donors,
        acceptors,
        Lipinski.NumRotatableBonds(mol),
        float(QED.qed(mol)),
        sum((weight > 500, logp > 5, donors > 5, acceptors > 10)),
        _alerts(mol, kinds.PAINS),
        _alerts(mol, kinds.BRENK),
        sa_score,
    )


def morgan_fingerprint(smiles: str, *, radius: int = 2, n_bits: int = 2048):
    if radius < 0 or n_bits <= 0:
        raise ValueError("Fingerprint radius must be nonnegative and n_bits positive")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(parse_molecule(standardize_smiles(smiles).canonical_smiles))


def fingerprint_array(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    output = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        morgan_fingerprint(smiles, radius=radius, n_bits=n_bits), output
    )
    return output


def maximum_tanimoto(candidates: list[str], reference: list[str]) -> np.ndarray:
    if not reference:
        raise ValueError("Reference set must contain at least one molecule")
    reference_fps = [morgan_fingerprint(smiles) for smiles in reference]
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(morgan_fingerprint(smiles), reference_fps))
            for smiles in candidates
        ],
        dtype=np.float64,
    )
