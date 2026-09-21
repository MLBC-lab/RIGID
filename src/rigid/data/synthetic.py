"""Small deterministic fixtures for software checks, never paper benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rigid.chem.standardize import parse_molecule

from .io import save_dataset
from .schema import CompoundRecord, GeneManifest, ProfileDataset, ProfileRecord, SourceProvenance
from .splits import assign_splits

SYNTHETIC_SMILES = (
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "Cc1ccncc1",
    "c1cncnc1",
    "c1ccoc1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CC1",
    "C1CCC1",
    "C1CCCC1",
    "C1CCCCC1",
    "C1CCCCCC1",
    "C1CCCCCCC1",
    "C1COCCN1",
    "C1CCNCC1",
    "C1CCOC1",
    "C1COC1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",
    "c1ccc2ncccc2c1",
    "c1ncc[nH]1",
    "CCO",
    "CC(=O)O",
    "CCN",
    "CCOC",
    "c1nccnc1",
    "O=C1CCCCC1",
)


def make_synthetic_dataset(
    directory: str | Path | None = None, *, seed: int = 17, n_genes: int = 32
) -> ProfileDataset:
    """Generate chemical-validity and end-to-end smoke-test data.

    Profiles, cell labels, and category labels are simulated. No profile is an
    experimental measurement and no label states a real mechanism of action.
    """
    if n_genes < 4:
        raise ValueError("Synthetic fixtures require at least four genes")
    rng = np.random.default_rng(seed)
    genes = GeneManifest(tuple(f"SYNTH_GENE_{index:04d}" for index in range(n_genes)), "synthetic")
    basis = rng.normal(0, 0.25, size=(6, n_genes))
    context_basis = rng.normal(0, 0.10, size=n_genes)
    compounds, profiles, expression = [], [], []
    for index, smiles in enumerate(SYNTHETIC_SMILES):
        compound_id = f"SYNTH_COMPOUND_{index:03d}"
        mol = parse_molecule(smiles)
        feature = np.array(
            [
                mol.GetNumHeavyAtoms() / 10,
                sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / 6,
                sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
                sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
                sum(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms()),
                mol.GetRingInfo().NumRings(),
            ]
        )
        category = int(np.argmax(feature[1:]))
        compounds.append(CompoundRecord(compound_id, smiles, (f"synthetic_category_{category}",)))
        for context in range(2):
            for replicate in range(2):
                profiles.append(
                    ProfileRecord(
                        f"{compound_id}_C{context}_R{replicate}",
                        compound_id,
                        f"SYNTH_CELL_{context}",
                        1.0,
                        24.0,
                    )
                )
                expression.append(
                    feature @ basis + context * context_basis + rng.normal(0, 0.03, n_genes)
                )
    provenance = SourceProvenance(
        kind="synthetic",
        description="Deterministic simulated expression profiles on simple valid molecules, for software testing only. Category labels are artificial and do not represent measured mechanisms of action.",
        transformations=(f"Synthetic linear feature model with Gaussian noise; seed={seed}",),
        license="CC0-1.0 for generated fixture data",
    )
    dataset = ProfileDataset(
        genes, tuple(compounds), tuple(profiles), np.asarray(expression), provenance
    )
    dataset = assign_splits(dataset, strategy="scaffold", seed=seed)
    if directory is not None:
        save_dataset(dataset, directory)
    return dataset
