# Data preparation and leakage controls

RIGID accepts curated compound-response tables with explicit gene order, chemical
identity, experimental context, and provenance. See [public datasets](datasets.md)
for source collections and [study materials](SCIENTIFIC_CONTRACT.md) for the
inputs needed to evaluate manuscript endpoints.

## Interchange directory

```text
dataset/
  genes.json
  compounds.csv
  profiles.csv
  dataset.json
```

`genes.json` fixes gene identifiers, their namespace, and their exact order. The
same order must be used during training, scoring, disease-signature construction,
and checkpoint loading. A matrix with reordered columns is rejected even when it
has the expected dimensions. Supply the actual landmark manifest for an L1000
analysis; the software never substitutes an assumed list of 978 identifiers.

```json
{
  "gene_ids": ["7157", "1956", "4609"],
  "namespace": "NCBI Gene"
}
```

This short example illustrates the schema; it is not the L1000 landmark list.
`GeneManifest.save()` adds a SHA-256 checksum of the namespace and ordered genes.
The checksum is validated on load. Identifiers must be unique and nonempty.

`compounds.csv` requires these fields:

| Column | Meaning |
| --- | --- |
| `compound_id` | Stable, unique perturbagen identifier |
| `smiles` | Parseable molecular structure, including a salt if present |
| `moa` | Optional semicolon-separated mechanism annotations |
| `aliases` | Optional semicolon-separated alternative identifiers |

MOA annotations may be absent or have multiple labels. Missing annotations are
not evidence of a different mechanism. Aliases can refer to another compound ID;
their transitive connections are respected during split construction.

`profiles.csv` requires:

```csv
profile_id,compound_id,cell_id,dose_um,time_h,split,7157,1956,4609
example-response,compound-001,example-cell,1.0,24.0,train,0.3,-0.4,0.1
```

Expression values must be finite. Profile IDs must be unique. Each compound ID
must resolve to `compounds.csv`. Cell labels must be explicit; dose is in
micromolar and must be nonnegative; time is in hours and must be positive.
Changing dose or time units requires an explicit conversion before import.
Gene columns must follow the manifest in order, and cannot have the same names
as metadata columns. The loader does not impute missing expression values.

Allowed splits are `train`, `validation`, `calibration`, `test`, `external`, and
`excluded`. A blank split is accepted during preparation but fails the leakage
audit until assigned. `assign_splits()` generates train/validation/test
partitions; a dedicated calibration cohort can be assigned explicitly and is
checked against all other partitions.

`dataset.json` stores the schema version, file checksums, counts, and source
provenance. It is written by `save_dataset()`. `load_dataset()` checks file hashes
by default. When hand-editing data, update the manifest intentionally; disabling
checksum verification is an explicit import option, not a repair of stale
provenance. The `provenance.kind` field must be
`synthetic`, `experimental`, or `unknown`. Missing provenance is `unknown`.

The JSON schemas in [schemas](schemas/) describe the two JSON files. Python
validation additionally checks table alignment, checksums, chemical validity,
gene order, context values, and cross-file references.

## Chemical identity

`standardize_smiles()` applies RDKit cleanup, selects the largest organic
fragment, neutralizes where possible, removes explicit hydrogen atoms, and
generates canonical isomeric SMILES. Original input SMILES remain in the record.
The number of removed fragments is available in the standardization result.

Record these preprocessing settings in the study protocol. Stereoisomers and
tautomers are retained as molecular structures. A separate non-isomeric connectivity
key groups stereoisomers conservatively for leakage checks. No tautomer canonicalization or
protonation-at-pH prediction is performed.

## Splits

```python
from rigid.data import assign_splits, audit_leakage, load_dataset, save_dataset

dataset = load_dataset("data/curated")
dataset = assign_splits(
    dataset,
    strategy="scaffold",
    seed=17,
    fractions=(0.7, 0.15, 0.15),
    acyclic_policy="group",
)
audit_leakage(dataset, scaffold=True).raise_for_leakage()
save_dataset(dataset, "data/partitioned")
```

Compound partitions group aliases, identical standardized parents, and matching
connectivity keys. Scaffold partitions also group exact Bemis–Murcko frameworks.
Every dose, cell, time, and replicate for a group stays in one partition. Splits
are seeded and approximately balance profile counts, without reading expression
values or MOA labels. Exact fractions are usually impossible with large groups.
Too few independent groups cause an error instead of an empty held-out split.
Calling `assign_splits()` intentionally replaces existing assignments.

For repeated seeded holdouts or rotating train/validation/test folds, use
[`SplitProtocol`](protocols.md). It materializes audited datasets and assignment
records while preserving declared external and calibration partitions.

An empty scaffold is common for acyclic molecules. Choose its policy explicitly:

| Policy | Treatment |
| --- | --- |
| `group` (default) | All acyclic parents belong to one split group |
| `compound` | Acyclic parents are grouped by connectivity; weaker novelty control |
| `exclude` | Acyclic groups are labeled `excluded` |

Exact scaffold isolation does not guarantee low fingerprint similarity. Report
`maximum_tanimoto()` separately when auditing close analogs. The leakage report
lists the offending identity, overlap type, partitions, and compound IDs. It
checks compound IDs, aliases, canonical structures, connectivity, and optionally
scaffolds. `excluded` profiles do not participate; calibration and external
partitions do.

## Train-only normalization

```python
from rigid.data import TrainStandardizer

normalizer = TrainStandardizer.fit(
    dataset.expression,
    dataset.split_labels,
    dataset.genes,
)
normalized = normalizer.transform(dataset.expression, dataset.genes)
normalizer.save("normalizer.json")
```

Only rows labeled `train` determine means and scales. Constant training genes
receive scale one. Held-out values cannot alter preprocessing statistics.
Normalizer state includes the gene manifest and fit-partition declaration for
checkpoint restoration. Disease reversal is constructed in differential-
expression space first, then transformed using the fitted training normalizer.

## GCT and GCTX

`read_gct()` reads GCT 1.2; GCT 1.3 is rejected because its metadata layout differs.
`read_gctx()` reads standard GCTX files with the optional `h5py` dependency. Both
return `(profile_ids, expression)` in profile-by-gene order. They import matrix
values only; source accession, quality filters, MOA, cell, dose, and time must be
joined from explicit metadata before creating a `ProfileDataset`.

```python
from rigid.data import GeneManifest, read_gctx

genes = GeneManifest.load("landmark_genes.json")
profile_ids, expression = read_gctx(
    "input.gctx",
    genes,
    allow_reorder=True,
)
```

`allow_reorder=True` explicitly selects manifest genes and restores their order.
The default requires an exact match. Missing genes and duplicate identifiers
are errors. GCTX selection reads only the requested gene columns, but the
resulting selected matrix is held in memory. Partition very large sources before
loading if the selected matrix will not fit in memory.

`prepare_matrix()` provides the complete matrix-to-dataset join:

```python
from rigid.data import prepare_matrix

dataset = prepare_matrix(
    "input.gctx",
    "landmark_genes.json",
    "compound_metadata.csv",
    "sample_metadata.csv",
    "data/curated",
    source_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742",
    source_license="Record the actual source dataset license here",
    allow_reorder=True,
)
```

Replace the illustrative source and license with the specific source used.
Sample metadata uses the five required profile metadata columns above and an
optional `split`. It must cover exactly the matrix's profile IDs once each.
Rows are joined by ID, not row position. Missing, duplicate, and extra metadata
rows are rejected. The prepared dataset records SHA-256 hashes for all four
inputs, the source supplied by the caller, and the import operation. Source
provenance is declared experimental by this entry point, but the software cannot
independently certify that designation. Use the synthetic fixture entry point
for software test data.

## Disease signatures and program definitions

The package can negate disease-minus-control signatures, interpolate between
targets, take a mean or median cohort consensus, and multiply genes belonging to
named programs by explicit weights. These operations do not infer cohorts,
disease labels, or gene sets. `ProgramSet` requires a source description and
checks that every member gene exists in the manifest.

Setting a program weight to zero removes its signal. Overlapping program
weights multiply. Program scores can use signed or absolute expression; the
choice must accompany an interpretation of stress burden. No universal stress
threshold or disease-specific gene list is silently installed.

## Graph feature contract

`graph_from_smiles()` creates 46 atom features and 14 bond features. Each bond is
represented twice, once per direction; `reverse_edge` gives the opposite edge.
`batch_graphs()` offsets atom and edge indices and supplies `graph_index` for
molecule pooling. Single-atom graphs are supported; empty molecules and empty
batches are rejected. `GraphBatch.to(device)` converts to PyTorch tensors.

Atom features encode element, total degree, formal charge, chirality, hydrogen
count, hybridization, aromaticity, ring membership, and scaled atomic mass. Bond
features encode bond type, conjugation, ring membership, and stereochemistry.
Categorical fields include an unknown value. The feature schema version is
`rigid-atom46-bond14-v1` and must agree with saved model weights.

## Sources

- [RDKit molecular standardization](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html)
- [RDKit Bemis–Murcko scaffolds](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html)
- [CMap matrix file formats](https://clue.io/connectopedia/category/Data)

These sources describe file formats and cheminformatics operations. Dataset
accessions, selected profile IDs, and preprocessing decisions belong in each
experiment's provenance record.
