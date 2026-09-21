# Nearest-neighbor reference catalogs

The nearest-neighbor baseline has two explicit information-access regimes:

| Mode | Available candidate responses | Exact held-out source recovery |
| --- | --- | --- |
| Training only, default | Query dataset training partition | Impossible for a strictly compound-held-out source |
| Independent reference catalog | Selected partitions of a separate response dataset | Possible when an independently measured response for the source is available |

Independent reference retrieval is an observed-response baseline, not a structure-only predictor. A matching candidate molecular catalog does not imply matching information access. Report both the molecular universe and the available response measurements when comparing methods.

## Configuration

```sh
rigid neighbors data/query --reference-dataset data/reference \
  --reference-config configs/reference_neighbors.json --top-k 50 \
  --output runs/neighbors.json
```

```python
from rigid.baselines import ReferenceNeighborConfig
from rigid.experiments import nearest_neighbor

result = nearest_neighbor(
    "data/query",
    split="test",
    k=50,
    reference_dataset="data/reference",
    reference_config=ReferenceNeighborConfig(
        reference_splits=("train",),
        match_cell=True,
        dose_tolerance_um=0.0,
        time_tolerance_h=0.0,
        aggregation="max",
        chunk_profiles=1024,
    ),
)
```

`ReferenceNeighborConfig.load(path)` reads the same fields from strict JSON; `configs/reference_neighbors.json` is a ready-to-use example. Unknown keys, duplicate keys, invalid partitions, nonfinite tolerances, and invalid types are rejected. The reference partitions are local labels in the separate catalog; choosing `train` does not import the query dataset's training partition. All reference rows must have identifiers independent of the query dataset.

`scaffold_holdout=True` also audits query scaffold partitions. `acyclic_policy` accepts `group` (default), `compound`, or `exclude`, matching the data split utilities. The selected policy and leakage audit are included in results. Changing this policy changes the benchmark definition and must be decided before evaluation.

Cell labels, dose in micromolar, and exposure time in hours match exactly by default. Absolute dose/time tolerances and cell matching are configurable and recorded. `match_cell=False` relaxes only the cell restriction. It does not remove the dose and time checks. The score is the maximum or mean Pearson correlation over eligible responses for a candidate. Maximum aggregation can favor compounds with more measured responses; record and inspect `reference_profile_count` or compare against mean aggregation.

The query dataset's **training partition alone** determines gene-wise means and scales. Reference data and held-out queries never fit normalization. Ordered gene manifests and namespaces must match. Working expression blocks are bounded by `chunk_profiles`; the loaded datasets, identity audit, and score/count arrays remain resident. No query-by-reference correlation matrix is allocated. Score ties preserve declared candidate order.

## Molecular identities and coverage

By default the molecular universe is the entire query `compounds.csv`, matching the supplied library rather than just its training compounds. The Python `candidate_ids` argument can restrict it to an explicitly declared sequence of compound IDs. The function never adds a source compound by consulting target labels. Canonically identical structures collapse to the first declared representative; the output records all identity members and the reference-to-candidate map. Exact recovery is evaluated on that standardized structural identity. Candidate and target MOA annotations are used only after ranking, for evaluation. Those labels are unioned across all query compound-table records for the same standardized structure, including aliases outside the selected candidate ID list. `annotation_identity_members` records their sources; selecting an unlabeled alias does not erase known annotations or change the evaluation denominator.

Each query reports the scoreable candidate count, missing-response count, fraction of the declared universe covered, eligible response count, and whether the source is declared and scoreable. A candidate with no matching response receives no score. A query with no eligible candidates returns an empty ranking and stays in the evaluation denominator. Rankings are never padded to `k`. Constant standardized reference profiles are excluded explicitly; a constant standardized query causes an error instead of silently disappearing.

## Source and independence checks

Prepare both datasets using the standard data schema. The reference dataset must declare experimental or synthetic provenance and describe its source. Experimental provenance requires source URLs or upstream input checksums. Files from both datasets are hashed and checked again before results are returned. A result containing either synthetic input remains marked synthetic, including mixed-source runs.

Retrieval rejects shared profile IDs, exact expression-vector copies under renamed IDs, and compound IDs or aliases assigned conflicting standardized structures. These checks cover all rows in both supplied datasets, not just the selected partitions. Compound identity overlap is intentionally permitted: an independent replicate of a held-out compound is the information this baseline uses. Float32 expression identity checks are not proof of independent biological measurements; transformed copies, shared upstream replicates, batch effects, and overlapping source releases still require a provenance audit.

Public sources and preparation guidance are described in the repository data documentation. Dataset-specific profile selections and source-release manifests should be obtained from the authors when not distributed publicly. This implementation does not infer an unpublished experimental protocol from a reported aggregate score. Freeze the catalog, partition selection, context rules, and aggregation setting before evaluating held-out targets.
