# Training and inference protocol

## Partition contract

`rigid train` requires nonempty `train` and `validation` partitions and refuses identity
leakage or unassigned rows. `--scaffold-holdout` additionally requires disjoint scaffold
groups. Repeated profiles, aliases, standardized connectivity, and stereoisomer identities
are grouped by the data layer before splitting. A separate `calibration` partition is
available for the calibration APIs; it is not used by the trainer.

Keep `acyclic_policy` consistent with the split protocol: `group` keeps acyclic
compounds together, `compound` groups them by connectivity identity, and `exclude`
omits their scaffold keys. The training configuration stores this choice in the
checkpoint; screening uses the same policy. The CLI's `--acyclic-policy` overrides
the configuration explicitly. Excluded source rows should carry `split: excluded`.

Training-expression means/scales and cell vocabulary are fitted on `train` only.
The cell encoder is a one-hot vocabulary with an unknown-cell slot, followed by
`log1p(dose_um)` and `log1p(time_h)`. Missing or negative doses, nonpositive times,
nonfinite expression, incorrect gene order and duplicate identifiers are rejected.
The saved context vocabulary and dose/time transforms are part of the model contract.

## Optimization sequence

1. Fit a D-MPNN molecular-to-expression regressor by MSE on standardized expression.
2. Independently initialize and fit an atom-message/Transformer response regressor.
3. Freeze both predictors. Fit the D-MPNN/target-MLP ranker using normalized graded
   relevance and listwise cross-entropy. Exact source compounds have relevance 1;
   overlapping supplied MOA labels use `moa_relevance`; other candidates have 0.
4. Select each model's checkpoint by validation loss. No test or external expression
   contributes to optimization, normalization, vocabulary fitting or early stopping.

Training candidate pools contain the positive compound, mechanism-related and same-scaffold
candidates when available, then seeded shuffled training compounds. `mechanism_candidates`
and `scaffold_candidates` configure the related-candidate limits; each defaults to two.
Validation pools use the held-out positive structure and a deterministic candidate subset.
Changing validation results can change the selected checkpoint, as intended. Final
evaluation should use the full declared candidate library and explicit requested k.

The forward predictor's auxiliary features on training examples are in-sample predictions.
This is not cross-fitted stacking. Before drawing
strong claims about stacking generalization, compare with compound/scaffold-aware out-of-fold
auxiliary predictions as a separately specified experiment. Validation and test responses
are never used as candidate features.

AdamW, dropout, batch sizes, hidden dimensions and early-stopping patience are specified in
strict JSON configuration. Unknown keys fail loudly. Seeds are set for Python, NumPy and
PyTorch; deterministic operations are requested. Unsupported deterministic CUDA operations
fail rather than silently relaxing the setting. Exact bitwise equality across devices,
BLAS implementations, or dependency versions is not promised.

## Auxiliary features and score decomposition

The ranker receives four interpretable features for each candidate:

1. Pearson correlation between the desired target and the learned forward response.
2. `1 / (1 + mean absolute program-score error)` using supplied program memberships.
3. `tanh(mean absolute predicted expression)` over the union of supplied stress genes.
4. Maximum Morgan-fingerprint Tanimoto to the training compound set.

Undefined constant-vector correlation is encoded as zero **as a model feature**; evaluation
metrics instead represent undefined correlations as missing. Program agreement is zero
when programs are absent. Stress is zero internally when absent, and unavailable in
user-facing annotations. Record these feature definitions and the selected program sets
with each experiment.

```text
final_score = compatibility_logit
              - stress_weight * predicted_stress
              + novelty_weight * (1 - maximum_training_tanimoto)
```

Returned rows retain all three contributions. Neither raw nor adjusted scores are
calibrated probabilities. Tune penalties only on validation data; calibration APIs require
their own explicitly declared partition and prohibit known forbidden IDs.

Program configuration is supplied as JSON:

```json
{
  "source": "Investigator-provided source and version for the memberships below",
  "programs": {"example_program": ["EXAMPLE_GENE_1", "EXAMPLE_GENE_2"]},
  "stress_programs": ["example_program"]
}
```

The identifiers above illustrate syntax; they are not biological gene-set recommendations.
Every actual member must exist in the exact model gene manifest.

## Screening and validation

`RIGIDPipeline.rank` accepts a single expression vector, an ordered gene manifest, context,
and a structure library. The scoring path reads no candidate experimental response,
candidate MOA label or ground-truth target source. It returns chemical properties/alerts,
predicted correlations from both operators, their absolute disagreement, similarity to
training structures and an auditable ranking decomposition.

`screen_dataset` wraps this API for held-out experiments and attaches source/MOA labels
only after scoring. It refuses seen target source structures. Both forward correlations
use standardized features and are labeled **predicted**. They are not the independently
measured external-correlation endpoint in the manuscript. The standalone evaluation APIs
accept context-matched empirical profiles when those are available.

The default primary policy keeps the independent predictor out of candidate selection.
An explicit [consensus policy](consensus.md), supplied through `rank --reranking`, can
include agreement and disagreement terms. That makes the second predictor a scoring
component, not an untouched validation control. Architecture independence is weaker
than independent data; agreement is not an uncertainty interval or empirical validation.

Large candidate libraries are scored in batches. [Prepared libraries](library.md) cache
graphs, embeddings, and context-specific forward responses within explicit budgets.
Benchmark actual workload time and memory before selecting batch/cache sizes or adding
mixed precision and distributed sampling.

## Checkpoints and traceability

Each model bundle contains only state dictionaries and primitive metadata, including
feature schema, ordered gene manifest, train-fitted normalizer, context vocabulary,
program definitions and model settings. Loading requires a SHA-256 sidecar and uses
`torch.load(weights_only=True)`. A checksum detects corruption, not a maliciously replaced
artifact; use trusted distribution and verify signatures for public releases.

Run manifests retain input checksums, software versions, configuration, split counts,
fit-profile IDs and available Git revision/dirty status. No environment-variable dumps,
credentials, private manuscript, or model-success claims are stored.
