# Inspecting, comparing and protecting runs

## Inspect a saved run

```bash
rigid inspect-run runs/research
```

The output contains input checksums, configuration, per-phase best validation epochs and
checkpoint integrity. It verifies the checkpoint without loading its tensors or importing
the model implementation. `--skip-weights` explicitly marks that verification as skipped.
Duplicate or discontinuous history epochs are reported as errors, not silently discarded.

## Compare two screening outputs

```bash
rigid compare runs/first/test.json runs/second/test.json \
  --resamples 5000 --seed 17 --output runs/comparison.json
```

The comparison requires the same input hashes, ordered genes, candidate count, requested
top-k, split and target IDs. Model checkpoints may differ. Legacy files without input
provenance must be regenerated before comparison. The result includes exact/MOA/family
retrieval changes, target-level paired bootstrap intervals and per-target top-k Jaccard.
All differences are **right minus left**. Missing annotations are excluded pairwise and
counted explicitly; fewer than two eligible targets produce no interval.

The bootstrap assumes independent target IDs. Multiple cell/dose/replicate profiles from
the same compound may violate that assumption. Prespecify a compound- or cohort-level
analysis when dependence is present. The convenience comparison does not claim to solve
experimental design, multiplicity across a large search, or causal biological validation.

## Data checks before a long run

```bash
rigid qc data/prepared --output runs/data-quality.json
rigid drift data/prepared --split test --output runs/test-shift.json
rigid inspect-library candidates.csv --output runs/library-manifest.json
```

Quality and drift findings are descriptive warnings. They are not an automatic instruction
to normalize test data differently or tune settings using test results. See [quality.md](quality.md)
for the train-reference definitions and [library.md](library.md) for explicit deduplication
and cache budgets. All report destinations are protected from accidental replacement.

## Interruption and recovery

```bash
rigid --verbose train data/prepared --config configs/research.json --output runs/research
rigid --verbose train data/prepared --config configs/research.json --output runs/research --resume
```

Resume is also available for `train-generator` and `train-vae`. Keep the data, configuration,
options and software unchanged. Phase-specific progress, heartbeat records and immutable
optimization snapshots live under `optimization/`. See [engine.md](engine.md) for the exact
transaction boundary and deterministic replay guarantees.

## Integrity boundaries

JSON configuration and metadata reject duplicate fields, nonfinite values and unsupported
schemas. Checkpoint verification hashes the same opened stream later passed to tensor-only
loading. Integrity records do not authorize untrusted code or establish scientific correctness.
Retain the run inputs, source revision and environment record alongside any reported result.
