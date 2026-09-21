# Data quality and distribution shift

Use these diagnostics before training and when introducing a new cohort. They
describe the supplied data; they do not establish biological validity, predict
clinical performance, or authorize removal of inconvenient observations.

```python
from rigid.data import load_dataset
from rigid.quality import audit_dataset, compare_to_training

dataset = load_dataset("data/prepared", chunk_size=4096, max_profiles=1_000_000)
quality = audit_dataset(dataset).to_dict()
drift = compare_to_training(dataset, "external").to_dict()
```

The command-line equivalents are `rigid qc DATASET --output quality.json` and
`rigid drift DATASET --split external --output drift.json`. Both produce strict
JSON: undefined statistics are `null`, not NaN or infinity. The report includes
its schema version, ordered-gene checksum, source type, and all thresholds.

## Contracts versus warnings

Hard data contracts are enforced on input: unique identifiers, valid molecular
structures, exactly ordered genes, finite float32-representable expression
values, finite nonnegative doses, positive times, valid split names, and complete
compound/profile joins. A supplied checksum manifest must cover all three data
files, with each digest verified by default. An absent `dataset.json` is accepted
for manually prepared interchange files but provides no provenance guarantee.

Quality warnings do not reject an otherwise valid dataset. The audit reports:

- Profile, compound and canonical-structure counts by partition.
- Cell counts, dose/time ranges and medians, and mechanism-label coverage.
- Per-gene population mean, standard deviation, range and constant-gene flags.
- Extreme marginal values and up to 20 example profile identifiers.
- Matched-context replicate counts and mean pairwise expression correlation.
- Unassigned partitions, small training sets, and unknown source provenance.

Replicate groups use canonical structure, exact cell identifier, dose and time.
This is a metadata-based grouping, not a claim that the measurements are
technical replicates. Multiple aliases of one structure are combined. The
report flags groups crossing partition boundaries. Pearson correlation excludes
pairs involving expression vectors constant across genes; its denominator is
reported as `informative_pairs`. No sampling is used: the mean is calculated
from the sum of normalized vectors without allocating a pairwise matrix.

Expression moments use float64 chunk accumulation. Outliers exceed the configured
number of marginal standard deviations from the **entire dataset** mean. This
is a descriptive audit, not a training transform. Heavy tails, small samples and
biological heterogeneity can all affect this statistic. Never feed these global
moments back into a fitted model; use `TrainStandardizer` instead.

## Training-reference drift

`compare_to_training` accepts `validation`, `calibration`, `test` or `external`.
Both the training and selected comparison partitions must be nonempty. Other
partitions have no effect on the report. It never changes a dataset or a model.

For each gene it computes:

| Field | Definition |
| --- | --- |
| `standardized_mean_shift` | `(comparison mean - training mean) / training population SD` |
| `standard_deviation_ratio` | Comparison population SD divided by training population SD |
| `population_stability_index` | Sum over bins of `(q - p) * log(q / p)` |
| `constant_reference_changed` | Any comparison value differs from a constant training mean beyond tolerance |

Histogram edges are quantiles of **training rows only**, with duplicate edges
collapsed and unbounded outer bins. For a constant training gene, three bins
surround the mean by the configured tolerance. Every bin receives the same
positive pseudocount before each histogram is normalized. The report records
the fitted interior edges. Pseudocounts make the index finite with empty bins;
they also make it sample-size sensitive. The index is not a p-value.

Mean shifts and SD ratios are undefined (`null`) when the training SD is below
tolerance. Such genes receive an explicit constant-reference change flag rather
than an arbitrary enormous standardized effect. A warning also identifies new
cell IDs, doses/times outside the training range, and unseen mechanism labels.
This context audit reports marginal coverage; it does not establish overlap of
every cell/dose/time combination.

## Threshold configuration

Defaults are operational starting points, **not validated biological cutoffs**.
Both APIs accept immutable threshold dataclasses:

```python
from rigid.quality import DriftThresholds, QualityThresholds

quality_options = QualityThresholds(
    constant_tolerance=1e-8,
    outlier_standard_deviations=6.0,
    minimum_replicate_correlation=0.5,
    minimum_training_profiles=20,
    chunk_size=4096,
)
drift_options = DriftThresholds(
    standardized_mean_shift=1.0,
    population_stability_index=0.2,
    histogram_bins=10,
    histogram_pseudocount=0.5,
    minimum_training_profiles=20,
    minimum_comparison_profiles=10,
)
```

Choose thresholds using domain knowledge and an evaluation plan fixed before
examining final test performance. Profile-level counts do not imply independent
samples: repeated compounds or contexts can substantially reduce effective
sample size. No hypothesis testing or multiplicity correction is implied.

## Ingestion and output safety

CSV ingestion converts bounded batches to float32 instead of retaining a Python
float object per matrix cell. The returned dataset is still in memory;
`chunk_size` is not an out-of-core guarantee. Use `max_profiles` to enforce an
explicit row limit. Numeric failures include their CSV line number.

`save_dataset` accepts a new path or an empty directory, uses exclusive file
creation, and refuses to overwrite existing contents. It writes the checksum
metadata last. A write failure leaves any partial files for inspection; the
operation is not a transactional directory replacement. Store a revised dataset
under a new versioned path and retain its provenance.
