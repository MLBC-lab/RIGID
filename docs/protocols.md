# Repeated splits and rotating folds

`SplitProtocol` separates dataset partitioning from model settings. The checked-in
`configs/protocol.json` is a starter protocol, not a validated study configuration.
Materialization writes data only: it does not launch training or select favorable runs.

```python
from rigid.protocols import SplitProtocol, materialize_splits

protocol = SplitProtocol.load("configs/protocol.json")
manifest = materialize_splits("data/curated", "data/protocol-01", protocol)
```

The output directory must be new and outside the source dataset. It contains
`split-0001`, `split-0002`, and subsequent complete RIGID dataset directories, each
with an `assignment.json`. A final `manifest.json` records configurations, source
and output checksums, software versions, compound groups, seed values, partition
counts, and hashes of the individual assignment records. An incomplete operation
may leave partial output; without the final manifest it is not a completed protocol.
No existing data is deleted or replaced.

Each assignment has a `partition_sha256` computed from its sorted profile-to-partition
map, independent of seed, repetition, or fold labels. The final manifest reports
`unique_assignment_count`, groups identical splits in `duplicate_assignments`, and
includes a warning when different allocation attempts produce the same partition.
These semantic hashes are separate from the complete assignment-file checksums.

## Configuration

| Setting | Contract |
| --- | --- |
| `mode` | `repeated_holdout` or `kfold` |
| `strategy` | `compound` or `scaffold` |
| `seed` | Integer from 0 through 2^32 - 1 |
| `repetitions` | Number of independent seeded allocation attempts |
| `folds` | Exactly 1 for holdout; at least 3 for rotating folds |
| `fractions` | Three positive train/validation/test fractions summing to 1 for holdout; omit or set null for k-fold |
| `acyclic_policy` | `group`, `compound`, or `exclude`; used by scaffold grouping |

For each repetition, the seed is `(seed + repetition_index) % 2**32` with a zero-based
index. A protocol contains `repetitions * folds` materializations. At most 10,000
are representable; materialization defaults to a stricter `max_splits=100` budget.
An explicit override is required for larger jobs. `max_profiles` forwards an input
ingestion limit. Each materialized split stores a full expression dataset, so disk
usage grows roughly with the number of splits. Planning itself is available as a
lazy iterator, without writing datasets:

```python
from rigid.data import load_dataset
from rigid.protocols import plan_splits

for assignment in plan_splits(load_dataset("data/curated"), protocol):
    print(assignment.split_id, assignment.profile_counts)
```

For five rotating folds and two repetitions:

```json
{
  "mode": "kfold",
  "strategy": "scaffold",
  "seed": 17,
  "repetitions": 2,
  "folds": 5,
  "fractions": null,
  "acyclic_policy": "group"
}
```

Within each repetition, whole groups are allocated to five approximately balanced
buckets. Each bucket is test once and validation once; the other buckets are train.
This is rotating train/validation/test evaluation, not nested cross-validation and
not unrestricted test-set tuning. At least as many eligible groups as folds are
required. All three active partitions must be nonempty.

## Identity and preserved partitions

Aliases, normalized structure identities, and connectivity identities always stay
together. Scaffold mode adds Bemis-Murcko grouping. Allocation uses group profile
counts, with seeded tie ordering, without inspecting expression or MOA annotations.
Large indivisible groups can make realized fractions differ substantially from the
requested values. Source row ordering does not determine assignments.

The allocator processes larger groups first and randomizes equal-size ties. Different
seeds can therefore produce identical assignments, especially when all group sizes
differ. Increasing repetitions does not guarantee additional distinct holdouts; inspect
the manifest's uniqueness counts before aggregating results.

Existing `train`, `validation`, `test`, and unassigned rows are resplit deliberately.
Existing `external`, `calibration`, and `excluded` rows retain their labels in every
output. A protected external/calibration group that overlaps an eligible group or
another protected partition causes an error; it is never silently moved into train.
Previously excluded rows remain excluded and do not participate in balancing.
The scaffold `exclude` policy additionally excludes eligible acyclic groups and lists
the affected profiles in the manifest. The input dataset remains unchanged.

Each output passes the ordinary identity/scaffold leakage audit. Existing excluded
rows are ignored by that audit; their identity overlap is not training exposure.
The `compound` acyclic policy is less strict than grouping all acyclic compounds;
record it when reporting chemical generalization.

## Analysis discipline

Freeze the split protocol, biological annotations, candidate catalog, model settings,
and primary metrics before examining final test results. Fit normalization and
vocabulary separately inside each train partition. Existing training commands do
this when supplied one materialized dataset. Keep external and calibration labels
untouched when selecting settings or comparing methods.

Repeated holdout test sets overlap; their metrics are correlated. Do not treat
`repetitions * folds` scores as that many independent biological observations.
Persist target-level predictions and use a prespecified grouped aggregation or
resampling scheme. Select the number of repetitions/folds based on the study and
compute budget, not which count produces the most favorable result.
