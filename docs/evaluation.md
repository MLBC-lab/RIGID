# Evaluation and scientific controls

RIGID evaluates **targets**, then summarizes across targets. A candidate is not an
independent replicate of its target. The implementation follows the manuscript's
reported metric equations and makes evaluation conventions explicit. Consult
[study materials](SCIENTIFIC_CONTRACT.md) for the target lists, data releases,
split assignments, fitted models, and configurations needed for the paper's endpoints.

Run `python examples/evaluation_protocol.py --seed 42` for an offline worked
example. It prints JSON covering separate validation/calibration/test partitions,
paired contrasts, calibrated confidence, conformal intervals, retrospective
designability, decoys, stress reranking, and cohort stability. Every observation is
explicitly synthetic, and repeated runs with the same seed produce the same report.

## Retrieval endpoints and annotation coverage

```python
from rigid.evaluation import CandidateAnnotation, TargetAnnotation, evaluate_rankings

targets = {
    "target-1": TargetAnnotation(
        source_compound="compound-a",
        moa={"kinase inhibitor"},
        profile_families={"cell-cycle response"},
    )
}
candidates = {
    "compound-a": CandidateAnnotation(
        moa={"kinase inhibitor"},
        profile_families={"cell-cycle response"},
        scaffold="scaffold-1",
        stress=0.2,
    ),
    "compound-b": CandidateAnnotation(moa={"other mechanism"}),
}
report = evaluate_rankings(
    {"target-1": ["compound-a", "compound-b"]},
    targets,
    candidates,
    k=2,
    training_scaffolds=["scaffold-1"],
    stress_threshold=0.5,
)
print(report.to_dict())
```

| Endpoint | Unit and numerator | Denominator |
| --- | --- | --- |
| Exact@k | Targets with the source compound among top k | Targets with a source-compound label |
| MOA@k | Targets with at least one candidate sharing a nonempty MOA label | Targets with at least one MOA label |
| ProfileFamily@k | Targets with at least one matching profile-family label | Targets with at least one family label |
| NewScaffold@k | Returned candidates with scaffolds absent from training | Returned candidates with known scaffolds |
| ScafDiv@k | Distinct known scaffolds in top k | Returned candidates with known scaffolds |
| StressHigh@k | Candidates with stress at or above a declared threshold | Returned candidates with measured stress |

Each rate retains its numerator and denominator. Undefined rates serialize as
`null`, never as fabricated zeroes or nonstandard JSON NaN. Missing annotations are
excluded from that endpoint and their counts remain visible. An annotated target
with no returned ranking remains an eligible failure. Empty strings cannot match
one another as MOA labels. Multi-label MOA success means any label intersection.
Identifiers and labels are otherwise matched exactly; canonicalization belongs in
data preparation.

The manuscript assumes k returned, fully annotated candidates. `strict_k=True`
enforces the candidate count. For smaller libraries the default evaluates the
actual returned set and reports both `requested_k` and `returned`; candidate rates
use the annotated returned denominator. Do not call these shortened rankings a
full top-50 experiment. Duplicate candidates and unknown identifiers raise errors.

Use canonical nonempty scaffold identifiers. If an acyclic compound has an empty
Bemis-Murcko scaffold string, decide on and document an acyclic representation in
data preparation; an empty string here means missing scaffold annotation. Always
supply the actual training scaffold set for novelty results. Omitting that set
produces an undefined novelty result; it does not label every candidate novel.

## Expression agreement and top-gene ranking

`pearson_correlation` computes Pearson r on aligned gene vectors. It returns `None`
for a constant profile or fewer than two genes. Non-finite inputs and mismatched
lengths raise errors. Gene order, assay context, dose, and time must be harmonized
upstream; equal vector lengths do not establish biological comparability.

The manuscript's target summary is:

```text
best_r(target) = max(correlation(target, candidate) for candidate in top_k)
R_med = median(best_r(target) for eligible targets)
```

`evaluate_rankings` implements this order of operations. It reports the count of
targets with defined correlations and the number of finite candidate correlations
per target. Missing response profiles do not become zero correlations. Pass
empirical profiles separately from model predictions and label the provenance of
each result. A learned compatibility score is not itself a predicted expression
profile and cannot be used as a correlation vector.

`ndcg` implements exponential gain `(2**relevance - 1)` and logarithmic discount.
For a truncated candidate ranking, pass the **full universe's** relevance values as
`ideal_relevance`; otherwise the ideal ranking is calculated only from supplied
items. All-zero relevance yields zero. Stable input-order tie breaking makes
identical inputs deterministic.

`top_gene_ndcg` ranks genes by absolute response magnitude and uses binary
membership in the target's `target_top_n` highest-magnitude nonzero genes as
relevance. This is an explicit metric convention: the paper describes
highest-magnitude gene overlap but does not specify the relevance cutoff or gain
encoding. This metric is unsigned; a sign-reversed profile can have perfect
top-gene NDCG. Pair it with signed Pearson correlation and report `k` and
`target_top_n`.

## Paired target bootstrap

`paired_bootstrap_ci(a, b)` resamples the same target indices for both methods and
returns a percentile interval for `statistic(a) - statistic(b)`. The median option
uses the difference of medians, not the median of target differences. The output
records the seed, target count, resample count, confidence, and interval method.
`compare_target_metrics` first aligns exact target-ID sets and retains only paired
eligible observations.

```python
from rigid.evaluation import compare_target_metrics

contrast = compare_target_metrics(
    {"t1": 1.0, "t2": 0.0, "t3": 1.0},
    {"t1": 0.0, "t2": 0.0, "t3": 1.0},
    statistic="mean",
    n_resamples=5000,
    confidence=0.95,
    seed=42,
)
```

This bootstrap assumes independent target units. If targets share donors,
experimental batches, or other dependence, construct a cluster bootstrap at that
unit instead of claiming these intervals correct for clustering. No comparison
function infers dominance from overlapping marginal intervals; inspect the paired
contrast directly.

## Calibration and uncertainty

Temperature scaling and conformal calibration use a separate **calibration** set.
They reject calls labeled train, validation, or test. Every fit requires unique
sample IDs and accepts `forbidden_ids` for training/validation/test leakage checks.
Prediction can take sample IDs to prevent reporting calibration rows as held-out
evaluation. The split label is an API guard, not proof of provenance: persist and
audit the split manifest as well.

`TemperatureScaler` fits one positive temperature to binary compatibility logits
by minimizing binary cross-entropy. Both outcome classes are required. The allowed
temperature range is [0.05, 100]. It is a one-parameter calibration method, not a
change to candidate encoding or a new training objective. Fit on outcomes matching
the reported event, such as target-level MOA@50; do not present candidate-pair
calibration as target-recovery calibration without a separate experiment.

`expected_calibration_error` reports equal-width binary-event ECE, including each
bin's count, confidence, and observed event frequency. A probability of exactly 1
belongs to the final bin. Empty bins are retained with `None` summaries. This ECE
compares event probabilities with event frequencies, not max-class confidence.

`SplitConformalRegressor` calibrates an absolute-residual interval for a scalar
target outcome. Its residual threshold is order statistic
`ceil((n_calibration + 1) * (1 - alpha))`. If this rank exceeds the available sample
count, it produces an unbounded interval rather than replacing the finite-sample
rule with a smaller empirical quantile. Coverage is marginal under exchangeability;
it is not guaranteed for each disease, mechanism, or individual target. Candidate
dependence, cohort shift, and use of predicted instead of observed outcomes change
the interpretation. `empirical_coverage` counts inclusive interval endpoints and
retains the number covered and total evaluated.

## Stress-aware ranking and designability

`rerank_candidates` exposes every term of:

```text
final_score = compatibility - stress_weight * stress + novelty_weight * novelty
```

Stress and novelty features must lie in [0, 1]. Weights are explicit and
nonnegative. Ties preserve input order; configure the input candidate order for
reproducible experiments. Each returned `RankedCandidate` retains compatibility,
stress penalty, novelty bonus, and final score. Tune weights on validation data.
Correlation, logits, and probabilities have different scales; declare the score
scale before choosing a penalty. Record the additive scoring formula and selected
weights as part of the evaluation protocol.

The pipeline also accepts an opt-in [consensus policy](consensus.md) that combines
both predicted correlations and their disagreement. Keep that experiment separate
from evaluation with an unused independent predictor.

`DesignabilityModel` follows the manuscript's validation-normalized weighted
component combination, including inverted stress. It requires explicit component
weights summing to one and `split="validation"`. Typical components are maximum
target correlation, MOA@k, profile-family success, scaffold diversity, and
stress-high fraction. The implementation uses validation min-max ranges, clips
new observations to [0, 1], maps validation-constant components to the neutral
value 0.5, inverts `stress_high`, and clips the weighted output. Normalization and
weight values were not fully specified in the manuscript; these conventions must
be retained in experiment metadata and tested for sensitivity. `to_dict()` records
fit ranges, explicit weights, and validation IDs.

MOA/family recovery components require evaluation labels. A retrospective
designability score containing those labels is not an immediately deployable
confidence predictor for an unlabeled disease target. For prospective use,
define observable proxy components and validate them separately. Do not silently
replace unknown recovery labels with successes or use test labels to fit ranges.

## Cohort stability, families, and negative controls

`cohort_robustness` calculates all pairwise top-k candidate-set Jaccards and returns
the intersections, unions, and eligible pair count. Two empty candidate sets yield
an undefined Jaccard because they provide no stability evidence. One empty and one
nonempty set yield zero. With MOA annotations, a separate shared-family rate uses
`intersection / min(number of families in each cohort)`. That denominator is an
explicit metric convention; report it alongside the family counts.

`summarize_families` reports unique scaffolds, unique MOA families, annotation
coverage, novel-scaffold candidate counts, scaffold Shannon entropy, and effective
scaffold count. MOA counts can exceed candidate counts when candidates have several
labels. These are descriptive families; they do not establish target binding,
mechanism validation, or therapeutic efficacy.

`benjamini_hochberg` applies BH FDR adjustment to the complete submitted hypothesis
family and restores the original order. Choose the tested family universe before
seeing results; adjusting only significant pathways is invalid. Inputs outside
[0, 1] or with missing p-values raise errors. The helper does not invent enrichment
p-values or a biological background universe.

`make_decoy_targets` provides deterministic within-profile gene shuffling,
whole-profile sign reversal, and mixtures of two distinct target profiles. These
controls test different hypotheses. Sign reversal can be biologically meaningful
and is not universally an impossible state. Low-reproducibility disease controls
require empirical replicate evidence; this code does not manufacture them.

`run_ablations` calls an experiment callback for the complete model and one
disabled component at a time. When a component changes fitting, the callback must
retrain from the same split and seed protocol. A post-hoc stress penalty change
must not be labeled an ablation of the learned molecular encoder.

## Optional native fingerprint kernel

`rigid.native` provides a dependency-free C++17 kernel and an equivalent NumPy
fallback. Fingerprints are packed `uint8` matrices, one row per molecule.
`pack_fingerprints` packs binary matrices with zero padding in the final byte.
Tanimoto is bit intersection divided by bit union; all-zero versus all-zero is
defined as 1. This numerical convention does not make missing chemical structures
valid inputs. Reject invalid molecules before fingerprinting.

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release --parallel
ctest --test-dir build/native -C Release --output-on-failure
cmake --install build/native --config Release --prefix src/rigid/_native
```

The loader searches `src/rigid/_native` in an editable installation, or its installed
package equivalent. Alternatively, set `RIGID_NATIVE_LIBRARY` to the absolute path
of the built DLL/shared library. An explicitly configured broken library raises;
automatic selection does not silently ignore it. With no native library,
`backend="auto"` selects NumPy. `backend="native"` requires a built library.

`maximum_training_similarity` uses query/reference blocks to bound intermediate
memory. It rejects an empty training library, for which maximum similarity is
undefined. The kernel checks null pointers and dimension overflow before access.
CTest covers the C ABI; Python tests compare the native output to an independent
binary-set calculation and the NumPy path, including empty rows and noncontiguous
inputs. Enable compiler-supported address/undefined-behavior sanitizers using
`-DRIGID_ENABLE_SANITIZERS=ON` for additional native diagnostics.
