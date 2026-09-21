# Repeated library screening

`PreparedLibrary` separates a candidate library's identity from target-specific
scoring. A library can be prepared once and reused for many expression signatures,
cell contexts, or compatible checkpoints. It never contains target expression
measurements and never uses mechanism-of-action labels to calculate scores.

```python
from rigid.data import load_dataset
from rigid.library import PreparedLibrary
from rigid.pipeline import RIGIDPipeline

dataset = load_dataset("data/curated")
pipeline = RIGIDPipeline.load("runs/main/model.pt")
library = PreparedLibrary.from_csv("data/candidates.csv")

for index, profile in enumerate(dataset.profiles):
    if profile.split == "external":
        ranked = pipeline.rank(
            dataset.expression[index],
            dataset.genes,
            profile,
            library,
            k=50,
            batch_size=128,
        )
        # Store or consume each target's results before processing the next.

print(library.summary())
library.save_manifest("runs/main/candidate-identities.json")
```

The existing `pipeline.rank(..., candidates=list_of_compounds)` API still works.
It creates a temporary prepared library; callers screening repeated targets should
retain a `PreparedLibrary` explicitly. `screen_dataset` already does so internally.

## Candidate identity

CSV input requires `compound_id` and `smiles` columns. Optional `aliases` and `moa`
columns contain semicolon-separated values. An optional UTF-8 byte-order mark is
accepted. Malformed rows report their input line number; invalid molecules are
errors, not silently omitted candidates.

Structures are standardized with the same chemical policy as dataset preparation.
Duplicate identifiers, ambiguous aliases mapping to different structures, and
duplicate standardized structures are rejected by default. Supply
`deduplicate=True` only when intentional:

* The first occurrence remains the representative candidate and keeps its position.
* Later source identifiers and aliases are retained as alias provenance.
* Biological annotations are merged for caller inspection, never scored.
* The summary records original and retained candidate counts.

The identity digest covers candidate order, representative IDs, canonical SMILES,
aliases, source IDs, and feature/fingerprint versions. Changing a mechanism label
does not change this digest because labels are not part of a screening identity.
Manifests preserve this identity, not evaluation annotations. Retain the source CSV
or dataset when biological labels are needed.

`save_manifest` writes a new, checksummed JSON artifact and refuses overwrites.
`from_manifest` checks integrity, canonicalization, required fields, and supported
feature versions. A checksum detects accidental changes; it is not a signature or
proof that a third-party library is trustworthy. CSV source checksums are recorded
separately from the canonical identity digest.

## Cache and memory policy

Defaults are intentionally finite:

| Cache | Default retained payload budget | Contents |
| --- | ---: | --- |
| Numeric rows | 128 MiB | Molecular embeddings, context-specific predictions, training similarity |
| Molecular graphs | 64 MiB | Atom, bond, edge, and reverse-edge arrays |
| Fingerprints | 16 MiB | 2,048-bit radius-2 Morgan fingerprints |
| Chemical descriptors | 1,024 candidates | QED, SA, physicochemical descriptors, alerts |

Least-recently-used entries are evicted when a budget is reached. Oversized entries
are computed without retention. Set a byte budget or the descriptor-entry limit to
zero to disable that cache. `cache_info()` exposes occupancy, payload limits,
hits, misses, and eviction counters. `clear_cache()` removes retained entries
without changing library identity.

These limits apply to retained data payloads, **not total process RAM**. Python and
RDKit objects have overhead; the source records, one training-similarity vector,
model parameters, temporary graph/prediction batches, and the requested top-k
results also occupy memory. Lower `batch_size` for large molecules or limited
accelerator memory. Transformer attention still grows quadratically with the
largest molecule in a batch. Libraries larger than the cache remain supported,
but repeated queries can require recomputation after eviction.

Maximum training similarity is computed with candidate and reference blocks. It
does not materialize a candidate-by-training-library similarity matrix. Reference
identities participate in the cache key, so changing the training library cannot
reuse stale similarity values.

## Model-state and context isolation

Prediction cache namespaces include the full current model tensor contents,
model configuration, class, chemical feature version, gene-manifest checksum,
execution device, and the encoded context where applicable. Fingerprints are
content-based; mutable filenames and Python object identities are not cache keys.
Changing parameters in place, loading a different checkpoint, changing the gene
manifest, or changing cell/dose/time produces a different namespace.

Only structure-dependent ranker embeddings are shared across all targets. Forward
responses can also be reused when structure and encoded context agree. Their cache
key does not include the target signature because these predictors do not receive
that signature. Target embeddings, auxiliary compatibility/program features,
stress penalties, and ranking-head scores are evaluated for the current query.
The pipeline refuses models switched back into training mode, where dropout would
invalidate this deterministic inference contract.

The implementation hashes model tensors at each query boundary. This costs a
linear model-state scan and a host transfer on accelerators; it provides explicit
invalidation even after in-place parameter changes. No embedding cache is pickled
or trusted from disk. Cached arrays returned to callers are copies, so modifying a
result cannot corrupt subsequent scores. Prepared-library cache instances are
process-local and are not intended for concurrent mutation by multiple threads.

## Ranking semantics

Scoring retains the existing additive decomposition:

```text
final score = compatibility - stress_weight × stress
                            + novelty_weight × (1 - maximum_training_similarity)
```

A bounded heap retains exactly the best `k` candidates across batches, using the
same input-order tie break as `rerank_candidates`. It stores no dense
target-by-candidate score matrix. Changing the computational batch size can change
last-bit floating-point rounding and therefore the order of nearly tied scores;
exact ties always preserve input order. Regression tests compare the streamed
implementation against the full legacy scoring calculation using matching
batches and check numeric agreement across batch sizes.

Chemical descriptors are evaluated only for retained candidates, then cached for
later queries. They remain reported flags rather than an implicit exclusion rule.
The screening artifact records the candidate-library identity checksum alongside
checkpoint and input checksums. It does not turn model scores into calibrated
probabilities or substitute predicted responses for experimental validation.
