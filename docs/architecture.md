# Architecture and maintenance boundaries

RIGID is a local research toolkit with a command interface and an importable Python API.
The package separates experimental definitions from optimization, model code, storage,
and reporting. A change in one layer should not silently change another layer's contract.

| Boundary | Responsibility | Must not do |
| --- | --- | --- |
| `data` and `quality` | Validate identities, ordered genes, splits and experimental context; report data quality | Fit preprocessing on held-out expression or invent biological annotations |
| `chem` and `library` | Standardize structures, construct graphs, retain identity provenance, bound reusable caches | Use mechanism labels or observed candidate responses as inference features |
| `models` and `generation` | Tensor transformations, objectives and legal graph actions | Read files, select datasets or manage optimization output directories |
| `engine` | Optimization state, cancellation, restart, early stopping and progress records | Define relevance labels or alter scientific candidate pools |
| `training` and `experiments` | Assemble a declared protocol and its phase-specific objectives | Select models using test results or silently resume a changed protocol |
| `pipeline` | Score libraries, expose contributions and retain exact top-k results | Treat ranking logits as probabilities or validator agreement as experimental evidence |
| `evaluation` and `runs` | Compute endpoints, eligible denominators and matched-target comparisons | Hide missing labels or compare mismatched input populations |
| `serialization`, `storage`, `checkpoint` | Parse unambiguous metadata and publish/verify artifacts | Deserialize arbitrary Python objects or silently replace immutable outputs |
| `commands` and `cli` | Translate arguments into public workflows and stable exit behavior | Duplicate scientific implementations or eagerly import optional ML dependencies for help |

## Decisions that constrain future changes

### Gene ordering is part of the model identity

A vector with the same width is not necessarily the same biological input. The ordered
gene manifest and namespace are checked at import, normalization and inference. Reordering
must be requested explicitly at ingestion, and the transformation belongs in provenance.
Silently sorting columns or matching only array shapes would make a numerically valid but
scientifically wrong run possible.

### Scientific definitions remain outside the optimizer

The reusable engine receives a loss callback and declared partition identities. Relevance,
candidate sampling, context encoding and generator exclusions belong to the experiment
layer. Engine refactors are checked against interrupted/uninterrupted training equivalence;
scientific changes require updated method documentation and separate experiments.

### Resume replays from a completed epoch

An interrupted partial epoch is not a reliable transactional boundary. Resume restores
optimizer, model, random generators and early-stopping state from the last complete epoch.
Protocol and software fingerprints must agree. The implementation does not pretend to resume
an arbitrary changed program from the middle of an optimizer update.

### Caches are bounded and invalidatable

Prepared libraries retain stable input-order identities and expose cache diagnostics.
Learned values are keyed by model state and feature/context information; chemical caches
have explicit memory or entry budgets. Labels do not enter scoring. Cached and uncached
paths are tested for numerical agreement, including source-order tie handling.

### Artifacts are immutable unless replacement is explicit

The storage layer writes to a same-directory temporary file, flushes it, then publishes it.
Create-if-absent publication uses a hard link so two cooperating writers cannot overwrite
one another. NTFS and ordinary local Linux filesystems support this operation. Unsupported
filesystems fail explicitly. Mutable state pointers use atomic replacement instead.

A checkpoint's integrity sidecar is its commit record. The payload and sidecar are separate
files, so a process termination between the two can leave an incomplete artifact. Readers
refuse it. Hashes detect corruption but do not authenticate a maliciously replaced pair.
Public distribution still requires trusted hosting and release-signature verification.

### Optional acceleration stays optional

The C++17 fingerprint ABI has a tested NumPy equivalent. Package import and pure-Python
evaluation remain usable without a compiler or GPU. A configured but broken native library
fails explicitly rather than producing silently different results.

## Command behavior

Successful commands write one JSON result to stdout; logging and errors go to stderr.
`--json-errors` makes execution errors machine-readable. Exit codes are 0 for success,
2 for input/execution errors and 130 for a keyboard interruption. Argument-parser usage
errors retain argparse's standard text and exit status. Optional profiling is published
without overwriting an existing artifact, even when the command is interrupted.

## Review checklist

Before merging a change, check the affected method definitions, serialization compatibility,
error behavior, resource budget and regression tests. Prefer an observable failing example
over a broad refactor justified only by aesthetics. New model claims need experimental
evidence; new engineering guarantees need tests that exercise their failure cases.
