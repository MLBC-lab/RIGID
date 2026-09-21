# Local verification record

Version 0.3.0 was verified on 2026-09-15 using Windows 11, CPython 3.12.14 and CPU-only PyTorch.
This record describes executed checks, not a claim of scientific reproduction or
remote CI status. The original manuscript remained unchanged.

## Results

| Check | Observed result |
| --- | --- |
| Complete Python suite, including configured native kernel | 496 passed; no skipped tests |
| Statement coverage | 92.20% (4,529 of 4,912 executable statements) |
| Branch coverage | 80.55% (1,445 of 1,794 branches) |
| Combined coverage.py measure | 89.08% |
| C++ Release build and CTest | Build successful; 1 of 1 test passed |
| Native / NumPy parity | Exact numerical parity tests passed |
| Ruff lint and formatting | Passed |
| Scoped mypy: configuration, provenance, checkpoint, storage, serialization and run inspection | Passed for 6 files |
| Scoped mypy: optimization, libraries, inference, quality, features and input contracts | Passed for 8 files with Python 3.12 checker target |
| Scoped mypy: configuration, consensus, run inspection, inference, split protocols and reference retrieval | Passed for 6 files with Python 3.12 checker target |
| Source distribution and wheel build | Successful |
| Installed-wheel smoke test | Imports, diagnostics, benchmark and complete synthetic demo passed |
| Dependency integrity (`pip check`) | No broken requirements |
| Dependency vulnerability audit | No known vulnerabilities reported after updating the isolated environment's pip |

The previous 0.2.0 baseline had 343 passing tests and 87.86% combined coverage. Version 0.3.0
adds 153 tests while increasing the covered implementation surface. No new runtime
dependencies were introduced in this review. The recorded dependency audit is from the
same verified environment; the new source was checked by tests and independent code review,
not by querying a vulnerability database for unpublished code.

The audit skips the editable RIGID distribution itself because it is not a published
package with advisory records. It is not a security audit of newly written source code.
Runtime dependencies, build tools and audit tools are listed in the environment inventory.
The snapshot [tested-windows-py312.txt](../requirements/tested-windows-py312.txt) records
exact installed versions but is not a hash-locked, platform-independent lockfile.

## Scientific regression checks

- Independent-reference retrieval recovers a held-out source from a distinct synthetic
  response while rejecting shared profile IDs and expression-vector copies. Missing
  references remain explicit; changing an alias cannot remove known MOA annotations.
- Primary scoring remains unchanged when the independent predictor changes. Consensus
  scoring uses the declared components and records that both predictors participate in
  selection; saved reports reject contradictory evaluator roles or score decompositions.
- Repeated/fold protocols preserve protected partitions and group identities, record
  duplicate assignments explicitly, and propagate acyclic policies through training
  and checkpoint-based screening.
- Interrupted/resumed optimization matches uninterrupted CPU execution bit for bit,
  including dropout, stochastic objectives, optimizer state, RNGs and all ranking phases.
- Prepared-library and uncached results agree; state/configuration/context changes
  invalidate learned caches and ties retain the original candidate order.
- Train-reference drift diagnostics do not use held-out data to fit the reference.
- Competing writers cannot replace an immutable artifact, and aborted staging leaves
  no published partial output.
- Aggregate rates are reconciled with per-target records, and matched-run comparisons
  reject changed annotation eligibility instead of silently dropping difficult targets.
- Altering all held-out test expression values and inserting a new held-out molecule,
  MOA label and unseen context leaves all three trained models' parameters bitwise unchanged.
- Training candidate generation and validation checkpoint selection do not use test/external
  compounds or their MOA annotations.
- Changing candidate MOA labels does not change scores; those labels are attached only for evaluation.
- Wrong gene order, checkpoint corruption, duplicate identities and protected-output
  overwrites are rejected.
- D-MPNN reverse-edge exclusion is checked against a hand-calculated example, and atom-order
  invariance, graph-batch isolation, gradients and tiny-problem fitting are tested.
- Generator teacher traces round-trip supported chemistry; unsupported chemistry is reported
  explicitly. The hybrid proposal-to-ranking example runs against actually trained tiny models.
- Calibration fit partitions are distinct from test partitions; missing labels and unknown
  novelty have explicit denominators rather than fabricated success values.

## Re-run

```bash
python -m pytest --cov=rigid --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy --follow-imports=skip src/rigid/config.py src/rigid/provenance.py src/rigid/checkpoint.py
python -m build
python tools/check_distribution.py --wheel dist/rigid_transcriptomics-0.3.0-py3-none-any.whl --output verification/wheel-check --demo
```

Build the optional native library and set `RIGID_NATIVE_LIBRARY` to its absolute path to
include native parity in the full test suite. Without it, one optional parity test skips.
See [evaluation.md](evaluation.md) for native commands and [development.md](development.md)
for environment diagnostics and temporary-directory restrictions.

The local `verification/` directory contains detailed coverage/JUnit records, a dependency
audit and a CycloneDX software-component inventory. Those machine-local artifacts, installed
packages, synthetic checkpoints and private inputs are intentionally excluded from Git and
the source ZIP. Cross-platform CI is configured but has not run on a public host. No GPU,
large-scale LINCS experiment, original ExPO checkpoint or clinical validation was performed.
