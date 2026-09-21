# Development and diagnostics

## Local environment

Install `.[ml,io,plots,dev]` into a virtual environment. `rigid doctor` reports the Python
platform, scientific package versions, optional dependencies, native-kernel availability
and CUDA availability. It does not collect secrets or upload telemetry.

Core data/evaluation use does not require PyTorch. Model imports and training require
the `ml` extra. GCTX needs `io`. Plotting dependencies are optional, not necessary for
the self-contained HTML report. Native acceleration is an optional separately built library;
ordinary wheels remain pure-Python and use the NumPy fallback by default.

## Tests

```bash
python -m pytest --cov=rigid --cov-report=term-missing --cov-report=xml
python -m ruff check .
python -m ruff format --check .
python -m build
python tools/check_distribution.py --wheel-dir dist --output verification/wheel-check --demo
```

All fixtures are small and synthetic; no network, private manuscript or real patient
data is required. Native parity tests skip when the compiled library is unavailable;
the native CI job explicitly builds and loads it, so that path cannot silently go untested.
Tests cover graph invariance, message exclusion, differentiability, optimization, generation
round trips, partition leakage, gene order, normalization, statistical denominators,
calibration fit partitions, checksums and end-to-end workflows.

The distribution check inspects the wheel for missing package files, unsafe paths and
development/private artifacts, installs it into a new directory without fetching dependencies,
and verifies imports come from that installation. It exercises diagnostics, fingerprint
benchmarking and, with `--demo`, the complete synthetic workflow. Use `--wheel PATH` when
`dist/` contains wheels from several versions. Dependencies must already be installed.

CI enforces at least 85% combined statement/branch coverage. Tests added during the
engineering review exercise resumable optimizer/RNG state, competing writers, cache
invalidation, input drift, nested JSON schemas and matched comparison denominators.

On machines restricting the system temporary directory, use a new local path such as
`python -m pytest --basetemp .test-tmp/local-run-1`. Never reuse a valuable directory as
pytest's `--basetemp`; pytest owns and may clear that directory.

## Profiling and debugging

```bash
rigid --verbose --profile runs/demo-profile.prof demo --output runs/profiled-demo
python -m pstats runs/demo-profile.prof
rigid benchmark --molecules 1000
```

`--verbose` provides epoch losses and failure tracebacks. Finite-loss and gradient-norm
checks stop unstable optimization. Checkpoint feature schemas and gene manifests catch
misaligned inputs. `rigid validate` provides detailed offending identity/scaffold groups.
Set `RIGID_NATIVE_LIBRARY` to an explicitly trusted built library to enable the C++ kernel;
never load a library supplied by an untrusted source.

## Common failures

| Error | Check |
| --- | --- |
| Gene order mismatch | Preserve `genes.json` from training; explicitly opt into reorder during import |
| Leakage audit failed | Inspect aliases, standardized structures and repeated profiles across splits |
| Too few independent groups | Use a larger dataset; do not split repetitions of the same molecule |
| Missing train/validation split | Import metadata or call `rigid split` into a new directory |
| Unsupported generator chemistry | Inspect recorded exclusions; charged/stereo/isotope chemistry is not silently simplified |
| Checkpoint checksum mismatch | Recover the trusted checkpoint and matching integrity record |
| CUDA unavailable/deterministic operation error | Use CPU or a verified supported CUDA environment |
| Output already exists | Choose a new run name; existing results are protected |
| GCTX unavailable | Install the `io` extra and verify HDF5 dimensions/metadata |

## Static typing

The project distributes `py.typed` and includes type annotations on public contracts.
Scientific dependency stubs vary substantially across NumPy/RDKit releases. The committed
mypy command checks the dependency-light configuration/provenance/checkpoint interfaces
without following external stubs; comprehensive runtime contracts and tests cover graph
and array shapes. Do not describe this as whole-project strict type-check coverage.

Expanded local checks cover the optimizer, prepared libraries, pipeline, quality diagnostics
and feature contracts with the Python 3.12 checker target. This accommodates the installed
NumPy stub syntax; the supported Python 3.11 runtime is exercised separately by the CI matrix.

## Release hygiene

Build from a clean reviewed Git revision, run the test matrix, inspect the wheel/sdist,
record dependency versions, and execute a smoke test from the installed wheel. The CI
workflow is supplied but cannot demonstrate a historical green build until run on the
actual remote host. Dependencies are constrained to supported major versions; this is
not a claim that every possible version combination has been tested.
