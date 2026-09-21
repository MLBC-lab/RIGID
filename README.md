# RIGID

**Robust Inverse Gene-Expression Design for Stress-Aware Transcriptomic Candidate Prioritization**

A research toolkit for signature-conditioned molecular ranking, graph-based
candidate generation, and stress-aware evaluation. Python provides the scientific pipeline;
an optional dependency-free C++17 utility provides packed-fingerprint comparisons.

Version 0.3 adds independent-reference retrieval, configurable repeated/fold splits,
and optional consensus ranking alongside resumable training and prepared-library reuse.
See the [architecture](docs/architecture.md)
and [changelog](CHANGELOG.md) for the design and compatibility boundaries.

## Start here

Python 3.11 or newer is required. A CPU is sufficient for the demonstration.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ml,io,plots,dev]"
rigid doctor
rigid --verbose demo --output runs/demo
```

Open `runs/demo/report.html` for the offline report. The run also produces a versioned
dataset, train/validation loss history, a checksummed model bundle, per-target rankings,
metrics with eligible denominators, and a provenance manifest. Re-running requires a
new output directory; existing results are not silently overwritten.

## What is implemented

| Area | Capabilities |
| --- | --- |
| Data | Ordered-gene manifests; chunked CSV, GCT 1.2 and GCTX import; explicit metadata joins; source checksums; molecular standardization; repeated compound/scaffold holdouts and rotating folds; identity leakage audits; train-only normalization; quality and training-reference drift reports |
| Ranking | Directed-bond D-MPNN; target MLP; joint compatibility head; graded listwise and pairwise losses; configurable mechanism/scaffold candidate sampling; primary or opt-in consensus reranking |
| Generation | Trainable graph action model with add/connect/stop actions, teacher forcing, ring closures, legal-action masks, conditional or unconditional sampling, and failure/uniqueness accounting |
| Forward validation | Separate D-MPNN response regressor and independently parameterized graph/Transformer control, with cell/dose/time context |
| Baselines | Configurable conditional fingerprint VAE with library retrieval; training-only or separately supplied observed-response nearest neighbors with context policies |
| Evaluation | Exact/MOA/profile-family top-k recovery; Pearson and NDCG; scaffold novelty/diversity; stress-high rate; paired bootstrap; BH correction; cohort Jaccard; conformal intervals; temperature scaling/ECE; validation-fitted retrospective designability |
| Operations | Strict JSON; epoch-boundary resume; protocol guards; progress logs; bounded candidate caches; immutable outputs; checkpoint integrity; run inspection/comparison; offline reports; profiling; native parity; CI and packaging |

The graph generator uses a GraphINVENT-inspired action policy. The supplied forward
controls are a D-MPNN regressor and a graph/Transformer model; ExPO is a separate model
and is not bundled. The fingerprint VAE ranks library molecules rather than decoding
SMILES. See [model definitions](docs/models.md) for architecture and checkpoint compatibility.

## Use experimental data

Supply an ordered landmark-gene manifest and input files with their source and data-use
terms recorded. The [dataset guide](docs/datasets.md) links the relevant public resources.

```bash
rigid prepare --matrix expression.gctx --genes genes.json --compounds compounds.csv \
  --metadata metadata.csv --source-url https://example.org/your-dataset \
  --source-license "Your dataset license" --output data/imported --allow-reorder
rigid split data/imported --strategy scaffold --output data/scaffold-split
rigid validate data/scaffold-split --scaffold
rigid --verbose train data/scaffold-split --config configs/research.json \
  --scaffold-holdout --programs programs.json --output runs/research
rigid rank data/scaffold-split --checkpoint runs/research/model.pt \
  --split test --top-k 50 --output runs/research/test.json
rigid report runs/research/test.json --output runs/research/test.html
```

The multiline examples use POSIX line continuation; in PowerShell use one line or backticks.
`--programs` is optional. Without supplied program definitions, program/stress features
are disabled and missing stress annotations remain missing. The checked-in research
configuration is a starting point for validation, not a verified study configuration.

## Documentation

- [Architecture and maintenance boundaries](docs/architecture.md)
- [Optimization engine and recovery](docs/engine.md)
- [Prepared libraries and cache budgets](docs/library.md)
- [Data-quality and distribution-shift diagnostics](docs/quality.md)
- [Run inspection, comparison and command workflows](docs/run-management.md)
- [Data formats, splits, import and target editing](docs/data.md)
- [Public datasets and study-specific inputs](docs/datasets.md)
- [Repeated holdouts and rotating folds](docs/protocols.md)
- [Model architectures and graph generation](docs/models.md)
- [Training and inference protocol](docs/training.md)
- [Generator and baseline experiments](docs/experiments.md)
- [Observed-response reference catalogs](docs/neighbors.md)
- [Primary and consensus ranking](docs/consensus.md)
- [End-to-end hybrid candidate design example](examples/design_candidates.py)
- [Offline advanced evaluation example](examples/evaluation_protocol.py)
- [Evaluation definitions and native kernel](docs/evaluation.md)
- [Research scope and study materials](docs/SCIENTIFIC_CONTRACT.md)
- [FAIR4RS and OpenSSF readiness](docs/FAIR4RS.md)
- [Development, tests, profiling and debugging](docs/development.md)
- [Executed local verification and coverage](docs/VALIDATION.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Citation](CITATION.cff)

## Development checks

```bash
python -m pytest --cov=rigid --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m build
```

`rigid qc`, `rigid drift`, and `rigid inspect-run` provide inspectable checks before and
after training. Add `--resume` to a training command to continue an unchanged interrupted
run. Cached screening preserves source-order ties and checks model identity before reuse.

The optional packed-fingerprint utility has a tested NumPy fallback. Library screening
uses RDKit fingerprint comparisons. See the evaluation guide for build instructions;
measure end-to-end performance before choosing an alternative backend.

## Research use and release status

To request study target/profile IDs, split manifests, model configurations and weights,
gene-program sets, figure sources, or target-level results, contact the manuscript's
corresponding authors or the repository maintainer. These materials are not bundled;
their availability and redistribution permissions must be confirmed with the provider.

Predicted similarity, chemical filters, and agreement between learned validators are not
evidence of efficacy, safety, or causal disease reversal. Experimental validation is required.
Calibration guarantees also depend on their stated exchangeability assumptions.

The source is MIT-licensed. Third-party packages and datasets retain their own licenses.
This local repository does not yet have a public release, archival DOI, or OpenSSF badge.
The release checklist records these remaining steps instead of claiming certification.
