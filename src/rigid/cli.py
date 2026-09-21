"""Command-line interface; expensive optional dependencies are imported lazily."""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .storage import publish_file, staged_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigid", description="RIGID transcriptomic candidate design"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", action="store_true", help="Show training progress and debug errors"
    )
    parser.add_argument("--profile", type=Path, help="Write a cProfile timing trace")
    parser.add_argument(
        "--json-errors", action="store_true", help="Write structured errors to stderr"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local dependencies and acceleration")
    demo = commands.add_parser("demo", help="Run a fully synthetic training and screening example")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=17)
    validate = commands.add_parser("validate", help="Check dataset schema, checksums, and leakage")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--scaffold", action="store_true")
    validate.add_argument(
        "--acyclic-policy", choices=("group", "compound", "exclude"), default="group"
    )
    split = commands.add_parser("split", help="Create a new group-preserving dataset split")
    split.add_argument("dataset", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--strategy", choices=("compound", "scaffold"), default="scaffold")
    split.add_argument("--seed", type=int, default=17)
    split.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    split.add_argument(
        "--acyclic-policy", choices=("group", "compound", "exclude"), default="group"
    )
    plans = commands.add_parser(
        "plan-splits", help="Materialize auditable repeated or k-fold datasets"
    )
    plans.add_argument("dataset", type=Path)
    plans.add_argument("--protocol", type=Path, required=True)
    plans.add_argument("--output", type=Path, required=True)
    plans.add_argument("--max-splits", type=int, default=100)
    prepare = commands.add_parser("prepare", help="Join GCT/GCTX data with explicit metadata")
    for name in ("matrix", "genes", "compounds", "metadata", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--source-url", required=True)
    prepare.add_argument("--source-license", required=True)
    prepare.add_argument("--allow-reorder", action="store_true")
    for command in ("train", "train-generator", "train-vae"):
        description = {
            "train": "joint ranker and forward predictors",
            "train-generator": "graph proposal generator",
            "train-vae": "conditional fingerprint VAE",
        }[command]
        train = commands.add_parser(command, help=f"Train {description} on declared splits")
        train.add_argument("dataset", type=Path)
        train.add_argument("--output", type=Path, required=True)
        train.add_argument("--config", type=Path)
        train.add_argument(
            "--resume", action="store_true", help="Continue an unchanged interrupted training run"
        )
        train.add_argument("--scaffold-holdout", action="store_true")
        train.add_argument(
            "--acyclic-policy",
            choices=("group", "compound", "exclude"),
            help="Override the configuration's acyclic scaffold policy",
        )
        if command == "train":
            train.add_argument("--programs", type=Path)
        if command == "train-generator":
            train.add_argument("--max-atoms", type=int, default=40)
            train.add_argument("--unconditional", action="store_true")
        if command == "train-vae":
            train.add_argument("--fingerprint-bits", type=int, default=2048)
            train.add_argument("--latent-dim", type=int)
            train.add_argument("--beta", type=float, default=0.01)
    rank = commands.add_parser(
        "rank", help="Screen a candidate library and evaluate held-out targets"
    )
    rank.add_argument("dataset", type=Path)
    rank.add_argument("--checkpoint", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--split", choices=("test", "validation", "external"), default="test")
    rank.add_argument("--top-k", type=int, default=50)
    rank.add_argument("--device", default="cpu")
    rank.add_argument("--reranking", type=Path, help="JSON primary or consensus scoring policy")
    for command in ("rank-vae", "neighbors"):
        baseline = commands.add_parser(
            command, help="Evaluate a clearly labeled reference baseline"
        )
        baseline.add_argument("dataset", type=Path)
        baseline.add_argument("--output", type=Path, required=True)
        baseline.add_argument("--split", choices=("test", "validation", "external"), default="test")
        if command == "rank-vae":
            baseline.add_argument("--checkpoint", type=Path, required=True)
            baseline.add_argument("--samples", type=int, default=32)
        else:
            baseline.add_argument("--top-k", type=int, default=50)
            baseline.add_argument("--scaffold-holdout", action="store_true")
            baseline.add_argument(
                "--acyclic-policy", choices=("group", "compound", "exclude"), default="group"
            )
            baseline.add_argument(
                "--reference-dataset",
                type=Path,
                help="Separate observed-response catalog for reference retrieval",
            )
            baseline.add_argument(
                "--reference-config",
                type=Path,
                help="JSON context, aggregation, and reference selection policy",
            )
    generate = commands.add_parser(
        "generate", help="Sample graph proposals with validity accounting"
    )
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=100)
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument(
        "--target", type=Path, help="JSON with gene_manifest and expression fields"
    )
    report = commands.add_parser("report", help="Render an offline HTML screening report")
    report.add_argument("screening", type=Path)
    report.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser(
        "fetch", help="Download one explicitly requested checksum-pinned file"
    )
    fetch.add_argument("url")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--max-bytes", type=int, default=2_000_000_000)
    benchmark = commands.add_parser("benchmark", help="Measure local packed-fingerprint throughput")
    benchmark.add_argument("--molecules", type=int, default=1000)
    for name in ("qc", "drift"):
        diagnostic = commands.add_parser(name, help="Inspect data quality or shift from training")
        diagnostic.add_argument("dataset", type=Path)
        diagnostic.add_argument("--output", type=Path, required=True)
        if name == "drift":
            diagnostic.add_argument(
                "--split", choices=("validation", "calibration", "test", "external"), default="test"
            )
    library = commands.add_parser(
        "inspect-library", help="Inspect and optionally record a candidate-library manifest"
    )
    library.add_argument("library", type=Path)
    library.add_argument("--deduplicate", action="store_true")
    library.add_argument("--output", type=Path)
    inspect = commands.add_parser(
        "inspect-run", help="Verify a run and summarize optimization history"
    )
    inspect.add_argument("run", type=Path)
    inspect.add_argument(
        "--skip-weights", action="store_true", help="Skip checkpoint integrity verification"
    )
    compare = commands.add_parser("compare", help="Compare matched-target screening results")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--resamples", type=int, default=2000)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--seed", type=int, default=17)
    return parser


def _execute(args: argparse.Namespace) -> dict:
    from .commands import dispatch

    return dispatch(args)


def _emit_error(error: BaseException, args: argparse.Namespace) -> None:
    if isinstance(error, FileExistsError):
        code = "output_exists"
    elif isinstance(error, FileNotFoundError):
        code = "not_found"
    elif isinstance(error, ImportError):
        code = "missing_dependency"
    elif isinstance(error, KeyboardInterrupt):
        code = "interrupted"
    else:
        code = "invalid_request" if isinstance(error, ValueError) else "execution_failed"
    message = str(error) or "Command interrupted"
    if args.json_errors:
        print(
            json.dumps({"error": {"code": code, "message": message, "command": args.command}}),
            file=sys.stderr,
        )
    else:
        print(f"rigid: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.profile and args.profile.exists():
        _emit_error(FileExistsError("Profile output exists; choose a new filename"), args)
        return 2
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    profiler = cProfile.Profile() if args.profile else None
    status, result = 0, None
    try:
        if profiler:
            profiler.enable()
        result = _execute(args)
    except KeyboardInterrupt as error:
        _emit_error(error, args)
        status = 130
    except (ValueError, OSError, RuntimeError, ImportError, FloatingPointError) as error:
        if args.verbose:
            logging.exception("Command failed")
        _emit_error(error, args)
        status = 2
    finally:
        if profiler:
            profiler.disable()
            try:
                with staged_file(args.profile) as temporary:
                    profiler.dump_stats(temporary)
                    publish_file(temporary, args.profile)
            except OSError as error:
                _emit_error(error, args)
                status = 2
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
    return status
