"""Read-only inspection, reporting, and acquisition command handlers."""

from argparse import Namespace

from rigid.provenance import atomic_json


def doctor(args: Namespace) -> dict:
    from rigid.diagnostics import doctor as inspect_environment

    return inspect_environment()


def benchmark(args: Namespace) -> dict:
    from rigid.diagnostics import benchmark_fingerprints

    return benchmark_fingerprints(molecules=args.molecules)


def fetch(args: Namespace) -> dict:
    from rigid.acquisition import fetch_file

    return {
        "download": str(
            fetch_file(args.url, args.output, args.sha256, maximum_bytes=args.max_bytes)
        )
    }


def report(args: Namespace) -> dict:
    from rigid.reporting import render_report

    render_report(args.screening, args.output)
    return {"report": str(args.output.resolve())}


def inspect_run(args: Namespace) -> dict:
    from rigid.runs import inspect_run as inspect

    return inspect(args.run, verify_weights=not args.skip_weights)


def compare(args: Namespace) -> dict:
    from rigid.runs import compare_screenings

    if args.output.exists():
        raise FileExistsError("Comparison output exists; choose a new filename")
    result = compare_screenings(
        args.left,
        args.right,
        n_resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    atomic_json(args.output, result, overwrite=False)
    return {"output": str(args.output.resolve()), "target_count": result["target_count"]}
