"""Command handlers grouped by workflow; scientific dependencies remain lazy."""

from argparse import Namespace
from collections.abc import Callable

from . import data, models, operations

HANDLERS: dict[str, Callable[[Namespace], dict]] = {
    "validate": data.validate,
    "split": data.split,
    "plan-splits": data.plan_splits,
    "prepare": data.prepare,
    "qc": data.quality,
    "drift": data.drift,
    "inspect-library": data.inspect_library,
    "train": models.train,
    "train-generator": models.train,
    "train-vae": models.train,
    "rank": models.rank,
    "rank-vae": models.baseline,
    "neighbors": models.baseline,
    "generate": models.generate,
    "demo": models.demo,
    "doctor": operations.doctor,
    "benchmark": operations.benchmark,
    "fetch": operations.fetch,
    "report": operations.report,
    "inspect-run": operations.inspect_run,
    "compare": operations.compare,
}


def dispatch(args: Namespace) -> dict:
    try:
        handler = HANDLERS[args.command]
    except KeyError:
        raise ValueError(f"Unsupported command: {args.command}") from None
    return handler(args)
