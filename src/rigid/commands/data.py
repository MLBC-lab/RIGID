"""Dataset preparation and quality-control command handlers."""

from argparse import Namespace

from rigid.provenance import atomic_json


def validate(args: Namespace) -> dict:
    from rigid.data import audit_leakage, load_dataset

    dataset = load_dataset(args.dataset)
    audit = audit_leakage(dataset, scaffold=args.scaffold, acyclic_policy=args.acyclic_policy)
    audit.raise_for_leakage()
    return {"source_kind": dataset.provenance.kind, "genes": len(dataset.genes), **audit.to_dict()}


def split(args: Namespace) -> dict:
    from rigid.data import assign_splits, load_dataset, save_dataset

    if args.output.exists():
        raise FileExistsError("Split output exists; choose a new directory")
    dataset = load_dataset(args.dataset)
    save_dataset(
        assign_splits(
            dataset,
            strategy=args.strategy,
            seed=args.seed,
            fractions=tuple(args.fractions),
            acyclic_policy=args.acyclic_policy,
        ),
        args.output,
    )
    return {"dataset": str(args.output.resolve())}


def plan_splits(args: Namespace) -> dict:
    from rigid.protocols import SplitProtocol, materialize_splits

    manifest = materialize_splits(
        args.dataset, args.output, SplitProtocol.load(args.protocol), max_splits=args.max_splits
    )
    return {"output": str(args.output.resolve()), "manifest": manifest}


def prepare(args: Namespace) -> dict:
    from rigid.data.preparation import prepare_matrix

    dataset = prepare_matrix(
        args.matrix,
        args.genes,
        args.compounds,
        args.metadata,
        args.output,
        source_url=args.source_url,
        source_license=args.source_license,
        allow_reorder=args.allow_reorder,
    )
    return {"dataset": str(args.output.resolve()), "profiles": len(dataset.profiles)}


def quality(args: Namespace) -> dict:
    from rigid.data import load_dataset
    from rigid.quality import audit_dataset

    if args.output.exists():
        raise FileExistsError("Quality report output exists; choose a new filename")
    result = audit_dataset(load_dataset(args.dataset)).to_dict()
    atomic_json(args.output, result, overwrite=False)
    return {"output": str(args.output.resolve()), "source_kind": result.get("dataset_kind")}


def drift(args: Namespace) -> dict:
    from rigid.data import load_dataset
    from rigid.quality import compare_to_training

    if args.output.exists():
        raise FileExistsError("Drift report output exists; choose a new filename")
    result = compare_to_training(load_dataset(args.dataset), args.split).to_dict()
    atomic_json(args.output, result, overwrite=False)
    return {"output": str(args.output.resolve()), "source_kind": result.get("dataset_kind")}


def inspect_library(args: Namespace) -> dict:
    from rigid.library import PreparedLibrary

    library = PreparedLibrary.from_csv(args.library, deduplicate=args.deduplicate)
    if args.output:
        library.save_manifest(args.output)
    return library.summary()
