"""Model fitting and screening workflows for the command interface."""

from argparse import Namespace
from dataclasses import replace

from rigid.config import TrainConfig
from rigid.provenance import atomic_json
from rigid.serialization import read_json, require_object


def train(args: Namespace) -> dict:
    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    if args.acyclic_policy is not None:
        config = replace(config, acyclic_policy=args.acyclic_policy)
    common = {"resume": args.resume, "scaffold_holdout": args.scaffold_holdout}
    if args.command == "train":
        from rigid.training import train as train_ranker

        programs = (
            require_object(read_json(args.programs), context="Programs") if args.programs else None
        )
        result = train_ranker(args.dataset, args.output, config, programs=programs, **common)
    elif args.command == "train-generator":
        from rigid.experiments import train_generator

        result = train_generator(
            args.dataset,
            args.output,
            config,
            max_atoms=args.max_atoms,
            conditional=not args.unconditional,
            **common,
        )
    else:
        from rigid.experiments import train_vae

        result = train_vae(
            args.dataset,
            args.output,
            config,
            fingerprint_bits=args.fingerprint_bits,
            latent_dim=args.latent_dim,
            beta=args.beta,
            **common,
        )
    return {"output": str(args.output.resolve()), "source_kind": result.get("source_kind")}


def rank(args: Namespace) -> dict:
    from rigid.consensus import RerankingConfig
    from rigid.pipeline import screen_dataset

    result = screen_dataset(
        args.checkpoint,
        args.dataset,
        args.output,
        split=args.split,
        k=args.top_k,
        device=args.device,
        reranking=RerankingConfig.load(args.reranking) if args.reranking else None,
    )
    return {"output": str(args.output.resolve()), "metrics": result["metrics"]}


def baseline(args: Namespace) -> dict:
    from rigid.experiments import nearest_neighbor, screen_vae

    if args.output.exists():
        raise FileExistsError("Baseline output exists; choose a new filename")
    if args.command == "rank-vae":
        result = screen_vae(args.checkpoint, args.dataset, split=args.split, samples=args.samples)
    else:
        from rigid.baselines import ReferenceNeighborConfig

        if args.reference_config and not args.reference_dataset:
            raise ValueError("--reference-config requires --reference-dataset")
        result = nearest_neighbor(
            args.dataset,
            split=args.split,
            k=args.top_k,
            scaffold_holdout=args.scaffold_holdout,
            acyclic_policy=args.acyclic_policy,
            reference_dataset=args.reference_dataset,
            reference_config=ReferenceNeighborConfig.load(args.reference_config)
            if args.reference_config
            else None,
        )
    atomic_json(args.output, result, overwrite=False)
    return {"output": str(args.output.resolve())}


def generate(args: Namespace) -> dict:
    from rigid.data.schema import GeneManifest
    from rigid.experiments import generate as sample

    if args.output.exists():
        raise FileExistsError("Generation output exists; choose a new filename")
    target = (
        require_object(read_json(args.target), context="Generation target") if args.target else None
    )
    if target is not None and set(target) != {"gene_manifest", "expression"}:
        raise ValueError("Generation target requires exactly gene_manifest and expression")
    result = sample(
        args.checkpoint,
        n=args.count,
        seed=args.seed,
        target=target["expression"] if target else None,
        genes=GeneManifest.from_dict(target["gene_manifest"]) if target else None,
    )
    atomic_json(args.output, result, overwrite=False)
    return {"output": str(args.output.resolve())}


def demo(args: Namespace) -> dict:
    from rigid.data import make_synthetic_dataset
    from rigid.pipeline import screen_dataset
    from rigid.reporting import render_report
    from rigid.training import train as train_ranker

    if args.output.exists():
        raise FileExistsError("Demo output exists; choose a new directory")
    args.output.mkdir(parents=True)
    make_synthetic_dataset(args.output / "data", seed=args.seed)
    config = TrainConfig(
        seed=args.seed,
        epochs=4,
        forward_epochs=3,
        batch_size=8,
        candidate_pool_size=16,
        hidden_dim=32,
        depth=2,
        dropout=0,
        patience=4,
        top_k=5,
    )
    train_ranker(args.output / "data", args.output / "training", config)
    screen_dataset(
        args.output / "training/model.pt", args.output / "data", args.output / "screening.json", k=5
    )
    render_report(args.output / "screening.json", args.output / "report.html")
    return {
        "source_kind": "synthetic",
        "reproduces_manuscript": False,
        "report": str((args.output / "report.html").resolve()),
    }
