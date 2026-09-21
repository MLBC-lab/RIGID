"""Sample molecular proposals and rank a hybrid library for one expression target.

Install RIGID with its ml extra before running this script. Both checkpoints must
come from actual training runs; the report preserves synthetic source labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from rigid.checkpoint import load_checkpoint
from rigid.chem.standardize import standardize_smiles
from rigid.data import CompoundRecord, ProfileRecord, TrainStandardizer, load_dataset
from rigid.experiments import generate
from rigid.generation import GeneratedMolecule, select_candidates
from rigid.pipeline import RIGIDPipeline
from rigid.provenance import dataset_hashes, environment_report, sha256_file


def _hybrid_candidates(compounds, generation_report):
    library = {}
    library_aliases = {}
    for compound in sorted(compounds, key=lambda item: item.compound_id):
        smiles = compound.canonical_smiles
        library_aliases.setdefault(smiles, []).append(compound.compound_id)
        if smiles not in library:
            # Screening candidates carry structure and identity, never MOA labels.
            library[smiles] = CompoundRecord(compound.compound_id, smiles)
    canonical_proposals = []
    standardization_rejections = []
    for index, row in enumerate(generation_report["proposals"]):
        proposal = GeneratedMolecule(**row)
        if not proposal.valid or proposal.smiles is None:
            canonical_proposals.append(proposal)
            continue
        try:
            smiles = standardize_smiles(proposal.smiles).canonical_smiles
        except (ValueError, RuntimeError) as exc:
            standardization_rejections.append({"attempt_index": index, "reason": str(exc)})
            canonical_proposals.append(
                GeneratedMolecule(
                    None,
                    False,
                    proposal.terminated,
                    proposal.steps,
                    proposal.log_probability,
                    str(exc),
                )
            )
            continue
        canonical_proposals.append(
            GeneratedMolecule(
                smiles, True, proposal.terminated, proposal.steps, proposal.log_probability
            )
        )
    selected = select_candidates(
        canonical_proposals, scorer=lambda _: 0.0, top_k=max(1, len(canonical_proposals))
    )
    origins = {compound.compound_id: "library" for compound in library.values()}
    generated = []
    known_ids = set(origins)
    for smiles, _ in selected.candidates:
        if smiles in library:
            origins[library[smiles].compound_id] = "library+generated"
            continue
        identifier = "RIGID_GENERATED_" + hashlib.sha256(smiles.encode()).hexdigest()[:24]
        if identifier in known_ids:
            raise ValueError("A generated candidate identifier collides with the input catalog")
        known_ids.add(identifier)
        generated.append(CompoundRecord(identifier, smiles))
        origins[identifier] = "generated"
    candidates = [*library.values(), *generated]
    return (
        candidates,
        origins,
        {
            "library_input_records": len(compounds),
            "library_unique_structures": len(library),
            "library_duplicate_groups": [ids for ids in library_aliases.values() if len(ids) > 1],
            "generated_valid_after_standardization": selected.valid,
            "generated_unique_structures": selected.unique,
            "generated_new_to_library": len(generated),
            "hybrid_candidate_count": len(candidates),
            "standardization_rejections": standardization_rejections,
        },
    )


def design_candidates(
    dataset_path: str | Path,
    ranker_checkpoint: str | Path,
    generator_checkpoint: str | Path,
    target_id: str,
    output: str | Path,
    *,
    count: int = 128,
    top_k: int = 50,
    seed: int = 17,
) -> dict:
    """Build a hybrid candidate pool and score it without recovery labels."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output exists; choose a new JSON filename")
    if type(count) is not int or count < 1 or type(top_k) is not int or top_k < 1:
        raise ValueError("count and top_k must be positive integers")
    dataset = load_dataset(dataset_path)
    indices = [i for i, profile in enumerate(dataset.profiles) if profile.profile_id == target_id]
    if len(indices) != 1:
        raise ValueError(f"Target profile {target_id!r} is absent or not unique")
    index = indices[0]
    profile = dataset.profiles[index]
    ranker_payload = load_checkpoint(ranker_checkpoint, device="cpu")
    generator_payload = load_checkpoint(generator_checkpoint, device="cpu")
    if generator_payload.get("kind") != "graph-generator":
        raise ValueError("Expected a graph-generator proposal checkpoint")
    pipeline = RIGIDPipeline(ranker_payload)
    pipeline.genes.require_match(dataset.genes)
    TrainStandardizer.from_dict(generator_payload["normalizer"]).genes.require_match(dataset.genes)
    torch.set_num_threads(ranker_payload["config"].get("num_threads", 1))
    conditional = generator_payload["model_config"]["condition_dim"] > 0
    generation_report = generate(
        generator_checkpoint,
        n=count,
        seed=seed,
        target=dataset.expression[index] if conditional else None,
        genes=dataset.genes if conditional else None,
    )
    candidates, origins, pool_report = _hybrid_candidates(dataset.compounds, generation_report)
    context = ProfileRecord(
        "query",
        "query",
        profile.cell_id,
        profile.dose_um,
        profile.time_h,
    )
    ranked = pipeline.rank(dataset.expression[index], dataset.genes, context, candidates, k=top_k)
    for row in ranked:
        row["origin"] = origins[row["candidate_id"]]
    sources = {
        "input": dataset.provenance.kind,
        "ranker_training": ranker_payload.get("training_source_kind", "unknown"),
        "generator_training": generator_payload.get("source_kind", "unknown"),
    }
    contains_synthetic = "synthetic" in sources.values()
    source_kind = (
        "synthetic"
        if contains_synthetic
        else "unknown"
        if "unknown" in sources.values()
        else "experimental"
    )
    result = {
        "schema_version": 1,
        "kind": "hybrid-proposal-ranking",
        "source_kind": source_kind,
        "contains_synthetic_data": contains_synthetic,
        "reproduces_manuscript": False,
        "scientific_status": (
            "Synthetic demonstration: at least one input or model-training source is synthetic."
            if contains_synthetic
            else "Model inference; candidates require independent validation."
        ),
        "target_profile_id": target_id,
        "target_split": profile.split,
        "context": {
            "cell_id": profile.cell_id,
            "dose_um": profile.dose_um,
            "time_h": profile.time_h,
        },
        "requested_top_k": top_k,
        "seed": seed,
        "provenance": {
            "source_kinds": sources,
            "input": dataset.provenance.to_dict(),
            "input_sha256": dataset_hashes(dataset_path),
            "gene_manifest": dataset.genes.to_dict(),
            "ranker_checkpoint_sha256": sha256_file(ranker_checkpoint),
            "generator_checkpoint_sha256": sha256_file(generator_checkpoint),
            "ranker_training_compound_count": len(ranker_payload["train_ids"]),
            "generator_training_compound_count": len(generator_payload["train_ids"]),
            "software": environment_report(),
        },
        "generation": generation_report,
        "candidate_pool": pool_report,
        "rankings": ranked,
        "notes": [
            "The pool combines the supplied library with unique valid generated proposals.",
            "No analog optimization is performed.",
            "Target source-compound and mechanism labels are not passed into ranking.",
            "Chemical descriptors and alerts are reported; no hard medicinal-chemistry cutoff is applied here.",
            "Compatibility and predicted-response scores are not experimental validation.",
        ],
    }
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against an output appearing during inference.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--ranker", required=True, type=Path, help="Trained ranker-bundle checkpoint"
    )
    parser.add_argument(
        "--generator", required=True, type=Path, help="Trained graph-generator checkpoint"
    )
    parser.add_argument("--target-id", required=True, help="Profile ID in the prepared dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        result = design_candidates(
            args.dataset,
            args.ranker,
            args.generator,
            args.target_id,
            args.output,
            count=args.count,
            top_k=args.top_k,
            seed=args.seed,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Design failed: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_kind": result["source_kind"],
                "generated_attempts": result["generation"]["attempted"],
                "hybrid_candidates": result["candidate_pool"]["hybrid_candidate_count"],
                "returned": len(result["rankings"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
