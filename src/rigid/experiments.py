"""Optional proposal training and explicit library-constrained baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .chem.properties import fingerprint_array
from .config import TrainConfig
from .data import GeneManifest, TrainStandardizer, audit_leakage, load_dataset
from .evaluation import CandidateAnnotation, TargetAnnotation, evaluate_rankings
from .generation import AutoregressiveGraphGenerator, GeneratorConfig, teacher_actions
from .models import ConditionalFingerprintVAE
from .provenance import atomic_json, run_manifest, sha256_file
from .training import (
    _finish_run,
    _fit,
    _phase_options,
    _prepare_output,
    _training_protocol,
    seed_everything,
)


def _training_data(dataset_path, config, scaffold_holdout=False):
    seed_everything(config)
    dataset = load_dataset(dataset_path)
    audit = audit_leakage(dataset, scaffold=scaffold_holdout, acyclic_policy=config.acyclic_policy)
    audit.raise_for_leakage()
    training = [i for i, profile in enumerate(dataset.profiles) if profile.split == "train"]
    validation = [i for i, profile in enumerate(dataset.profiles) if profile.split == "validation"]
    if not training or not validation:
        raise ValueError("Nonempty train and validation splits are required")
    return dataset, audit, training, validation


def _run_record(dataset_path, dataset, audit, config, training, validation):
    manifest = run_manifest(dataset_path, config.to_dict(), source_kind=dataset.provenance.kind)
    manifest.update(
        {
            "leakage_audit": audit.to_dict(),
            "gene_manifest": dataset.genes.to_dict(),
            "training_profile_ids": [dataset.profiles[i].profile_id for i in training],
            "validation_profile_ids": [dataset.profiles[i].profile_id for i in validation],
            "scientific_status": "trainable models; pretrained weights not bundled",
            "selection_partition": "validation",
            "test_used_for_training_or_selection": False,
        }
    )
    return manifest


def train_generator(
    dataset_path: str | Path,
    output: str | Path,
    config: TrainConfig,
    *,
    max_atoms: int = 40,
    conditional: bool = True,
    scaffold_holdout: bool = False,
    resume: bool = False,
) -> dict:
    """Fit graph proposals on training profiles, selecting by held-out trace loss.

    Unsupported chemistry is excluded with compound/profile-level reasons.
    Test molecules are not inspected by the graph-trace encoder. Conditional
    training uses each eligible profile; unconditional training uses each
    eligible compound once to avoid weighting by replicate counts.
    """
    if type(conditional) is not bool:
        raise ValueError("conditional must be a boolean")
    protocol = _training_protocol(
        dataset_path,
        config,
        "graph-generator",
        max_atoms=max_atoms,
        conditional=conditional,
        scaffold_holdout=scaffold_holdout,
    )
    output, completed = _prepare_output(output, protocol, resume)
    if completed is not None:
        return completed
    dataset, audit, training, validation = _training_data(dataset_path, config, scaffold_holdout)
    settings = GeneratorConfig(
        hidden_dim=config.hidden_dim,
        condition_dim=len(dataset.genes) if conditional else 0,
        message_steps=config.depth,
        max_nodes=max_atoms,
        max_steps=max(128, 4 * max_atoms),
        dropout=config.dropout,
    )
    compounds = dataset.compound_lookup
    exclusion_report = {}
    selected = {}
    for split, indices in (("train", training), ("validation", validation)):
        reasons = {}
        for compound_id in sorted({dataset.profiles[i].compound_id for i in indices}):
            try:
                teacher_actions(compounds[compound_id].canonical_smiles, settings)
            except (ValueError, RuntimeError) as exc:
                reasons[compound_id] = str(exc)
        accepted = [i for i in indices if dataset.profiles[i].compound_id not in reasons]
        if not conditional:
            seen = set()
            unique = []
            for index in accepted:
                compound_id = dataset.profiles[index].compound_id
                if compound_id not in seen:
                    seen.add(compound_id)
                    unique.append(index)
            accepted = unique
        selected[split] = accepted
        exclusion_report[split] = {
            "input_profiles": len(indices),
            "input_compounds": len({dataset.profiles[i].compound_id for i in indices}),
            "training_examples_retained": len(accepted),
            "excluded_compounds": reasons,
            "excluded_profile_ids": [
                dataset.profiles[i].profile_id
                for i in indices
                if dataset.profiles[i].compound_id in reasons
            ],
            "reason_counts": dict(Counter(reasons.values())),
        }
    atomic_json(output / "exclusions.json", exclusion_report)
    training, validation = selected["train"], selected["validation"]
    if not training or not validation:
        raise ValueError(
            "No supported generator examples remain in train or validation; see exclusions.json"
        )
    # Only retained training examples determine conditional feature transforms.
    training_set = set(training)
    fitted_labels = [
        "train" if i in training_set else "excluded" for i in range(len(dataset.profiles))
    ]
    normalizer = TrainStandardizer.fit(dataset.expression, fitted_labels, dataset.genes)
    y = torch.tensor(normalizer.transform(dataset.expression, dataset.genes), device=config.device)
    model = AutoregressiveGraphGenerator(settings).to(config.device)

    def loss_function(indices, is_training, epoch):
        losses = []
        for index in indices:
            smiles = compounds[dataset.profiles[index].compound_id].canonical_smiles
            condition = y[index : index + 1] if conditional else None
            losses.append(model.teacher_forcing_loss([smiles], condition))
        return torch.stack(losses).mean()

    history = _fit(
        model,
        training,
        validation,
        loss_function,
        config,
        config.epochs,
        "generator",
        **_phase_options(output, "generator", protocol, resume),
    )
    manifest = _run_record(dataset_path, dataset, audit, config, training, validation)
    manifest.update(
        {
            "kind": "graph-generator",
            "model_config": model.config,
            "conditional": conditional,
            "exclusions": exclusion_report,
            "objective": "mean per-example mean-action cross-entropy",
            "example_weighting": "one per profile" if conditional else "one per compound",
        }
    )
    train_ids = sorted({dataset.profiles[i].compound_id for i in training})
    payload = {
        "schema_version": 1,
        "kind": "graph-generator",
        "config": config.to_dict(),
        "model_config": model.config,
        "state_dict": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "source_kind": dataset.provenance.kind,
        "train_smiles": [compounds[cid].canonical_smiles for cid in train_ids],
        "train_ids": train_ids,
    }
    _finish_run(output, payload, manifest, history, protocol)
    return manifest


def _load_model(checkpoint, kind, constructor):
    payload = load_checkpoint(checkpoint, device="cpu")
    if payload.get("kind") != kind:
        raise ValueError(f"Expected checkpoint kind {kind!r}")
    model = constructor(payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


@torch.inference_mode()
def generate(
    checkpoint: str | Path,
    n: int = 32,
    seed: int = 0,
    target: np.ndarray | None = None,
    genes: GeneManifest | tuple[str, ...] | None = None,
) -> dict:
    """Load a checksummed graph-generator checkpoint and report every proposal.

    Conditional checkpoints require one raw target vector and its ordered gene
    manifest. Generated proposals are not ranked or asserted to reverse disease.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    model, payload = _load_model(
        checkpoint,
        "graph-generator",
        lambda values: AutoregressiveGraphGenerator(GeneratorConfig(**values)),
    )
    normalizer = TrainStandardizer.from_dict(payload["normalizer"])
    condition = None
    if model.settings.condition_dim:
        if target is None or genes is None:
            raise ValueError("Conditional generation requires target and ordered genes")
        manifest = (
            genes
            if isinstance(genes, GeneManifest)
            else GeneManifest(tuple(genes), normalizer.genes.namespace)
        )
        condition = normalizer.transform(np.asarray(target), manifest)
        if condition.ndim != 1:
            raise ValueError("Generation expects one target expression vector")
        condition = torch.tensor(condition[None])
    elif target is not None or genes is not None:
        raise ValueError("This checkpoint is unconditional; do not supply a target")
    proposals = model.sample(n, conditions=condition, seed=seed)
    valid = [item for item in proposals if item.valid and item.smiles is not None]
    unique = sorted({item.smiles for item in valid})
    training = set(payload["train_smiles"])
    return {
        "schema_version": 1,
        "kind": "graph-proposals",
        "source_kind": payload["source_kind"],
        "reproduces_manuscript": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": seed,
        "attempted": len(proposals),
        "valid": len(valid),
        "unique": len(unique),
        "validity": len(valid) / len(proposals),
        "uniqueness": len(unique) / len(valid) if valid else 0.0,
        "new_unique_structures": sum(smiles not in training for smiles in unique),
        "failure_reasons": dict(
            Counter(item.rejection_reason for item in proposals if not item.valid)
        ),
        "proposals": [asdict(item) for item in proposals],
        "notes": [
            "Proposals require downstream compatibility scoring and chemical filters.",
            "New structures are assessed by exact canonical SMILES, not scaffold novelty.",
        ],
    }


def train_vae(
    dataset_path: str | Path,
    output: str | Path,
    config: TrainConfig,
    *,
    fingerprint_bits: int = 2048,
    latent_dim: int | None = None,
    beta: float = 0.01,
    scaffold_holdout: bool = False,
    resume: bool = False,
) -> dict:
    """Fit the explicitly defined conditional fingerprint-VAE library baseline."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if type(fingerprint_bits) is not int or fingerprint_bits < 1:
        raise ValueError("fingerprint_bits must be a positive integer")
    protocol = _training_protocol(
        dataset_path,
        config,
        "fingerprint-vae",
        fingerprint_bits=fingerprint_bits,
        latent_dim=latent_dim,
        beta=beta,
        scaffold_holdout=scaffold_holdout,
    )
    output, completed = _prepare_output(output, protocol, resume)
    if completed is not None:
        return completed
    dataset, audit, training, validation = _training_data(dataset_path, config, scaffold_holdout)
    normalizer = TrainStandardizer.fit(dataset.expression, dataset.split_labels, dataset.genes)
    y = torch.tensor(normalizer.transform(dataset.expression, dataset.genes), device=config.device)
    compounds = dataset.compound_lookup
    selected_ids = {dataset.profiles[i].compound_id for i in training + validation}
    fingerprints = {
        cid: fingerprint_array(compounds[cid].canonical_smiles, n_bits=fingerprint_bits)
        for cid in sorted(selected_ids)
    }
    model = ConditionalFingerprintVAE(
        profile_dim=len(dataset.genes),
        fingerprint_dim=fingerprint_bits,
        hidden_dim=config.hidden_dim,
        latent_dim=max(4, config.hidden_dim // 2) if latent_dim is None else latent_dim,
        dropout=config.dropout,
    ).to(config.device)

    def compute_loss(indices):
        x = torch.tensor(
            np.stack([fingerprints[dataset.profiles[i].compound_id] for i in indices]),
            device=config.device,
        )
        output = model(x, y[indices])
        return model.loss(output, x, beta=beta)[0]

    def loss_function(indices, is_training, epoch):
        if is_training:
            return compute_loss(indices)
        devices = (
            [torch.device(config.device).index or 0] if config.device.startswith("cuda") else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed + 100000 + indices[0])
            return compute_loss(indices)

    history = _fit(
        model,
        training,
        validation,
        loss_function,
        config,
        config.epochs,
        "fingerprint-vae",
        **_phase_options(output, "fingerprint-vae", protocol, resume),
    )
    manifest = _run_record(dataset_path, dataset, audit, config, training, validation)
    fingerprint_schema = {"kind": "Morgan", "radius": 2, "bits": fingerprint_bits}
    manifest.update(
        {
            "kind": "fingerprint-vae",
            "model_config": model.config,
            "fingerprint_schema": fingerprint_schema,
            "beta": beta,
            "objective": "summed Bernoulli reconstruction plus beta KL; mean across profiles",
            "validation_sampling": "fixed seed per validation batch across epochs",
        }
    )
    train_ids = sorted({dataset.profiles[i].compound_id for i in training})
    _finish_run(
        output,
        {
            "schema_version": 1,
            "kind": "fingerprint-vae",
            "config": config.to_dict(),
            "model_config": model.config,
            "state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "fingerprint_schema": fingerprint_schema,
            "source_kind": dataset.provenance.kind,
            "beta": beta,
            "train_ids": train_ids,
            "train_smiles": [compounds[cid].canonical_smiles for cid in train_ids],
        },
        manifest,
        history,
        protocol,
    )
    return manifest


def _evaluation_targets(dataset, split):
    if split not in {"validation", "test", "external"}:
        raise ValueError("Evaluation requires validation, test, or external targets")
    indices = [i for i, profile in enumerate(dataset.profiles) if profile.split == split]
    if not indices:
        raise ValueError(f"No targets in split {split}")
    return indices


def _ranking_evaluation(dataset, rankings, selected, candidates, train_ids, k):
    lookup = dataset.compound_lookup
    targets = {
        dataset.profiles[i].profile_id: TargetAnnotation(
            dataset.profiles[i].compound_id, frozenset(lookup[dataset.profiles[i].compound_id].moa)
        )
        for i in selected
    }
    annotations = {
        item.compound_id: CandidateAnnotation(moa=frozenset(item.moa), scaffold=item.scaffold)
        for item in candidates
    }
    training_scaffolds = [lookup[cid].scaffold for cid in train_ids if cid in lookup]
    return evaluate_rankings(
        {tid: [row["candidate_id"] for row in rows] for tid, rows in rankings.items()},
        targets,
        annotations,
        k=k,
        training_scaffolds=training_scaffolds,
    ).to_dict()


@torch.inference_mode()
def screen_vae(
    checkpoint: str | Path,
    dataset_path: str | Path,
    split: str = "test",
    samples: int = 32,
) -> dict:
    """Rank the supplied molecular catalog without reading candidate response profiles."""
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    model, payload = _load_model(
        checkpoint, "fingerprint-vae", lambda values: ConditionalFingerprintVAE(**values)
    )
    dataset = load_dataset(dataset_path)
    normalizer = TrainStandardizer.from_dict(payload["normalizer"])
    normalizer.genes.require_match(dataset.genes)
    selected = _evaluation_targets(dataset, split)
    compounds = dataset.compound_lookup
    train_smiles = set(payload["train_smiles"])
    if any(
        compounds[dataset.profiles[i].compound_id].canonical_smiles in train_smiles
        for i in selected
    ):
        raise ValueError("Held-out target source structures overlap checkpoint training structures")
    candidates = sorted(dataset.compounds, key=lambda compound: compound.compound_id)
    if len({item.canonical_smiles for item in candidates}) != len(candidates):
        raise ValueError("Candidate structures must be unique after standardization")
    fps = torch.tensor(
        np.stack(
            [
                fingerprint_array(
                    item.canonical_smiles,
                    radius=payload["fingerprint_schema"]["radius"],
                    n_bits=payload["fingerprint_schema"]["bits"],
                )
                for item in candidates
            ]
        )
    )
    y = normalizer.transform(dataset.expression[selected], dataset.genes)
    rng = torch.Generator().manual_seed(payload["config"]["seed"])
    k = payload["config"]["top_k"]
    rankings = {}
    for index, target in zip(selected, y, strict=True):
        predictions = model.sample_fingerprints(torch.tensor(target[None]), samples, rng)[0]
        chunks = []
        for offset in range(0, len(fps), 1024):
            local = fps[offset : offset + 1024]
            intersection = predictions @ local.T
            union = predictions.sum(-1, keepdim=True) + local.sum(-1)[None] - intersection
            chunks.append((intersection / union.clamp_min(1e-8)).mean(0))
        scores = torch.cat(chunks).numpy()
        ordered = np.argsort(-scores, kind="stable")[:k]
        rankings[dataset.profiles[index].profile_id] = [
            {
                "rank": rank,
                "candidate_id": candidates[item].compound_id,
                "smiles": candidates[item].canonical_smiles,
                "score": float(scores[item]),
            }
            for rank, item in enumerate(ordered, 1)
        ]
    return {
        "schema_version": 1,
        "kind": "fingerprint-vae-ranking",
        "source_kind": dataset.provenance.kind,
        "checkpoint_source_kind": payload["source_kind"],
        "reproduces_manuscript": False,
        "split": split,
        "candidate_count": len(candidates),
        "samples": samples,
        "requested_k": k,
        "checkpoint_sha256": sha256_file(checkpoint),
        "rankings": rankings,
        "metrics": _ranking_evaluation(
            dataset, rankings, selected, candidates, payload["train_ids"], k
        ),
        "notes": [
            "Scores are Monte Carlo mean soft-Tanimoto similarities.",
            "This is a library-constrained fingerprint VAE, not a molecule decoder.",
        ],
    }


def nearest_neighbor(
    dataset_path: str | Path,
    split: str = "test",
    k: int = 50,
    *,
    scaffold_holdout: bool = False,
    acyclic_policy: str = "group",
    reference_dataset: str | Path | None = None,
    reference_config=None,
    candidate_ids=None,
) -> dict:
    """Rank training compounds by their best observed training-profile correlation.

    No held-out candidate profiles enter the retrieval catalog. Consequently a
    strictly compound-held-out source cannot be recovered exactly by this
    training-only baseline; the candidate-set difference is reported explicitly.
    Supply reference_dataset to use an independently measured response catalog;
    this explicitly changes the information-access regime and is reported.
    """
    if reference_dataset is not None:
        from .baselines import reference_nearest_neighbor

        return reference_nearest_neighbor(
            dataset_path,
            reference_dataset,
            split,
            k,
            config=reference_config,
            candidate_ids=candidate_ids,
            scaffold_holdout=scaffold_holdout,
            acyclic_policy=acyclic_policy,
        )
    if reference_config is not None or candidate_ids is not None:
        raise ValueError("reference_config and candidate_ids require reference_dataset")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(acyclic_policy, str) or acyclic_policy not in {
        "group",
        "compound",
        "exclude",
    }:
        raise ValueError("acyclic_policy must be group, compound, or exclude")
    dataset = load_dataset(dataset_path)
    audit = audit_leakage(dataset, scaffold=scaffold_holdout, acyclic_policy=acyclic_policy)
    audit.raise_for_leakage()
    selected = _evaluation_targets(dataset, split)
    training = [i for i, profile in enumerate(dataset.profiles) if profile.split == "train"]
    if not training:
        raise ValueError("Nearest-neighbor retrieval requires observed training profiles")
    normalizer = TrainStandardizer.fit(dataset.expression, dataset.split_labels, dataset.genes)
    reference = normalizer.transform(dataset.expression[training], dataset.genes).astype(np.float64)
    reference -= reference.mean(-1, keepdims=True)
    norms = np.linalg.norm(reference, axis=-1)
    valid = norms > 0
    excluded_profiles = [
        dataset.profiles[index].profile_id
        for index, keep in zip(training, valid, strict=True)
        if not keep
    ]
    training = [index for index, keep in zip(training, valid, strict=True) if keep]
    if not training:
        raise ValueError("All standardized reference profiles have undefined Pearson correlation")
    reference = reference[valid] / norms[valid, None]
    compounds = dataset.compound_lookup
    train_ids = sorted({dataset.profiles[i].compound_id for i in training})
    positions = {
        cid: [j for j, i in enumerate(training) if dataset.profiles[i].compound_id == cid]
        for cid in train_ids
    }
    rankings = {}
    for index in selected:
        target = normalizer.transform(dataset.expression[index], dataset.genes).astype(np.float64)
        target -= target.mean()
        norm = np.linalg.norm(target)
        if norm == 0:
            raise ValueError(
                f"Target {dataset.profiles[index].profile_id} has undefined Pearson correlation"
            )
        correlations = reference @ (target / norm)
        scores = np.asarray([max(correlations[positions[cid]]) for cid in train_ids])
        ordered = np.argsort(-scores, kind="stable")[:k]
        rankings[dataset.profiles[index].profile_id] = [
            {
                "rank": rank,
                "candidate_id": train_ids[item],
                "smiles": compounds[train_ids[item]].canonical_smiles,
                "score": float(np.clip(scores[item], -1, 1)),
            }
            for rank, item in enumerate(ordered, 1)
        ]
    candidates = [compounds[cid] for cid in train_ids]
    return {
        "schema_version": 1,
        "kind": "observed-training-nearest-neighbor",
        "leakage_audit": audit.to_dict(),
        "source_kind": dataset.provenance.kind,
        "reproduces_manuscript": False,
        "split": split,
        "candidate_count": len(candidates),
        "requested_k": k,
        "candidate_pool": "training compounds only",
        "ranking_profile_ids": [dataset.profiles[i].profile_id for i in training],
        "excluded_constant_reference_profile_ids": excluded_profiles,
        "rankings": rankings,
        "metrics": _ranking_evaluation(dataset, rankings, selected, candidates, train_ids, k),
        "notes": [
            "Per-compound score is the maximum Pearson correlation over its observed training profiles.",
            "Gene standardization is fitted on training profiles only.",
            "Constant standardized reference profiles are excluded because Pearson correlation is undefined.",
            "No context matching is performed; dose, time, and cell differences can confound retrieval.",
            "Exact held-out recovery is impossible in a training-compounds-only catalog.",
            "This catalog is smaller than the full library used by model-based screening.",
        ],
    }
