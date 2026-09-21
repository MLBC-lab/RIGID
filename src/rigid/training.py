"""Deterministic reference training with held-out model selection and audit records."""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .chem.graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    batch_graphs,
    graph_from_smiles,
)
from .chem.properties import maximum_tanimoto
from .config import TrainConfig
from .data import TrainStandardizer, audit_leakage, load_dataset
from .engine import fit_model, optimization_step, protocol_fingerprint
from .features import AuxiliarySpec, ContextEncoder, auxiliary_features
from .models.forward import GraphTransformerPredictor, NeuralProfilePredictor
from .models.losses import listwise_rank_loss
from .models.ranker import RIGIDRanker
from .provenance import atomic_json, dataset_hashes, environment_report, run_manifest, sha256_file
from .serialization import read_json, require_object

LOGGER = logging.getLogger(__name__)


def seed_everything(config: TrainConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")


def _step(
    loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, config: TrainConfig
) -> float:
    return optimization_step(loss, model, optimizer, config)


def _fit(
    model: nn.Module,
    train_indices: list[int],
    validation_indices: list[int],
    loss_function,
    config: TrainConfig,
    epochs: int,
    phase: str,
    *,
    state_directory: str | Path | None = None,
    resume: bool = False,
    protocol: dict | None = None,
) -> list[dict]:
    return fit_model(
        model,
        train_indices,
        validation_indices,
        loss_function,
        config,
        epochs,
        phase,
        state_directory=state_directory,
        resume=resume,
        protocol=protocol,
    )


def _training_protocol(dataset_path: str | Path, config: TrainConfig, kind: str, **options) -> dict:
    package = Path(__file__).resolve().parent
    return {
        "protocol_version": 1,
        "kind": kind,
        "configuration": config.to_dict(),
        "input_sha256": dataset_hashes(dataset_path),
        "options": options,
        "software": environment_report(),
        "implementation_sha256": {
            str(path.relative_to(package)).replace("\\", "/"): sha256_file(path)
            for path in sorted(package.rglob("*.py"))
        },
    }


def _prepare_output(output: str | Path, protocol: dict, resume: bool) -> tuple[Path, dict | None]:
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    session_path = output / "training-session.json"
    fingerprint = protocol_fingerprint(protocol)
    if not resume:
        if any(output.iterdir()):
            raise FileExistsError("Training output must be empty; use a new run directory")
        atomic_json(
            session_path,
            {"schema_version": 1, "fingerprint": fingerprint, "protocol": protocol},
            overwrite=False,
        )
        return output, None
    if not session_path.is_file():
        raise FileNotFoundError("No resumable training session found in this output directory")
    session = require_object(read_json(session_path), context="Training session")
    if session.get("schema_version") != 1 or session.get("fingerprint") != fingerprint:
        raise ValueError(
            "Resume protocol mismatch: configuration, data, options, or software changed"
        )
    if (output / "model.pt").exists():
        payload = load_checkpoint(output / "model.pt")
        if payload.get("training_protocol_fingerprint") != fingerprint:
            raise ValueError("Final model does not match this training session")
        manifest = require_object(read_json(output / "manifest.json"), context="Training manifest")
        read_json(output / "history.json")
        return output, manifest
    return output, None


def _phase_options(output: Path, phase: str, protocol: dict, resume: bool) -> dict:
    state_directory = output / "optimization" / phase
    return {
        "state_directory": state_directory,
        "resume": resume and (state_directory / "latest.json").is_file(),
        "protocol": protocol,
    }


def _finish_run(
    output: Path, payload: dict, manifest: dict, history: list[dict], protocol: dict
) -> None:
    fingerprint = protocol_fingerprint(protocol)
    manifest["training_protocol_fingerprint"] = fingerprint
    manifest["optimization_state"] = {
        "directory": "optimization",
        "resume_boundary": "completed epoch",
        "completed_phases": list(dict.fromkeys(row["phase"] for row in history)),
    }
    payload["training_protocol_fingerprint"] = fingerprint
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "history.json", history)
    save_checkpoint(output / "model.pt", payload)


def train(
    dataset_path: str | Path,
    output: str | Path,
    config: TrainConfig,
    *,
    programs: dict | None = None,
    scaffold_holdout: bool = False,
    resume: bool = False,
) -> dict:
    protocol = _training_protocol(
        dataset_path,
        config,
        "ranker-bundle",
        programs=programs or {},
        scaffold_holdout=scaffold_holdout,
    )
    output, completed = _prepare_output(output, protocol, resume)
    if completed is not None:
        return completed
    seed_everything(config)
    dataset = load_dataset(dataset_path)
    audit = audit_leakage(dataset, scaffold=scaffold_holdout, acyclic_policy=config.acyclic_policy)
    audit.raise_for_leakage()
    training = [i for i, p in enumerate(dataset.profiles) if p.split == "train"]
    validation = [i for i, p in enumerate(dataset.profiles) if p.split == "validation"]
    if not training or not validation:
        raise ValueError("Nonempty train and validation splits are required")
    normalizer = TrainStandardizer.fit(dataset.expression, dataset.split_labels, dataset.genes)
    expression = normalizer.transform(dataset.expression, dataset.genes)
    context_encoder = ContextEncoder.fit([dataset.profiles[i] for i in training])
    contexts = context_encoder.transform(list(dataset.profiles))
    spec = AuxiliarySpec.from_dict(programs or {}, dataset.genes)
    compounds = dataset.compound_lookup
    graphs = {c.compound_id: graph_from_smiles(c.canonical_smiles) for c in dataset.compounds}
    device = config.device
    y = torch.tensor(expression, device=device)
    context_tensor = torch.tensor(contexts, device=device)
    shared = dict(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        profile_dim=len(dataset.genes),
        hidden_dim=config.hidden_dim,
        context_dim=context_encoder.dimension,
        dropout=config.dropout,
    )
    forward = NeuralProfilePredictor(**shared).to(device)
    independent = GraphTransformerPredictor(**shared).to(device)
    history = []
    for phase, model in (("forward", forward), ("independent", independent)):

        def profile_loss(indices, is_training, epoch):
            graph = batch_graphs([graphs[dataset.profiles[i].compound_id] for i in indices])
            prediction = model(graph, context_tensor[indices])
            return nn.functional.mse_loss(prediction, y[indices])

        history.extend(
            _fit(
                model,
                training,
                validation,
                profile_loss,
                config,
                config.forward_epochs,
                phase,
                **_phase_options(output, phase, protocol, resume),
            )
        )
        model.requires_grad_(False)

    ranker = RIGIDRanker(
        ATOM_FEATURE_DIM,
        BOND_FEATURE_DIM,
        profile_dim=len(dataset.genes),
        hidden_dim=config.hidden_dim,
        auxiliary_dim=4,
        depth=config.depth,
        dropout=config.dropout,
    ).to(device)
    train_ids = sorted({dataset.profiles[i].compound_id for i in training})
    train_smiles = [compounds[cid].canonical_smiles for cid in train_ids]
    validation_pool_ids = sorted(
        set(train_ids) | {dataset.profiles[i].compound_id for i in validation}
    )
    all_similarity = maximum_tanimoto(
        [compounds[cid].canonical_smiles for cid in validation_pool_ids], train_smiles
    )
    similarity = dict(zip(validation_pool_ids, all_similarity, strict=True))
    if len(train_ids) < 2 or config.candidate_pool_size < 2:
        raise ValueError("Ranking needs at least two training compounds and candidate slots")

    def candidate_ids(index: int, is_training: bool, epoch: int) -> list[str]:
        positive = dataset.profiles[index].compound_id
        source = train_ids if is_training else validation_pool_ids
        seed_material = (
            f"{config.seed}:{dataset.profiles[index].profile_id}:{epoch if is_training else 0}"
        )
        sample_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
        rng = np.random.default_rng(sample_seed)
        others = [cid for cid in source if cid != positive]
        rng.shuffle(others)
        related = [cid for cid in others if set(compounds[cid].moa) & set(compounds[positive].moa)]
        scaffold = [
            cid for cid in others if compounds[cid].scaffold == compounds[positive].scaffold
        ]
        preferred = list(
            dict.fromkeys(
                [
                    positive,
                    *related[: config.mechanism_candidates],
                    *scaffold[: config.scaffold_candidates],
                    *others,
                ]
            )
        )
        return preferred[: config.candidate_pool_size]

    def ranking_loss(indices, is_training, epoch):
        losses = []
        for index in indices:
            ids = candidate_ids(index, is_training, epoch)
            graph = batch_graphs([graphs[cid] for cid in ids])
            with torch.no_grad():
                predicted = forward(graph, context_tensor[index : index + 1].expand(len(ids), -1))
            auxiliary = auxiliary_features(
                expression[index],
                predicted.cpu().numpy(),
                np.array([similarity[cid] for cid in ids]),
                dataset.genes,
                spec,
            )
            scores = ranker(graph, y[index : index + 1], torch.tensor(auxiliary, device=device))
            positive = compounds[dataset.profiles[index].compound_id]
            relevance = [
                1.0
                if cid == positive.compound_id
                else config.moa_relevance
                if set(compounds[cid].moa) & set(positive.moa)
                else 0.0
                for cid in ids
            ]
            losses.append(
                listwise_rank_loss(scores[None], torch.tensor([relevance], device=device))
            )
        return torch.stack(losses).mean()

    history.extend(
        _fit(
            ranker,
            training,
            validation,
            ranking_loss,
            config,
            config.epochs,
            "ranker",
            **_phase_options(output, "ranker", protocol, resume),
        )
    )
    manifest = run_manifest(dataset_path, config.to_dict(), source_kind=dataset.provenance.kind)
    manifest.update(
        {
            "leakage_audit": audit.to_dict(),
            "gene_manifest": dataset.genes.to_dict(),
            "training_profile_ids": [dataset.profiles[i].profile_id for i in training],
            "validation_profile_ids": [dataset.profiles[i].profile_id for i in validation],
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "programs": spec.to_dict(),
            "context_cells": context_encoder.cells,
            "scientific_status": "trainable models; pretrained weights not bundled",
        }
    )
    payload = {
        "schema_version": 1,
        "kind": "ranker-bundle",
        "config": config.to_dict(),
        "training_source_kind": dataset.provenance.kind,
        "scaffold_holdout": scaffold_holdout,
        "training_compounds": [
            {
                "compound_id": compounds[cid].compound_id,
                "smiles": compounds[cid].canonical_smiles,
                "aliases": compounds[cid].aliases,
            }
            for cid in train_ids
        ],
        "normalizer": normalizer.to_dict(),
        "context_cells": context_encoder.cells,
        "programs": spec.to_dict(),
        "train_smiles": train_smiles,
        "train_ids": train_ids,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "models": {
            name: {"config": model.config, "state_dict": model.state_dict()}
            for name, model in (
                ("ranker", ranker),
                ("forward", forward),
                ("independent", independent),
            )
        },
    }
    _finish_run(output, payload, manifest, history, protocol)
    return manifest
