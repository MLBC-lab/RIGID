"""Library screening with no candidate-response or mechanism labels at inference."""

from __future__ import annotations

import heapq
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .chem.graph import FEATURE_SCHEMA_VERSION
from .chem.standardize import scaffold_key, standardize_smiles
from .consensus import RerankingConfig
from .data import TrainStandardizer, load_dataset
from .data.schema import CompoundRecord, GeneManifest, ProfileRecord
from .evaluation import CandidateAnnotation, TargetAnnotation, evaluate_rankings, rerank_candidates
from .features import AuxiliarySpec, ContextEncoder, auxiliary_features, correlations
from .library import PreparedLibrary, prediction_namespace
from .models.forward import GraphTransformerPredictor, NeuralProfilePredictor
from .models.ranker import RIGIDRanker
from .provenance import atomic_json, dataset_hashes, sha256_file


class RIGIDPipeline:
    ranker: RIGIDRanker
    forward: NeuralProfilePredictor
    independent: GraphTransformerPredictor

    def __init__(self, payload: dict, *, device: str = "cpu") -> None:
        if payload.get("kind") != "ranker-bundle":
            raise ValueError("Expected a RIGID ranker checkpoint bundle")
        if payload.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Checkpoint chemical features differ from this software version")
        self.device = device
        self.checkpoint_sha256: str | None = None
        self.normalizer = TrainStandardizer.from_dict(payload["normalizer"])
        self.genes = self.normalizer.genes
        self.context = ContextEncoder(tuple(payload["context_cells"]))
        self.spec = AuxiliarySpec.from_dict(payload["programs"], self.genes)
        self.train_smiles = payload["train_smiles"]
        self.train_ids = payload["train_ids"]
        self.training_source_kind = payload.get("training_source_kind", "unknown")
        self.scaffold_holdout = payload.get("scaffold_holdout", False)
        self.training_compounds = [
            CompoundRecord(**record)
            for record in payload.get(
                "training_compounds",
                [
                    dict(compound_id=identifier, smiles=smiles)
                    for identifier, smiles in zip(self.train_ids, self.train_smiles, strict=True)
                ],
            )
        ]
        self.config = payload["config"]
        self.acyclic_policy = self.config.get("acyclic_policy", "group")
        if self.acyclic_policy not in {"group", "compound", "exclude"}:
            raise ValueError("Checkpoint acyclic policy is invalid")
        for name, constructor in (
            ("ranker", RIGIDRanker),
            ("forward", NeuralProfilePredictor),
            ("independent", GraphTransformerPredictor),
        ):
            record = payload["models"][name]
            model = constructor(**record["config"]).to(device)
            model.load_state_dict(record["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            setattr(self, name, model)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> RIGIDPipeline:
        pipeline = cls(load_checkpoint(path, device=device), device=device)
        pipeline.checkpoint_sha256 = sha256_file(path)
        return pipeline

    def prepare_library(self, candidates: list[CompoundRecord], **cache_options) -> PreparedLibrary:
        """Prepare one reusable library for repeated target screening."""
        return PreparedLibrary(candidates, **cache_options)

    @torch.inference_mode()
    def rank(
        self,
        target: np.ndarray,
        genes: GeneManifest,
        context: ProfileRecord,
        candidates: list[CompoundRecord] | PreparedLibrary,
        *,
        k: int = 50,
        batch_size: int = 128,
        stress_weight: float | None = None,
        novelty_weight: float | None = None,
        reranking: RerankingConfig | None = None,
    ) -> list[dict]:
        if (
            not candidates
            or isinstance(k, bool)
            or not isinstance(k, int)
            or k < 1
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("Candidates, k, and batch size must be nonempty/positive")
        library = (
            candidates
            if isinstance(candidates, PreparedLibrary)
            else self.prepare_library(candidates)
        )
        if library.feature_schema != FEATURE_SCHEMA_VERSION:
            raise ValueError("Prepared library feature schema differs from this pipeline")
        if any(
            module.training
            for model in (self.ranker, self.forward, self.independent)
            for module in model.modules()
        ):
            raise RuntimeError("Screening models must remain in evaluation mode")
        sw = self.config["stress_weight"] if stress_weight is None else stress_weight
        nw = self.config["novelty_weight"] if novelty_weight is None else novelty_weight
        if reranking is not None and not isinstance(reranking, RerankingConfig):
            raise ValueError("reranking must be a RerankingConfig")
        if reranking is not None and (stress_weight is not None or novelty_weight is not None):
            raise ValueError("Specify weights in reranking or legacy weight arguments, not both")
        policy = (reranking or RerankingConfig()).resolve(sw, nw)
        target = self.normalizer.transform(target, genes)
        if target.ndim != 1:
            raise ValueError("A ranking query is one expression vector")
        train_similarity = library.training_similarity(self.train_smiles)
        target_tensor = torch.tensor(target[None], device=self.device)
        target_embedding = self.ranker.target_encoder(target_tensor)
        context_array = self.context.transform([context])
        context_tensor = torch.tensor(context_array, device=self.device)
        embedding_key = prediction_namespace(
            self.ranker.molecule_encoder,
            stage="ranker-molecule",
            genes_sha256=self.genes.sha256,
            device=self.device,
        )
        response_keys = {
            name: prediction_namespace(
                model,
                stage=name,
                genes_sha256=self.genes.sha256,
                device=self.device,
                context=context_array,
            )
            for name, model in (("forward", self.forward), ("independent", self.independent))
        }

        def encode(indices):
            return self.ranker.molecule_encoder(library.graph_batch(indices)).cpu().numpy()

        def predict(model, indices):
            return (
                model(library.graph_batch(indices), context_tensor.expand(len(indices), -1))
                .cpu()
                .numpy()
            )

        retained: list[tuple[float, int, dict]] = []
        for indices in library.batches(batch_size):
            local = [library.records[index] for index in indices]
            embeddings = library.cached_rows(embedding_key, indices, encode)
            response = library.cached_rows(
                response_keys["forward"], indices, lambda missing: predict(self.forward, missing)
            )
            control = library.cached_rows(
                response_keys["independent"],
                indices,
                lambda missing: predict(self.independent, missing),
            )
            auxiliary = auxiliary_features(
                target,
                response,
                train_similarity[list(indices)],
                self.genes,
                self.spec,
            )
            compatibility = self.ranker.score_embeddings(
                torch.tensor(embeddings, device=self.device),
                target_embedding,
                torch.tensor(auxiliary, device=self.device),
            )
            primary = correlations(target, response)
            independent = correlations(target, control)
            raw_compatibility = compatibility.cpu().numpy()
            components = policy.components(raw_compatibility, primary, independent)
            ordered = rerank_candidates(
                [record.compound_id for record in local],
                components["combined"],
                auxiliary[:, 2],
                stress_weight=policy.stress_weight,
                novelty_scores=1 - train_similarity[list(indices)],
                novelty_weight=policy.novelty_weight,
            )
            lookup = {record.compound_id: position for position, record in enumerate(local)}
            for item in ordered:
                position = lookup[item.candidate_id]
                index = indices[position]
                row = {
                    **asdict(item),
                    "compatibility": float(raw_compatibility[position]),
                    **{
                        name: float(values[position])
                        for name, values in components.items()
                        if name != "combined"
                    },
                    "smiles": library.records[index].canonical_smiles,
                    "predicted_profile_correlation": float(primary[position]),
                    "independent_profile_correlation": float(independent[position]),
                    "validator_disagreement": abs(
                        float(primary[position]) - float(independent[position])
                    ),
                    "max_training_tanimoto": float(train_similarity[index]),
                    "predicted_stress": float(auxiliary[position, 2])
                    if self.spec.stress_programs
                    else None,
                }
                entry = (item.score, -index, row)
                if len(retained) < k:
                    heapq.heappush(retained, entry)
                elif entry[:2] > retained[0][:2]:
                    heapq.heapreplace(retained, entry)
        result = []
        for position, (_, negative_index, row) in enumerate(sorted(retained, reverse=True), 1):
            result.append(
                {
                    "rank": position,
                    **row,
                    "chemical_properties": library.properties(-negative_index).to_dict(),
                }
            )
        return result


def screen_dataset(
    checkpoint: str | Path,
    dataset_path: str | Path,
    output: str | Path,
    *,
    split: str = "test",
    k: int = 50,
    device: str = "cpu",
    reranking: RerankingConfig | None = None,
) -> dict:
    """Screen targets and separately attach held-out source/MOA evaluation labels."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Screening output exists; choose a new filename")
    dataset = load_dataset(dataset_path)
    pipeline = RIGIDPipeline.load(checkpoint, device=device)
    if reranking is not None and not isinstance(reranking, RerankingConfig):
        raise ValueError("reranking must be a RerankingConfig")
    policy = (reranking or RerankingConfig()).resolve(
        pipeline.config["stress_weight"], pipeline.config["novelty_weight"]
    )
    pipeline.genes.require_match(dataset.genes)
    selected = [i for i, p in enumerate(dataset.profiles) if p.split == split]
    if not selected:
        raise ValueError(f"No targets in split {split}")
    if split not in {"test", "external", "validation"}:
        raise ValueError("Screening evaluation must use validation, test, or external targets")
    compounds = dataset.compound_lookup
    training_aliases = {
        alias.strip().casefold()
        for compound in pipeline.training_compounds
        for alias in (compound.compound_id, *compound.aliases)
    }
    training_connectivity = {
        standardize_smiles(compound.canonical_smiles).connectivity_key
        for compound in pipeline.training_compounds
    }
    training_scaffold_keys = {
        scaffold_key(smiles, pipeline.acyclic_policy) for smiles in pipeline.train_smiles
    }
    training_scaffold_keys.discard(None)
    for i in selected:
        source = compounds[dataset.profiles[i].compound_id]
        aliases = {alias.strip().casefold() for alias in (source.compound_id, *source.aliases)}
        connectivity = standardize_smiles(source.canonical_smiles).connectivity_key
        if aliases & training_aliases or connectivity in training_connectivity:
            raise ValueError(
                "Held-out target source identities or structures overlap checkpoint training compounds"
            )
        if (
            pipeline.scaffold_holdout
            and scaffold_key(source.canonical_smiles, pipeline.acyclic_policy)
            in training_scaffold_keys
        ):
            raise ValueError("Held-out target scaffolds overlap checkpoint training scaffolds")
    rows, targets = {}, {}
    library = pipeline.prepare_library(list(dataset.compounds))
    for i in selected:
        profile = dataset.profiles[i]
        rows[profile.profile_id] = pipeline.rank(
            dataset.expression[i], dataset.genes, profile, library, k=k, reranking=policy
        )
        source = compounds[profile.compound_id]
        targets[profile.profile_id] = TargetAnnotation(source.compound_id, frozenset(source.moa))
    candidates = {
        c.compound_id: CandidateAnnotation(moa=frozenset(c.moa), scaffold=c.scaffold)
        for c in dataset.compounds
    }
    train_scaffolds = [standardize_smiles(smiles).scaffold for smiles in pipeline.train_smiles]
    metrics = evaluate_rankings(
        {tid: [r["candidate_id"] for r in ranked] for tid, ranked in rows.items()},
        targets,
        candidates,
        k=k,
        training_scaffolds=train_scaffolds,
    ).to_dict()
    metrics["predicted_forward_median_best_correlation"] = float(
        np.median(
            [max(r["predicted_profile_correlation"] for r in ranked) for ranked in rows.values()]
        )
    )
    metrics["independent_forward_median_best_correlation"] = float(
        np.median(
            [max(r["independent_profile_correlation"] for r in ranked) for ranked in rows.values()]
        )
    )
    result = {
        "schema_version": 1,
        "source_kind": dataset.provenance.kind,
        "training_source_kind": pipeline.training_source_kind,
        "reproduces_manuscript": False,
        "reranking": policy.to_dict(),
        "independent_predictor_role": "scoring_component"
        if policy.mode == "consensus"
        else "held-out-from-ranking",
        "split": split,
        "requested_k": k,
        "candidate_count": len(candidates),
        "library_identity_sha256": library.sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_sha256": dataset_hashes(dataset_path),
        "gene_manifest_sha256": dataset.genes.sha256,
        "rankings": rows,
        "metrics": metrics,
        "notes": [
            "Compatibility is an uncalibrated ranking score, not a probability.",
            "Forward correlations use predicted standardized responses, not independent experiments.",
            "The independent architecture shares training data; it is not external validation.",
            "Consensus mode uses both predictors for selection; neither is then an unused validation control."
            if policy.mode == "consensus"
            else "The independent predictor is not used to select candidates.",
            "Missing family/stress annotations remain missing in retrieval metrics.",
            "Chemical alerts are screening flags, not proof of safety.",
        ],
    }
    atomic_json(output, result, overwrite=False)
    return result
