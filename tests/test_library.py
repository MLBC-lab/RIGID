"""Reusable screening must preserve scores, identities, and bounded memory."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from rigid.checkpoint import save_checkpoint
from rigid.chem.graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, FEATURE_SCHEMA_VERSION
from rigid.chem.properties import maximum_tanimoto
from rigid.data import CompoundRecord, TrainStandardizer, make_synthetic_dataset
from rigid.evaluation import rerank_candidates
from rigid.features import ContextEncoder, auxiliary_features
from rigid.library import PreparedLibrary, model_fingerprint, prediction_namespace
from rigid.models.forward import GraphTransformerPredictor, NeuralProfilePredictor
from rigid.models.ranker import RIGIDRanker
from rigid.pipeline import RIGIDPipeline


@pytest.fixture
def compounds():
    return [
        CompoundRecord("ethanol", "CCO", aliases=("alcohol-1",)),
        CompoundRecord("amine", "CCN"),
        CompoundRecord("benzene", "c1ccccc1"),
        CompoundRecord("pyridine", "c1ccncc1"),
        CompoundRecord("cyclohexane", "C1CCCCC1"),
    ]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("library-models")
    dataset = make_synthetic_dataset(root / "data", n_genes=8)
    train_profiles = [profile for profile in dataset.profiles if profile.split == "train"]
    context = ContextEncoder.fit(train_profiles)
    normalizer = TrainStandardizer.fit(dataset.expression, dataset.split_labels, dataset.genes)
    shared = dict(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        profile_dim=8,
        hidden_dim=16,
        dropout=0,
    )
    models = {
        "ranker": RIGIDRanker(**shared, auxiliary_dim=4),
        "forward": NeuralProfilePredictor(**shared, context_dim=context.dimension),
        "independent": GraphTransformerPredictor(**shared, context_dim=context.dimension),
    }
    ids = {profile.compound_id for profile in train_profiles}
    training = [compound for compound in dataset.compounds if compound.compound_id in ids]
    payload = {
        "schema_version": 1,
        "kind": "ranker-bundle",
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "normalizer": normalizer.to_dict(),
        "context_cells": context.cells,
        "programs": {
            "programs": {"test-only-stress": list(dataset.genes.gene_ids[:3])},
            "stress_programs": ["test-only-stress"],
            "source": "Synthetic unit test fixture",
        },
        "train_smiles": [compound.canonical_smiles for compound in training],
        "train_ids": [compound.compound_id for compound in training],
        "training_source_kind": "synthetic",
        "config": {"stress_weight": 0.2, "novelty_weight": 0.05},
        "models": {
            name: {"config": model.config, "state_dict": model.state_dict()}
            for name, model in models.items()
        },
    }
    return root, dataset, payload


def _pipeline(bundle):
    _, dataset, payload = bundle
    pipeline = RIGIDPipeline(copy.deepcopy(payload))
    index = next(index for index, profile in enumerate(dataset.profiles) if profile.split == "test")
    return pipeline, dataset, index


def _rank(pipeline, dataset, index, library, **kwargs):
    return pipeline.rank(
        dataset.expression[index], dataset.genes, dataset.profiles[index], library, **kwargs
    )


def test_identity_validation_and_explicit_deduplication(compounds):
    duplicate = CompoundRecord("also-ethanol", "OCC", moa=("annotation-only",), aliases=("EtOH",))
    with pytest.raises(ValueError, match="structures"):
        PreparedLibrary([*compounds, duplicate])
    library = PreparedLibrary([*compounds, duplicate], deduplicate=True)
    assert len(library) == len(compounds)
    assert library.records[0].compound_id == "ethanol"
    assert set(library.records[0].aliases) == {"also-ethanol", "EtOH", "alcohol-1"}
    assert library.summary()["merged_record_count"] == 1
    with pytest.raises(ValueError, match="identifiers"):
        PreparedLibrary([compounds[0], compounds[0]], deduplicate=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        PreparedLibrary([compounds[0], replace(compounds[1], aliases=("ALCOHOL-1",))])


@pytest.mark.parametrize(
    "kwargs",
    [{"cache_bytes": -1}, {"cache_bytes": True}, {"property_cache_size": 1.5}, {"deduplicate": 1}],
)
def test_invalid_cache_configuration_rejected(compounds, kwargs):
    with pytest.raises(ValueError):
        PreparedLibrary(compounds, **kwargs)


def test_manifest_roundtrip_preserves_identity_and_source_provenance(compounds, tmp_path):
    library = PreparedLibrary([*compounds, CompoundRecord("second-name", "OCC")], deduplicate=True)
    path = tmp_path / "library.json"
    library.save_manifest(path)
    restored = PreparedLibrary.from_manifest(path)
    assert restored.sha256 == library.sha256
    assert restored.summary()["source_record_count"] == len(compounds) + 1
    assert [record.compound_id for record in restored.records] == [
        record.compound_id for record in library.records
    ]
    with pytest.raises(FileExistsError):
        library.save_manifest(path)
    relabeled = PreparedLibrary(
        [replace(compound, moa=("never-scored",)) for compound in compounds]
    )
    assert PreparedLibrary(compounds).sha256 == relabeled.sha256
    assert PreparedLibrary(reversed(compounds)).sha256 != relabeled.sha256
    value = json.loads(path.read_text())
    value["candidates"][0]["compound_id"] = "tampered"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        PreparedLibrary.from_manifest(path)


def test_csv_reader_reports_bad_rows_and_records_checksum(tmp_path):
    path = tmp_path / "library.csv"
    path.write_text(
        "compound_id,smiles,aliases,moa\na,CCO,EtOH;alcohol,label\nb,CCN,,\n", encoding="utf-8-sig"
    )
    library = PreparedLibrary.from_csv(path)
    assert library.records[0].aliases == ("EtOH", "alcohol")
    assert len(library.source_sha256) == 64
    output = tmp_path / "manifest.json"
    library.save_manifest(output)
    assert PreparedLibrary.from_manifest(output).source_sha256 == library.source_sha256
    path.write_text("compound_id,smiles\na,CCO,extra\n")
    with pytest.raises(ValueError, match="row 2"):
        PreparedLibrary.from_csv(path)
    path.write_text("compound_id,smiles\na,invalid-molecule\n")
    with pytest.raises(ValueError, match="row 2"):
        PreparedLibrary.from_csv(path)


def test_array_cache_is_bounded_isolated_and_namespaced(compounds):
    library = PreparedLibrary(compounds, cache_bytes=16)
    calls = []

    def compute(indices):
        calls.append(indices)
        return np.array([[index, -index] for index in indices], dtype=np.float32)

    first = library.cached_rows("model-a", (0, 1), compute)
    first[:] = 99
    np.testing.assert_array_equal(
        library.cached_rows("model-a", (0, 1), compute), [[0, 0], [1, -1]]
    )
    assert len(calls) == 1
    library.cached_rows("model-a", (2, 3), compute)
    assert library.cache_info()["arrays"]["payload_bytes"] == 16
    assert library.cache_info()["arrays"]["evictions"] == 2
    library.cached_rows("model-b", (2, 3), compute)
    assert len(calls) == 3
    library.clear_cache()
    assert library.cache_info()["arrays"]["entries"] == 0


def test_oversized_rows_and_zero_cache_still_compute_correctly(compounds):
    library = PreparedLibrary(compounds, cache_bytes=0)
    result = library.cached_rows("test", (0, 2), lambda indices: np.ones((len(indices), 100)))
    assert result.shape == (2, 100)
    assert library.cache_info()["arrays"]["entries"] == 0
    with pytest.raises(ValueError, match="finite"):
        library.cached_rows("bad", (0,), lambda _: np.array([[np.nan]]))
    with pytest.raises(ValueError, match="one numeric row"):
        library.cached_rows("bad", (0,), lambda _: np.zeros((2, 3)))
    with pytest.raises(ValueError, match="duplicate indices"):
        library.cached_rows("bad", (0, 0), lambda _: np.zeros((2, 3)))
    with pytest.raises(IndexError):
        library.graph_batch((-1,))
    with pytest.raises(ValueError):
        list(library.batches(False))


def test_chemistry_caches_and_blocked_similarity_match_reference(compounds):
    library = PreparedLibrary(
        compounds, graph_cache_bytes=1200, fingerprint_cache_bytes=512, property_cache_size=1
    )
    graph = library.graph_batch((0, 1))
    assert graph.num_graphs == 2
    assert library.cache_info()["graphs"]["payload_bytes"] <= 1200
    library.properties(0)
    library.properties(1)
    assert library.cache_info()["properties"]["entries"] == 1
    reference = ["CCO", "CCCl", "CCCC"]
    actual = library.training_similarity(reference, batch_size=2, reference_batch_size=1)
    expected = maximum_tanimoto([record.canonical_smiles for record in compounds], reference)
    np.testing.assert_array_equal(actual, expected)
    info = library.cache_info()
    assert info["fingerprints"]["payload_bytes"] <= 512
    library.training_similarity(reference)
    assert library.cache_info()["arrays"]["hits"] >= len(compounds)
    with pytest.raises(ValueError, match="nonempty"):
        library.training_similarity([])


@torch.inference_mode()
def test_streamed_ranking_matches_uncached_full_reference(bundle):
    pipeline, dataset, index = _pipeline(bundle)
    library = PreparedLibrary(dataset.compounds)
    target = pipeline.normalizer.transform(dataset.expression[index], dataset.genes)
    context = torch.tensor(pipeline.context.transform([dataset.profiles[index]]))
    similarity = maximum_tanimoto(
        [c.canonical_smiles for c in dataset.compounds], pipeline.train_smiles
    )
    score_batches, feature_batches = [], []
    for indices in library.batches(3):
        graph = library.graph_batch(indices)
        response = pipeline.forward(graph, context.expand(len(indices), -1)).numpy()
        features = auxiliary_features(
            target, response, similarity[list(indices)], dataset.genes, pipeline.spec
        )
        score_batches.append(
            pipeline.ranker(graph, torch.tensor(target[None]), torch.tensor(features)).numpy()
        )
        feature_batches.append(features)
    scores, auxiliary = np.concatenate(score_batches), np.concatenate(feature_batches)
    expected = rerank_candidates(
        [record.compound_id for record in library.records],
        scores,
        auxiliary[:, 2],
        stress_weight=0.2,
        novelty_scores=1 - similarity,
        novelty_weight=0.05,
        k=9,
    )
    actual = _rank(pipeline, dataset, index, library, k=9, batch_size=3)
    assert [row["candidate_id"] for row in actual] == [row.candidate_id for row in expected]
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for field in ("score", "compatibility", "stress_penalty", "novelty_bonus"):
            assert actual_row[field] == pytest.approx(asdict(expected_row)[field], abs=1e-6)


def test_streamed_ties_preserve_input_order_across_batches(bundle):
    pipeline, dataset, index = _pipeline(bundle)
    with torch.no_grad():
        for parameter in pipeline.ranker.ranking_head.parameters():
            parameter.zero_()
    library = PreparedLibrary(reversed(dataset.compounds), cache_bytes=0, graph_cache_bytes=0)
    actual = _rank(
        pipeline, dataset, index, library, k=7, batch_size=3, stress_weight=0, novelty_weight=0
    )
    assert [row["candidate_id"] for row in actual] == [
        record.compound_id for record in library.records[:7]
    ]


def test_repeated_queries_reuse_models_and_context_changes_invalidate_predictions(bundle):
    pipeline, dataset, index = _pipeline(bundle)
    library = pipeline.prepare_library(list(dataset.compounds))
    counts = {"ranker": 0, "forward": 0, "independent": 0}

    def counter(name):
        def increment(*_):
            counts[name] += 1

        return increment

    hooks = [
        pipeline.ranker.molecule_encoder.register_forward_hook(counter("ranker")),
        pipeline.forward.register_forward_hook(counter("forward")),
        pipeline.independent.register_forward_hook(counter("independent")),
    ]
    try:
        first = _rank(pipeline, dataset, index, library, k=5, batch_size=9)
        before = counts.copy()
        second = _rank(pipeline, dataset, index, library, k=5, batch_size=5)
        assert [row["candidate_id"] for row in first] == [row["candidate_id"] for row in second]
        np.testing.assert_allclose(
            [row["score"] for row in first], [row["score"] for row in second], atol=1e-6
        )
        assert counts == before
        profile = replace(dataset.profiles[index], dose_um=50)
        pipeline.rank(dataset.expression[index], dataset.genes, profile, library, k=5)
        assert counts["ranker"] == before["ranker"]
        assert counts["forward"] > before["forward"]
        assert counts["independent"] > before["independent"]
    finally:
        for hook in hooks:
            hook.remove()


def test_model_mutation_and_new_checkpoint_cannot_reuse_stale_embeddings(bundle, tmp_path):
    pipeline, dataset, index = _pipeline(bundle)
    library = PreparedLibrary(dataset.compounds)
    _rank(pipeline, dataset, index, library, k=5)
    previous = model_fingerprint(pipeline.ranker.molecule_encoder)
    with torch.no_grad():
        next(pipeline.ranker.molecule_encoder.parameters()).add_(0.25)
    assert model_fingerprint(pipeline.ranker.molecule_encoder) != previous
    cached = _rank(pipeline, dataset, index, library, k=5)
    fresh = _rank(pipeline, dataset, index, list(dataset.compounds), k=5)
    assert cached == fresh
    payload = copy.deepcopy(bundle[2])
    first_path = tmp_path / "first.pt"
    save_checkpoint(first_path, payload)
    payload["models"]["ranker"]["state_dict"] = pipeline.ranker.state_dict()
    second_path = tmp_path / "second.pt"
    save_checkpoint(second_path, payload)
    first = RIGIDPipeline.load(first_path)
    second = RIGIDPipeline.load(second_path)
    assert first.checkpoint_sha256 != second.checkpoint_sha256
    assert _rank(second, dataset, index, library, k=5) == _rank(
        second, dataset, index, list(dataset.compounds), k=5
    )


def test_model_namespace_covers_configuration_context_and_feature_identity(bundle):
    pipeline, _, _ = _pipeline(bundle)
    model = pipeline.forward
    first = prediction_namespace(
        model,
        stage="forward",
        genes_sha256="genes-a",
        device="cpu",
        context=np.zeros((1, 2), np.float32),
    )
    second = prediction_namespace(
        model,
        stage="forward",
        genes_sha256="genes-b",
        device="cpu",
        context=np.zeros((1, 2), np.float32),
    )
    third = prediction_namespace(
        model,
        stage="forward",
        genes_sha256="genes-a",
        device="cpu",
        context=np.ones((1, 2), np.float32),
    )
    assert len({first, second, third}) == 3
    before = model_fingerprint(model)
    model.config["extra-deployment-parameter"] = 2
    assert model_fingerprint(model) != before
    before = model_fingerprint(model)
    model.encoder.depth += 1
    assert model_fingerprint(model) != before


def test_training_reference_changes_invalidate_similarity(compounds):
    library = PreparedLibrary(compounds)
    first = library.training_similarity(["CCO"])
    second = library.training_similarity(["CCN"])
    assert first[0] == second[1] == 1.0
    assert second[0] < 1.0
    assert first[1] < 1.0
    np.testing.assert_array_equal(
        second, maximum_tanimoto([compound.canonical_smiles for compound in compounds], ["CCN"])
    )


def test_unknown_manifest_fields_and_duplicate_json_keys_are_rejected(compounds, tmp_path):
    library = PreparedLibrary(compounds)
    path = tmp_path / "library.json"
    library.save_manifest(path)
    value = json.loads(path.read_text())
    value["unrecognized"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fields"):
        PreparedLibrary.from_manifest(path)
    path.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        PreparedLibrary.from_manifest(path)


def test_cached_scoring_refuses_training_mode_and_invalid_weights(bundle):
    pipeline, dataset, index = _pipeline(bundle)
    library = PreparedLibrary(dataset.compounds)
    with pytest.raises(ValueError, match="Weights"):
        _rank(pipeline, dataset, index, library, novelty_weight=np.nan)
    pipeline.forward.train()
    with pytest.raises(RuntimeError, match="evaluation mode"):
        _rank(pipeline, dataset, index, library)
