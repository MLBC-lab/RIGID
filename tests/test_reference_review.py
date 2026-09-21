"""Independent checks of reference eligibility, identity, and reported evidence."""

from dataclasses import replace

import numpy as np
import pytest

from rigid.baselines import ReferenceNeighborConfig, reference_nearest_neighbor
from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    SourceProvenance,
    save_dataset,
)


def _inputs():
    compounds = tuple(
        CompoundRecord(cid, smiles, (moa,) if moa else ())
        for cid, smiles, moa in (
            ("a", "CC", ""),
            ("b", "CO", ""),
            ("c", "CCC", "mechanism-c"),
            ("d", "CCCC", "mechanism-d"),
            ("empty", "CCO", ""),
        )
    )
    genes = GeneManifest(("g1", "g2", "g3", "g4"), "synthetic")
    provenance = SourceProvenance(kind="synthetic", description="Review-only generated values")
    query = ProfileDataset(
        genes,
        compounds,
        tuple(
            ProfileRecord("query-" + cid, cid, "cell", 1, 24, split)
            for cid, split in (("a", "train"), ("b", "train"), ("c", "test"), ("d", "test"))
        ),
        np.array([[-1] * 4, [1] * 4, [2, 1, -1, -2], [-1, 2, -2, 1]]),
        provenance,
    )
    reference = ProfileDataset(
        genes,
        compounds,
        tuple(ProfileRecord("reference-" + cid, cid, "cell", 1, 24, "train") for cid in ("c", "d")),
        np.array([[2.1, 0.9, -1.05, -2.2], [-1.1, 2.1, -1.9, 1.1]]),
        provenance,
    )
    return query, reference


def _run(path, query, reference, **kwargs):
    save_dataset(query, path / "query")
    save_dataset(reference, path / "reference")
    return reference_nearest_neighbor(path / "query", path / "reference", k=1, **kwargs)


def test_canonical_alias_order_cannot_discard_known_moa(tmp_path):
    query, reference = _inputs()
    query = replace(query, compounds=(*query.compounds, CompoundRecord("c-alias", "CCC")))
    direct = _run(tmp_path / "direct", query, reference, candidate_ids=["c", "c-alias", "d"])
    alias = _run(tmp_path / "alias", query, reference, candidate_ids=["c-alias", "c", "d"])
    assert direct["metrics"]["exact"] == alias["metrics"]["exact"]
    assert direct["metrics"]["same_moa"] == alias["metrics"]["same_moa"]
    assert alias["metrics"]["same_moa"]["numerator"] == 2


def test_alias_only_selection_preserves_known_canonical_moa(tmp_path):
    query, reference = _inputs()
    query = replace(query, compounds=(*query.compounds, CompoundRecord("c-alias", "CCC")))
    direct = _run(tmp_path / "direct", query, reference, candidate_ids=["c", "d"])
    alias = _run(tmp_path / "alias", query, reference, candidate_ids=["c-alias", "d"])
    assert direct["metrics"]["exact"] == alias["metrics"]["exact"]
    assert direct["metrics"]["same_moa"] == alias["metrics"]["same_moa"]
    assert alias["metrics"]["same_moa"]["numerator"] == 2


def test_partly_missing_reference_keeps_all_targets_and_explicit_coverage(tmp_path):
    query, reference = _inputs()
    reference = replace(
        reference, profiles=reference.profiles[:1], expression=reference.expression[:1]
    )
    result = _run(tmp_path, query, reference)
    assert result["metrics"]["exact"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert result["metrics"]["target_count"] == 2
    assert result["coverage"]["query-c"]["source_scoreable"] is True
    assert result["coverage"]["query-d"]["source_scoreable"] is False
    assert all(row["unscoreable_count"] == 4 for row in result["coverage"].values())
    assert result["metrics"]["median_target_correlation"] is None
    assert result["metrics"]["correlation_target_count"] == 0


def test_missing_moa_stays_missing_without_removing_exact_denominator(tmp_path):
    query, reference = _inputs()
    query = replace(query, compounds=tuple(replace(record, moa=()) for record in query.compounds))
    result = _run(tmp_path, query, reference)
    assert result["metrics"]["same_moa"] == {"numerator": 0, "denominator": 0, "value": None}
    assert result["metrics"]["exact"]["denominator"] == 2


@pytest.mark.parametrize(
    "query_kind,reference_kind,expected",
    [
        ("synthetic", "synthetic", "synthetic"),
        ("synthetic", "experimental", "synthetic"),
        ("experimental", "synthetic", "synthetic"),
        ("experimental", "experimental", "experimental"),
        ("unknown", "experimental", "unknown"),
        ("unknown", "synthetic", "synthetic"),
    ],
)
def test_source_labels_do_not_hide_synthetic_or_unknown_inputs(
    tmp_path, query_kind, reference_kind, expected
):
    query, reference = _inputs()

    def labeled(dataset, kind):
        return replace(
            dataset,
            provenance=SourceProvenance(
                kind=kind,
                description="Synthetic values exercising declared-source metadata",
                source_urls=("https://example.org/review-fixture",),
            ),
        )

    result = _run(tmp_path, labeled(query, query_kind), labeled(reference, reference_kind))
    assert result["source_kind"] == expected
    assert result["query_source_kind"] == query_kind
    assert result["reference_source_kind"] == reference_kind
    assert result["mixed_source_kinds"] is (query_kind != reference_kind)
    assert result["reproduces_manuscript"] is False


def test_reference_unit_work_is_chunked(tmp_path, monkeypatch):
    import rigid.baselines as baselines

    query, reference = _inputs()
    original = baselines._unit_rows
    counts = []

    def record(values, normalizer):
        counts.append(len(values))
        return original(values, normalizer)

    monkeypatch.setattr(baselines, "_unit_rows", record)
    result = _run(tmp_path, query, reference, config=ReferenceNeighborConfig(chunk_profiles=1))
    assert counts and max(counts) == 1
    assert result["metrics"]["exact"]["numerator"] == 2


def test_reference_mutation_during_scoring_prevents_result_publication(tmp_path, monkeypatch):
    import rigid.baselines as baselines

    query, reference = _inputs()
    original = baselines._context_matches
    modified = False

    def change_source(target, response, config):
        nonlocal modified
        if not modified:
            (tmp_path / "reference" / "changed.txt").write_text("changed", encoding="utf-8")
            modified = True
        return original(target, response, config)

    monkeypatch.setattr(baselines, "_context_matches", change_source)
    with pytest.raises(ValueError, match="changed during reference retrieval"):
        _run(tmp_path, query, reference)
