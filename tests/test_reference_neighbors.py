"""Independent observed-response retrieval and its information-access boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rigid.baselines import ReferenceNeighborConfig, reference_nearest_neighbor  # noqa: E402
from rigid.data import (  # noqa: E402
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    SourceProvenance,
    save_dataset,
)
from rigid.experiments import nearest_neighbor  # noqa: E402


def datasets():
    compounds = tuple(
        CompoundRecord(cid, smiles, ("fixture_moa",))
        for cid, smiles in (("a", "CC"), ("b", "CO"), ("c", "CCC"), ("d", "CCCC"), ("e", "CCO"))
    )
    genes = GeneManifest(("g1", "g2", "g3", "g4"), "synthetic")
    query = ProfileDataset(
        genes,
        compounds,
        tuple(
            ProfileRecord("q_" + cid, cid, "cell", 1, 24, split)
            for cid, split in (("a", "train"), ("b", "train"), ("c", "test"), ("d", "test"))
        ),
        np.array([[-1] * 4, [1] * 4, [2, 1, -1, -2], [-1, 2, -2, 1]]),
        SourceProvenance(kind="synthetic", description="Generated query fixtures"),
    )
    reference = ProfileDataset(
        genes,
        compounds,
        tuple(ProfileRecord("r_" + cid, cid, "cell", 1, 24, "train") for cid in ("a", "c", "d")),
        np.array([[2, -2, 2, -2], [2.2, 1.1, -1.05, -1.9], [-1.1, 2.3, -1.8, 1.1]]),
        SourceProvenance(kind="synthetic", description="Separately generated response fixtures"),
    )
    return query, reference


def run(tmp_path, query=None, reference=None, **kwargs):
    default_query, default_reference = datasets()
    save_dataset(query or default_query, tmp_path / "query")
    save_dataset(reference or default_reference, tmp_path / "reference")
    return nearest_neighbor(
        tmp_path / "query", k=kwargs.pop("k", 1), reference_dataset=tmp_path / "reference", **kwargs
    )


def test_independent_reference_supports_nonzero_exact_source_recovery(tmp_path):
    result = run(tmp_path)
    assert result["metrics"]["exact"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert result["rankings"]["q_c"][0]["candidate_id"] == "c"
    assert result["candidate_count"] == 5
    assert result["coverage"]["q_c"]["eligible_count"] == 3
    assert result["coverage"]["q_c"]["unscoreable_count"] == 2
    assert result["coverage"]["q_c"]["candidate_coverage"] == 0.6
    assert result["normalizer"]["n_training_profiles"] == 2
    assert result["normalizer"]["mean"] == [0.0] * 4
    assert result["normalizer"]["scale"] == [1.0] * 4
    assert result["source_kind"] == "synthetic"
    assert result["input_sha256"] and result["reference_input_sha256"]
    assert "observed candidate responses" in result["access_regime"]
    json.dumps(result, allow_nan=False)


def test_identical_profile_id_rejected_even_outside_selected_reference_split(tmp_path):
    query, reference = datasets()
    profiles = (
        replace(reference.profiles[0], profile_id="q_c", split="excluded"),
        *reference.profiles[1:],
    )
    with pytest.raises(ValueError, match="profile IDs overlap"):
        run(tmp_path, query, replace(reference, profiles=profiles))


def test_expression_copy_with_renamed_id_is_rejected(tmp_path):
    query, reference = datasets()
    expression = reference.expression.copy()
    expression[1] = query.expression[2]
    with pytest.raises(ValueError, match="duplicates a query dataset expression"):
        run(tmp_path, query, replace(reference, expression=expression))


def test_training_row_copies_also_rejected(tmp_path):
    query, reference = datasets()
    expression = reference.expression.copy()
    expression[1] = query.expression[0]
    with pytest.raises(ValueError, match="duplicates a query dataset expression"):
        run(tmp_path, query, replace(reference, expression=expression))


@pytest.mark.parametrize(
    "field,value", [("cell_id", "another_cell"), ("dose_um", 1.1), ("time_h", 25)]
)
def test_strict_context_mismatch_keeps_empty_targets_in_denominator(tmp_path, field, value):
    query, reference = datasets()
    reference = replace(
        reference, profiles=tuple(replace(p, **{field: value}) for p in reference.profiles)
    )
    result = run(tmp_path, query, reference)
    assert result["rankings"] == {"q_c": [], "q_d": []}
    assert result["metrics"]["exact"] == {"numerator": 0, "denominator": 2, "value": 0.0}
    assert result["coverage"]["q_c"]["eligible_count"] == 0
    assert result["coverage"]["q_c"]["unscoreable_count"] == 5


def test_explicit_context_tolerances_and_cell_relaxation(tmp_path):
    query, reference = datasets()
    reference = replace(
        reference,
        profiles=tuple(
            replace(p, cell_id="other", dose_um=1.25, time_h=25) for p in reference.profiles
        ),
    )
    config = ReferenceNeighborConfig(match_cell=False, dose_tolerance_um=0.25, time_tolerance_h=1)
    result = run(tmp_path, query, reference, reference_config=config)
    assert result["metrics"]["exact"]["value"] == 1.0
    assert result["configuration"] == config.to_dict()


def test_declared_catalog_does_not_add_known_target_compounds(tmp_path):
    result = run(tmp_path, candidate_ids=["a", "e"], k=50)
    assert result["candidate_ids"] == ["a", "e"]
    assert result["reference_compounds_outside_catalog"] == ["b", "c", "d"]
    assert result["metrics"]["exact"]["value"] == 0
    assert all(
        [row["candidate_id"] for row in rows] == ["a"] for rows in result["rankings"].values()
    )
    assert result["coverage"]["q_c"]["source_in_declared_catalog"] is False
    assert result["coverage"]["q_c"]["source_scoreable"] is False
    assert result["coverage"]["q_c"]["returned"] == 1


def test_unscoreable_catalog_is_not_filled_with_fabricated_scores(tmp_path):
    result = run(tmp_path, candidate_ids=["e"])
    assert all(rows == [] for rows in result["rankings"].values())
    assert result["metrics"]["exact"]["denominator"] == 2
    assert result["coverage"]["q_c"]["candidate_coverage"] == 0.0


def test_external_aliases_map_to_declared_canonical_structure(tmp_path):
    query, reference = datasets()
    compounds = tuple(
        replace(c, compound_id="external-c", aliases=("c",)) if c.compound_id == "c" else c
        for c in reference.compounds
    )
    profiles = tuple(
        replace(p, compound_id="external-c") if p.compound_id == "c" else p
        for p in reference.profiles
    )
    result = run(tmp_path, query, replace(reference, compounds=compounds, profiles=profiles))
    assert result["reference_to_candidate_ids"]["external-c"] == "c"
    assert result["rankings"]["q_c"][0]["candidate_id"] == "c"
    assert result["metrics"]["exact"]["value"] == 1.0


def test_declared_alias_representative_preserves_structural_exact_recovery(tmp_path):
    query, reference = datasets()
    query = replace(query, compounds=(*query.compounds, CompoundRecord("c_alias", "CCC")))
    result = run(tmp_path, query, reference, candidate_ids=["c_alias", "c", "d"])
    assert result["candidate_ids"] == ["c_alias", "d"]
    assert result["candidate_identity_members"]["c_alias"] == ["c_alias", "c"]
    assert result["rankings"]["q_c"][0]["candidate_id"] == "c_alias"
    assert result["metrics"]["exact"]["value"] == 1.0


@pytest.mark.parametrize("conflict", ["id", "alias"])
def test_conflicting_compound_identity_is_rejected(tmp_path, conflict):
    query, reference = datasets()
    old = reference.compounds[0]
    replacement = CompoundRecord("a", "CN") if conflict == "id" else replace(old, aliases=("C",))
    with pytest.raises(ValueError, match="Conflicting compound identity or alias"):
        run(tmp_path, query, replace(reference, compounds=(replacement, *reference.compounds[1:])))


def test_max_and_mean_aggregate_all_eligible_replicates(tmp_path):
    query, reference = datasets()
    reference = replace(
        reference,
        profiles=(*reference.profiles, replace(reference.profiles[1], profile_id="r_c_negative")),
        expression=np.vstack([reference.expression, -reference.expression[1]]),
    )
    first = run(tmp_path / "max", query, reference, k=5)
    second = run(
        tmp_path / "mean",
        query,
        reference,
        k=5,
        reference_config=ReferenceNeighborConfig(aggregation="mean"),
    )
    max_row = next(row for row in first["rankings"]["q_c"] if row["candidate_id"] == "c")
    mean_row = next(row for row in second["rankings"]["q_c"] if row["candidate_id"] == "c")
    assert max_row["reference_profile_count"] == mean_row["reference_profile_count"] == 2
    assert max_row["score"] > 0.99
    assert mean_row["score"] == pytest.approx(0, abs=1e-12)


def test_chunk_size_does_not_change_scores_or_tie_order(tmp_path):
    first = run(tmp_path / "one", k=5, reference_config=ReferenceNeighborConfig(chunk_profiles=1))
    second = run(
        tmp_path / "many", k=5, reference_config=ReferenceNeighborConfig(chunk_profiles=100)
    )
    for tid in first["rankings"]:
        assert [r["candidate_id"] for r in first["rankings"][tid]] == [
            r["candidate_id"] for r in second["rankings"][tid]
        ]
        np.testing.assert_allclose(
            [r["score"] for r in first["rankings"][tid]],
            [r["score"] for r in second["rankings"][tid]],
            atol=1e-14,
        )


def test_constant_reference_profiles_are_excluded_explicitly(tmp_path):
    query, reference = datasets()
    expression = reference.expression.copy()
    expression[0] = 3
    result = run(tmp_path, query, replace(reference, expression=expression))
    assert result["excluded_constant_reference_profile_ids"] == ["r_a"]
    assert result["coverage"]["q_c"]["eligible_count"] == 2


def test_constant_query_is_a_hard_error_not_a_dropped_target(tmp_path):
    query, reference = datasets()
    expression = query.expression.copy()
    expression[2] = 7
    with pytest.raises(ValueError, match="Target q_c has undefined"):
        run(tmp_path, replace(query, expression=expression), reference)


def test_reference_partition_selection_is_explicit(tmp_path):
    query, reference = datasets()
    reference = replace(
        reference, profiles=tuple(replace(p, split="external") for p in reference.profiles)
    )
    with pytest.raises(ValueError, match="No reference profiles"):
        run(tmp_path / "default", query, reference)
    result = run(
        tmp_path / "external",
        query,
        reference,
        reference_config=ReferenceNeighborConfig(reference_splits=("external",)),
    )
    assert result["metrics"]["exact"]["value"] == 1.0


def test_query_test_values_do_not_fit_normalization(tmp_path):
    query, reference = datasets()
    changed = query.expression.copy()
    changed[2:] = changed[2:] * 50 + 9
    first = run(tmp_path / "first", query, reference)
    second = run(tmp_path / "second", replace(query, expression=changed), reference)
    assert first["normalizer"] == second["normalizer"]


def test_moa_labels_never_change_retrieval_scores(tmp_path):
    query, reference = datasets()
    first = run(tmp_path / "first", query, reference)
    query = replace(query, compounds=tuple(replace(c, moa=("changed",)) for c in query.compounds))
    reference = replace(reference, compounds=tuple(replace(c, moa=()) for c in reference.compounds))
    second = run(tmp_path / "second", query, reference)
    assert first["rankings"] == second["rankings"]


def test_mixed_sources_keep_synthetic_warning(tmp_path):
    query, reference = datasets()
    reference = replace(
        reference,
        provenance=SourceProvenance(
            kind="experimental",
            description="Public reference fixture metadata",
            source_urls=("https://example.org/dataset",),
        ),
    )
    result = run(tmp_path, query, reference)
    assert result["source_kind"] == "synthetic"
    assert result["mixed_source_kinds"] is True
    assert result["reference_source_kind"] == "experimental"


@pytest.mark.parametrize(
    "provenance",
    [
        SourceProvenance(),
        SourceProvenance(kind="synthetic"),
        SourceProvenance(kind="experimental", description="No traceable source"),
    ],
)
def test_reference_provenance_required(tmp_path, provenance):
    query, reference = datasets()
    with pytest.raises(ValueError, match="provenance|source"):
        run(tmp_path, query, replace(reference, provenance=provenance))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_splits": "train"},
        {"reference_splits": []},
        {"reference_splits": ["excluded"]},
        {"reference_splits": ["train", "train"]},
        {"match_cell": 1},
        {"aggregation": []},
        {"aggregation": "median"},
        {"dose_tolerance_um": -1},
        {"time_tolerance_h": float("inf")},
        {"dose_tolerance_um": True},
        {"chunk_profiles": True},
        {"chunk_profiles": 0},
    ],
)
def test_invalid_reference_configuration(kwargs):
    with pytest.raises(ValueError):
        ReferenceNeighborConfig(**kwargs)


def test_reference_config_strict_json_loading(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(ReferenceNeighborConfig().to_dict()))
    assert ReferenceNeighborConfig.load(path) == ReferenceNeighborConfig()
    for invalid in (
        '{"unknown":1}',
        '{"match_cell":true,"match_cell":false}',
        "[]",
        '{"dose_tolerance_um":NaN}',
    ):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            ReferenceNeighborConfig.load(path)


@pytest.mark.parametrize("candidate_ids", ["c", [], ["unknown"], ["c", "c"], [None]])
def test_invalid_candidate_selection(tmp_path, candidate_ids):
    with pytest.raises(ValueError, match="candidate_ids"):
        run(tmp_path, candidate_ids=candidate_ids)


def test_reference_parameters_rejected_without_reference_dataset(tmp_path):
    with pytest.raises(ValueError, match="require reference_dataset"):
        nearest_neighbor(tmp_path, reference_config=ReferenceNeighborConfig())
    with pytest.raises(ValueError, match="require reference_dataset"):
        nearest_neighbor(tmp_path, candidate_ids=["a"])


def test_same_dataset_cannot_be_its_own_reference(tmp_path):
    with pytest.raises(ValueError, match="separate dataset"):
        reference_nearest_neighbor(tmp_path, tmp_path)


def test_gene_order_mismatch_rejected(tmp_path):
    query, reference = datasets()
    reference = replace(reference, genes=GeneManifest(("g2", "g1", "g3", "g4"), "synthetic"))
    with pytest.raises(ValueError, match="gene order"):
        run(tmp_path, query, reference)


def test_configured_acyclic_scaffold_policy_is_audited(tmp_path):
    with pytest.raises(ValueError, match="leakage|Leakage"):
        run(tmp_path / "group", scaffold_holdout=True)
    result = run(tmp_path / "compound", scaffold_holdout=True, acyclic_policy="compound")
    assert result["leakage_audit"]["acyclic_policy"] == "compound"


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"split": "train"}, {"config": {}}, {"acyclic_policy": "guess"}]
)
def test_invalid_public_parameters_rejected_before_data_access(tmp_path, kwargs):
    with pytest.raises(ValueError):
        reference_nearest_neighbor(tmp_path / "query", tmp_path / "reference", **kwargs)


def test_reference_integrity_failure_is_rejected(tmp_path):
    query, reference = datasets()
    save_dataset(query, tmp_path / "query")
    save_dataset(reference, tmp_path / "reference")
    path = tmp_path / "reference" / "profiles.csv"
    path.write_text(path.read_text().replace("r_a", "renamed"))
    with pytest.raises(ValueError, match="checksum|Checksum"):
        reference_nearest_neighbor(tmp_path / "query", tmp_path / "reference")


def test_alias_annotations_merge_after_ranking(tmp_path):
    query, reference = datasets()
    query = replace(query, compounds=(*query.compounds, CompoundRecord("c_alias", "CCC")))
    result = run(tmp_path, query, reference, candidate_ids=["c_alias", "c", "d"])
    assert result["metrics"]["same_moa"]["value"] == 1.0


def test_target_annotations_use_all_known_canonical_identity_labels(tmp_path):
    query, reference = datasets()
    query = replace(
        query,
        compounds=(
            *(replace(c, moa=()) if c.compound_id == "c" else c for c in query.compounds),
            CompoundRecord("c_alias", "CCC", ("fixture_moa",)),
        ),
    )
    result = run(tmp_path, query, reference, candidate_ids=["c", "d"])
    assert result["candidate_identity_members"]["c"] == ["c"]
    assert result["annotation_identity_members"]["c"] == ["c", "c_alias"]
    assert result["metrics"]["same_moa"] == {"numerator": 2, "denominator": 2, "value": 1.0}


def test_reference_neighbors_complete_cli_workflow(tmp_path, capsys):
    from rigid.cli import main

    query, reference = datasets()
    save_dataset(query, tmp_path / "query")
    save_dataset(reference, tmp_path / "reference")
    config_path = tmp_path / "reference-config.json"
    config_path.write_text(json.dumps(ReferenceNeighborConfig().to_dict()))
    output = tmp_path / "neighbors.json"
    assert (
        main(
            [
                "neighbors",
                str(tmp_path / "query"),
                "--reference-dataset",
                str(tmp_path / "reference"),
                "--reference-config",
                str(config_path),
                "--top-k",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    result = json.loads(output.read_text())
    assert result["metrics"]["exact"]["value"] == 1.0
    assert result["source_kind"] == "synthetic"
    assert result["coverage"]["q_c"]["unscoreable_count"] == 2
    assert result["configuration"] == ReferenceNeighborConfig().to_dict()


def test_cli_rejects_reference_config_without_catalog(tmp_path, capsys):
    from rigid.cli import main

    output = tmp_path / "must-not-exist.json"
    assert (
        main(
            [
                "--json-errors",
                "neighbors",
                str(tmp_path / "query"),
                "--reference-config",
                str(tmp_path / "config.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().err)
    assert "requires --reference-dataset" in result["error"]["message"]
    assert not output.exists()
