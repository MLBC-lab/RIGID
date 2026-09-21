"""Preprocessing contracts and empirical checks of leakage protection."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    ProgramSet,
    TrainStandardizer,
    assign_splits,
    audit_leakage,
    consensus_signature,
    edit_programs,
    interpolate_targets,
    load_dataset,
    make_synthetic_dataset,
    prepare_matrix,
    read_gct,
    read_gctx,
    reverse_disease_signature,
    save_dataset,
)


def small_dataset(compounds=None, splits=("train", "validation", "test")):
    if compounds is None:
        compounds = (
            CompoundRecord("a", "CCO"),
            CompoundRecord("b", "c1ccccc1"),
            CompoundRecord("c", "C1CCCCC1"),
        )
    profiles = tuple(
        ProfileRecord(f"p{i}", compound.compound_id, "cell", 1.0, 24.0, split)
        for i, (compound, split) in enumerate(zip(compounds, splits, strict=True))
    )
    return ProfileDataset(
        GeneManifest(("g1", "g2")), compounds, profiles, np.arange(len(profiles) * 2).reshape(-1, 2)
    )


def test_gene_manifest_is_ordered_and_checksummed(tmp_path):
    genes = GeneManifest(("a", "b"), "entrez")
    genes.save(tmp_path / "genes.json")
    assert GeneManifest.load(tmp_path / "genes.json") == genes
    with pytest.raises(ValueError, match="order"):
        genes.require_match(("b", "a"))
    with pytest.raises(ValueError, match="duplicate"):
        GeneManifest(("a", "a"))
    corrupt = genes.to_dict()
    corrupt["gene_ids"] = ["a", "c"]
    with pytest.raises(ValueError, match="checksum"):
        GeneManifest.from_dict(corrupt)


@pytest.mark.parametrize("values", [np.array([[np.nan, 1]]), np.array([[np.inf, 1]])])
def test_nonfinite_profiles_rejected(values):
    dataset = small_dataset()
    with pytest.raises(ValueError, match="NaN|infinite"):
        replace(dataset, profiles=dataset.profiles[:1], expression=values)


@pytest.mark.parametrize(
    "changes",
    [{"dose_um": -1}, {"dose_um": float("nan")}, {"time_h": 0}, {"cell_id": ""}, {"split": "val"}],
)
def test_context_metadata_validated(changes):
    with pytest.raises(ValueError):
        replace(small_dataset().profiles[0], **changes)


def test_csv_roundtrip_preserves_genes_metadata_and_provenance(tmp_path):
    dataset = make_synthetic_dataset(tmp_path, n_genes=8)
    loaded = load_dataset(tmp_path)
    np.testing.assert_array_equal(loaded.expression, dataset.expression)
    assert loaded.genes == dataset.genes
    assert loaded.profiles == dataset.profiles
    assert loaded.provenance.kind == "synthetic"
    assert not loaded.expression.flags.writeable
    assert len(loaded.subset("train").profiles) > 0


def test_checksum_prevents_silent_data_changes(tmp_path):
    save_dataset(small_dataset(), tmp_path)
    with (tmp_path / "profiles.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("n_profiles", 10000), ("n_compounds", 10000), ("gene_manifest_sha256", "wrong")],
)
def test_metadata_inconsistency_is_rejected(tmp_path, field, value):
    save_dataset(small_dataset(), tmp_path)
    path = tmp_path / "dataset.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_dataset(tmp_path)


def test_reordered_csv_genes_rejected_even_when_dimensions_match(tmp_path):
    save_dataset(small_dataset(), tmp_path)
    path = tmp_path / "profiles.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        row[-1], row[-2] = row[-2], row[-1]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="order"):
        load_dataset(tmp_path, verify_checksums=False)


def test_alias_parent_and_scaffold_leaks_are_reported():
    compounds = (
        CompoundRecord("a", "Cc1ccccc1", aliases=("old-id",)),
        CompoundRecord("b", "Oc1ccccc1", aliases=("old-id",)),
        CompoundRecord("c", "Cc1ccccc1.[Cl-]"),
    )
    report = audit_leakage(small_dataset(compounds))
    assert not report.ok
    kinds = {item.kind for item in report.violations}
    assert {"alias", "canonical_structure", "connectivity", "scaffold"} <= kinds
    with pytest.raises(ValueError, match="Leakage"):
        report.raise_for_leakage()


def test_same_compound_multiple_contexts_stays_in_one_split():
    dataset = make_synthetic_dataset(n_genes=4)
    assigned = assign_splits(dataset, seed=123)
    assert audit_leakage(assigned).ok
    assert assigned.split_labels == assign_splits(dataset, seed=123).split_labels
    assert all(assigned.subset(split).profiles for split in ("train", "validation", "test"))
    groups = {}
    for profile in assigned.profiles:
        groups.setdefault(profile.compound_id, set()).add(profile.split)
    assert all(len(values) == 1 for values in groups.values())


def test_scaffold_split_groups_aliases_transitively():
    compounds = (
        CompoundRecord("a", "CCO", aliases=("b",)),
        CompoundRecord("b", "CCN"),
        CompoundRecord("c", "c1ccccc1"),
        CompoundRecord("d", "C1CCCCC1"),
    )
    dataset = small_dataset(compounds, ("", "", "", ""))
    assigned = assign_splits(dataset, strategy="compound", fractions=(0.34, 0.33, 0.33))
    assert assigned.profiles[0].split == assigned.profiles[1].split
    assert audit_leakage(assigned, scaffold=False).ok


def test_too_few_independent_groups_fail_explictly():
    dataset = small_dataset(
        (CompoundRecord("a", "CCO"), CompoundRecord("b", "CCN"), CompoundRecord("c", "CCC"))
    )
    with pytest.raises(ValueError, match="independent groups"):
        assign_splits(dataset)


def test_unassigned_profiles_fail_audit():
    report = audit_leakage(small_dataset(splits=("", "validation", "test")))
    assert not report.ok
    assert report.unassigned_profile_ids == ("p0",)


def test_normalizer_ignores_held_out_extreme_values_and_roundtrips(tmp_path):
    genes = GeneManifest(("g1", "g2"))
    matrix = np.array([[1, 2], [3, 2], [100000, -100000]])
    standardizer = TrainStandardizer.fit(matrix, ("train", "train", "test"), genes)
    np.testing.assert_allclose(standardizer.mean, [2, 2])
    np.testing.assert_allclose(standardizer.scale, [1, 1])
    standardizer.save(tmp_path / "normalizer.json")
    reloaded = TrainStandardizer.load(tmp_path / "normalizer.json")
    transformed = reloaded.transform(matrix, genes)
    np.testing.assert_allclose(reloaded.inverse_transform(transformed, genes), matrix)
    with pytest.raises(ValueError, match="order"):
        reloaded.transform(matrix, GeneManifest(("g2", "g1")))
    with pytest.raises(ValueError, match="No training"):
        TrainStandardizer.fit(matrix, ("test", "test", "test"), genes)


def test_disease_reversal_and_program_edits_are_explicit():
    genes = GeneManifest(("g1", "g2", "g3"))
    target = np.array([1, -2, 3])
    programs = ProgramSet(
        genes, {"stress": ("g1",), "cycle": ("g3",)}, source="synthetic test definitions"
    )
    np.testing.assert_array_equal(reverse_disease_signature(target, genes), [-1, 2, -3])
    np.testing.assert_array_equal(
        edit_programs(target, genes, programs, {"stress": 0, "cycle": 2}), [0, -2, 6]
    )
    np.testing.assert_array_equal(interpolate_targets(target, -target, genes), [0, 0, 0])
    np.testing.assert_array_equal(programs.scores(target, genes, absolute=True), [1, 3])
    np.testing.assert_array_equal(consensus_signature(np.array([target, target]), genes), target)
    with pytest.raises(ValueError, match="missing"):
        ProgramSet(genes, {"stress": ("unknown",)}, source="test")


def test_gct_import_requires_explicit_reordering(tmp_path):
    path = tmp_path / "expression.gct"
    path.write_text(
        "#1.2\n2\t2\nName\tDescription\tp1\tp2\ng2\tna\t2\t4\ng1\tna\t1\t3\n", encoding="utf-8"
    )
    genes = GeneManifest(("g1", "g2"))
    with pytest.raises(ValueError, match="order"):
        read_gct(path, genes)
    ids, matrix = read_gct(path, genes, allow_reorder=True)
    assert ids == ("p1", "p2")
    np.testing.assert_array_equal(matrix, [[1, 2], [3, 4]])


def test_gctx_import_selects_manifest_genes_in_exact_order(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "expression.gctx"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("0/META/ROW/id", data=np.array([b"g2", b"g1", b"extra"]))
        handle.create_dataset("0/META/COL/id", data=np.array([b"p1", b"p2"]))
        handle.create_dataset("0/DATA/0/matrix", data=[[2, 1, 10], [4, 3, 20]])
    ids, matrix = read_gctx(path, GeneManifest(("g1", "g2")), allow_reorder=True)
    assert ids == ("p1", "p2")
    np.testing.assert_array_equal(matrix, [[1, 2], [3, 4]])


def preparation_inputs(tmp_path):
    matrix = tmp_path / "input.gct"
    matrix.write_text(
        "#1.2\n2\t2\nName\tDescription\tp2\tp1\ng1\tna\t3\t1\ng2\tna\t4\t2\n", encoding="utf-8"
    )
    genes = tmp_path / "landmarks.json"
    GeneManifest(("g1", "g2")).save(genes)
    compounds = tmp_path / "compound_metadata.csv"
    compounds.write_text("compound_id,smiles,moa\na,CCO,\nb,c1ccccc1,\n", encoding="utf-8")
    metadata = tmp_path / "sample_metadata.csv"
    metadata.write_text(
        "profile_id,compound_id,cell_id,dose_um,time_h,split\np1,a,cell,1,24,train\np2,b,cell,1,24,test\n",
        encoding="utf-8",
    )
    return matrix, genes, compounds, metadata


def test_prepare_matrix_joins_ids_in_matrix_order(tmp_path):
    inputs = preparation_inputs(tmp_path)
    dataset = prepare_matrix(
        *inputs,
        tmp_path / "prepared",
        source_url="https://example.org/fixture",
        source_license="test-only",
    )
    assert tuple(profile.profile_id for profile in dataset.profiles) == ("p2", "p1")
    assert dataset.profiles[0].compound_id == "b"
    np.testing.assert_array_equal(dataset.expression, [[3, 4], [1, 2]])
    assert dataset.provenance.kind == "experimental"
    assert len(dataset.provenance.input_sha256) == 4
    np.testing.assert_array_equal(
        load_dataset(tmp_path / "prepared").expression, dataset.expression
    )


@pytest.mark.parametrize(
    "invalid_metadata,error",
    [
        ("p1,a,cell,1,24,train\np1,b,cell,1,24,test\n", "Duplicate"),
        ("p1,a,cell,1,24,train\n", "coverage differs"),
        ("p1,a,cell,1,24,train\np2,b,cell,1,24,test\np3,b,cell,1,24,test\n", "coverage differs"),
    ],
)
def test_prepare_rejects_duplicate_missing_or_extra_metadata(tmp_path, invalid_metadata, error):
    inputs = preparation_inputs(tmp_path)
    inputs[-1].write_text(
        "profile_id,compound_id,cell_id,dose_um,time_h,split\n" + invalid_metadata, encoding="utf-8"
    )
    with pytest.raises(ValueError, match=error):
        prepare_matrix(
            *inputs,
            tmp_path / "prepared",
            source_url="https://example.org/fixture",
            source_license="test-only",
        )


def test_calibration_is_isolated_by_leakage_audit():
    dataset = small_dataset(splits=("train", "calibration", "test"))
    assert audit_leakage(dataset).ok
    duplicate = replace(dataset.profiles[1], compound_id="a")
    dataset = replace(dataset, profiles=(dataset.profiles[0], duplicate, dataset.profiles[2]))
    assert not audit_leakage(dataset).ok


def test_prepare_preserves_existing_output(tmp_path):
    inputs = preparation_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "profiles.csv"
    sentinel.write_text("existing research data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_matrix(
            *inputs, output, source_url="https://example.org/fixture", source_license="test-only"
        )
    assert sentinel.read_text(encoding="utf-8") == "existing research data"
