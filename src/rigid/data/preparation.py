"""Join source matrices to explicit sample and perturbagen metadata."""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlsplit

from .io import PROFILE_COLUMNS, attach_source_hashes, save_dataset
from .matrices import read_gct, read_gctx
from .schema import CompoundRecord, GeneManifest, ProfileDataset, ProfileRecord, SourceProvenance


def _rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path.name} contains duplicate column names")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed {path.name} row at line {line}")
            rows.append({key: value.strip() for key, value in row.items()})
    return rows


def prepare_matrix(
    matrix_path: str | Path,
    genes_path: str | Path,
    compounds_csv: str | Path,
    metadata_csv: str | Path,
    output: str | Path,
    *,
    source_url: str,
    source_license: str,
    allow_reorder: bool = False,
) -> ProfileDataset:
    """Prepare an experimental matrix with a one-to-one profile metadata join.

    Source URL and license are recorded exactly as supplied. Their presence does
    not certify access rights, validate measurements, or identify paper inputs.
    """
    url = urlsplit(source_url)
    if url.scheme not in {"https", "http"} or not url.netloc:
        raise ValueError("source_url must be an explicit HTTP(S) source URL")
    if not source_license.strip():
        raise ValueError("source_license must be explicitly supplied")
    matrix_path, genes_path = Path(matrix_path), Path(genes_path)
    compounds_csv, metadata_csv = Path(compounds_csv), Path(metadata_csv)
    inputs = [matrix_path, genes_path, compounds_csv, metadata_csv]
    if len(set(path.resolve() for path in inputs)) != len(inputs):
        raise ValueError("Matrix, gene manifest, compounds, and metadata must be distinct inputs")
    output_path = Path(output).resolve()
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError("Prepared output must be an empty directory or a new path")
    output_files = {
        output_path / name
        for name in ("genes.json", "compounds.csv", "profiles.csv", "dataset.json")
    }
    if any(path.resolve() in output_files for path in inputs):
        raise ValueError(
            "Prepared output would overwrite an input file; choose a different directory"
        )
    genes = GeneManifest.load(genes_path)
    if matrix_path.suffix.lower() == ".gct":
        profile_ids, matrix = read_gct(matrix_path, genes, allow_reorder=allow_reorder)
    elif matrix_path.suffix.lower() == ".gctx":
        profile_ids, matrix = read_gctx(matrix_path, genes, allow_reorder=allow_reorder)
    else:
        raise ValueError("Matrix must have a .gct or .gctx extension")
    metadata = _rows(metadata_csv, set(PROFILE_COLUMNS))
    lookup: dict[str, ProfileRecord] = {}
    for row in metadata:
        profile = ProfileRecord(
            row["profile_id"],
            row["compound_id"],
            row["cell_id"],
            float(row["dose_um"]),
            float(row["time_h"]),
            row.get("split", ""),
        )
        if profile.profile_id in lookup:
            raise ValueError(f"Duplicate metadata for profile {profile.profile_id!r}")
        lookup[profile.profile_id] = profile
    missing = set(profile_ids) - set(lookup)
    extra = set(lookup) - set(profile_ids)
    if missing or extra:
        raise ValueError(
            f"Matrix/metadata coverage differs: {len(missing)} profiles missing metadata {sorted(missing)[:5]}; {len(extra)} metadata rows absent from matrix {sorted(extra)[:5]}"
        )
    compounds = []
    for row in _rows(compounds_csv, {"compound_id", "smiles"}):
        compounds.append(
            CompoundRecord(
                row["compound_id"],
                row["smiles"],
                tuple(value.strip() for value in row.get("moa", "").split(";") if value.strip()),
                tuple(
                    value.strip() for value in row.get("aliases", "").split(";") if value.strip()
                ),
            )
        )
    provenance = SourceProvenance(
        kind="experimental",
        description="User-supplied experimental expression matrix joined to explicit profile and compound metadata. Source designation has not been independently verified.",
        source_urls=(source_url,),
        license=source_license,
        transformations=(
            f"Imported {matrix_path.suffix.lower()[1:]} matrix; manifest gene selection/reordering={allow_reorder}; exact profile_id join",
        ),
    )
    dataset = ProfileDataset(
        genes,
        tuple(compounds),
        tuple(lookup[identifier] for identifier in profile_ids),
        matrix,
        provenance,
    )
    dataset = attach_source_hashes(dataset, inputs)
    save_dataset(dataset, output)
    return dataset
