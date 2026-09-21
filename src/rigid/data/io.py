"""Strict CSV interchange, explicit provenance, and checksum validation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .schema import (
    SCHEMA_VERSION,
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    SourceProvenance,
)

PROFILE_COLUMNS = ("profile_id", "compound_id", "cell_id", "dose_um", "time_h")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(
    directory: str | Path,
    *,
    verify_checksums: bool = True,
    chunk_size: int = 4096,
    max_profiles: int | None = None,
) -> ProfileDataset:
    """Load a dataset, converting CSV expression values in bounded numeric chunks.

    The returned dataset remains an in-memory array. ``chunk_size`` bounds the
    conversion buffer, not total dataset memory. ``max_profiles`` can enforce
    an explicit ingestion limit before loading an unexpectedly large source.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if max_profiles is not None and (
        isinstance(max_profiles, bool) or not isinstance(max_profiles, int) or max_profiles < 1
    ):
        raise ValueError("max_profiles must be a positive integer when supplied")
    directory = Path(directory)
    genes = GeneManifest.load(directory / "genes.json")
    metadata_path = directory / "dataset.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a JSON object")
    if metadata.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema version")
    if metadata.get("gene_manifest_sha256", genes.sha256) != genes.sha256:
        raise ValueError("Dataset metadata references a different gene manifest")
    checksums = metadata.get("files_sha256")
    if checksums is not None and (
        not isinstance(checksums, dict)
        or set(checksums) != {"genes.json", "compounds.csv", "profiles.csv"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in checksums.values()
        )
    ):
        raise ValueError(
            "Dataset checksum manifest must cover all three data files with SHA-256 digests"
        )
    if verify_checksums:
        for filename, checksum in (checksums or {}).items():
            if filename not in {"genes.json", "compounds.csv", "profiles.csv"}:
                raise ValueError("Dataset checksum manifest contains an unexpected path")
            if _checksum(directory / filename) != checksum:
                raise ValueError(f"Dataset checksum mismatch for {filename}")
    compounds = []
    with (directory / "compounds.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"compound_id", "smiles"}.issubset(reader.fieldnames):
            raise ValueError("compounds.csv requires compound_id and smiles columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate compounds.csv column names")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed compounds.csv row at line {line_number}")
            compounds.append(
                CompoundRecord(
                    row["compound_id"].strip(),
                    row["smiles"].strip(),
                    tuple(
                        value.strip() for value in row.get("moa", "").split(";") if value.strip()
                    ),
                    tuple(
                        value.strip()
                        for value in row.get("aliases", "").split(";")
                        if value.strip()
                    ),
                )
            )
    profiles: list[ProfileRecord] = []
    chunks: list[NDArray[np.float32]] = []
    buffer: NDArray[np.float32] = np.empty((chunk_size, len(genes)), dtype=np.float32)
    buffered = 0
    with (directory / "profiles.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(PROFILE_COLUMNS).issubset(columns):
            raise ValueError("profiles.csv metadata columns are missing or duplicated")
        observed_genes = tuple(
            column for column in columns if column not in (*PROFILE_COLUMNS, "split")
        )
        genes.require_match(observed_genes)
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed profiles.csv row at line {line_number}")
            try:
                if max_profiles is not None and len(profiles) >= max_profiles:
                    raise ValueError(f"Profile limit of {max_profiles} exceeded")
                values: NDArray[np.float64] = np.asarray(
                    [float(row[gene]) for gene in genes.gene_ids], dtype=np.float64
                )
                if not np.isfinite(values).all() or np.any(
                    np.abs(values) > np.finfo(np.float32).max
                ):
                    raise ValueError(
                        "Expression values must be finite and representable as float32"
                    )
                profiles.append(
                    ProfileRecord(
                        row["profile_id"].strip(),
                        row["compound_id"].strip(),
                        row["cell_id"].strip(),
                        float(row["dose_um"]),
                        float(row["time_h"]),
                        row.get("split", "").strip(),
                    )
                )
                buffer[buffered] = values
                buffered += 1
                if buffered == chunk_size:
                    chunks.append(buffer)
                    buffer = np.empty_like(buffer)
                    buffered = 0
            except ValueError as error:
                raise ValueError(f"Invalid profiles.csv line {line_number}: {error}") from error
    if buffered:
        chunks.append(buffer[:buffered])
    expression = np.concatenate(chunks) if chunks else np.empty((0, len(genes)), dtype=np.float32)
    provenance = SourceProvenance.from_dict(metadata.get("provenance", {}))
    if metadata.get("n_profiles", len(profiles)) != len(profiles):
        raise ValueError("Dataset metadata profile count does not match profiles.csv")
    if metadata.get("n_compounds", len(compounds)) != len(compounds):
        raise ValueError("Dataset metadata compound count does not match compounds.csv")
    return ProfileDataset(
        genes,
        tuple(compounds),
        tuple(profiles),
        expression,
        provenance,
    )


def save_dataset(dataset: ProfileDataset, directory: str | Path) -> Path:
    """Write a new dataset; never replace an existing directory's contents.

    ``dataset.json`` is written last, after data files and checksums. If an I/O
    failure interrupts creation, partial files remain available for inspection.
    """
    if set(dataset.genes.gene_ids) & {*PROFILE_COLUMNS, "split"}:
        raise ValueError("Gene identifiers must not collide with reserved profile metadata columns")
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Dataset output must be an empty directory or a new path")
    provenance = dataset.provenance.to_dict()
    json.dumps(provenance, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "genes.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dataset.genes.to_dict(), indent=2, allow_nan=False) + "\n")
    with (directory / "compounds.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("compound_id", "smiles", "moa", "aliases"))
        for compound in dataset.compounds:
            writer.writerow(
                (
                    compound.compound_id,
                    compound.smiles,
                    ";".join(compound.moa),
                    ";".join(compound.aliases),
                )
            )
    with (directory / "profiles.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((*PROFILE_COLUMNS, "split", *dataset.genes.gene_ids))
        for profile, expression in zip(dataset.profiles, dataset.expression, strict=True):
            writer.writerow(
                (
                    profile.profile_id,
                    profile.compound_id,
                    profile.cell_id,
                    profile.dose_um,
                    profile.time_h,
                    profile.split,
                    *[float(value) for value in expression],
                )
            )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "gene_manifest_sha256": dataset.genes.sha256,
        "provenance": provenance,
        "files_sha256": {
            name: _checksum(directory / name)
            for name in ("genes.json", "compounds.csv", "profiles.csv")
        },
        "n_profiles": len(dataset.profiles),
        "n_compounds": len(dataset.compounds),
    }
    with (directory / "dataset.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return directory


def attach_source_hashes(dataset: ProfileDataset, paths: list[str | Path]) -> ProfileDataset:
    hashes = {Path(path).name: _checksum(Path(path)) for path in paths}
    if len(hashes) != len(paths):
        raise ValueError("Input basenames must be unique for provenance hashes")
    return replace(
        dataset,
        provenance=replace(
            dataset.provenance, input_sha256={**dataset.provenance.input_sha256, **hashes}
        ),
    )
