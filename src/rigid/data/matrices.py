"""Import expression matrices without guessing gene identifiers or metadata."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .schema import GeneManifest


def _select_genes(
    matrix: np.ndarray, observed: tuple[str, ...], genes: GeneManifest, allow_reorder: bool
) -> np.ndarray:
    if any(not value.strip() for value in observed):
        raise ValueError("Matrix gene identifiers must be nonempty")
    if len(set(observed)) != len(observed):
        raise ValueError("Matrix contains duplicate gene identifiers")
    if allow_reorder:
        missing = set(genes.gene_ids) - set(observed)
        if missing:
            raise ValueError(f"Matrix is missing manifest genes: {sorted(missing)[:5]}")
        index = {gene: position for position, gene in enumerate(observed)}
        matrix = matrix[:, [index[gene] for gene in genes.gene_ids]]
    else:
        genes.require_match(observed)
    if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > np.finfo(np.float32).max):
        raise ValueError("Imported expression values must be finite and representable as float32")
    return matrix.astype(np.float32)


def read_gct(
    path: str | Path, genes: GeneManifest, *, allow_reorder: bool = False
) -> tuple[tuple[str, ...], np.ndarray]:
    """Read GCT 1.2 as profiles-by-genes; explicit reorder can select landmarks.

    GCT 1.3 has separate metadata semantics and is rejected rather than silently
    interpreting metadata as expression values.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        if handle.readline().strip() != "#1.2":
            raise ValueError("Only GCT 1.2 is supported by this importer; convert 1.3 explicitly")
        shape = handle.readline().strip().split()
        if len(shape) != 2:
            raise ValueError("GCT 1.2 dimensions must contain gene and profile counts")
        n_genes, n_profiles = map(int, shape)
        if n_genes < 1 or n_profiles < 1:
            raise ValueError("GCT dimensions must be positive")
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
        profile_ids = tuple(header[2:])
        if (
            len(profile_ids) != n_profiles
            or len(set(profile_ids)) != n_profiles
            or any(not identifier.strip() for identifier in profile_ids)
        ):
            raise ValueError("GCT profile identifiers are duplicated or dimensionally inconsistent")
        rows, observed = [], []
        for line_number, row in enumerate(reader, start=4):
            if len(row) != n_profiles + 2:
                raise ValueError(f"Malformed GCT row at line {line_number}")
            observed.append(row[0])
            try:
                rows.append([float(value) for value in row[2:]])
            except ValueError as error:
                raise ValueError(f"Invalid GCT expression value at line {line_number}") from error
        if len(rows) != n_genes:
            raise ValueError("GCT gene count differs from declared dimensions")
    matrix = _select_genes(np.asarray(rows).T, tuple(observed), genes, allow_reorder)
    return profile_ids, matrix


def read_gctx(
    path: str | Path, genes: GeneManifest, *, allow_reorder: bool = False
) -> tuple[tuple[str, ...], np.ndarray]:
    """Read standard GCTX layout with an optional h5py dependency."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("GCTX import requires h5py; install RIGID's io extra") from error

    def decode(values) -> tuple[str, ...]:
        return tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values
        )

    with h5py.File(path, "r") as handle:
        observed = decode(handle["0/META/ROW/id"][:])
        profile_ids = decode(handle["0/META/COL/id"][:])
        if (
            not profile_ids
            or len(set(profile_ids)) != len(profile_ids)
            or any(not identifier.strip() for identifier in profile_ids)
        ):
            raise ValueError("GCTX profile identifiers must be nonempty and unique")
        source = handle["0/DATA/0/matrix"]
        if source.shape != (len(profile_ids), len(observed)):
            raise ValueError("GCTX matrix must use standard profile-by-gene storage")
        if allow_reorder:
            if len(set(observed)) != len(observed):
                raise ValueError("GCTX contains duplicate gene identifiers")
            lookup = {gene: index for index, gene in enumerate(observed)}
            missing = set(genes.gene_ids) - set(lookup)
            if missing:
                raise ValueError(f"GCTX is missing manifest genes: {sorted(missing)[:5]}")
            selected = np.array([lookup[gene] for gene in genes.gene_ids])
            ordered = np.argsort(selected)
            matrix = source[:, selected[ordered]][:, np.argsort(ordered)]
            observed = genes.gene_ids
        else:
            matrix = source[:]
    return profile_ids, _select_genes(matrix, observed, genes, False)
