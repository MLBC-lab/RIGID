"""Portable provenance and atomic artifact writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .storage import write_text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text(path, payload, overwrite=overwrite)


def environment_report() -> dict:
    packages: dict[str, str | None] = {}
    for name in ("rigid-transcriptomics", "numpy", "torch", "rdkit", "h5py", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def git_state(root: str | Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=10, text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def dataset_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    return {p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}


def run_manifest(dataset_path: str | Path, config: dict, *, source_kind: str) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "software": environment_report(),
        "source": git_state(Path(__file__).resolve().parents[2]),
        "input_sha256": dataset_hashes(dataset_path),
        "configuration": config,
        "source_kind": source_kind,
        "reproduces_manuscript": False,
    }
