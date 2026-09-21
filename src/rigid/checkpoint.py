"""Checksummed tensor-only checkpoints; never load arbitrary Python objects."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import BinaryIO

from .provenance import atomic_json, sha256_file
from .serialization import read_json, require_object
from .storage import publish_file, staged_file


def _integrity_record(path: Path) -> dict:
    record = require_object(
        read_json(path.with_suffix(path.suffix + ".sha256.json"), maximum_bytes=64 * 1024),
        context="Checkpoint integrity record",
    )
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint checksum/integrity schema")
    if record.get("format") != "torch-state-dict":
        raise ValueError("Unsupported checkpoint integrity format")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Checkpoint integrity record requires a lowercase SHA-256")
    if "bytes" in record and (type(record["bytes"]) is not int or record["bytes"] < 0):
        raise ValueError("Checkpoint byte count must be a nonnegative integer")
    return record


def _verify_stream(stream: BinaryIO, expected: dict) -> str:
    if "bytes" in expected and os.fstat(stream.fileno()).st_size != expected["bytes"]:
        raise ValueError("Checkpoint checksum/size mismatch; refusing to load")
    accumulator = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        accumulator.update(block)
    digest = accumulator.hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("Checkpoint checksum mismatch; refusing to load")
    stream.seek(0)
    return digest


def inspect_checkpoint(path: str | Path) -> dict:
    """Verify integrity without importing PyTorch or deserializing model content."""
    path = Path(path)
    expected = _integrity_record(path)
    with path.open("rb") as stream:
        digest = _verify_stream(stream, expected)
        size = os.fstat(stream.fileno()).st_size
    return {
        "filename": path.name,
        "sha256": digest,
        "bytes": size,
        "format": "torch-state-dict",
        "integrity_verified": True,
    }


def save_checkpoint(path: str | Path, payload: dict) -> None:
    import torch

    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("Checkpoint payload must use schema_version=1")
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + ".sha256.json").exists():
        raise FileExistsError("Checkpoint or integrity sidecar exists; choose a new filename")
    with staged_file(path) as temporary:
        torch.save(payload, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        record = {
            "sha256": sha256_file(temporary),
            "bytes": temporary.stat().st_size,
            "format": "torch-state-dict",
            "schema_version": 1,
        }
        publish_file(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".sha256.json"), record, overwrite=False)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> dict:
    import torch

    path = Path(path)
    expected = _integrity_record(path)
    with path.open("rb") as stream:
        _verify_stream(stream, expected)
        result = torch.load(stream, map_location=device, weights_only=True)
    if (
        not isinstance(result, dict)
        or type(result.get("schema_version")) is not int
        or result["schema_version"] != 1
    ):
        raise ValueError("Unsupported checkpoint schema")
    return result
