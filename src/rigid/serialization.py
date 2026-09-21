"""Strict JSON boundaries for configuration and recorded experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite floating-point range")
    return number


def read_json(path: str | Path, *, maximum_bytes: int | None = None) -> Any:
    """Read UTF-8 JSON, rejecting ambiguous keys and JavaScript numeric extensions."""
    source = Path(path)
    if maximum_bytes is not None:
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be a positive integer")
        with source.open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes:
            raise ValueError(f"JSON file exceeds {maximum_bytes} bytes: {source.name}")
    else:
        raw = source.read_bytes()
    try:
        return json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=_float,
        )
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON in {source.name}: {error}") from error


def require_object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value
