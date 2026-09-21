"""Shared validation for numerical evaluation inputs."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vector(values: ArrayLike, name: str, *, nonempty: bool = True) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (nonempty and not array.size):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}1-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def same_shape(*arrays: np.ndarray) -> None:
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("Inputs must have the same shape")


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def check_fit_partition(
    split: str,
    expected: str,
    sample_ids: Sequence[str],
    n: int,
    forbidden_ids: Sequence[str],
) -> tuple[str, ...]:
    if split != expected:
        raise ValueError(f"Fit requires split='{expected}', received {split!r}")
    ids = tuple(sample_ids)
    if len(ids) != n or len(set(ids)) != n or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("sample_ids must be unique nonempty strings, one per fit observation")
    overlap = set(ids).intersection(forbidden_ids)
    if overlap:
        raise ValueError(f"Fit partition overlaps forbidden data: {sorted(overlap)[:5]}")
    return ids
