"""Optional C++ packed-fingerprint Tanimoto kernel with a NumPy fallback."""

import ctypes
import os
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def pack_fingerprints(fingerprints: ArrayLike) -> np.ndarray:
    """Pack a binary rows-by-bits matrix; unused padding bits are zero."""
    bits = np.asarray(fingerprints)
    if bits.ndim != 2 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("fingerprints must be a binary rows-by-bits matrix")
    return np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")


def _packed(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be a 2-D uint8 packed-fingerprint array")
    return np.ascontiguousarray(array)


def _load_library() -> ctypes.CDLL | None:
    explicit = os.environ.get("RIGID_NATIVE_LIBRARY")
    if explicit:
        paths = [Path(explicit).expanduser()]
    else:
        directory = Path(__file__).resolve().parent / "_native"
        paths = [
            directory / name
            for name in ("rigid_kernels.dll", "librigid_kernels.so", "librigid_kernels.dylib")
        ]
    for path in paths:
        if path.is_file():
            library = ctypes.CDLL(str(path.resolve()))
            function = library.rigid_tanimoto_pairwise
            byte_pointer = ctypes.POINTER(ctypes.c_uint8)
            function.argtypes = [
                byte_pointer,
                ctypes.c_size_t,
                byte_pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
            ]
            function.restype = ctypes.c_int
            return library
    if explicit:
        raise FileNotFoundError(f"RIGID_NATIVE_LIBRARY does not exist: {explicit}")
    return None


def native_available() -> bool:
    """Return whether a loadable native kernel is configured."""
    try:
        return _load_library() is not None
    except (OSError, AttributeError):
        return False


def pairwise_tanimoto(a: ArrayLike, b: ArrayLike, *, backend: str = "auto") -> np.ndarray:
    """Pairwise binary Tanimoto; two all-zero fingerprints have similarity 1.

    The Python path handles one query row at a time to avoid an (n,m,bits) tensor.
    The native path writes the same float64 rows-by-rows matrix without copies of
    contiguous uint8 inputs. An explicitly configured broken library raises.
    """
    left, right = _packed(a, "a"), _packed(b, "b")
    if left.shape[1] != right.shape[1]:
        raise ValueError("Packed fingerprint widths differ")
    if backend not in {"auto", "python", "native"}:
        raise ValueError("backend must be auto, python, or native")
    library = None if backend == "python" else _load_library()
    if backend == "native" and library is None:
        raise RuntimeError("Native kernel unavailable; build with CMake or use backend='python'")
    result = np.empty((len(left), len(right)), dtype=np.float64)
    if library is not None:
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        code = library.rigid_tanimoto_pairwise(
            left.ctypes.data_as(byte_pointer),
            len(left),
            right.ctypes.data_as(byte_pointer),
            len(right),
            left.shape[1],
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if code:
            raise RuntimeError(f"Native Tanimoto kernel failed with error code {code}")
        return result
    for row, fingerprint in enumerate(left):
        intersection = _POPCOUNT[np.bitwise_and(fingerprint, right)].sum(axis=1, dtype=np.uint64)
        union = _POPCOUNT[np.bitwise_or(fingerprint, right)].sum(axis=1, dtype=np.uint64)
        result[row] = np.divide(intersection, union, out=np.ones(len(right)), where=union != 0)
    return result


def maximum_training_similarity(
    candidates: ArrayLike,
    training: ArrayLike,
    *,
    backend: str = "auto",
    block_size: int = 512,
) -> np.ndarray:
    """Maximum training-set similarity with bounded pairwise-matrix memory."""
    from .evaluation._validation import positive_integer

    queries, references = _packed(candidates, "candidates"), _packed(training, "training")
    positive_integer(block_size, "block_size")
    if not len(references):
        raise ValueError("Maximum training similarity is undefined for an empty training set")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("Packed fingerprint widths differ")
    maxima = np.zeros(len(queries))
    for q in range(0, len(queries), block_size):
        for t in range(0, len(references), block_size):
            similarities = pairwise_tanimoto(
                queries[q : q + block_size], references[t : t + block_size], backend=backend
            )
            maxima[q : q + block_size] = np.maximum(
                maxima[q : q + block_size], similarities.max(axis=1)
            )
    return maxima
