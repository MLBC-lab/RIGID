"""NumPy/native parity and zero-padding semantics for packed fingerprints."""

import numpy as np
import pytest

from rigid.native import (
    maximum_training_similarity,
    native_available,
    pack_fingerprints,
    pairwise_tanimoto,
)


def test_packing_padding_and_known_similarity():
    bits = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0]])
    packed = pack_fingerprints(bits)
    np.testing.assert_array_equal(packed[:, 0], [0, 5, 1, 2])
    result = pairwise_tanimoto(packed, packed, backend="python")
    np.testing.assert_allclose(np.diag(result), 1)
    assert result[1, 2] == 0.5
    assert result[1, 3] == 0
    assert result[0, 0] == 1
    np.testing.assert_array_equal(result, result.T)


def test_python_kernel_matches_independent_binary_sets():
    rng = np.random.default_rng(31)
    bits_a, bits_b = rng.integers(0, 2, (17, 129)), rng.integers(0, 2, (11, 129))
    actual = pairwise_tanimoto(
        pack_fingerprints(bits_a), pack_fingerprints(bits_b), backend="python"
    )
    expected = np.array(
        [[np.logical_and(a, b).sum() / np.logical_or(a, b).sum() for b in bits_b] for a in bits_a]
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape_a,shape_b", [((0, 4), (3, 4)), ((2, 4), (0, 4)), ((0, 0), (0, 0))])
def test_zero_rows(shape_a, shape_b):
    output = pairwise_tanimoto(
        np.zeros(shape_a, dtype=np.uint8), np.zeros(shape_b, dtype=np.uint8), backend="python"
    )
    assert output.shape == (shape_a[0], shape_b[0])


def test_zero_bit_width():
    np.testing.assert_array_equal(
        pairwise_tanimoto(
            np.zeros((3, 0), dtype=np.uint8), np.zeros((2, 0), dtype=np.uint8), backend="python"
        ),
        np.ones((3, 2)),
    )


def test_blockwise_training_maximum():
    packed = pack_fingerprints([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    result = maximum_training_similarity(packed, packed[1:], block_size=1, backend="python")
    np.testing.assert_allclose(result, [0.5, 1, 1])
    with pytest.raises(ValueError, match="empty training"):
        maximum_training_similarity(packed, packed[:0])


@pytest.mark.parametrize("values", [[[1, 2, 0]], [[float("nan")]], [1, 0]])
def test_packing_rejects_non_binary(values):
    with pytest.raises(ValueError):
        pack_fingerprints(values)


def test_invalid_packed_and_backend_inputs():
    with pytest.raises(ValueError):
        pairwise_tanimoto(np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        pairwise_tanimoto(np.zeros((1, 2), np.uint8), np.zeros((1, 3), np.uint8))
    with pytest.raises(ValueError):
        pairwise_tanimoto(np.zeros((1, 2), np.uint8), np.zeros((1, 2), np.uint8), backend="gpu")


def test_explicit_missing_native_library_is_not_silently_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGID_NATIVE_LIBRARY", str(tmp_path / "missing-kernel.so"))
    assert not native_available()
    with pytest.raises(FileNotFoundError):
        pairwise_tanimoto(np.zeros((1, 2), np.uint8), np.zeros((1, 2), np.uint8), backend="auto")


@pytest.mark.skipif(not native_available(), reason="Optional C++ kernel is not built or configured")
def test_native_matches_python_including_noncontiguous_input():
    rng = np.random.default_rng(102)
    a = rng.integers(0, 256, (51, 32), dtype=np.uint8)[::2, ::2]
    b = rng.integers(0, 256, (35, 16), dtype=np.uint8)
    np.testing.assert_array_equal(
        pairwise_tanimoto(a, b, backend="native"), pairwise_tanimoto(a, b, backend="python")
    )
    np.testing.assert_array_equal(
        pairwise_tanimoto(np.zeros((2, 0), np.uint8), np.zeros((3, 0), np.uint8), backend="native"),
        np.ones((2, 3)),
    )
    assert pairwise_tanimoto(a[:0], b, backend="native").shape == (0, 35)
