import hashlib
import io
from urllib.request import Request

import numpy as np
import pytest

from rigid.acquisition import _HTTPSRedirectHandler, fetch_file
from rigid.checkpoint import load_checkpoint, save_checkpoint
from rigid.config import TrainConfig
from rigid.data.schema import GeneManifest, ProfileRecord
from rigid.features import AuxiliarySpec, ContextEncoder, auxiliary_features
from rigid.provenance import atomic_json, sha256_file


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": True},
        {"seed": -1},
        {"dropout": 1},
        {"learning_rate": float("nan")},
        {"hidden_dim": 7},
        {"device": 4},
        {"device": "cuda:banana"},
        {"candidate_pool_size": 1},
        {"gradient_clip": 0},
    ],
)
def test_configuration_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_configuration_strict_json(tmp_path):
    path = tmp_path / "config.json"
    atomic_json(path, TrainConfig().to_dict())
    assert TrainConfig.load(path) == TrainConfig()
    atomic_json(path, {"lerning_rate": 1})
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.load(path)


def test_atomic_json_does_not_destroy_existing_on_nonfinite(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"ok": True})
    before = sha256_file(path)
    with pytest.raises(ValueError):
        atomic_json(path, {"not_ok": float("nan")})
    assert sha256_file(path) == before


def test_checkpoint_integrity_and_nonoverwrite(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"schema_version": 1, "tensor": torch.arange(4)})
    assert torch.equal(load_checkpoint(path)["tensor"], torch.arange(4))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, {"schema_version": 1})
    path.write_bytes(path.read_bytes() + b"damage")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path)


class FakeResponse(io.BytesIO):
    url = "https://example.org/file"


def _fake_network(monkeypatch, payload):
    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse(payload)

    monkeypatch.setattr("rigid.acquisition.build_opener", lambda *args: FakeOpener())


def test_download_integrity_limits_and_private_query_redaction(tmp_path, monkeypatch):
    data = b"tiny fixture"
    _fake_network(monkeypatch, data)
    digest = hashlib.sha256(data).hexdigest()
    destination = tmp_path / "fixture.bin"
    fetch_file("https://example.org/file?token=secret", destination, digest)
    assert destination.read_bytes() == data
    assert "secret" not in destination.with_suffix(".bin.source.json").read_text()
    with pytest.raises(FileExistsError):
        fetch_file("https://example.org/file", destination, digest)
    with pytest.raises(ValueError, match="checksum"):
        fetch_file("https://example.org/file", tmp_path / "bad", "0" * 64)
    with pytest.raises(ValueError, match="size limit"):
        fetch_file("https://example.org/file", tmp_path / "huge", digest, maximum_bytes=2)
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "huge").exists()
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize(
    "url", ["http://example.org/x", "file:///etc/passwd", "https://name:password@example.org/x"]
)
def test_download_rejects_unsafe_transport(tmp_path, url):
    with pytest.raises(ValueError):
        fetch_file(url, tmp_path / "x", "0" * 64)


def test_redirect_rejects_downgrade_before_following():
    handler = _HTTPSRedirectHandler()
    with pytest.raises(ValueError, match="HTTPS"):
        handler.redirect_request(
            Request("https://example.org"), None, 302, "", {}, "http://example.org"
        )


def test_context_uses_training_cells_and_unknown_slot():
    a = ProfileRecord("p1", "c1", "cellA", 1, 24, "train")
    b = ProfileRecord("p2", "c2", "cellB", 10, 48, "test")
    encoder = ContextEncoder.fit([a])
    actual = encoder.transform([a, b])
    np.testing.assert_array_equal(actual[:, :2], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="training"):
        ContextEncoder.fit([a, b])


def test_auxiliary_features_explicit_programs():
    genes = GeneManifest(("a", "b", "c"))
    spec = AuxiliarySpec.from_dict(
        {"programs": {"stress": ["a", "b"]}, "stress_programs": ["stress"], "source": "synthetic"},
        genes,
    )
    actual = auxiliary_features(
        np.array([1.0, 0.0, -1.0]), np.array([[1.0, 0.0, -1.0]]), np.array([0.3]), genes, spec
    )
    np.testing.assert_allclose(actual, [[1, 1, np.tanh(0.5), 0.3]])
    empty = AuxiliarySpec.from_dict({}, genes)
    actual = auxiliary_features(np.zeros(3), np.zeros((1, 3)), np.array([0]), genes, empty)
    np.testing.assert_array_equal(actual, np.zeros((1, 4)))
    with pytest.raises(ValueError):
        AuxiliarySpec.from_dict({"programs": {"x": ["missing"]}, "source": "test"}, genes)


def test_environment_doctor_and_measured_benchmark():
    from rigid.diagnostics import benchmark_fingerprints, doctor

    report = doctor()
    assert report["packages"]["numpy"]
    measured = benchmark_fingerprints(molecules=10, bits=64)
    assert measured["synthetic"] is True
    assert measured["seconds"] > 0
    assert measured["result_checksum"] >= 10
    with pytest.raises(ValueError):
        benchmark_fingerprints(molecules=0)
