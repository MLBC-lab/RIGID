import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from rigid.checkpoint import inspect_checkpoint, load_checkpoint, save_checkpoint
from rigid.provenance import atomic_json
from rigid.serialization import read_json
from rigid.storage import staged_file, write_text


@pytest.mark.parametrize(
    "payload",
    ['{"x":1,"x":2}', '{"a":{"b":1,"b":2}}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e9999}', "{"],
)
def test_strict_json_rejects_ambiguous_or_nonfinite_input(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        read_json(path)


def test_json_budget_and_bom(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'\xef\xbb\xbf{"value": 3}')
    assert read_json(path) == {"value": 3}
    with pytest.raises(ValueError, match="exceeds"):
        read_json(path, maximum_bytes=2)


def test_atomic_output_competing_writers(tmp_path):
    path = tmp_path / "record.json"

    def publish(index):
        try:
            atomic_json(path, {"writer": index, "payload": str(index) * 10000}, overwrite=False)
            return index
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = [result for result in pool.map(publish, range(8)) if result is not None]
    assert len(winners) == 1
    value = read_json(path)
    assert value == {"writer": winners[0], "payload": str(winners[0]) * 10000}
    assert not list(tmp_path.glob("*.tmp"))


def test_aborted_stage_does_not_publish(tmp_path):
    target = tmp_path / "result.txt"
    with pytest.raises(RuntimeError):
        with staged_file(target) as staging:
            staging.write_text("partial")
            raise RuntimeError("cancelled")
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_nonoverwrite_and_atomic_replace(tmp_path):
    target = tmp_path / "result.txt"
    write_text(target, "first")
    with pytest.raises(FileExistsError):
        write_text(target, "second")
    assert target.read_text() == "first"
    write_text(target, "second", overwrite=True)
    assert target.read_text() == "second"


def test_checkpoint_competing_writers_remain_consistent(tmp_path):
    torch = pytest.importorskip("torch")
    target = tmp_path / "model.pt"

    def publish(index):
        try:
            save_checkpoint(target, {"schema_version": 1, "weights": torch.tensor([index])})
            return index
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = [result for result in pool.map(publish, range(4)) if result is not None]
    assert len(winners) == 1
    assert load_checkpoint(target)["weights"].item() == winners[0]
    assert inspect_checkpoint(target)["integrity_verified"]


@pytest.mark.parametrize(
    "bad_record",
    [
        {},
        {"schema_version": True},
        {"schema_version": 1, "format": "pickle", "sha256": "0" * 64},
        {"schema_version": 1, "format": "torch-state-dict", "sha256": "invalid"},
    ],
)
def test_checkpoint_integrity_schema_is_checked_without_loading(tmp_path, bad_record):
    target = tmp_path / "model.pt"
    target.with_suffix(".pt.sha256.json").write_text(json.dumps(bad_record))
    with pytest.raises(ValueError):
        inspect_checkpoint(target)
