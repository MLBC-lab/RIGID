import json
import subprocess
import sys

import pytest

from rigid.cli import main
from rigid.data import make_synthetic_dataset
from rigid.provenance import atomic_json


def test_help_stays_lightweight_without_ml_imports():
    code = "from rigid.cli import _parser; import sys; _parser(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_structured_error_output_and_nonfinite_config(tmp_path, capsys):
    atomic_json(tmp_path / "bad.json", {"bad_field": True})
    assert (
        main(
            [
                "--json-errors",
                "train",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "run"),
                "--config",
                str(tmp_path / "bad.json"),
            ]
        )
        == 2
    )
    result = capsys.readouterr()
    assert not result.out
    assert json.loads(result.err)["error"]["code"] == "invalid_request"


def test_qc_drift_inspection_cli(tmp_path, capsys):
    make_synthetic_dataset(tmp_path / "data", n_genes=8)
    assert main(["qc", str(tmp_path / "data"), "--output", str(tmp_path / "qc.json")]) == 0
    assert json.loads(capsys.readouterr().out)["source_kind"] == "synthetic"
    assert (
        main(
            [
                "drift",
                str(tmp_path / "data"),
                "--split",
                "test",
                "--output",
                str(tmp_path / "drift.json"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads((tmp_path / "qc.json").read_text())["dataset_kind"] == "synthetic"
    assert json.loads((tmp_path / "drift.json").read_text())["dataset_kind"] == "synthetic"
    before = (tmp_path / "qc.json").read_bytes()
    assert main(["qc", str(tmp_path / "data"), "--output", str(tmp_path / "qc.json")]) == 2
    assert before == (tmp_path / "qc.json").read_bytes()


def test_library_cli_manifest_and_explicit_deduplication(tmp_path, capsys):
    csv = tmp_path / "library.csv"
    csv.write_text("compound_id,smiles,aliases\na,CCO,ethanol\nb,OCC,alternate\n", encoding="utf-8")
    assert main(["inspect-library", str(csv)]) == 2
    capsys.readouterr()
    assert (
        main(
            [
                "inspect-library",
                str(csv),
                "--deduplicate",
                "--output",
                str(tmp_path / "manifest.json"),
            ]
        )
        == 0
    )
    assert (tmp_path / "manifest.json").exists()
    capsys.readouterr()


def test_profile_is_preserved_when_command_raises(tmp_path, monkeypatch, capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr("rigid.cli._execute", interrupted)
    target = tmp_path / "cancelled.prof"
    assert main(["--json-errors", "--profile", str(target), "doctor"]) == 130
    assert target.stat().st_size > 0
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "interrupted"


@pytest.mark.integration
def test_resume_flag_for_completed_generator_cli(tmp_path, capsys):
    pytest.importorskip("torch")
    from rigid.config import TrainConfig

    make_synthetic_dataset(tmp_path / "data", n_genes=4)
    config = TrainConfig(
        epochs=1, forward_epochs=1, batch_size=32, hidden_dim=16, depth=1, dropout=0
    )
    atomic_json(tmp_path / "config.json", config.to_dict())
    args = [
        "train-generator",
        str(tmp_path / "data"),
        "--output",
        str(tmp_path / "run"),
        "--config",
        str(tmp_path / "config.json"),
        "--unconditional",
    ]
    assert main(args) == 0
    capsys.readouterr()
    before = (tmp_path / "run/model.pt").read_bytes()
    assert main([*args, "--resume"]) == 0
    capsys.readouterr()
    assert before == (tmp_path / "run/model.pt").read_bytes()
    assert main(["inspect-run", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint"]["integrity_verified"]
