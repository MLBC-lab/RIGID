"""CLI smoke tests exercise actual preparation and training entry points."""

import json

import pytest

from rigid.cli import main
from rigid.config import TrainConfig
from rigid.data import make_synthetic_dataset


def test_version_and_missing_command(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--version"])
    assert exited.value.code == 0
    assert "0." in capsys.readouterr().out
    with pytest.raises(SystemExit) as exited:
        main([])
    assert exited.value.code == 2


def test_validate_reports_synthetic_provenance(tmp_path, capsys):
    make_synthetic_dataset(tmp_path / "data", n_genes=4)
    assert main(["validate", str(tmp_path / "data"), "--scaffold"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["source_kind"] == "synthetic"


def test_unknown_config_returns_readable_error(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text('{"typo_epochs": 1}', encoding="utf-8")
    status = main(
        [
            "train",
            str(tmp_path / "missing-data"),
            "--config",
            str(path),
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert status == 2
    assert "unknown configuration keys" in capsys.readouterr().err


@pytest.mark.integration
def test_train_rank_report_cli_workflow(tmp_path, capsys):
    pytest.importorskip("torch")
    make_synthetic_dataset(tmp_path / "data", n_genes=4)
    config = TrainConfig(
        epochs=1,
        forward_epochs=1,
        hidden_dim=16,
        batch_size=8,
        candidate_pool_size=8,
        depth=2,
        dropout=0,
        num_threads=1,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert (
        main(
            [
                "train",
                str(tmp_path / "data"),
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["source_kind"] == "synthetic"
    assert (
        main(
            [
                "rank",
                str(tmp_path / "data"),
                "--checkpoint",
                str(tmp_path / "run/model.pt"),
                "--output",
                str(tmp_path / "screening.json"),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            ["report", str(tmp_path / "screening.json"), "--output", str(tmp_path / "report.html")]
        )
        == 0
    )
    capsys.readouterr()
    assert "synthetic" in (tmp_path / "report.html").read_text(encoding="utf-8")
    before = (tmp_path / "report.html").read_bytes()
    assert (
        main(
            ["report", str(tmp_path / "screening.json"), "--output", str(tmp_path / "report.html")]
        )
        == 2
    )
    assert (tmp_path / "report.html").read_bytes() == before


def test_split_refuses_existing_output(tmp_path, capsys):
    make_synthetic_dataset(tmp_path / "data", n_genes=4)
    before = (tmp_path / "data/profiles.csv").read_bytes()
    assert main(["split", str(tmp_path / "data"), "--output", str(tmp_path / "data")]) == 2
    assert "exists" in capsys.readouterr().err
    assert (tmp_path / "data/profiles.csv").read_bytes() == before


def test_profiling_refuses_existing_artifact(tmp_path, capsys):
    output = tmp_path / "existing.prof"
    output.write_bytes(b"preserve me")
    assert main(["--profile", str(output), "doctor"]) == 2
    assert "exists" in capsys.readouterr().err
    assert output.read_bytes() == b"preserve me"
