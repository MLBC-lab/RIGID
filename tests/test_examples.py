"""The advanced evaluation example runs deterministically without network access."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run_example(seed: int) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    runner = """
import runpy
import socket
import sys

def deny_network(*args, **kwargs):
    raise RuntimeError('The synthetic evaluation example must run without network access')

socket.create_connection = deny_network
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.getaddrinfo = deny_network
sys.argv = ['evaluation_protocol.py', '--seed', sys.argv[1], '--resamples', '100']
runpy.run_path('examples/evaluation_protocol.py', run_name='__main__')
"""
    process = subprocess.run(
        [sys.executable, "-c", runner, str(seed)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(process.stdout)


def test_evaluation_protocol_has_disjoint_fit_partitions_and_finite_outputs():
    result = _run_example(42)
    assert result["data_origin"] == "synthetic"
    assert result["claimed_manuscript_reproduction"] is False
    protocol = result["protocol"]
    partitions = protocol["partitions"]
    ids = [sample_id for group in partitions.values() for sample_id in group]
    assert len(ids) == len(set(ids))
    assert protocol["partition_counts"] == {"validation": 40, "calibration": 160, "test": 64}
    assert result["calibration"]["fit_partition"] == "calibration"
    assert result["calibration"]["evaluation_partition"] == "test"
    assert result["conformal"]["test_coverage"]["denominator"] == len(partitions["test"])
    assert (
        result["retrospective_designability"]["fit"]["validation_ids"] == partitions["validation"]
    )
    assert result["retrospective_designability"]["requires_known_target_labels"] is True
    for contrast in result["paired_comparisons"].values():
        assert contrast["n_targets"] == 64
        assert contrast["n_resamples"] == 100
        assert np.isfinite([contrast["estimate"], contrast["lower"], contrast["upper"]]).all()
    assert 0 <= result["retrospective_designability"]["test_min"] <= 1
    assert 0 <= result["retrospective_designability"]["test_max"] <= 1
    assert len(result["reranked_candidates"]) == 5
    assert result["cohort_stability"]["eligible_pair_count"] == 3


def test_evaluation_protocol_is_reproducible_and_seed_sensitive():
    first = _run_example(31)
    repeated = _run_example(31)
    different = _run_example(32)
    assert first == repeated
    assert first["paired_comparisons"] != different["paired_comparisons"]
