"""Read-only run inspection and matched-target screening comparisons."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .checkpoint import inspect_checkpoint
from .provenance import sha256_file
from .serialization import read_json, require_object


def _number(
    value: Any, name: str, *, lower: float | None = None, upper: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} exceeds the finite numeric range") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} is outside its supported range")
    return number


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_screening(path: str | Path) -> dict:
    """Validate a recorded library-screening artifact before analysis or HTML output."""
    record = require_object(read_json(path), context="Screening result")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported screening schema")
    k = _positive_integer(record.get("requested_k"), "requested_k")
    count = _positive_integer(record.get("candidate_count"), "candidate_count")
    if not isinstance(record.get("source_kind"), str) or record["source_kind"] not in {
        "synthetic",
        "experimental",
        "unknown",
    }:
        raise ValueError("Screening source_kind must be synthetic, experimental, or unknown")
    if not isinstance(record.get("split"), str) or not record["split"]:
        raise ValueError("Screening split is missing")
    rankings = require_object(record.get("rankings"), context="rankings")
    policy = None
    if "reranking" in record:
        from .consensus import RerankingConfig

        policy = RerankingConfig.from_dict(record["reranking"])
        if policy.stress_weight is None or policy.novelty_weight is None:
            raise ValueError("Recorded reranking policy must contain resolved weights")
        role = "scoring_component" if policy.mode == "consensus" else "held-out-from-ranking"
        if record.get("independent_predictor_role") != role:
            raise ValueError("Predictor role disagrees with the recorded scoring policy")
    if not rankings:
        raise ValueError("Screening result contains no targets")
    for target, rows in rankings.items():
        if not target or not isinstance(rows, list) or not 0 < len(rows) <= min(k, count):
            raise ValueError(f"Invalid or empty ranking for {target!r}")
        seen, previous = set(), math.inf
        for position, row in enumerate(rows, 1):
            row = require_object(row, context=f"Candidate row for {target}")
            if type(row.get("rank")) is not int or row["rank"] != position:
                raise ValueError(f"Ranks for {target} must be consecutive and one-based")
            cid = row.get("candidate_id")
            if not isinstance(cid, str) or not cid or cid in seen:
                raise ValueError(f"Candidate IDs for {target} must be nonempty and unique")
            seen.add(cid)
            if not isinstance(row.get("smiles"), str) or not row["smiles"]:
                raise ValueError("Candidate SMILES is missing")
            score = _number(row.get("score"), "score")
            if score > previous:
                raise ValueError(f"Scores for {target} are not in descending order")
            previous = score
            for key in ("predicted_profile_correlation", "independent_profile_correlation"):
                _number(row.get(key), key, lower=-1, upper=1)
            if policy is not None:
                assert policy.stress_weight is not None and policy.novelty_weight is not None
                compatibility = _number(row.get("compatibility"), "compatibility")
                primary = row["predicted_profile_correlation"]
                independent = row["independent_profile_correlation"]
                expected_parts = policy.components([compatibility], [primary], [independent])
                for key in (
                    "compatibility_contribution",
                    "consensus_bonus",
                    "disagreement_penalty",
                ):
                    if not math.isclose(
                        _number(row.get(key), key),
                        expected_parts[key][0],
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(f"Recorded {key} differs from the scoring policy")
                stress = row.get("predicted_stress")
                if stress is not None:
                    stress = _number(stress, "predicted_stress", lower=0, upper=1)
                similarity = _number(
                    row.get("max_training_tanimoto"), "max_training_tanimoto", lower=0, upper=1
                )
                stress_penalty = policy.stress_weight * (stress or 0)
                novelty_bonus = policy.novelty_weight * (1 - similarity)
                for key, expected_component in (
                    ("stress_penalty", stress_penalty),
                    ("novelty_bonus", novelty_bonus),
                ):
                    if not math.isclose(
                        _number(row.get(key), key), expected_component, rel_tol=1e-7, abs_tol=1e-9
                    ):
                        raise ValueError(f"Recorded {key} differs from the scoring policy")
                if not math.isclose(
                    score,
                    expected_parts["combined"][0] - stress_penalty + novelty_bonus,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                ):
                    raise ValueError("Recorded score differs from the scoring policy")
            properties = require_object(
                row.get("chemical_properties"), context="chemical_properties"
            )
            _number(properties.get("qed"), "QED", lower=0, upper=1)
    metrics = require_object(record.get("metrics"), context="metrics")
    for name, value in metrics.items():
        if isinstance(value, dict) and "denominator" in value:
            if "value" not in value:
                raise ValueError(f"Rate {name} must explicitly record its value")
            numerator, denominator = value.get("numerator"), value.get("denominator")
            if (
                type(numerator) is not int
                or type(denominator) is not int
                or not 0 <= numerator <= denominator
            ):
                raise ValueError(f"Invalid rate counts for {name}")
            expected = numerator / denominator if denominator else None
            actual = value.get("value")
            if expected is None:
                if actual is not None:
                    raise ValueError(f"Undefined rate {name} must have a null value")
            elif not math.isclose(_number(actual, name), expected, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"Rate value for {name} disagrees with its counts")
    if not isinstance(record.get("notes"), list) or any(
        not isinstance(n, str) for n in record["notes"]
    ):
        raise ValueError("Screening notes must be a list of strings")
    _digest(record.get("checkpoint_sha256"), "checkpoint_sha256")
    if "gene_manifest_sha256" in record:
        _digest(record["gene_manifest_sha256"], "gene_manifest_sha256")
    if "input_sha256" in record:
        for name, digest in require_object(record["input_sha256"], context="input_sha256").items():
            _digest(digest, f"input_sha256.{name}")
    if "target_count" in metrics and _positive_integer(
        metrics["target_count"], "target_count"
    ) != len(rankings):
        raise ValueError("Recorded target_count differs from the rankings")
    if "k" in metrics and _positive_integer(metrics["k"], "metric k") != k:
        raise ValueError("Metric k differs from requested_k")
    if "per_target" not in metrics:
        raise ValueError("Screening metrics must include per_target records")
    if "per_target" in metrics:
        targets = _target_metrics(record)
        for target, row in targets.items():
            if "requested_k" in row and row["requested_k"] != k:
                raise ValueError(f"Target {target} requested_k differs from the screening")
            if "returned" in row and row["returned"] != len(rankings[target]):
                raise ValueError(f"Target {target} returned count differs from its ranking")
        for metric in ("exact", "same_moa", "profile_family"):
            if metric not in metrics:
                continue
            aggregate = require_object(metrics[metric], context=f"Aggregate {metric}")
            eligible = [row[metric] for row in targets.values() if row[metric] is not None]
            if aggregate.get("numerator") != sum(eligible) or aggregate.get("denominator") != len(
                eligible
            ):
                raise ValueError(f"Aggregate {metric} disagrees with per-target counts")
    return record


def inspect_run(directory: str | Path, *, verify_weights: bool = True) -> dict:
    """Summarize optimization records without importing the model implementation."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("Run inspection requires a directory")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = require_object(read_json(manifest_path), context="Run manifest")
        manifest_digest = sha256_file(manifest_path)
    else:
        session_path = root / "training-session.json"
        session = require_object(read_json(session_path), context="Training session")
        if type(session.get("schema_version")) is not int or session["schema_version"] != 1:
            raise ValueError("Unsupported training session schema")
        protocol = require_object(session.get("protocol"), context="Training protocol")
        manifest = {
            "schema_version": 1,
            "configuration": protocol.get("configuration", {}),
            "input_sha256": protocol.get("input_sha256", {}),
        }
        manifest_digest = None
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported run manifest schema")
    history_path = root / "history.json"
    history = read_json(history_path) if history_path.exists() else []
    if not isinstance(history, list):
        raise ValueError("Training history must be a list")
    phases: dict[str, list[dict]] = defaultdict(list)
    for row in history:
        row = require_object(row, context="History row")
        if not isinstance(row.get("phase"), str) or not row["phase"]:
            raise ValueError("History row requires a phase")
        _positive_integer(row.get("epoch"), "epoch")
        _number(row.get("train_loss"), "train_loss")
        _number(row.get("validation_loss"), "validation_loss")
        phases[row["phase"]].append(row)
    summary = {}
    for phase, rows in phases.items():
        epochs = [r["epoch"] for r in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise ValueError(f"Epoch sequence for {phase} is incomplete or duplicated")
        best = min(rows, key=lambda r: r["validation_loss"])
        summary[phase] = {
            "completed_epochs": len(rows),
            "best_epoch": best["epoch"],
            "best_validation_loss": best["validation_loss"],
            "last_train_loss": rows[-1]["train_loss"],
        }
    checkpoint = root / "model.pt"
    weights = (
        inspect_checkpoint(checkpoint)
        if checkpoint.exists() and verify_weights
        else {"exists": checkpoint.exists(), "integrity_verified": False}
    )
    progress = {}
    optimization = root / "optimization"
    if optimization.is_dir() and not optimization.is_symlink():
        for phase_path in sorted(optimization.iterdir()):
            heartbeat = phase_path / "heartbeat.json"
            if phase_path.is_dir() and not phase_path.is_symlink() and heartbeat.is_file():
                event = require_object(
                    read_json(heartbeat, maximum_bytes=1024 * 1024), context="Training heartbeat"
                )
                progress[phase_path.name] = {
                    key: event.get(key)
                    for key in ("event", "timestamp", "completed_epochs", "error_type", "error")
                }
    return {
        "schema_version": 1,
        "run_name": root.name,
        "source_kind": manifest.get("source_kind", "unknown"),
        "created_at": manifest.get("created_at"),
        "source": manifest.get("source"),
        "configuration": manifest.get("configuration", {}),
        "phases": summary,
        "input_sha256": manifest.get("input_sha256", {}),
        "checkpoint": weights,
        "manifest_sha256": manifest_digest,
        "recorded_state": "exported" if checkpoint.exists() else "not_exported",
        "phase_progress": progress,
        "progress_note": "Heartbeat records describe last activity, not current process liveness.",
        "reproduces_manuscript": manifest.get("reproduces_manuscript", False),
    }


def _target_metrics(record: dict) -> dict[str, dict]:
    values = record["metrics"].get("per_target")
    if not isinstance(values, list):
        raise ValueError("Comparison requires recorded per-target metrics")
    result = {}
    for value in values:
        value = require_object(value, context="Per-target metric")
        target = value.get("target_id")
        if not isinstance(target, str) or target in result:
            raise ValueError("Per-target metrics contain a missing or duplicate target ID")
        for field in ("exact", "same_moa", "profile_family"):
            if field not in value:
                raise ValueError(f"Per-target metric must explicitly record {field}")
            if value.get(field) is not None and type(value[field]) is not bool:
                raise ValueError(f"{field} must be a boolean or null")
        result[target] = value
    if set(result) != set(record["rankings"]):
        raise ValueError("Metric and ranking target identifiers differ")
    return result


def compare_screenings(
    left_path: str | Path,
    right_path: str | Path,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> dict:
    """Compare matched inputs and targets; all differences are right minus left."""
    from .evaluation import paired_bootstrap_ci

    if type(n_resamples) is not int or n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Use positive resamples and confidence strictly between zero and one")
    left, right = load_screening(left_path), load_screening(right_path)
    for key in ("input_sha256", "gene_manifest_sha256"):
        if not left.get(key) or not right.get(key):
            raise ValueError(
                f"Comparison requires {key}; rerun legacy screening with provenance enabled"
            )
    for key in (
        "requested_k",
        "candidate_count",
        "split",
        "source_kind",
        "input_sha256",
        "gene_manifest_sha256",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Screening comparison requires identical {key}")
    a, b = _target_metrics(left), _target_metrics(right)
    if set(a) != set(b):
        raise ValueError("Screening runs must cover exactly the same target identifiers")
    comparisons = {}
    for metric in ("exact", "same_moa", "profile_family"):
        if any((a[target][metric] is None) != (b[target][metric] is None) for target in a):
            raise ValueError(
                f"Matched screening inputs have different annotation eligibility for {metric}"
            )
        eligible = [
            t for t in sorted(a) if a[t].get(metric) is not None and b[t].get(metric) is not None
        ]
        result: dict[str, Any] = {
            "eligible_targets": len(eligible),
            "excluded_targets": len(a) - len(eligible),
        }
        if eligible:
            av, bv = [int(a[t][metric]) for t in eligible], [int(b[t][metric]) for t in eligible]
            result.update({"left_mean": sum(av) / len(av), "right_mean": sum(bv) / len(bv)})
        result["right_minus_left"] = (
            asdict(
                paired_bootstrap_ci(
                    bv, av, n_resamples=n_resamples, confidence=confidence, seed=seed
                )
            )
            if len(eligible) >= 2
            else None
        )
        comparisons[metric] = result
    overlap = {}
    for target in sorted(a):
        first = {r["candidate_id"] for r in left["rankings"][target]}
        second = {r["candidate_id"] for r in right["rankings"][target]}
        overlap[target] = len(first & second) / len(first | second)
    return {
        "schema_version": 1,
        "direction": "right minus left",
        "target_count": len(a),
        "source_kind": left["source_kind"],
        "left_sha256": sha256_file(left_path),
        "right_sha256": sha256_file(right_path),
        "requested_k": left["requested_k"],
        "comparisons": comparisons,
        "top_k_jaccard_by_target": overlap,
        "notes": [
            "Bootstrap intervals resample target IDs and assume independent targets.",
            "Replicated targets require a prespecified grouped analysis instead.",
            "Fewer than two eligible targets do not receive an uncertainty interval.",
            "These comparisons do not establish biological efficacy.",
        ],
    }
