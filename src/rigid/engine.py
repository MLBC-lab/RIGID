"""Single-process optimization with recoverable, checksummed epoch boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .provenance import atomic_json, sha256_file
from .serialization import read_json, require_object

LOGGER = logging.getLogger(__name__)
LossFunction = Callable[[list[int], bool, int], torch.Tensor]
EpochObserver = Callable[[Mapping[str, Any]], None]
_SNAPSHOT_NAME = re.compile(r"state-[0-9]{8}-[0-9a-f]{16}\.pt")


class TrainingCancelled(RuntimeError):
    """Optimization was cancelled; the last completed epoch remains resumable."""


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    """Hash a canonical JSON protocol; nonfinite and nonserializable values fail."""
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _rng_state(shuffle: np.random.Generator, device: str) -> dict:
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "shuffle": copy.deepcopy(shuffle.bit_generator.state),
        "torch": torch.get_rng_state().clone(),
    }
    if device.startswith("cuda"):
        result["cuda"] = [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
    if device == "mps":
        result["mps"] = torch.mps.get_rng_state().cpu().clone()
    return result


def _restore_rng(state: dict, shuffle: np.random.Generator) -> None:
    random.setstate(state["python"])
    algorithm, values, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (algorithm, np.asarray(values, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )
    shuffle.bit_generator.state = state["shuffle"]
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


@contextmanager
def _writer_lock(directory: Path) -> Iterator[None]:
    """An OS lock is released on process death; the lock file itself may remain."""
    with (directory / ".writer.lock").open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another trainer is writing {directory}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _StateStore:
    def __init__(self, directory: Path, phase: str) -> None:
        self.directory = directory
        self.phase = phase
        self.pointer = directory / "latest.json"

    def read(self) -> dict:
        pointer = require_object(read_json(self.pointer), context="Training checkpoint pointer")
        name = pointer.get("checkpoint", "")
        if pointer.get("schema_version") != 1 or not _SNAPSHOT_NAME.fullmatch(name):
            raise ValueError("Invalid training checkpoint pointer")
        path = self.directory / name
        if path.is_symlink() or sha256_file(path) != pointer.get("sha256"):
            raise ValueError("Training checkpoint checksum mismatch")
        state = load_checkpoint(path)
        if state.get("kind") != "optimization-state":
            raise ValueError("Expected an optimization-state checkpoint")
        return state

    def commit(self, state: dict) -> None:
        name = f"state-{state['completed_epochs']:08d}-{uuid.uuid4().hex[:16]}.pt"
        path = self.directory / name
        previous = None
        if self.pointer.exists():
            previous = require_object(
                read_json(self.pointer), context="Training checkpoint pointer"
            ).get("checkpoint")
        save_checkpoint(path, state)
        atomic_json(
            self.pointer,
            {"schema_version": 1, "checkpoint": name, "sha256": sha256_file(path)},
        )
        # Retain the active and previous generation. Unreferenced snapshots are disposable.
        for candidate in self.directory.iterdir():
            if (
                _SNAPSHOT_NAME.fullmatch(candidate.name)
                and candidate.name not in {name, previous}
                and not candidate.is_symlink()
            ):
                try:
                    candidate.unlink()
                    candidate.with_suffix(".pt.sha256.json").unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove obsolete optimization snapshot %s", candidate)

    def event(self, event: str, completed_epochs: int, **details: Any) -> None:
        record = {
            "schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "phase": self.phase,
            "event": event,
            "completed_epochs": completed_epochs,
            **details,
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.directory / "heartbeat.json", record)


def optimization_step(
    loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, config: TrainConfig
) -> float:
    """Apply one finite scalar training loss and reject nonfinite gradients."""
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("Training loss must be a scalar tensor")
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss; inspect input data and learning rate")
    if not loss.requires_grad:
        raise ValueError("Training loss must be connected to a gradient graph")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Training loss is not connected to any model parameter")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Optimizer produced nonfinite model parameters")
    return float(loss.detach())


def _validate_partitions(training: Sequence[int], validation: Sequence[int]) -> None:
    for name, indices in (("training", training), ("validation", validation)):
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError(f"{name} indices must be nonempty nonnegative integers")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate {name} indices")
    if set(training) & set(validation):
        raise ValueError("Training and validation partitions must be disjoint")


def fit_model(
    model: nn.Module,
    train_indices: list[int],
    validation_indices: list[int],
    loss_function: LossFunction,
    config: TrainConfig,
    epochs: int,
    phase: str,
    *,
    state_directory: str | Path | None = None,
    resume: bool = False,
    protocol: Mapping[str, Any] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_epoch: EpochObserver | None = None,
) -> list[dict]:
    """Fit using training gradients and validation-only early stopping.

    Resume repeats any unfinished epoch from its committed boundary, including
    Python, NumPy, Torch, and shuffle RNG state. The loss callback must not mutate
    external state or use private RNGs unless that state is derived from the
    supplied epoch/index. ``on_epoch`` is observational and must not consume RNGs.
    """
    _validate_partitions(train_indices, validation_indices)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be nonempty")
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    if state_directory is None and resume:
        raise ValueError("resume requires a state_directory")
    if state_directory is not None and not protocol:
        raise ValueError("Persistent training requires an explicit data/objective protocol")
    contract = {
        "engine_version": 1,
        "phase": phase,
        "epochs": epochs,
        "configuration": config.to_dict(),
        "training_indices": train_indices,
        "validation_indices": validation_indices,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_representation": repr(model),
        "model_config": getattr(model, "config", None),
        "model_structure": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()
        },
        "trainable_parameters": [
            name for name, value in model.named_parameters() if value.requires_grad
        ],
        "torch_version": str(torch.__version__),
        "runtime": {
            "num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "protocol": dict(protocol or {}),
    }
    fingerprint = protocol_fingerprint(contract)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    shuffle = np.random.default_rng(config.seed)

    def run(store: _StateStore | None) -> list[dict]:
        state = {
            "schema_version": 1,
            "kind": "optimization-state",
            "protocol_fingerprint": fingerprint,
            "protocol": contract,
            "completed_epochs": 0,
            "complete": False,
            "best_loss": None,
            "stale_epochs": 0,
            "best_state": None,
            "model_state": _cpu_copy(model.state_dict()),
            "optimizer_state": _cpu_copy(optimizer.state_dict()),
            "rng_state": _rng_state(shuffle, config.device),
            "history": [],
        }
        if resume:
            assert store is not None
            state = store.read()
            if state.get("protocol_fingerprint") != fingerprint:
                raise ValueError("Resume protocol mismatch: configuration, data, or model changed")
            model.load_state_dict(state["model_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])
            _restore_rng(state["rng_state"], shuffle)
        elif store:
            if store.pointer.exists():
                raise FileExistsError(
                    "Optimization state exists; explicitly resume or use a new path"
                )
            store.commit(state)
        if store:
            store.event("resumed" if resume else "started", state["completed_epochs"])
        if state["complete"]:
            model.load_state_dict(state["best_state"], strict=True)
            model.eval()
            if store:
                store.event("completed", state["completed_epochs"], reused=True)
            return copy.deepcopy(state["history"])

        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise TrainingCancelled(f"{phase} cancelled at an epoch boundary")

        try:
            for epoch in range(state["completed_epochs"], epochs):
                check_cancel()
                started = time.perf_counter()
                last_heartbeat = started
                if store:
                    store.event("epoch_started", epoch, epoch=epoch + 1)
                model.train()
                shuffled = shuffle.permutation(train_indices).tolist()
                train_sum = 0.0
                for offset in range(0, len(shuffled), config.batch_size):
                    check_cancel()
                    indices = shuffled[offset : offset + config.batch_size]
                    loss = loss_function(indices, True, epoch)
                    train_sum += optimization_step(loss, model, optimizer, config) * len(indices)
                    if store and time.perf_counter() - last_heartbeat >= 30:
                        store.event(
                            "training", epoch, epoch=epoch + 1, examples=offset + len(indices)
                        )
                        last_heartbeat = time.perf_counter()
                model.eval()
                optimizer.zero_grad(set_to_none=True)
                validation_sum = 0.0
                with torch.no_grad():
                    for offset in range(0, len(validation_indices), config.batch_size):
                        check_cancel()
                        indices = validation_indices[offset : offset + config.batch_size]
                        loss = loss_function(indices, False, epoch)
                        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                            raise ValueError("Validation loss must be a scalar tensor")
                        value = float(loss)
                        if not math.isfinite(value):
                            raise FloatingPointError("Nonfinite validation loss")
                        validation_sum += value * len(indices)
                        if store and time.perf_counter() - last_heartbeat >= 30:
                            store.event(
                                "validation", epoch, epoch=epoch + 1, examples=offset + len(indices)
                            )
                            last_heartbeat = time.perf_counter()
                validation_loss = validation_sum / len(validation_indices)
                train_loss = train_sum / len(train_indices)
                if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
                    raise FloatingPointError("Nonfinite aggregated epoch loss")
                improved = state["best_loss"] is None or validation_loss < state["best_loss"]
                stale = 0 if improved else state["stale_epochs"] + 1
                record = {
                    "phase": phase,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "seconds": time.perf_counter() - started,
                }
                next_state = {
                    **state,
                    "completed_epochs": epoch + 1,
                    "complete": stale >= config.patience or epoch + 1 == epochs,
                    "best_loss": validation_loss if improved else state["best_loss"],
                    "stale_epochs": stale,
                    "best_state": _cpu_copy(model.state_dict())
                    if improved
                    else state["best_state"],
                    "model_state": _cpu_copy(model.state_dict()),
                    "optimizer_state": _cpu_copy(optimizer.state_dict()),
                    "rng_state": _rng_state(shuffle, config.device),
                    "history": [*state["history"], record],
                }
                if store:
                    store.commit(next_state)
                state = next_state
                if store:
                    store.event("epoch_completed", epoch + 1, **record)
                LOGGER.info(
                    "%s epoch %d/%d train=%.5f validation=%.5f",
                    phase,
                    epoch + 1,
                    epochs,
                    record["train_loss"],
                    validation_loss,
                )
                if on_epoch:
                    on_epoch(copy.deepcopy(record))
                if state["complete"]:
                    break
        except BaseException as exc:
            model.load_state_dict(state["model_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(state["rng_state"], shuffle)
            model.eval()
            if store:
                try:
                    store.event(
                        "cancelled"
                        if isinstance(exc, (KeyboardInterrupt, TrainingCancelled))
                        else "failed",
                        state["completed_epochs"],
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                except OSError:
                    LOGGER.exception("Could not write training failure status")
            raise
        model.load_state_dict(state["best_state"], strict=True)
        model.eval()
        if store:
            store.event(
                "completed",
                state["completed_epochs"],
                early_stopped=state["stale_epochs"] >= config.patience,
            )
        return copy.deepcopy(state["history"])

    if state_directory is None:
        return run(None)
    directory = Path(state_directory)
    directory.mkdir(parents=True, exist_ok=True)
    with _writer_lock(directory):
        return run(_StateStore(directory, phase))
