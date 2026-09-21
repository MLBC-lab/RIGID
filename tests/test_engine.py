"""Recovery, selection isolation, and failure behavior of the training engine."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from rigid.checkpoint import load_checkpoint
from rigid.config import TrainConfig
from rigid.data import (
    CompoundRecord,
    GeneManifest,
    ProfileDataset,
    ProfileRecord,
    SourceProvenance,
    save_dataset,
)
from rigid.engine import TrainingCancelled, _writer_lock, fit_model
from rigid.provenance import atomic_json
from rigid.training import seed_everything, train


def settings(**changes):
    return replace(
        TrainConfig(
            epochs=4,
            forward_epochs=3,
            batch_size=2,
            hidden_dim=8,
            depth=1,
            dropout=0.3,
            patience=8,
            num_threads=1,
        ),
        **changes,
    )


def toy(config, *, interrupt_epoch=None, loss_override=None):
    seed_everything(config)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(config.dropout), nn.Tanh(), nn.Linear(8, 1))
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 20
    y = x.sum(-1, keepdim=True) / 2
    observations = []
    batch_calls = 0

    def loss(indices, is_training, epoch):
        nonlocal batch_calls
        observations.append((tuple(indices), is_training, torch.is_grad_enabled(), model.training))
        if epoch == interrupt_epoch and is_training:
            batch_calls += 1
            if batch_calls == 2:
                raise KeyboardInterrupt("simulated interruption after an optimizer step")
        if loss_override is not None:
            replacement = loss_override(model, is_training)
            if replacement is not None:
                return replacement
        prediction = model(x[indices])
        if is_training:
            noise = torch.randn_like(prediction) * 0.01
            noise += float(np.random.normal()) * 0.01 + random.random() * 0.01
            prediction = prediction + noise
        return nn.functional.mse_loss(prediction, y[indices])

    return model, loss, observations


def losses(history):
    return [(row["epoch"], row["train_loss"], row["validation_loss"]) for row in history]


def call_fit(model, loss, config, path=None, **kwargs):
    return fit_model(
        model,
        list(range(6)),
        [6, 7],
        loss,
        config,
        config.epochs,
        "toy",
        state_directory=path,
        protocol={"dataset_sha256": "toy-v1", "objective": "mse"},
        **kwargs,
    )


def test_resume_matches_uninterrupted_dropout_sampling_and_rng(tmp_path):
    config = settings()
    reference, reference_loss, _ = toy(config)
    expected = call_fit(reference, reference_loss, config, tmp_path / "reference")
    expected_random = (random.random(), float(np.random.random()), torch.rand(3))

    interrupted, interrupt_loss, _ = toy(config, interrupt_epoch=1)
    with pytest.raises(KeyboardInterrupt):
        call_fit(interrupted, interrupt_loss, config, tmp_path / "resumed")
    status = json.loads((tmp_path / "resumed/heartbeat.json").read_text())
    assert status["event"] == "cancelled" and status["completed_epochs"] == 1
    resumed, resumed_loss, observations = toy(replace(config, seed=99))
    actual = call_fit(resumed, resumed_loss, config, tmp_path / "resumed", resume=True)
    assert losses(actual) == losses(expected)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    actual_random = (random.random(), float(np.random.random()), torch.rand(3))
    assert expected_random[:2] == actual_random[:2]
    torch.testing.assert_close(expected_random[2], actual_random[2], rtol=0, atol=0)
    assert all(grad == training == mode for _, training, grad, mode in observations)
    assert all(
        set(indices) <= set(range(6)) for indices, training, _, _ in observations if training
    )
    assert len(list((tmp_path / "resumed").glob("state-*.pt"))) == 2
    events = [
        json.loads(line) for line in (tmp_path / "resumed/events.jsonl").read_text().splitlines()
    ]
    assert [
        event["completed_epochs"] for event in events if event["event"] == "epoch_completed"
    ] == [1, 2, 3, 4]


def test_completed_phase_is_reused_without_loss_evaluation(tmp_path):
    config = settings(epochs=2)
    model, loss, _ = toy(config)
    expected = call_fit(model, loss, config, tmp_path)
    restored, _, _ = toy(config)

    def forbidden(*args):
        raise AssertionError("Completed phase must not recompute losses")

    actual = call_fit(restored, forbidden, config, tmp_path, resume=True)
    assert actual == expected
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["config", "data", "partition", "model", "dropout"])
def test_resume_refuses_changed_protocol(tmp_path, change):
    config = settings(epochs=1)
    model, loss, _ = toy(config)
    call_fit(model, loss, config, tmp_path)
    protocol = {"dataset_sha256": "toy-v1", "objective": "mse"}
    training = list(range(6))
    if change == "config":
        config = replace(config, learning_rate=0.002)
    elif change == "data":
        protocol["dataset_sha256"] = "toy-v2"
    elif change == "partition":
        training.reverse()
    elif change == "model":
        model = nn.Linear(4, 1)
    else:
        model[1].p = 0.7
    with pytest.raises(ValueError, match="protocol mismatch"):
        fit_model(
            model,
            training,
            [6, 7],
            loss,
            config,
            1,
            "toy",
            state_directory=tmp_path,
            resume=True,
            protocol=protocol,
        )


@pytest.mark.parametrize(
    "kind", ["nan-training", "nan-validation", "vector", "detached", "unconnected", "gradient"]
)
def test_invalid_losses_restore_last_boundary_and_record_failure(tmp_path, kind):
    config = settings(epochs=1)

    def invalid(model, training):
        parameter = next(model.parameters())
        if kind == "nan-training" and training:
            return parameter.sum() * float("nan")
        if kind == "nan-validation" and not training:
            return torch.tensor(float("nan"))
        if kind == "vector" and training:
            return parameter[0]
        if kind == "detached" and training:
            return torch.tensor(1.0)
        if kind == "unconnected" and training:
            return torch.tensor(1.0, requires_grad=True)
        if kind == "gradient" and training:
            return (parameter.sum() * 0).sqrt()
        return None

    model, loss, _ = toy(config, loss_override=invalid)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises((ValueError, FloatingPointError, RuntimeError)):
        call_fit(model, loss, config, tmp_path)
    status = json.loads((tmp_path / "heartbeat.json").read_text())
    assert status["event"] == "failed" and status["completed_epochs"] == 0
    for name, value in initial.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "training,validation",
    [([], [1]), ([0], []), ([0], [0]), ([0, 0], [1]), ([0], [1, 1]), ([-1], [2]), ([True], [2])],
)
def test_invalid_partitions_are_rejected(training, validation):
    config = settings()
    model, loss, _ = toy(config)
    with pytest.raises(ValueError):
        fit_model(model, training, validation, loss, config, 1, "toy")


def test_cancellation_preserves_committed_epoch(tmp_path):
    config = settings()
    model, loss, _ = toy(config)
    cancelled = False

    def after_epoch(record):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(TrainingCancelled):
        call_fit(
            model, loss, config, tmp_path, should_cancel=lambda: cancelled, on_epoch=after_epoch
        )
    pointer = json.loads((tmp_path / "latest.json").read_text())
    state = load_checkpoint(tmp_path / pointer["checkpoint"])
    assert state["completed_epochs"] == 1 and not state["complete"]


def test_early_stopping_patience_is_restored(tmp_path):
    config = settings(epochs=8, patience=2)
    model, original, _ = toy(config)

    def constant_validation(indices, training, epoch):
        return original(indices, training, epoch) if training else torch.tensor(1.0)

    def interrupt(record):
        if record["epoch"] == 2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        call_fit(model, constant_validation, config, tmp_path, on_epoch=interrupt)
    result = call_fit(model, constant_validation, config, tmp_path, resume=True)
    assert len(result) == 3
    assert json.loads((tmp_path / "heartbeat.json").read_text())["early_stopped"] is True


def test_snapshot_integrity_and_pointer_path_are_checked(tmp_path):
    config = settings(epochs=1)
    model, loss, _ = toy(config)
    call_fit(model, loss, config, tmp_path)
    pointer = json.loads((tmp_path / "latest.json").read_text())
    checkpoint = tmp_path / pointer["checkpoint"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        call_fit(model, loss, config, tmp_path, resume=True)
    atomic_json(tmp_path / "latest.json", {**pointer, "checkpoint": "../outside.pt"})
    with pytest.raises(ValueError, match="pointer"):
        call_fit(model, loss, config, tmp_path, resume=True)


def test_only_one_writer_can_own_state_directory(tmp_path):
    with _writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another trainer"):
            with _writer_lock(tmp_path):
                pass
    with _writer_lock(tmp_path):
        pass


def _dataset():
    entries = [
        ("a", "CC", "train"),
        ("b", "CO", "train"),
        ("c", "CCC", "validation"),
        ("d", "CCCC", "test"),
    ]
    profiles = tuple(
        ProfileRecord(f"{cid}_{rep}", cid, "synthetic", 1, 24, split)
        for cid, _, split in entries
        for rep in range(2)
    )
    return ProfileDataset(
        GeneManifest(("g1", "g2", "g3", "g4"), "synthetic"),
        tuple(CompoundRecord(cid, smiles, ("synthetic",)) for cid, smiles, _ in entries),
        profiles,
        np.random.default_rng(10).normal(size=(len(profiles), 4)).astype(np.float32),
        SourceProvenance(kind="synthetic"),
    )


@pytest.mark.integration
def test_full_pipeline_resumes_across_phases_without_changing_parameters(tmp_path, monkeypatch):
    import rigid.training as training_module

    dataset = _dataset()
    save_dataset(dataset, tmp_path / "data")
    config = settings(epochs=2, forward_epochs=2, candidate_pool_size=4)
    train(tmp_path / "data", tmp_path / "reference", config)
    original_fit = training_module._fit
    interrupted = False

    def interrupt_fit(model, training, validation, loss, config, epochs, phase, **kwargs):
        def wrapped(indices, is_training, epoch):
            nonlocal interrupted
            if phase == "independent" and epoch == 1 and is_training and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("interrupt independent phase")
            return loss(indices, is_training, epoch)

        return original_fit(model, training, validation, wrapped, config, epochs, phase, **kwargs)

    monkeypatch.setattr(training_module, "_fit", interrupt_fit)
    with pytest.raises(KeyboardInterrupt):
        train(tmp_path / "data", tmp_path / "resumed", config)
    monkeypatch.setattr(training_module, "_fit", original_fit)
    actual = train(tmp_path / "data", tmp_path / "resumed", config, resume=True)
    assert actual["optimization_state"]["completed_phases"] == ["forward", "independent", "ranker"]
    expected_payload = load_checkpoint(tmp_path / "reference/model.pt")
    actual_payload = load_checkpoint(tmp_path / "resumed/model.pt")
    for name, record in expected_payload["models"].items():
        for parameter, value in record["state_dict"].items():
            torch.testing.assert_close(
                value, actual_payload["models"][name]["state_dict"][parameter], rtol=0, atol=0
            )
    reference_history = json.loads((tmp_path / "reference/history.json").read_text())
    actual_history = json.loads((tmp_path / "resumed/history.json").read_text())
    assert losses(actual_history) == losses(reference_history)
    before = (tmp_path / "resumed/model.pt").read_bytes()
    assert train(tmp_path / "data", tmp_path / "resumed", config, resume=True) == json.loads(
        json.dumps(actual)
    )
    assert (tmp_path / "resumed/model.pt").read_bytes() == before
    with pytest.raises(ValueError, match="protocol mismatch"):
        train(tmp_path / "data", tmp_path / "resumed", replace(config, epochs=3), resume=True)
    modified = replace(dataset, expression=dataset.expression + 1)
    save_dataset(modified, tmp_path / "different-data")
    with pytest.raises(ValueError, match="protocol mismatch"):
        train(tmp_path / "different-data", tmp_path / "resumed", config, resume=True)


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["vae", "generator"])
def test_optional_trainers_resume_exactly_and_guard_options(tmp_path, monkeypatch, kind):
    import rigid.experiments as experiments

    save_dataset(_dataset(), tmp_path / "data")
    config = settings(epochs=2, forward_epochs=1)
    trainer = experiments.train_vae if kind == "vae" else experiments.train_generator
    options = {"fingerprint_bits": 32, "latent_dim": 4} if kind == "vae" else {"max_atoms": 8}
    trainer(tmp_path / "data", tmp_path / "reference", config, **options)
    original_fit = experiments._fit

    def interrupt_fit(model, training, validation, loss, config, epochs, phase, **kwargs):
        def wrapped(indices, is_training, epoch):
            if epoch == 1 and is_training:
                raise KeyboardInterrupt("interrupt optional trainer")
            return loss(indices, is_training, epoch)

        return original_fit(model, training, validation, wrapped, config, epochs, phase, **kwargs)

    monkeypatch.setattr(experiments, "_fit", interrupt_fit)
    with pytest.raises(KeyboardInterrupt):
        trainer(tmp_path / "data", tmp_path / "resumed", config, **options)
    monkeypatch.setattr(experiments, "_fit", original_fit)
    trainer(tmp_path / "data", tmp_path / "resumed", config, resume=True, **options)
    reference = load_checkpoint(tmp_path / "reference/model.pt")
    resumed = load_checkpoint(tmp_path / "resumed/model.pt")
    for name, value in reference["state_dict"].items():
        torch.testing.assert_close(value, resumed["state_dict"][name], rtol=0, atol=0)
    expected_history = json.loads((tmp_path / "reference/history.json").read_text())
    resumed_history = json.loads((tmp_path / "resumed/history.json").read_text())
    assert losses(expected_history) == losses(resumed_history)
    changed_options = {**options, "beta": 0.1} if kind == "vae" else {"max_atoms": 9}
    with pytest.raises(ValueError, match="protocol mismatch"):
        trainer(tmp_path / "data", tmp_path / "resumed", config, resume=True, **changed_options)
