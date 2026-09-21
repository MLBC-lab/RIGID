"""Validated, serializable experiment settings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .serialization import read_json, require_object


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    epochs: int = 30
    forward_epochs: int = 20
    batch_size: int = 16
    candidate_pool_size: int = 64
    mechanism_candidates: int = 2
    scaffold_candidates: int = 2
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 5.0
    patience: int = 8
    moa_relevance: float = 0.25
    stress_weight: float = 0.2
    novelty_weight: float = 0.05
    top_k: int = 50
    device: str = "cpu"
    num_threads: int = 1
    deterministic: bool = True
    acyclic_policy: str = "group"

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "forward_epochs",
            "batch_size",
            "candidate_pool_size",
            "hidden_dim",
            "depth",
            "patience",
            "top_k",
            "num_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four for the transformer control")
        if self.candidate_pool_size < 2:
            raise ValueError("candidate_pool_size must be at least two")
        for name in ("mechanism_candidates", "scaffold_candidates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in (
            "dropout",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "moa_relevance",
            "stress_weight",
            "novelty_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.dropout >= 1 or self.moa_relevance > 1:
            raise ValueError("dropout must be < 1 and moa_relevance <= 1")
        if not isinstance(self.device, str):
            raise ValueError("device must be a string")
        import re

        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must be cpu, cuda[:index], or mps")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        if not isinstance(self.acyclic_policy, str) or self.acyclic_policy not in {
            "group",
            "compound",
            "exclude",
        }:
            raise ValueError("acyclic_policy must be group, compound, or exclude")

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        values = require_object(read_json(path, maximum_bytes=1024 * 1024), context="configuration")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
