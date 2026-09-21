"""Context and predicted-response features shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data.schema import GeneManifest, ProfileRecord


@dataclass(frozen=True)
class ContextEncoder:
    cells: tuple[str, ...]

    @classmethod
    def fit(cls, profiles: list[ProfileRecord]) -> ContextEncoder:
        if not profiles or any(p.split != "train" for p in profiles):
            raise ValueError("Context vocabulary must be fitted on training profiles only")
        return cls(tuple(sorted({p.cell_id for p in profiles})))

    @property
    def dimension(self) -> int:
        return len(self.cells) + 3

    def transform(self, profiles: list[ProfileRecord]) -> np.ndarray:
        lookup = {cell: i for i, cell in enumerate(self.cells)}
        result: NDArray[np.float32] = np.zeros((len(profiles), self.dimension), dtype=np.float32)
        for i, profile in enumerate(profiles):
            result[i, lookup.get(profile.cell_id, len(self.cells))] = 1
            result[i, -2:] = np.log1p([profile.dose_um, profile.time_h])
        return result


@dataclass(frozen=True)
class AuxiliarySpec:
    programs: dict[str, tuple[str, ...]]
    stress_programs: tuple[str, ...]
    source: str

    @classmethod
    def from_dict(cls, value: dict, genes: GeneManifest) -> AuxiliarySpec:
        if not isinstance(value, dict):
            raise ValueError("Program specification must be an object")
        unknown = set(value) - {"programs", "stress_programs", "source"}
        if unknown:
            raise ValueError(f"Unknown program specification fields: {sorted(unknown, key=str)}")
        definitions = value.get("programs", {})
        if not isinstance(definitions, dict):
            raise ValueError("Program definitions must be an object mapping names to gene lists")
        programs = {}
        for name, members in definitions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Program names must be nonempty strings")
            if not isinstance(members, (list, tuple)) or any(
                not isinstance(member, str) or not member.strip() for member in members
            ):
                raise ValueError(f"Program {name} requires a list of nonempty gene identifiers")
            if not members or len(set(members)) != len(members):
                raise ValueError("Each program requires unique, nonempty gene members")
            if set(members) - set(genes.gene_ids):
                raise ValueError(f"Program {name} contains genes outside the model manifest")
            programs[name] = tuple(members)
        selected_stress = value.get("stress_programs", ())
        if not isinstance(selected_stress, (list, tuple)) or any(
            not isinstance(name, str) or not name.strip() for name in selected_stress
        ):
            raise ValueError("stress_programs must be a list of nonempty program names")
        if len(set(selected_stress)) != len(selected_stress):
            raise ValueError("stress_programs must not contain duplicate names")
        stress = tuple(selected_stress)
        source = value.get("source", "No gene programs supplied" if not programs else "")
        if not isinstance(source, str):
            raise ValueError("Program source must be a string")
        if programs and not source.strip():
            raise ValueError("Gene program definitions require a source")
        if set(stress) - set(programs):
            raise ValueError("Stress programs must be named in the program definitions")
        return cls(programs, stress, source)

    def to_dict(self) -> dict:
        return {
            "programs": self.programs,
            "stress_programs": self.stress_programs,
            "source": self.source,
        }


def correlations(target: np.ndarray, responses: np.ndarray) -> np.ndarray:
    centered = responses - responses.mean(axis=-1, keepdims=True)
    target = target - target.mean()
    denominator = np.linalg.norm(centered, axis=-1) * np.linalg.norm(target)
    return np.divide(
        centered @ target, denominator, out=np.zeros(len(responses)), where=denominator > 1e-12
    ).clip(-1, 1)


def auxiliary_features(
    target: np.ndarray,
    predicted: np.ndarray,
    train_similarity: np.ndarray,
    genes: GeneManifest,
    spec: AuxiliarySpec,
) -> np.ndarray:
    """[Predicted correlation, program agreement, bounded stress, train similarity]."""
    result: NDArray[np.float32] = np.zeros((len(predicted), 4), dtype=np.float32)
    result[:, 0] = correlations(target, predicted)
    lookup = {gene: i for i, gene in enumerate(genes.gene_ids)}
    if spec.programs:
        program_indices = [[lookup[g] for g in members] for members in spec.programs.values()]
        target_programs = np.array([target[index].mean() for index in program_indices])
        predicted_programs = np.stack([predicted[:, index].mean(1) for index in program_indices], 1)
        result[:, 1] = 1 / (1 + np.abs(predicted_programs - target_programs).mean(1))
    if spec.stress_programs:
        stress_indices = sorted(
            {lookup[g] for name in spec.stress_programs for g in spec.programs[name]}
        )
        result[:, 2] = np.tanh(np.abs(predicted[:, stress_indices]).mean(1))
    result[:, 3] = train_similarity
    return result
