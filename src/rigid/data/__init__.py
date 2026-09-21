"""Input schemas, deterministic preprocessing, and leakage-aware partitions."""

from .io import attach_source_hashes, load_dataset, save_dataset
from .matrices import read_gct, read_gctx
from .normalization import TrainStandardizer
from .preparation import prepare_matrix
from .schema import CompoundRecord, GeneManifest, ProfileDataset, ProfileRecord, SourceProvenance
from .splits import LeakageReport, assign_splits, audit_leakage, compound_groups
from .synthetic import make_synthetic_dataset
from .targets import (
    ProgramSet,
    consensus_signature,
    edit_programs,
    interpolate_targets,
    reverse_disease_signature,
)

__all__ = [
    "CompoundRecord",
    "GeneManifest",
    "LeakageReport",
    "ProfileDataset",
    "ProfileRecord",
    "ProgramSet",
    "SourceProvenance",
    "TrainStandardizer",
    "assign_splits",
    "attach_source_hashes",
    "audit_leakage",
    "compound_groups",
    "consensus_signature",
    "edit_programs",
    "interpolate_targets",
    "load_dataset",
    "make_synthetic_dataset",
    "prepare_matrix",
    "read_gct",
    "read_gctx",
    "reverse_disease_signature",
    "save_dataset",
]
