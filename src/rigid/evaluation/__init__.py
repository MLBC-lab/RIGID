"""Scientific evaluation without test-set fitting or implicit missing-label matches."""

from .calibration import (
    SplitConformalRegressor,
    TemperatureScaler,
    empirical_coverage,
    expected_calibration_error,
)
from .controls import make_decoy_targets, run_ablations, summarize_families
from .metrics import (
    CandidateAnnotation,
    EvaluationReport,
    Rate,
    TargetAnnotation,
    TargetMetrics,
    evaluate_rankings,
    ndcg,
    pearson_correlation,
    top_gene_ndcg,
)
from .ranking import DesignabilityModel, RankedCandidate, rerank_candidates
from .statistics import (
    BootstrapInterval,
    benjamini_hochberg,
    cohort_robustness,
    compare_target_metrics,
    jaccard,
    paired_bootstrap_ci,
)

__all__ = [
    "BootstrapInterval",
    "CandidateAnnotation",
    "DesignabilityModel",
    "EvaluationReport",
    "RankedCandidate",
    "Rate",
    "SplitConformalRegressor",
    "TargetAnnotation",
    "TargetMetrics",
    "TemperatureScaler",
    "benjamini_hochberg",
    "cohort_robustness",
    "compare_target_metrics",
    "empirical_coverage",
    "evaluate_rankings",
    "expected_calibration_error",
    "jaccard",
    "make_decoy_targets",
    "ndcg",
    "paired_bootstrap_ci",
    "pearson_correlation",
    "rerank_candidates",
    "run_ablations",
    "summarize_families",
    "top_gene_ndcg",
]
