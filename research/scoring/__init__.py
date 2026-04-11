"""Research-side scoring model upgrade helpers.

This package is intentionally analysis-only. It does not change live strategy,
policy, execution, or canonical research schemas.
"""

from .contracts import (
    DEFAULT_SCORING_CONTRACT,
    ScoringContract,
    available_feature_columns,
    contract_coverage_summary,
    missing_feature_columns,
)
from .dataset import (
    audit_scoring_dataset_export,
    build_scoring_modeling_dataset,
    build_scoring_modeling_dataset_with_audit,
)
from .diagnostics import (
    categorical_cardinality_summary,
    column_availability_summary,
    feature_missingness_summary,
    numeric_constant_feature_summary,
    target_availability_summary,
    train_test_schema_check,
)
from .evaluation import (
    compare_shadow_to_baseline,
    evaluate_current_score,
    evaluate_prediction_buckets,
    evaluate_train_test_score,
    promotion_readiness_summary,
    segmented_score_evaluation,
)
from .models import (
    ShadowModelBundle,
    generate_shadow_predictions,
    inspect_shadow_model_coefficients,
    train_shadow_models,
)
from .predictions import SCORE_OUTPUT_COLUMNS, build_prediction_output
from .score_recipes import (
    BLENDED_RECIPE,
    EXPECTED_R_ONLY_RECIPE,
    WIN_PROB_ONLY_RECIPE,
    ScoreRecipe,
    compose_shadow_score,
)
from .splits import TimeSplitResult, time_train_test_split

__all__ = [
    "DEFAULT_SCORING_CONTRACT",
    "SCORE_OUTPUT_COLUMNS",
    "ScoringContract",
    "ScoreRecipe",
    "ShadowModelBundle",
    "TimeSplitResult",
    "audit_scoring_dataset_export",
    "available_feature_columns",
    "build_prediction_output",
    "build_scoring_modeling_dataset",
    "build_scoring_modeling_dataset_with_audit",
    "categorical_cardinality_summary",
    "column_availability_summary",
    "compare_shadow_to_baseline",
    "compose_shadow_score",
    "contract_coverage_summary",
    "evaluate_current_score",
    "evaluate_prediction_buckets",
    "evaluate_train_test_score",
    "feature_missingness_summary",
    "generate_shadow_predictions",
    "inspect_shadow_model_coefficients",
    "missing_feature_columns",
    "numeric_constant_feature_summary",
    "promotion_readiness_summary",
    "segmented_score_evaluation",
    "target_availability_summary",
    "time_train_test_split",
    "train_shadow_models",
    "train_test_schema_check",
    "BLENDED_RECIPE",
    "EXPECTED_R_ONLY_RECIPE",
    "WIN_PROB_ONLY_RECIPE",
]
