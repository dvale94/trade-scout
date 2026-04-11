"""Research-side scoring model upgrade helpers.

This package is intentionally analysis-only. It does not change live strategy,
policy, execution, or canonical research schemas.
"""

from .contracts import (
    DEFAULT_SCORING_CONTRACT,
    ScoringContract,
    available_feature_columns,
)
from .dataset import build_scoring_modeling_dataset
from .evaluation import (
    compare_shadow_to_baseline,
    evaluate_current_score,
    evaluate_prediction_buckets,
)
from .models import ShadowModelBundle, generate_shadow_predictions, train_shadow_models
from .predictions import SCORE_OUTPUT_COLUMNS, build_prediction_output
from .splits import TimeSplitResult, time_train_test_split

__all__ = [
    "DEFAULT_SCORING_CONTRACT",
    "SCORE_OUTPUT_COLUMNS",
    "ScoringContract",
    "ShadowModelBundle",
    "TimeSplitResult",
    "available_feature_columns",
    "build_prediction_output",
    "build_scoring_modeling_dataset",
    "compare_shadow_to_baseline",
    "evaluate_current_score",
    "evaluate_prediction_buckets",
    "generate_shadow_predictions",
    "time_train_test_split",
    "train_shadow_models",
]
