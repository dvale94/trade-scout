"""Runtime shadow scoring package.

This package is inference-oriented only. Research dataset building, training,
and evaluation stay under ``research/scoring``.
"""

from .features import build_shadow_score_features
from .models import RuntimeShadowScoreConfig, RuntimeShadowScoreOutput
from .registry import active_shadow_score_config
from .scorer import RuntimeShadowScorer

__all__ = [
    "RuntimeShadowScoreConfig",
    "RuntimeShadowScoreOutput",
    "RuntimeShadowScorer",
    "active_shadow_score_config",
    "build_shadow_score_features",
]
