from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeShadowScoreConfig:
    score_model_version: str
    feature_set_version: str
    score_recipe_version: str
    score_recipe_mode: str
    enabled: bool = True
    expected_r_intercept: float | None = None
    expected_r_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class RuntimeShadowScoreOutput:
    score_model_version: str
    feature_set_version: str
    score_recipe_version: str
    score_recipe_mode: str
    pred_expected_r: float | None
    pred_win_prob: float | None
    pred_expected_mfe: float | None
    score_shadow: float | None
    score_shadow_bucket: str | None

    def to_context_updates(self) -> dict[str, Any]:
        return {
            "score_model_version": self.score_model_version,
            "feature_set_version": self.feature_set_version,
            "score_recipe_version": self.score_recipe_version,
            "score_recipe_mode": self.score_recipe_mode,
            "pred_expected_r": self.pred_expected_r,
            "pred_win_prob": self.pred_win_prob,
            "pred_expected_mfe": self.pred_expected_mfe,
            "score_shadow": self.score_shadow,
            "score_shadow_bucket": self.score_shadow_bucket,
        }
