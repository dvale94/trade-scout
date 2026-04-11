from __future__ import annotations

from .features import build_shadow_score_features
from .models import RuntimeShadowScoreConfig, RuntimeShadowScoreOutput


class RuntimeShadowScorer:
    def __init__(self, config: RuntimeShadowScoreConfig):
        self.config = config

    def score(self, intent_row, context_row) -> RuntimeShadowScoreOutput:
        if not self.config.enabled:
            return self._output(None, None, None)

        features = build_shadow_score_features(intent_row, context_row)
        pred_expected_r = self._predict_expected_r(features)

        if self.config.score_recipe_mode == "expected_r_only":
            score_shadow = pred_expected_r
        else:
            score_shadow = None

        return self._output(
            pred_expected_r=pred_expected_r,
            pred_win_prob=None,
            pred_expected_mfe=None,
            score_shadow=score_shadow,
        )

    def _predict_expected_r(self, features: dict[str, object]) -> float | None:
        weights = self.config.expected_r_weights or {}
        intercept = self.config.expected_r_intercept
        if intercept is None and not weights:
            return None

        total = float(intercept or 0.0)
        used_feature = False
        for feature_name, weight in weights.items():
            value = _float_or_none(features.get(feature_name))
            if value is None:
                continue
            total += float(weight) * value
            used_feature = True

        if weights and not used_feature and intercept is None:
            return None
        return round(total, 6)

    def _output(
        self,
        pred_expected_r: float | None,
        pred_win_prob: float | None,
        pred_expected_mfe: float | None,
        score_shadow: float | None = None,
    ) -> RuntimeShadowScoreOutput:
        return RuntimeShadowScoreOutput(
            score_model_version=self.config.score_model_version,
            feature_set_version=self.config.feature_set_version,
            score_recipe_version=self.config.score_recipe_version,
            score_recipe_mode=self.config.score_recipe_mode,
            pred_expected_r=pred_expected_r,
            pred_win_prob=pred_win_prob,
            pred_expected_mfe=pred_expected_mfe,
            score_shadow=score_shadow,
            score_shadow_bucket=_shadow_score_bucket(score_shadow),
        )


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _shadow_score_bucket(score_shadow: float | None) -> str | None:
    if score_shadow is None:
        return None
    if score_shadow < 0:
        return "NEGATIVE"
    if score_shadow < 0.25:
        return "LOW_POSITIVE"
    if score_shadow < 0.75:
        return "MEDIUM_POSITIVE"
    return "HIGH_POSITIVE"
