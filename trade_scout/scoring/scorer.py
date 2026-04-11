from __future__ import annotations

from .buckets import DEFAULT_SHADOW_SCORE_BUCKET_CONFIG, shadow_score_bucket
from .features import build_shadow_score_features
from .models import RuntimeShadowScoreConfig, RuntimeShadowScoreOutput

SHADOW_SCORE_STATUS_DISABLED = "DISABLED"
SHADOW_SCORE_STATUS_UNCONFIGURED = "UNCONFIGURED"
SHADOW_SCORE_STATUS_SCORED = "SCORED"
SHADOW_SCORE_STATUS_ERROR = "ERROR"
SUPPORTED_SCORER_RECIPE_MODES = {"expected_r_only"}


class RuntimeShadowScorer:
    def __init__(self, config: RuntimeShadowScoreConfig):
        if config.enabled and config.score_recipe_mode not in SUPPORTED_SCORER_RECIPE_MODES:
            supported = ", ".join(sorted(SUPPORTED_SCORER_RECIPE_MODES))
            raise ValueError(f"unsupported runtime shadow score recipe mode={config.score_recipe_mode!r}; supported={supported}")
        self.config = config

    def score(self, intent_row, context_row) -> RuntimeShadowScoreOutput:
        if not self.config.enabled:
            return self._output(None, None, None, status=SHADOW_SCORE_STATUS_DISABLED)

        features = build_shadow_score_features(intent_row, context_row)
        pred_expected_r = self._predict_expected_r(features)

        if self.config.score_recipe_mode == "expected_r_only":
            score_shadow = pred_expected_r
        else:
            score_shadow = None

        status = SHADOW_SCORE_STATUS_SCORED if score_shadow is not None else SHADOW_SCORE_STATUS_UNCONFIGURED

        return self._output(
            pred_expected_r=pred_expected_r,
            pred_win_prob=None,
            pred_expected_mfe=None,
            score_shadow=score_shadow,
            status=status,
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
        status: str = SHADOW_SCORE_STATUS_UNCONFIGURED,
    ) -> RuntimeShadowScoreOutput:
        return RuntimeShadowScoreOutput(
            score_model_version=self.config.score_model_version,
            feature_set_version=self.config.feature_set_version,
            score_recipe_version=self.config.score_recipe_version,
            score_recipe_mode=self.config.score_recipe_mode,
            score_bucket_version=self.config.score_bucket_version,
            shadow_score_status=status,
            pred_expected_r=pred_expected_r,
            pred_win_prob=pred_win_prob,
            pred_expected_mfe=pred_expected_mfe,
            score_shadow=score_shadow,
            score_shadow_bucket=shadow_score_bucket(
                score_shadow,
                DEFAULT_SHADOW_SCORE_BUCKET_CONFIG,
            ),
        )


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
