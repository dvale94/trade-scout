from __future__ import annotations

import json
import os

from .models import RuntimeShadowScoreConfig


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _float_env_optional(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return float(raw)


def _weights_env(name: str) -> dict[str, float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object mapping feature names to weights")
    return {str(key): float(value) for key, value in parsed.items()}


def active_shadow_score_config() -> RuntimeShadowScoreConfig:
    """Return the active runtime shadow score config.

    The first runtime candidate is expected-R-only. If no approved coefficients
    are configured yet, outputs remain versioned and null-safe.
    """
    return RuntimeShadowScoreConfig(
        score_model_version=os.getenv("SHADOW_SCORE_MODEL_VERSION", "shadow_score_v1_expected_r_only").strip(),
        feature_set_version=os.getenv("SHADOW_SCORE_FEATURE_SET_VERSION", "score_features_v1").strip(),
        score_recipe_version=os.getenv("SHADOW_SCORE_RECIPE_VERSION", "score_recipe_expected_r_v1").strip(),
        score_recipe_mode=os.getenv("SHADOW_SCORE_RECIPE_MODE", "expected_r_only").strip(),
        enabled=_bool_env("SHADOW_SCORING_ENABLED", True),
        expected_r_intercept=_float_env_optional("SHADOW_SCORE_EXPECTED_R_INTERCEPT"),
        expected_r_weights=_weights_env("SHADOW_SCORE_EXPECTED_R_WEIGHTS_JSON"),
    )
