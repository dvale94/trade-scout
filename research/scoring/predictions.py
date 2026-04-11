from __future__ import annotations

import numpy as np
import pandas as pd

from .score_recipes import BLENDED_RECIPE, ScoreRecipe, compose_shadow_score

SCORE_OUTPUT_COLUMNS = (
    "alert_id",
    "signal_ts_utc",
    "score_model_version",
    "feature_set_version",
    "score_recipe_version",
    "score_recipe_mode",
    "pred_win_prob",
    "pred_expected_r",
    "pred_expected_mfe",
    "score_shadow",
    "score_shadow_bucket",
)


def _shadow_buckets(score: pd.Series, buckets: int = 5) -> pd.Series:
    numeric = pd.to_numeric(score, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(pd.NA, index=score.index, dtype="object")
    if numeric.nunique(dropna=True) <= 1:
        return pd.Series("all", index=score.index, dtype="object")
    return pd.qcut(numeric, q=min(buckets, numeric.nunique(dropna=True)), duplicates="drop").astype("object")


def build_prediction_output(
    frame: pd.DataFrame,
    *,
    score_model_version: str,
    feature_set_version: str,
    score_recipe: ScoreRecipe = BLENDED_RECIPE,
    pred_win_prob: pd.Series | np.ndarray | None = None,
    pred_expected_r: pd.Series | np.ndarray | None = None,
    pred_expected_mfe: pd.Series | np.ndarray | None = None,
    score_shadow: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build version-aware shadow prediction output rows."""
    out = pd.DataFrame(index=frame.index)
    out["alert_id"] = frame["alert_id"].values if "alert_id" in frame.columns else pd.NA
    out["signal_ts_utc"] = frame["signal_ts_utc"] if "signal_ts_utc" in frame.columns else pd.NA
    out["score_model_version"] = score_model_version
    out["feature_set_version"] = feature_set_version
    out["score_recipe_version"] = score_recipe.recipe_version
    out["score_recipe_mode"] = score_recipe.mode
    out["pred_win_prob"] = pred_win_prob if pred_win_prob is not None else np.nan
    out["pred_expected_r"] = pred_expected_r if pred_expected_r is not None else np.nan
    out["pred_expected_mfe"] = pred_expected_mfe if pred_expected_mfe is not None else np.nan

    if score_shadow is None:
        out["score_shadow"] = compose_shadow_score(
            pred_expected_r=out["pred_expected_r"],
            pred_win_prob=out["pred_win_prob"],
            recipe=score_recipe,
            index=out.index,
        )
    else:
        out["score_shadow"] = score_shadow

    out["score_shadow_bucket"] = _shadow_buckets(out["score_shadow"])
    return out[list(SCORE_OUTPUT_COLUMNS)].reset_index(drop=True)
