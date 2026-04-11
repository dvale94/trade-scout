from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ScoreRecipeMode = Literal["expected_r_only", "win_prob_only", "blended"]


@dataclass(frozen=True)
class ScoreRecipe:
    recipe_version: str
    mode: ScoreRecipeMode
    expected_r_weight: float = 1.0
    win_prob_weight: float = 1.0
    win_prob_center: float = 0.5


EXPECTED_R_ONLY_RECIPE = ScoreRecipe("score_recipe_expected_r_v1", "expected_r_only")
WIN_PROB_ONLY_RECIPE = ScoreRecipe("score_recipe_win_prob_v1", "win_prob_only")
BLENDED_RECIPE = ScoreRecipe("score_recipe_blended_v1", "blended")


def compose_shadow_score(
    *,
    pred_expected_r: pd.Series | np.ndarray | None,
    pred_win_prob: pd.Series | np.ndarray | None,
    recipe: ScoreRecipe = BLENDED_RECIPE,
    index: pd.Index | None = None,
) -> pd.Series:
    if index is None:
        if pred_expected_r is not None:
            index = pd.RangeIndex(len(pred_expected_r))
        elif pred_win_prob is not None:
            index = pd.RangeIndex(len(pred_win_prob))
        else:
            return pd.Series(dtype="float64")

    expected_r = (
        pd.Series(pred_expected_r, index=index, dtype="float64")
        if pred_expected_r is not None
        else pd.Series(np.nan, index=index, dtype="float64")
    )
    win_prob = (
        pd.Series(pred_win_prob, index=index, dtype="float64")
        if pred_win_prob is not None
        else pd.Series(np.nan, index=index, dtype="float64")
    )

    if recipe.mode == "expected_r_only":
        return expected_r
    if recipe.mode == "win_prob_only":
        return win_prob
    if recipe.mode == "blended":
        return (recipe.expected_r_weight * expected_r) + (
            recipe.win_prob_weight * (win_prob - recipe.win_prob_center)
        )
    raise ValueError(f"Unsupported score recipe mode: {recipe.mode}")
