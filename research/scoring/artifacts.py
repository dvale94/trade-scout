from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .score_recipes import ScoreRecipe


@dataclass(frozen=True)
class ScoringModelMetadata:
    score_model_version: str
    feature_set_version: str
    score_recipe_version: str
    score_recipe_mode: str
    feature_columns: tuple[str, ...]
    classification_target: str | None
    regression_target: str | None
    mfe_target: str | None
    train_start_ts_utc: str | None
    train_end_ts_utc: str | None
    train_rows: int
    test_rows: int | None
    classifier_available: bool
    regressor_available: bool
    mfe_regressor_available: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_scoring_model_metadata(
    *,
    score_model_version: str,
    feature_set_version: str,
    score_recipe: ScoreRecipe,
    feature_columns: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    classification_target: str | None = None,
    regression_target: str | None = None,
    mfe_target: str | None = None,
    classifier_available: bool = False,
    regressor_available: bool = False,
    mfe_regressor_available: bool = False,
    ts_col: str = "signal_ts_utc",
) -> ScoringModelMetadata:
    ts = pd.to_datetime(train[ts_col], errors="coerce", utc=True) if ts_col in train.columns else pd.Series(dtype="datetime64[ns, UTC]")
    return ScoringModelMetadata(
        score_model_version=score_model_version,
        feature_set_version=feature_set_version,
        score_recipe_version=score_recipe.recipe_version,
        score_recipe_mode=score_recipe.mode,
        feature_columns=tuple(feature_columns),
        classification_target=classification_target,
        regression_target=regression_target,
        mfe_target=mfe_target,
        train_start_ts_utc=ts.min().isoformat() if not ts.dropna().empty else None,
        train_end_ts_utc=ts.max().isoformat() if not ts.dropna().empty else None,
        train_rows=int(len(train)),
        test_rows=int(len(test)) if test is not None else None,
        classifier_available=classifier_available,
        regressor_available=regressor_available,
        mfe_regressor_available=mfe_regressor_available,
    )
