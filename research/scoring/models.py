from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DEFAULT_SCORING_CONTRACT, ScoringContract, available_feature_columns
from .predictions import build_prediction_output


def _require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local research env
        raise ImportError(
            "scikit-learn is required for research-side scoring models. "
            "Install research dependencies with requirements-research.txt."
        ) from exc

    return {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


@dataclass
class ShadowModelBundle:
    score_model_version: str
    feature_set_version: str
    feature_columns: list[str]
    classification_target: str | None
    regression_target: str | None
    mfe_target: str | None
    classifier: Any | None = None
    regressor: Any | None = None
    mfe_regressor: Any | None = None


def _build_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    contract: ScoringContract,
) -> Any:
    sk = _require_sklearn()
    numeric = [col for col in contract.numeric_feature_columns if col in feature_columns]
    categorical = [col for col in contract.categorical_feature_columns if col in feature_columns]

    numeric_pipe = sk["Pipeline"](
        steps=[
            ("imputer", sk["SimpleImputer"](strategy="median")),
            ("scaler", sk["StandardScaler"]()),
        ]
    )
    categorical_pipe = sk["Pipeline"](
        steps=[
            ("imputer", sk["SimpleImputer"](strategy="most_frequent")),
            ("onehot", sk["OneHotEncoder"](handle_unknown="ignore")),
        ]
    )
    return sk["ColumnTransformer"](
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def _fit_model(
    train: pd.DataFrame,
    *,
    target_col: str,
    feature_columns: list[str],
    contract: ScoringContract,
    model_kind: str,
) -> Any | None:
    if target_col not in train.columns or not feature_columns:
        return None
    y = train[target_col]
    valid = y.notna()
    if not valid.any():
        return None

    sk = _require_sklearn()
    X = train.loc[valid, feature_columns].copy()
    y_valid = y.loc[valid]
    preprocessor = _build_preprocessor(X, feature_columns, contract)

    if model_kind == "classification":
        y_bool = y_valid.astype(bool).astype(int)
        if y_bool.nunique() < 2:
            return None
        estimator = sk["LogisticRegression"](max_iter=1000, class_weight="balanced")
    else:
        y_num = pd.to_numeric(y_valid, errors="coerce")
        valid_num = y_num.notna()
        X = X.loc[valid_num]
        y_valid = y_num.loc[valid_num]
        if len(y_valid) < 2:
            return None
        estimator = sk["Ridge"](alpha=1.0)

    pipe = sk["Pipeline"](steps=[("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X, y_bool if model_kind == "classification" else y_valid)
    return pipe


def train_shadow_models(
    train: pd.DataFrame,
    *,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
    score_model_version: str = "shadow_score_v1",
    classification_target: str = "profitable",
    regression_target: str = "pnl_R",
    mfe_target: str = "mfe_R",
) -> ShadowModelBundle:
    """Train first-pass interpretable shadow score models."""
    feature_columns = available_feature_columns(train, contract)
    classifier = _fit_model(
        train,
        target_col=classification_target,
        feature_columns=feature_columns,
        contract=contract,
        model_kind="classification",
    )
    regressor = _fit_model(
        train,
        target_col=regression_target,
        feature_columns=feature_columns,
        contract=contract,
        model_kind="regression",
    )
    mfe_regressor = _fit_model(
        train,
        target_col=mfe_target,
        feature_columns=feature_columns,
        contract=contract,
        model_kind="regression",
    )
    return ShadowModelBundle(
        score_model_version=score_model_version,
        feature_set_version=contract.feature_set_version,
        feature_columns=feature_columns,
        classification_target=classification_target if classifier is not None else None,
        regression_target=regression_target if regressor is not None else None,
        mfe_target=mfe_target if mfe_regressor is not None else None,
        classifier=classifier,
        regressor=regressor,
        mfe_regressor=mfe_regressor,
    )


def generate_shadow_predictions(frame: pd.DataFrame, bundle: ShadowModelBundle) -> pd.DataFrame:
    X = frame[bundle.feature_columns].copy() if bundle.feature_columns else pd.DataFrame(index=frame.index)

    pred_win_prob = None
    if bundle.classifier is not None and not X.empty:
        pred_win_prob = bundle.classifier.predict_proba(X)[:, 1]

    pred_expected_r = None
    if bundle.regressor is not None and not X.empty:
        pred_expected_r = bundle.regressor.predict(X)

    pred_expected_mfe = None
    if bundle.mfe_regressor is not None and not X.empty:
        pred_expected_mfe = bundle.mfe_regressor.predict(X)

    if pred_expected_r is not None and pred_win_prob is not None:
        score_shadow = np.asarray(pred_expected_r) + (np.asarray(pred_win_prob) - 0.5)
    elif pred_expected_r is not None:
        score_shadow = pred_expected_r
    else:
        score_shadow = pred_win_prob

    return build_prediction_output(
        frame,
        score_model_version=bundle.score_model_version,
        feature_set_version=bundle.feature_set_version,
        pred_win_prob=pred_win_prob,
        pred_expected_r=pred_expected_r,
        pred_expected_mfe=pred_expected_mfe,
        score_shadow=score_shadow,
    )
