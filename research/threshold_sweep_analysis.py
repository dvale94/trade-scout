from __future__ import annotations

from typing import Sequence

import pandas as pd


TARGET_REALIZED_COLUMNS = ["pnl_R", "realized_r", "outcome_r", "outcome_R"]
TARGET_MFE_COLUMNS = ["mfe_R", "mfe_r"]
TARGET_MAE_COLUMNS = ["mae_R", "mae_r"]


def summarize_threshold_sweep(
    df: pd.DataFrame,
    feature_col: str,
    thresholds: Sequence[float],
    direction: str = "gte",
) -> pd.DataFrame:
    """Summarize simple feature-threshold sweeps using existing trade outcome fields."""
    if not thresholds:
        return pd.DataFrame(
            columns=[
                "feature",
                "direction",
                "threshold",
                "n",
                "coverage_pct",
                "win_rate",
                "avg_realized_r",
                "median_realized_r",
                "avg_mfe",
                "avg_mae",
            ]
        )

    direction_normalized = _normalize_direction(direction)
    feature = _numeric_series(df, feature_col)
    realized = _first_available_numeric_series(df, TARGET_REALIZED_COLUMNS)
    mfe = _first_available_numeric_series(df, TARGET_MFE_COLUMNS)
    mae = _first_available_numeric_series(df, TARGET_MAE_COLUMNS)

    valid_feature_mask = feature.notna()
    total_count = int(valid_feature_mask.sum()) if valid_feature_mask.any() else int(len(df))

    rows: list[dict[str, float | int | str | None]] = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        if direction_normalized == "lte":
            mask = feature.le(threshold_value)
        else:
            mask = feature.ge(threshold_value)

        selected_idx = df.index[mask.fillna(False)]
        realized_group = realized.loc[selected_idx].dropna()
        mfe_group = mfe.loc[selected_idx].dropna()
        mae_group = mae.loc[selected_idx].dropna()

        rows.append(
            {
                "feature": feature_col,
                "direction": direction_normalized,
                "threshold": threshold_value,
                "n": int(len(selected_idx)),
                "coverage_pct": (float(len(selected_idx) / total_count) if total_count else 0.0),
                "win_rate": float((realized_group > 0).mean()) if not realized_group.empty else None,
                "avg_realized_r": float(realized_group.mean()) if not realized_group.empty else None,
                "median_realized_r": float(realized_group.median()) if not realized_group.empty else None,
                "avg_mfe": float(mfe_group.mean()) if not mfe_group.empty else None,
                "avg_mae": float(mae_group.mean()) if not mae_group.empty else None,
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64", index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _normalize_direction(direction: str) -> str:
    normalized = str(direction).strip().lower()
    if normalized not in {"gte", "lte"}:
        return "gte"
    return normalized
