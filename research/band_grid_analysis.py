from __future__ import annotations

from typing import Sequence

import pandas as pd


TARGET_REALIZED_COLUMNS = ["pnl_R", "realized_r", "outcome_r", "outcome_R"]
TARGET_MFE_COLUMNS = ["mfe_R", "mfe_r"]
TARGET_MAE_COLUMNS = ["mae_R", "mae_r"]


def summarize_band_grid(
    df: pd.DataFrame,
    feature_a: str,
    feature_a_bins: Sequence[float],
    feature_b: str,
    feature_b_bins: Sequence[float],
) -> pd.DataFrame:
    """Summarize rectangular two-feature bands using existing trade outcome fields."""
    if len(feature_a_bins) < 2 or len(feature_b_bins) < 2:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "feature_a_low",
                "feature_a_high",
                "feature_b_low",
                "feature_b_high",
                "n",
                "coverage_pct",
                "win_rate",
                "avg_realized_r",
                "median_realized_r",
                "avg_mfe",
                "avg_mae",
            ]
        )

    a_values = _numeric_series(df, feature_a)
    b_values = _numeric_series(df, feature_b)
    realized = _first_available_numeric_series(df, TARGET_REALIZED_COLUMNS)
    mfe = _first_available_numeric_series(df, TARGET_MFE_COLUMNS)
    mae = _first_available_numeric_series(df, TARGET_MAE_COLUMNS)

    valid_grid_mask = a_values.notna() & b_values.notna()
    total_count = int(valid_grid_mask.sum()) if valid_grid_mask.any() else int(len(df))

    rows: list[dict[str, float | int | str | None]] = []
    for a_low, a_high in zip(feature_a_bins[:-1], feature_a_bins[1:]):
        for b_low, b_high in zip(feature_b_bins[:-1], feature_b_bins[1:]):
            a_mask = a_values.ge(float(a_low)) & a_values.lt(float(a_high))
            b_mask = b_values.ge(float(b_low)) & b_values.lt(float(b_high))
            selected_idx = df.index[(a_mask & b_mask).fillna(False)]

            realized_group = realized.loc[selected_idx].dropna()
            mfe_group = mfe.loc[selected_idx].dropna()
            mae_group = mae.loc[selected_idx].dropna()

            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "feature_a_low": float(a_low),
                    "feature_a_high": float(a_high),
                    "feature_b_low": float(b_low),
                    "feature_b_high": float(b_high),
                    "n": int(len(selected_idx)),
                    "coverage_pct": (float(len(selected_idx) / total_count) if total_count else 0.0),
                    "win_rate": float((realized_group > 0).mean()) if not realized_group.empty else None,
                    "avg_realized_r": float(realized_group.mean()) if not realized_group.empty else None,
                    "median_realized_r": float(realized_group.median()) if not realized_group.empty else None,
                    "avg_mfe": float(mfe_group.mean()) if not mfe_group.empty else None,
                    "avg_mae": float(mae_group.mean()) if not mae_group.empty else None,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["feature_a_low", "feature_b_low"])
        .reset_index(drop=True)
    )


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64", index=df.index)
    return pd.to_numeric(df[column], errors="coerce")
