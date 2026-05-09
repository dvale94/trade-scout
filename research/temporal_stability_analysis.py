from __future__ import annotations

from typing import Any

import pandas as pd


UNKNOWN = "unknown"
TARGET_REALIZED_COLUMNS = ["pnl_R", "realized_r", "outcome_r", "outcome_R"]
TARGET_MFE_COLUMNS = ["mfe_R", "mfe_r"]
TARGET_MAE_COLUMNS = ["mae_R", "mae_r"]


def summarize_temporal_performance(
    df: pd.DataFrame,
    cohort_col: str,
    time_col: str,
    freq: str = "W",
) -> pd.DataFrame:
    """Summarize cohort performance across calendar time buckets."""
    prepared = _prepare_temporal_frame(df, cohort_col, time_col)
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                cohort_col,
                "time_bucket",
                "n",
                "win_rate",
                "avg_realized_r",
                "median_realized_r",
                "avg_mfe",
                "avg_mae",
            ]
        )

    prepared["time_bucket"] = prepared["_time"].dt.to_period(freq).astype(str)
    return _summarize_grouped(prepared, [cohort_col, "time_bucket"])


def summarize_rolling_performance(
    df: pd.DataFrame,
    cohort_col: str,
    time_col: str,
    window_size: int,
) -> pd.DataFrame:
    """Summarize rolling trade-count windows for each cohort."""
    prepared = _prepare_temporal_frame(df, cohort_col, time_col)
    if prepared.empty or window_size <= 0:
        return pd.DataFrame(
            columns=[
                cohort_col,
                "window_end_trade_n",
                "window_end_time",
                "n",
                "win_rate",
                "avg_realized_r",
                "median_realized_r",
                "avg_mfe",
                "avg_mae",
            ]
        )

    rows: list[dict[str, Any]] = []
    for cohort_value, cohort_df in prepared.groupby(cohort_col, observed=False, dropna=False):
        cohort_sorted = cohort_df.sort_values("_time").reset_index(drop=True)
        for end_idx in range(window_size - 1, len(cohort_sorted)):
            window_df = cohort_sorted.iloc[end_idx - window_size + 1 : end_idx + 1].copy()
            summary = _summarize_window(window_df)
            summary.update(
                {
                    cohort_col: cohort_value,
                    "window_end_trade_n": int(end_idx + 1),
                    "window_end_time": window_df["_time"].iloc[-1],
                }
            )
            rows.append(summary)

    return pd.DataFrame(rows).sort_values([cohort_col, "window_end_trade_n"]).reset_index(drop=True)


def summarize_cumulative_performance(
    df: pd.DataFrame,
    cohort_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Build a simple cumulative expectancy curve per cohort over trade order."""
    prepared = _prepare_temporal_frame(df, cohort_col, time_col)
    if prepared.empty:
        return pd.DataFrame(
            columns=[cohort_col, "trade_n", "time", "cum_R", "cum_avg_realized_r"]
        )

    realized = _first_available_numeric_series(prepared, TARGET_REALIZED_COLUMNS)
    rows: list[pd.DataFrame] = []
    for cohort_value, cohort_df in prepared.groupby(cohort_col, observed=False, dropna=False):
        cohort_sorted = cohort_df.sort_values("_time").copy()
        cohort_realized = realized.loc[cohort_sorted.index]
        cohort_sorted = cohort_sorted.loc[cohort_realized.notna()].copy()
        cohort_realized = cohort_realized.loc[cohort_sorted.index]
        if cohort_sorted.empty:
            continue
        cohort_sorted["trade_n"] = range(1, len(cohort_sorted) + 1)
        cohort_sorted["cum_R"] = cohort_realized.cumsum().values
        cohort_sorted["cum_avg_realized_r"] = cohort_realized.expanding().mean().values
        cohort_sorted[cohort_col] = cohort_value
        cohort_sorted["time"] = cohort_sorted["_time"]
        rows.append(cohort_sorted[[cohort_col, "trade_n", "time", "cum_R", "cum_avg_realized_r"]])

    if not rows:
        return pd.DataFrame(columns=[cohort_col, "trade_n", "time", "cum_R", "cum_avg_realized_r"])
    return pd.concat(rows, ignore_index=True)


def _prepare_temporal_frame(df: pd.DataFrame, cohort_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    if cohort_col not in out.columns:
        out[cohort_col] = UNKNOWN
    out["_time"] = pd.to_datetime(out.get(time_col), errors="coerce", utc=True)
    out = out[out["_time"].notna()].copy()
    return out


def _summarize_grouped(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    realized = _first_available_numeric_series(df, TARGET_REALIZED_COLUMNS)
    mfe = _first_available_numeric_series(df, TARGET_MFE_COLUMNS)
    mae = _first_available_numeric_series(df, TARGET_MAE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for group_key, group_df in df.groupby(group_cols, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_cols))
        row = {column: value for column, value in zip(group_cols, key_values)}
        realized_group = realized.loc[group_df.index].dropna()
        mfe_group = mfe.loc[group_df.index].dropna()
        mae_group = mae.loc[group_df.index].dropna()
        row.update(
            {
                "n": int(len(group_df)),
                "win_rate": float((realized_group > 0).mean()) if not realized_group.empty else None,
                "avg_realized_r": float(realized_group.mean()) if not realized_group.empty else None,
                "median_realized_r": float(realized_group.median()) if not realized_group.empty else None,
                "avg_mfe": float(mfe_group.mean()) if not mfe_group.empty else None,
                "avg_mae": float(mae_group.mean()) if not mae_group.empty else None,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _summarize_window(df: pd.DataFrame) -> dict[str, Any]:
    realized = _first_available_numeric_series(df, TARGET_REALIZED_COLUMNS).dropna()
    mfe = _first_available_numeric_series(df, TARGET_MFE_COLUMNS).dropna()
    mae = _first_available_numeric_series(df, TARGET_MAE_COLUMNS).dropna()
    return {
        "n": int(len(df)),
        "win_rate": float((realized > 0).mean()) if not realized.empty else None,
        "avg_realized_r": float(realized.mean()) if not realized.empty else None,
        "median_realized_r": float(realized.median()) if not realized.empty else None,
        "avg_mfe": float(mfe.mean()) if not mfe.empty else None,
        "avg_mae": float(mae.mean()) if not mae.empty else None,
    }


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


def _group_key_values(group_key: Any, group_col_count: int) -> tuple[Any, ...]:
    if group_col_count == 1:
        if isinstance(group_key, tuple):
            return (group_key[0],)
        return (group_key,)
    return tuple(group_key)
