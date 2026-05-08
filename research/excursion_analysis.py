from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


UNKNOWN = "unknown"
TARGET_REALIZED_COLUMNS = ["pnl_R", "realized_r", "outcome_r", "outcome_R"]
TARGET_MFE_COLUMNS = ["mfe_R", "mfe_r"]
DEFAULT_THRESHOLDS = (0.25, 0.5, 1.0, 1.5, 2.0)
CAPTURE_WINSOR_QUANTILES = (0.05, 0.95)


def summarize_excursion_thresholds(
    df: pd.DataFrame,
    group_cols: list[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize grouped MFE threshold reach rates using available excursion fields."""
    grouped = _with_group_columns(df, group_cols)
    if grouped.empty:
        return pd.DataFrame(columns=[*group_cols, "n", *[_threshold_column_name(x) for x in thresholds]])

    mfe = _first_available_numeric_series(grouped, TARGET_MFE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped.groupby(group_cols, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_cols))
        row = {column: value for column, value in zip(group_cols, key_values)}
        row["n"] = int(len(group_df))

        group_mfe = mfe.loc[group_df.index].dropna()
        for threshold in thresholds:
            row[_threshold_column_name(threshold)] = (
                float((group_mfe >= float(threshold)).mean()) if not group_mfe.empty else None
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_capture_efficiency(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    winsor_quantiles: tuple[float, float] = CAPTURE_WINSOR_QUANTILES,
) -> pd.DataFrame:
    """Summarize realized-vs-MFE capture ratios with safe zero handling."""
    grouped = _with_group_columns(df, group_cols)
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "n",
                "median_capture_ratio",
                "mean_capture_ratio",
                "p25_capture_ratio",
                "p75_capture_ratio",
            ]
        )

    realized = _first_available_numeric_series(grouped, TARGET_REALIZED_COLUMNS)
    mfe = _first_available_numeric_series(grouped, TARGET_MFE_COLUMNS)
    valid_mfe = mfe.where(mfe.gt(0))
    capture_ratio = realized.div(valid_mfe).replace([float("inf"), float("-inf")], pd.NA)
    capture_ratio = _winsorize_series(capture_ratio, winsor_quantiles)

    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped.groupby(group_cols, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_cols))
        row = {column: value for column, value in zip(group_cols, key_values)}
        row["n"] = int(len(group_df))

        group_ratio = capture_ratio.loc[group_df.index].dropna()
        if not group_ratio.empty:
            row["median_capture_ratio"] = float(group_ratio.median())
            row["mean_capture_ratio"] = float(group_ratio.mean())
            row["p25_capture_ratio"] = float(group_ratio.quantile(0.25))
            row["p75_capture_ratio"] = float(group_ratio.quantile(0.75))
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    for column in (
        "median_capture_ratio",
        "mean_capture_ratio",
        "p25_capture_ratio",
        "p75_capture_ratio",
    ):
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _threshold_column_name(threshold: float) -> str:
    rendered = str(float(threshold)).replace(".", "_")
    rendered = rendered.replace("_0", "") if rendered.endswith("_0") else rendered
    return f"pct_reached_{rendered}R"


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


def _winsorize_series(series: pd.Series, quantiles: tuple[float, float]) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series

    lower_q, upper_q = quantiles
    if lower_q < 0 or upper_q > 1 or lower_q >= upper_q:
        return series

    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    clipped = series.copy()
    clipped.loc[valid.index] = valid.clip(lower=lower, upper=upper)
    return clipped


def _with_group_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in group_cols:
        if column not in out.columns:
            out[column] = UNKNOWN
    return out


def _group_key_values(group_key: Any, group_col_count: int) -> tuple[Any, ...]:
    if group_col_count == 1:
        if isinstance(group_key, tuple):
            return (group_key[0],)
        return (group_key,)
    return tuple(group_key)
