from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


UNKNOWN = "unknown"


def summarize_feature_distributions(
    df: pd.DataFrame,
    cohort_col: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Summarize simple numeric feature distributions by cohort."""
    prepared = _with_cohort_column(df, cohort_col)
    if prepared.empty:
        return pd.DataFrame(
            columns=["feature", cohort_col, "count", "mean", "median", "std", "p25", "p75"]
        )

    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        values = _numeric_series(prepared, feature)
        for cohort_value, cohort_df in prepared.groupby(cohort_col, observed=False, dropna=False):
            cohort_values = values.loc[cohort_df.index].dropna()
            rows.append(
                {
                    "feature": feature,
                    cohort_col: cohort_value,
                    "count": int(cohort_values.shape[0]),
                    "mean": float(cohort_values.mean()) if not cohort_values.empty else None,
                    "median": float(cohort_values.median()) if not cohort_values.empty else None,
                    "std": float(cohort_values.std(ddof=1)) if cohort_values.shape[0] > 1 else None,
                    "p25": float(cohort_values.quantile(0.25)) if not cohort_values.empty else None,
                    "p75": float(cohort_values.quantile(0.75)) if not cohort_values.empty else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", cohort_col]).reset_index(drop=True)


def compare_two_cohorts(
    df: pd.DataFrame,
    cohort_col: str,
    cohort_a: str,
    cohort_b: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Compare two cohorts using simple mean deltas and an IQR-overlap proxy."""
    prepared = _with_cohort_column(df, cohort_col)
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "cohort_a_mean",
                "cohort_b_mean",
                "mean_delta",
                "standardized_delta",
                "overlap_proxy",
            ]
        )

    cohort_series = prepared[cohort_col].fillna(UNKNOWN).astype(str)
    cohort_a_df = prepared.loc[cohort_series.eq(str(cohort_a))].copy()
    cohort_b_df = prepared.loc[cohort_series.eq(str(cohort_b))].copy()

    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        values = _numeric_series(prepared, feature)
        a_values = values.loc[cohort_a_df.index].dropna()
        b_values = values.loc[cohort_b_df.index].dropna()

        a_mean = float(a_values.mean()) if not a_values.empty else None
        b_mean = float(b_values.mean()) if not b_values.empty else None
        mean_delta = (float(a_mean - b_mean) if a_mean is not None and b_mean is not None else None)

        rows.append(
            {
                "feature": feature,
                "cohort_a_mean": a_mean,
                "cohort_b_mean": b_mean,
                "mean_delta": mean_delta,
                "standardized_delta": _standardized_delta(a_values, b_values),
                "overlap_proxy": _iqr_overlap_proxy(a_values, b_values),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("feature")
        .reset_index(drop=True)
    )


def _numeric_series(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(dtype="float64", index=df.index)
    return pd.to_numeric(df[feature], errors="coerce")


def _with_cohort_column(df: pd.DataFrame, cohort_col: str) -> pd.DataFrame:
    out = df.copy()
    if cohort_col not in out.columns:
        out[cohort_col] = UNKNOWN
    return out


def _standardized_delta(a_values: pd.Series, b_values: pd.Series) -> float | None:
    if a_values.shape[0] < 2 or b_values.shape[0] < 2:
        return None

    a_std = float(a_values.std(ddof=1))
    b_std = float(b_values.std(ddof=1))
    pooled_n = (a_values.shape[0] - 1) + (b_values.shape[0] - 1)
    if pooled_n <= 0:
        return None

    pooled_var = (
        ((a_values.shape[0] - 1) * (a_std ** 2)) + ((b_values.shape[0] - 1) * (b_std ** 2))
    ) / pooled_n
    if pooled_var <= 0:
        return None
    return float((a_values.mean() - b_values.mean()) / (pooled_var ** 0.5))


def _iqr_overlap_proxy(a_values: pd.Series, b_values: pd.Series) -> float | None:
    if a_values.empty or b_values.empty:
        return None

    a_low, a_high = float(a_values.quantile(0.25)), float(a_values.quantile(0.75))
    b_low, b_high = float(b_values.quantile(0.25)), float(b_values.quantile(0.75))
    union_low = min(a_low, b_low)
    union_high = max(a_high, b_high)
    union_width = union_high - union_low
    if union_width <= 0:
        return 1.0

    overlap_low = max(a_low, b_low)
    overlap_high = min(a_high, b_high)
    overlap_width = max(0.0, overlap_high - overlap_low)
    return float(overlap_width / union_width)
