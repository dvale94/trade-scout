from __future__ import annotations
import pandas as pd


_DEFAULT_EXCURSION_BINS = [-float("inf"), 0.0, 0.5, 1.0, 1.5, 2.5, float("inf")]
_DEFAULT_EXCURSION_LABELS = ["LT_0", "0_TO_0_5", "0_5_TO_1_0", "1_0_TO_1_5", "1_5_TO_2_5", "2_5_PLUS"]


def _coerce_excursion_metrics(trades: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    if trades.empty:
        empty = pd.Series(dtype="float64")
        return empty, empty, empty

    def _col(name: str) -> pd.Series:
        if name not in trades.columns:
            return pd.Series(index=trades.index, dtype="float64")
        return pd.to_numeric(trades[name], errors="coerce")

    return _col("mfe_R"), _col("mae_R"), _col("pnl_R")


def summarize_mfe_mae(trades: pd.DataFrame) -> dict:
    mfe, mae, _ = _coerce_excursion_metrics(trades)
    mfe_valid = mfe.dropna()
    mae_valid = mae.dropna()
    trade_count = int(max(len(mfe_valid), len(mae_valid)))

    def _pct_ge(series: pd.Series, threshold: float) -> float | None:
        if series.empty:
            return None
        return float((series >= threshold).mean())

    return {
        "trades": trade_count,
        "avg_mfe_R": float(mfe_valid.mean()) if not mfe_valid.empty else None,
        "median_mfe_R": float(mfe_valid.median()) if not mfe_valid.empty else None,
        "avg_mae_R": float(mae_valid.mean()) if not mae_valid.empty else None,
        "median_mae_R": float(mae_valid.median()) if not mae_valid.empty else None,
        "pct_reached_0_5R": _pct_ge(mfe_valid, 0.5),
        "pct_reached_1_0R": _pct_ge(mfe_valid, 1.0),
        "pct_reached_1_5R": _pct_ge(mfe_valid, 1.5),
        "pct_reached_2_5R": _pct_ge(mfe_valid, 2.5),
        "pct_mae_ge_0_5R": _pct_ge(mae_valid, 0.5),
        "pct_mae_ge_1_0R": _pct_ge(mae_valid, 1.0),
    }


def excursion_buckets(
    trades: pd.DataFrame,
    column: str = "mfe_R",
    bins=None,
    labels=None,
) -> pd.DataFrame:
    if trades.empty or column not in trades.columns:
        return pd.DataFrame()

    series = pd.to_numeric(trades[column], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame()

    bucket_bins = bins if bins is not None else _DEFAULT_EXCURSION_BINS
    bucket_labels = labels if labels is not None else _DEFAULT_EXCURSION_LABELS

    bucketed = pd.cut(series, bins=bucket_bins, labels=bucket_labels, include_lowest=True)
    counts = bucketed.value_counts(sort=False, dropna=False)
    total = int(counts.sum())

    return pd.DataFrame(
        {
            "bucket": counts.index.astype(str),
            "count": counts.values.astype(int),
            "pct": (counts.values / total) if total else 0.0,
        }
    )


def grouped_mfe_mae_summary(
    df: pd.DataFrame,
    by,
    min_trades: int = 1,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if isinstance(by, str):
        group_cols = [by]
    else:
        group_cols = list(by)

    if not group_cols or any(col not in df.columns for col in group_cols):
        return pd.DataFrame()

    rows = []
    for group_key, group_df in df.groupby(group_cols, observed=False, dropna=False):
        summary = summarize_mfe_mae(group_df)
        if summary["trades"] < min_trades:
            continue

        if len(group_cols) == 1:
            row = {group_cols[0]: group_key}
        else:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(group_cols, group_key))
        row.update(summary)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["trades"], ascending=[False]).reset_index(drop=True)


def edge_diagnostics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    mfe, mae, _ = _coerce_excursion_metrics(trades)
    mfe = mfe.dropna()
    mae = mae.dropna()

    def pct_gt(series: pd.Series, x: float) -> float | None:
        if series.empty:
            return None
        return float((series > x).mean())

    def pct_lt(series: pd.Series, x: float) -> float | None:
        if series.empty:
            return None
        return float((series < x).mean())

    return {
        "mfe_gt_1R": pct_gt(mfe, 1.0),
        "mfe_gt_2R": pct_gt(mfe, 2.0),
        "mae_lt_1R": pct_lt(mae, 1.0),
    }
