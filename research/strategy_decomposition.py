from __future__ import annotations

from typing import Any

import pandas as pd


UNKNOWN = "unknown"
UNCLASSIFIED = "unclassified"
SUBTYPE_COLUMNS = [
    "breakout_quality",
    "pullback_structure",
    "volatility_context",
    "trend_context",
    "liquidity_volume_context",
    "setup_family",
]
TARGET_COLUMN_CANDIDATES = [
    "pnl_R",
    "realized_r",
    "outcome_r",
    "outcome_R",
    "mfe_R",
    "mfe_r",
    "mae_R",
    "mae_r",
]
DECOMPOSITION_MODES = {"fixed", "quantile"}


def safe_col(df: pd.DataFrame, col: str, default: Any = None) -> pd.Series:
    """Return an existing column or an index-aligned default-filled series."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def derive_decomposition_features(
    df: pd.DataFrame,
    *,
    decomposition_mode: str = "fixed",
    quantile_count: int = 3,
) -> pd.DataFrame:
    """
    Derive null-safe intermediate research features for setup decomposition.

    This helper is feature-only by design. It must not use outcome or target
    columns when deriving subtype inputs.
    """
    out = df.copy()
    resolved_mode = _normalize_decomposition_mode(decomposition_mode)

    breakout_distance = _numeric_col(out, "breakout_distance_atr")
    ma20_distance = _abs_numeric_col(out, "ma20_distance_atr")
    ma50_distance = _abs_numeric_col(out, "ma50_distance_atr")
    ma200_distance = _abs_numeric_col(out, "ma200_distance_atr")
    ma20_slope = _numeric_col(out, "ma20_slope_pct")
    ma50_slope = _numeric_col(out, "ma50_slope_pct")
    vol_z = _numeric_col(out, "vol_z")
    extended_atr = _numeric_col(out, "extended_atr")
    ma20 = _numeric_col(out, "ma20")
    ma50 = _numeric_col(out, "ma50")
    ma200 = _numeric_col(out, "ma200")

    out["source_setup_family"] = safe_col(out, "setup_family", default=None)
    out["anchor_regime_normalized"] = (
        safe_col(out, "anchor_regime", default=UNKNOWN).fillna(UNKNOWN).astype(str).str.strip().str.lower()
    )
    out["breakout_distance_atr_num"] = breakout_distance
    out["ma20_distance_atr_abs"] = ma20_distance
    out["ma50_distance_atr_abs"] = ma50_distance
    out["ma200_distance_atr_abs"] = ma200_distance
    out["ma20_slope_pct_num"] = ma20_slope
    out["ma50_slope_pct_num"] = ma50_slope
    out["vol_z_num"] = vol_z
    out["extended_atr_num"] = extended_atr
    out["trend_stack_up"] = ma20.gt(ma50) & ma50.gt(ma200)
    out["trend_stack_down"] = ma20.lt(ma50) & ma50.lt(ma200)
    out["short_term_slope_up"] = ma20_slope.gt(0)
    out["intermediate_slope_up"] = ma50_slope.gt(0)
    out["decomposition_mode"] = resolved_mode

    if resolved_mode == "quantile":
        out["breakout_quality_quantile"] = derive_quantile_buckets(
            out,
            "breakout_distance_atr_num",
            labels=["tight", "clean", "extended"],
            quantile_count=quantile_count,
        )
        out["pullback_structure_quantile"] = derive_quantile_buckets(
            out,
            "ma20_distance_atr_abs",
            labels=["shallow", "moderate", "deep"],
            quantile_count=quantile_count,
        )
        out["liquidity_volume_context_quantile"] = derive_quantile_buckets(
            out,
            "vol_z_num",
            labels=["thin_relative_volume", "normal_relative_volume", "high_relative_volume"],
            quantile_count=quantile_count,
        )

    return out


def assign_setup_subtypes(
    df: pd.DataFrame,
    *,
    decomposition_mode: str = "fixed",
    quantile_count: int = 3,
) -> pd.DataFrame:
    """
    Add deterministic research subtype labels without using outcome columns.

    Subtype assignment must remain explainable and target-free so the labels can
    be used for post-hoc research without leaking trade outcomes into the setup
    taxonomy.
    """
    resolved_mode = _normalize_decomposition_mode(decomposition_mode)
    out = derive_decomposition_features(
        df,
        decomposition_mode=resolved_mode,
        quantile_count=quantile_count,
    )

    breakout_distance = _numeric_col(out, "breakout_distance_atr_num")
    ma20_distance = _numeric_col(out, "ma20_distance_atr_abs")
    ma50_distance = _numeric_col(out, "ma50_distance_atr_abs")
    vol_z = _numeric_col(out, "vol_z_num")
    extended_atr = _numeric_col(out, "extended_atr_num")
    anchor_regime = safe_col(out, "anchor_regime_normalized", default=UNKNOWN)
    trend_stack_up = _bool_col(out, "trend_stack_up")
    trend_stack_down = _bool_col(out, "trend_stack_down")
    short_term_slope_up = _bool_col(out, "short_term_slope_up")
    intermediate_slope_up = _bool_col(out, "intermediate_slope_up")

    if resolved_mode == "quantile":
        out["breakout_quality"] = safe_col(out, "breakout_quality_quantile", default=UNKNOWN).fillna(UNKNOWN)
        out.loc[breakout_distance.lt(0), "breakout_quality"] = "below_breakout_level"

        out["pullback_structure"] = safe_col(out, "pullback_structure_quantile", default=UNKNOWN).fillna(UNKNOWN)
    else:
        out["breakout_quality"] = UNKNOWN
        # tight/clean/extended are breakout-distance buckets, not outcome grades.
        out.loc[breakout_distance.lt(0), "breakout_quality"] = "below_breakout_level"
        out.loc[breakout_distance.between(0, 0.35, inclusive="both"), "breakout_quality"] = "tight"
        out.loc[breakout_distance.gt(0.35) & breakout_distance.le(0.8), "breakout_quality"] = "clean"
        out.loc[breakout_distance.gt(0.8), "breakout_quality"] = "extended"

        out["pullback_structure"] = UNKNOWN
        # shallow/moderate/deep are MA-distance pullback proxies.
        out.loc[ma20_distance.le(0.35) & ma50_distance.le(0.9), "pullback_structure"] = "shallow"
        out.loc[
            ma20_distance.gt(0.35) & ma20_distance.le(0.8) & ma50_distance.le(1.2),
            "pullback_structure",
        ] = "moderate"
        out.loc[ma50_distance.gt(1.2), "pullback_structure"] = "deep"

    out["volatility_context"] = UNKNOWN
    out.loc[(vol_z.ge(1.0)) | (extended_atr.ge(1.25)), "volatility_context"] = "high_volatility"
    out.loc[(vol_z.le(-0.5)) & (extended_atr.le(0.85)), "volatility_context"] = "compressed"
    out.loc[
        out["volatility_context"].eq(UNKNOWN) & (vol_z.notna() | extended_atr.notna()),
        "volatility_context",
    ] = "normal_volatility"

    out["trend_context"] = UNKNOWN
    out.loc[anchor_regime.eq("bull"), "trend_context"] = "bull_anchor"
    out.loc[anchor_regime.eq("bear"), "trend_context"] = "bear_anchor"
    out.loc[
        out["trend_context"].eq(UNKNOWN) & trend_stack_up & short_term_slope_up & intermediate_slope_up,
        "trend_context",
    ] = "stacked_uptrend"
    out.loc[out["trend_context"].eq(UNKNOWN) & trend_stack_up, "trend_context"] = "uptrend"
    out.loc[out["trend_context"].eq(UNKNOWN) & trend_stack_down, "trend_context"] = "downtrend"
    out.loc[
        out["trend_context"].eq(UNKNOWN) & (trend_stack_up | trend_stack_down),
        "trend_context",
    ] = "mixed_trend"

    if resolved_mode == "quantile":
        out["liquidity_volume_context"] = (
            safe_col(out, "liquidity_volume_context_quantile", default=UNKNOWN).fillna(UNKNOWN)
        )
    else:
        out["liquidity_volume_context"] = UNKNOWN
        out.loc[vol_z.ge(1.0), "liquidity_volume_context"] = "high_relative_volume"
        out.loc[vol_z.le(-0.75), "liquidity_volume_context"] = "thin_relative_volume"
        out.loc[
            out["liquidity_volume_context"].eq(UNKNOWN) & vol_z.notna(),
            "liquidity_volume_context",
        ] = "normal_relative_volume"

    # setup_family here is a derived research label, not the original runtime setup_family.
    out["setup_family"] = UNCLASSIFIED
    out.loc[
        out["breakout_quality"].isin(["tight", "clean"])
        & out["pullback_structure"].eq("shallow")
        & out["trend_context"].isin(["bull_anchor", "stacked_uptrend", "uptrend"]),
        "setup_family",
    ] = "trend_continuation_shallow_pullback"
    out.loc[
        out["breakout_quality"].isin(["tight", "clean"])
        & out["pullback_structure"].eq("moderate")
        & out["trend_context"].isin(["bull_anchor", "stacked_uptrend", "uptrend"]),
        "setup_family",
    ] = "trend_continuation_moderate_pullback"
    out.loc[
        out["breakout_quality"].eq("extended")
        & out["volatility_context"].eq("high_volatility"),
        "setup_family",
    ] = "momentum_extension"
    out.loc[out["pullback_structure"].eq("deep"), "setup_family"] = "deep_pullback_reclaim"
    out.loc[out["breakout_quality"].eq("below_breakout_level"), "setup_family"] = "pre_breakout_retest"

    return out


def summarize_subtype_expectancy(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    decomposition_mode: str = "fixed",
    quantile_count: int = 3,
) -> pd.DataFrame:
    """Summarize grouped expectancy metrics using already-recorded outcome fields."""
    prepared = assign_setup_subtypes(
        df,
        decomposition_mode=decomposition_mode,
        quantile_count=quantile_count,
    )
    grouped = _with_group_columns(prepared, group_cols)
    if grouped.empty:
        return pd.DataFrame(columns=[*group_cols, "n"])

    realized = _first_available_numeric_series(grouped, TARGET_COLUMN_CANDIDATES[:4])
    win_mask = _first_available_win_mask(grouped, realized)
    mfe = _first_available_numeric_series(grouped, TARGET_COLUMN_CANDIDATES[4:6])
    mae = _first_available_numeric_series(grouped, TARGET_COLUMN_CANDIDATES[6:8])

    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped.groupby(group_cols, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_cols))
        row = {column: value for column, value in zip(group_cols, key_values)}
        row["n"] = int(len(group_df))

        realized_group = realized.loc[group_df.index].dropna()
        win_group = win_mask.loc[group_df.index].dropna()
        mfe_group = mfe.loc[group_df.index].dropna()
        mae_group = mae.loc[group_df.index].dropna()

        if not win_group.empty:
            row["win_rate"] = float(win_group.mean())
        if not realized_group.empty:
            row["avg_realized_r"] = float(realized_group.mean())
            row["median_realized_r"] = float(realized_group.median())
        if not mfe_group.empty:
            row["avg_mfe"] = float(mfe_group.mean())
        if not mae_group.empty:
            row["avg_mae"] = float(mae_group.mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_exit_efficiency(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    decomposition_mode: str = "fixed",
    quantile_count: int = 3,
) -> pd.DataFrame:
    """Estimate realized-vs-MFE monetization efficiency where summary fields exist."""
    prepared = assign_setup_subtypes(
        df,
        decomposition_mode=decomposition_mode,
        quantile_count=quantile_count,
    )
    grouped = _with_group_columns(prepared, group_cols)
    if grouped.empty:
        return pd.DataFrame(columns=[*group_cols, "n"])

    realized = _first_available_numeric_series(grouped, TARGET_COLUMN_CANDIDATES[:4])
    mfe = _first_available_numeric_series(grouped, TARGET_COLUMN_CANDIDATES[4:6])
    valid_denominator = mfe.where(mfe.gt(0))
    efficiency = realized.div(valid_denominator)

    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped.groupby(group_cols, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_cols))
        row = {column: value for column, value in zip(group_cols, key_values)}
        row["n"] = int(len(group_df))

        efficiency_group = efficiency.loc[group_df.index].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not efficiency_group.empty:
            row["mean_exit_efficiency"] = float(efficiency_group.mean())
            row["median_exit_efficiency"] = float(efficiency_group.median())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(safe_col(df, col, default=None), errors="coerce")


def derive_quantile_buckets(
    df: pd.DataFrame,
    column: str,
    *,
    labels: list[str],
    quantile_count: int = 3,
    default: str = UNKNOWN,
) -> pd.Series:
    """
    Derive explainable percentile buckets for a numeric feature.

    This is research-only helper logic layered on top of fixed thresholds so we
    can compare balanced decompositions without changing the default taxonomy.
    """
    values = _numeric_col(df, column)
    out = pd.Series(default, index=df.index, dtype="object")
    valid = values.dropna()
    if valid.empty or quantile_count <= 1 or len(labels) != quantile_count:
        return out

    distinct_count = int(valid.nunique(dropna=True))
    if distinct_count < quantile_count:
        return out

    ranked = valid.rank(method="first")
    try:
        bucketed = pd.qcut(ranked, q=quantile_count, labels=labels)
    except ValueError:
        return out
    out.loc[bucketed.index] = bucketed.astype(str)
    return out


def _abs_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    return _numeric_col(df, col).abs()


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    series = safe_col(df, col, default=False)
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _with_group_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in group_cols:
        if column not in out.columns:
            out[column] = UNKNOWN
    return out


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


def _first_available_win_mask(df: pd.DataFrame, realized: pd.Series) -> pd.Series:
    for column in ("win", "won", "profitable", "is_winner"):
        if column not in df.columns:
            continue
        raw = safe_col(df, column, default=None)
        if raw.dtype == bool:
            return raw
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().any():
            return numeric.gt(0)
        text = raw.fillna("").astype(str).str.strip().str.upper()
        return text.isin({"TRUE", "T", "YES", "Y", "1", "WIN", "WON"})
    return realized.gt(0)


def _group_key_values(group_key: Any, group_col_count: int) -> tuple[Any, ...]:
    if group_col_count == 1:
        if isinstance(group_key, tuple):
            return (group_key[0],)
        return (group_key,)
    return tuple(group_key)


def _normalize_decomposition_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in DECOMPOSITION_MODES:
        return "fixed"
    return normalized
