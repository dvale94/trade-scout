from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


OUTSIDE_CANDIDATE_BAND = "outside_candidate_band"
TARGET_REALIZED_COLUMNS = ["pnl_R", "realized_r", "outcome_r", "outcome_R"]
TARGET_MFE_COLUMNS = ["mfe_R", "mfe_r"]
TARGET_MAE_COLUMNS = ["mae_R", "mae_r"]
DEFAULT_CANDIDATE_BANDS = [
    {
        "name": "clean_mid_pullback",
        "ranges": {
            "breakout_distance_atr": {"gte": 0.35, "lt": 0.50},
            "ma20_distance_atr": {"gte": 1.5, "lt": 2.0},
        },
    },
    {
        "name": "clean_upper_shallow",
        "ranges": {
            "breakout_distance_atr": {"gte": 0.50, "lt": 0.80},
            "ma20_distance_atr": {"gte": 1.0, "lt": 1.5},
        },
    },
    {
        "name": "clean_upper_mid",
        "ranges": {
            "breakout_distance_atr": {"gte": 0.50, "lt": 0.80},
            "ma20_distance_atr": {"gte": 1.5, "lt": 2.0},
        },
    },
]


def assign_candidate_bands(
    df: pd.DataFrame,
    band_specs: Sequence[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Assign named candidate rectangular bands to a trade frame."""
    out = df.copy()
    out["candidate_band"] = OUTSIDE_CANDIDATE_BAND
    specs = list(DEFAULT_CANDIDATE_BANDS if band_specs is None else band_specs)

    for spec in specs:
        band_name = str(spec.get("name", OUTSIDE_CANDIDATE_BAND))
        ranges = spec.get("ranges", {})
        if not isinstance(ranges, dict) or not band_name:
            continue

        mask = pd.Series(True, index=out.index, dtype=bool)
        for feature, bounds in ranges.items():
            series = _numeric_series(out, feature)
            if not isinstance(bounds, dict):
                mask &= False
                continue
            if "gte" in bounds:
                mask &= series.ge(float(bounds["gte"]))
            if "gt" in bounds:
                mask &= series.gt(float(bounds["gt"]))
            if "lte" in bounds:
                mask &= series.le(float(bounds["lte"]))
            if "lt" in bounds:
                mask &= series.lt(float(bounds["lt"]))
        out.loc[mask.fillna(False), "candidate_band"] = band_name

    return out


def summarize_candidate_bands(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize candidate bands optionally segmented by additional grouping columns."""
    prepared = assign_candidate_bands(df) if "candidate_band" not in df.columns else df.copy()
    group_columns = ["candidate_band", *(group_cols or [])]
    prepared = _with_group_columns(prepared, group_columns)
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "n",
                "win_rate",
                "avg_realized_r",
                "median_realized_r",
                "avg_mfe",
                "avg_mae",
            ]
        )

    realized = _first_available_numeric_series(prepared, TARGET_REALIZED_COLUMNS)
    mfe = _first_available_numeric_series(prepared, TARGET_MFE_COLUMNS)
    mae = _first_available_numeric_series(prepared, TARGET_MAE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for group_key, group_df in prepared.groupby(group_columns, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_columns))
        row = {column: value for column, value in zip(group_columns, key_values)}

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

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64", index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


def _with_group_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in group_cols:
        if column not in out.columns:
            out[column] = "unknown"
    return out


def _group_key_values(group_key: Any, group_col_count: int) -> tuple[Any, ...]:
    if group_col_count == 1:
        if isinstance(group_key, tuple):
            return (group_key[0],)
        return (group_key,)
    return tuple(group_key)
