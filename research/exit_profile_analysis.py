from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


UNKNOWN = "unknown"
TARGET_REALIZED_COLUMNS = ["pnl_R", "realized_r", "outcome_r", "outcome_R"]
TARGET_MFE_COLUMNS = ["mfe_R", "mfe_r"]
DEFAULT_PROFILES = (
    "baseline_realized",
    "tp_0_5R_full",
    "tp_1R_full",
    "tp_1_5R_full",
    "tp_2R_full",
    "half_at_1R_rest_baseline",
    "half_at_1R_rest_breakeven",
)


def simulate_exit_profiles(
    df: pd.DataFrame,
    group_cols: list[str],
    profiles: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Simulate grouped exit-profile summaries using trade-level outcome aggregates only.

    These profiles are intentionally simple and conservative. MFE/MAE fields do
    not preserve intratrade path ordering, so no profile here infers candle-by-
    candle sequencing beyond "threshold was reached at some point".
    """
    grouped = _with_group_columns(df, group_cols)
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "profile",
                "n",
                "avg_simulated_r",
                "median_simulated_r",
                "win_rate",
                "avg_delta_vs_baseline",
                "median_delta_vs_baseline",
            ]
        )

    selected_profiles = list(DEFAULT_PROFILES if profiles is None else profiles)
    realized = _first_available_numeric_series(grouped, TARGET_REALIZED_COLUMNS)
    mfe = _first_available_numeric_series(grouped, TARGET_MFE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped.groupby(group_cols, observed=False, dropna=False):
        key_values = _group_key_values(group_key, len(group_cols))
        base_row = {column: value for column, value in zip(group_cols, key_values)}

        baseline_group = realized.loc[group_df.index]
        for profile in selected_profiles:
            simulated = _simulate_profile(profile, baseline_group, mfe.loc[group_df.index])
            valid_simulated = simulated.dropna()
            valid_baseline = baseline_group.dropna()
            delta = (simulated - baseline_group).dropna()

            row = {
                **base_row,
                "profile": profile,
                "n": int(len(group_df)),
                "avg_simulated_r": float(valid_simulated.mean()) if not valid_simulated.empty else None,
                "median_simulated_r": float(valid_simulated.median()) if not valid_simulated.empty else None,
                "win_rate": float((valid_simulated > 0).mean()) if not valid_simulated.empty else None,
                "avg_delta_vs_baseline": float(delta.mean()) if not delta.empty else None,
                "median_delta_vs_baseline": float(delta.median()) if not delta.empty else None,
            }
            if valid_baseline.empty and profile == "baseline_realized":
                row["avg_simulated_r"] = None
                row["median_simulated_r"] = None
                row["win_rate"] = None
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values([*group_cols, "profile"]).reset_index(drop=True)


def _simulate_profile(profile: str, baseline: pd.Series, mfe: pd.Series) -> pd.Series:
    if profile == "baseline_realized":
        return baseline.copy()
    if profile == "tp_0_5R_full":
        return _tp_full_profile(baseline, mfe, 0.5)
    if profile == "tp_1R_full":
        return _tp_full_profile(baseline, mfe, 1.0)
    if profile == "tp_1_5R_full":
        return _tp_full_profile(baseline, mfe, 1.5)
    if profile == "tp_2R_full":
        return _tp_full_profile(baseline, mfe, 2.0)
    if profile == "half_at_1R_rest_baseline":
        return _half_at_1r_rest_baseline(baseline, mfe)
    if profile == "half_at_1R_rest_breakeven":
        return _half_at_1r_rest_breakeven(baseline, mfe)
    return baseline.copy()


def _tp_full_profile(baseline: pd.Series, mfe: pd.Series, threshold: float) -> pd.Series:
    simulated = baseline.copy()
    reached = mfe >= threshold
    simulated = simulated.where(~reached, other=float(threshold))
    return simulated


def _half_at_1r_rest_baseline(baseline: pd.Series, mfe: pd.Series) -> pd.Series:
    simulated = baseline.copy()
    reached = mfe >= 1.0
    # Half exits at +1R; the remaining half keeps the recorded baseline result.
    replacement = 0.5 + (0.5 * baseline)
    simulated = simulated.where(~reached, other=replacement)
    return simulated


def _half_at_1r_rest_breakeven(baseline: pd.Series, mfe: pd.Series) -> pd.Series:
    simulated = baseline.copy()
    reached = mfe >= 1.0
    # Once 1R is reached, assume half is banked and the runner exits flat.
    simulated = simulated.where(~reached, other=0.5)
    return simulated


def _first_available_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype="float64", index=df.index)


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
