from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


_DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -----------------------------
# Schema / coverage helpers
# -----------------------------
def requested_columns_present(trades: pd.DataFrame, columns: Sequence[str]) -> dict[str, bool]:
    present = set(trades.columns)
    return {col: col in present for col in columns}


def non_null_column_coverage(trades: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    total_rows = int(len(trades))

    for col in columns:
        if col not in trades.columns:
            rows.append(
                {
                    "column": col,
                    "present": False,
                    "non_null_rows": 0,
                    "coverage_pct": 0.0,
                }
            )
            continue

        non_null_rows = int(trades[col].notna().sum())
        rows.append(
            {
                "column": col,
                "present": True,
                "non_null_rows": non_null_rows,
                "coverage_pct": float(non_null_rows / total_rows) if total_rows else 0.0,
            }
        )

    return pd.DataFrame(rows)


def split_core_and_advanced_rows(
    trades: pd.DataFrame,
    *,
    core_columns: Sequence[str] = ("pnl_R",),
    advanced_feature_columns: Sequence[str] = (),
) -> dict[str, object]:
    if trades.empty:
        return {
            "core_rows": pd.DataFrame(),
            "advanced_rows": pd.DataFrame(),
            "missing_core_columns": list(core_columns),
            "missing_advanced_columns": list(advanced_feature_columns),
            "core_row_count": 0,
            "advanced_row_count": 0,
        }

    core_present = [col for col in core_columns if col in trades.columns]
    advanced_present = [col for col in advanced_feature_columns if col in trades.columns]
    missing_core = [col for col in core_columns if col not in trades.columns]
    missing_advanced = [col for col in advanced_feature_columns if col not in trades.columns]

    core_rows = trades.copy()
    for col in core_present:
        core_rows = core_rows[core_rows[col].notna()]

    advanced_rows = core_rows.copy()
    for col in advanced_present:
        advanced_rows = advanced_rows[advanced_rows[col].notna()]

    return {
        "core_rows": core_rows.copy(),
        "advanced_rows": advanced_rows.copy(),
        "missing_core_columns": missing_core,
        "missing_advanced_columns": missing_advanced,
        "core_row_count": int(len(core_rows)),
        "advanced_row_count": int(len(advanced_rows)),
    }


# -----------------------------
# Type coercion helpers
# -----------------------------
def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def to_dt(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)
    return out


def to_num(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def to_boolish(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if out[c].dtype == object:
            out[c] = out[c].map(
                lambda x: True if str(x).strip().lower() in {"true", "1", "yes"} else (
                    False if str(x).strip().lower() in {"false", "0", "no"} else x
                )
            )
        try:
            out[c] = out[c].astype("boolean")
        except Exception:
            pass
    return out


# -----------------------------
# Frame preparation
# -----------------------------
def prepare_trade_frames(
    trades: pd.DataFrame,
    alerts_intents: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    telemetry: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    prepared_trades = to_dt(
        trades,
        [
            "signal_ts_utc",
            "entry_ts_utc",
            "exit_ts_utc",
            "bar_start_ts_utc",
            "bar_end_ts_utc",
            "first_touch_ts_utc",
            "entry_fill_ts_utc",
        ],
    )
    prepared_trades = to_num(
        prepared_trades,
        [
            "score",
            "signal_px",
            "entry_low",
            "entry_high",
            "stop_px",
            "t1_px",
            "t2_px",
            "entry_px",
            "exit_px",
            "pnl_R",
            "mfe_R",
            "mae_R",
            "mfe_r",
            "mae_r",
            "mfe_px",
            "mae_px",
            "time_in_trade_s",
            "time_to_t1_s",
            "time_to_t2_s",
            "time_to_stop_s",
            "atr",
            "ma20",
            "ma50",
            "ma200",
            "breakout_level",
            "vol_z",
            "extended_atr",
            "stop_distance",
            "timeout_seconds",
            "entry_fill_px",
            "slippage_pct",
            "signal_hour_utc",
            "signal_dow_utc",
        ],
    )
    prepared_trades = to_boolish(prepared_trades, ["entered", "t1_touched", "t2_touched", "stop_touched"])

    for src, dst in [
        ("time_in_trade_s", "time_in_trade_min"),
        ("time_to_stop_s", "time_to_stop_min"),
        ("time_to_t1_s", "time_to_t1_min"),
        ("time_to_t2_s", "time_to_t2_min"),
    ]:
        if src in prepared_trades.columns:
            prepared_trades[dst] = prepared_trades[src] / 60.0

    if "signal_ts_utc" in prepared_trades.columns:
        ts = prepared_trades["signal_ts_utc"]
        if "signal_hour_utc" not in prepared_trades.columns or prepared_trades["signal_hour_utc"].isna().all():
            prepared_trades["signal_hour_utc"] = ts.dt.hour
        if "signal_dow_utc" not in prepared_trades.columns or prepared_trades["signal_dow_utc"].isna().all():
            prepared_trades["signal_dow_utc"] = ts.dt.dayofweek
        if "signal_day_of_week" not in prepared_trades.columns:
            prepared_trades["signal_day_of_week"] = ts.dt.day_name().str[:3]

    if "entry_high" in prepared_trades.columns and "entry_fill_px" in prepared_trades.columns:
        eh = pd.to_numeric(prepared_trades["entry_high"], errors="coerce")
        ef = pd.to_numeric(prepared_trades["entry_fill_px"], errors="coerce")
        prepared_trades["entry_efficiency_pct"] = np.where(
            eh > 0,
            ((ef - eh) / eh) * 100.0,
            np.nan,
        )

    if "mfe_R" in prepared_trades.columns and "pnl_R" in prepared_trades.columns:
        pnl = pd.to_numeric(prepared_trades["pnl_R"], errors="coerce")
        mfe = pd.to_numeric(prepared_trades["mfe_R"], errors="coerce")
        prepared_trades["gave_back_R"] = np.where(
            pnl.notna() & mfe.notna(),
            mfe - pnl,
            np.nan,
        )

    if "time_in_trade_min" in prepared_trades.columns:
        t = pd.to_numeric(prepared_trades["time_in_trade_min"], errors="coerce")
        prepared_trades["trade_duration_bucket"] = pd.cut(
            t,
            bins=[-np.inf, 5, 15, 30, 60, 120, np.inf],
            labels=["<=5m", "5-15m", "15-30m", "30-60m", "60-120m", ">120m"],
        )

    prepared_alerts = alerts_intents.copy() if alerts_intents is not None else pd.DataFrame()
    if not prepared_alerts.empty:
        prepared_alerts = to_dt(prepared_alerts, ["signal_ts_utc"])
        prepared_alerts = to_num(
            prepared_alerts,
            [
                "signal_px",
                "entry_low",
                "entry_high",
                "stop_px",
                "t1_px",
                "t2_px",
                "timeout_seconds",
                "stop_distance",
                "t1_R",
                "t2_R",
            ],
        )

    prepared_outcomes = outcomes.copy() if outcomes is not None else pd.DataFrame()
    if not prepared_outcomes.empty:
        prepared_outcomes = to_dt(
            prepared_outcomes,
            ["entry_ts_utc", "exit_ts_utc", "first_touch_ts_utc", "entry_fill_ts_utc"],
        )
        prepared_outcomes = to_num(
            prepared_outcomes,
            [
                "entry_px",
                "exit_px",
                "pnl_R",
                "mfe_R",
                "mae_R",
                "mfe_r",
                "mae_r",
                "mfe_px",
                "mae_px",
                "time_in_trade_s",
                "time_to_t1_s",
                "time_to_t2_s",
                "time_to_stop_s",
                "entry_fill_px",
                "slippage_pct",
            ],
        )
        prepared_outcomes = to_boolish(prepared_outcomes, ["entered", "t1_touched", "t2_touched", "stop_touched"])

    prepared_telemetry = telemetry.copy() if telemetry is not None else pd.DataFrame()
    if not prepared_telemetry.empty:
        prepared_telemetry = to_dt(prepared_telemetry, ["timestamp_utc"])

    return {
        "trades": prepared_trades,
        "alerts_intents": prepared_alerts,
        "outcomes": prepared_outcomes,
        "telemetry": prepared_telemetry,
    }


# -----------------------------
# Population helpers
# -----------------------------
def filter_strategy_version(trades: pd.DataFrame, contains: str = "v3") -> pd.DataFrame:
    if trades.empty or "strategy_version" not in trades.columns:
        return trades.copy()
    filtered = trades[trades["strategy_version"].astype(str).str.contains(contains, case=False, na=False)].copy()
    return filtered if not filtered.empty else trades.copy()


def split_trade_population(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {"all": trades.copy(), "closed": pd.DataFrame(), "entered": pd.DataFrame()}
    closed = trades[trades["pnl_R"].notna()].copy() if "pnl_R" in trades.columns else pd.DataFrame()
    entered = trades[trades["entered"] == True].copy() if "entered" in trades.columns else pd.DataFrame()
    return {"all": trades.copy(), "closed": closed, "entered": entered}


def safe_div(a, b):
    if b in [0, None] or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b


# -----------------------------
# Core metrics
# -----------------------------
def expectancy(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    wins = s[s > 0]
    losses = s[s < 0]
    win_rate = len(wins) / len(s)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))


def expectancy_std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=1))


def expectancy_t_stat(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    stdev = s.std(ddof=1)
    if stdev == 0 or pd.isna(stdev):
        return np.nan
    return float((s.mean() / stdev) * np.sqrt(len(s)))


def bootstrap_expectancy_ci(
    series: pd.Series,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(s) == 0:
        return pd.Series(
            {
                "trades": 0,
                "expectancy_R": np.nan,
                "ci_low_R": np.nan,
                "ci_high_R": np.nan,
                "ci_width_R": np.nan,
                "ci_excludes_zero": np.nan,
            }
        )

    rng = np.random.default_rng(random_state)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(s, size=len(s), replace=True)
        means[i] = float(sample.mean())

    alpha = 1 - ci
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    center = float(np.mean(s))
    return pd.Series(
        {
            "trades": int(len(s)),
            "expectancy_R": center,
            "ci_low_R": low,
            "ci_high_R": high,
            "ci_width_R": high - low,
            "ci_excludes_zero": bool((low > 0) or (high < 0)),
        }
    )


def core_metrics(df: pd.DataFrame) -> pd.Series:
    if df.empty or "pnl_R" not in df.columns:
        return pd.Series(dtype=object)

    s = pd.to_numeric(df["pnl_R"], errors="coerce").dropna()
    wins = s[s > 0]
    losses = s[s < 0]
    cum = s.cumsum()

    return pd.Series(
        {
            "trades": int(len(s)),
            "win_rate": float(len(wins) / len(s)) if len(s) else np.nan,
            "avg_R": float(s.mean()) if len(s) else np.nan,
            "median_R": float(s.median()) if len(s) else np.nan,
            "avg_win_R": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss_R": float(losses.mean()) if len(losses) else np.nan,
            "expectancy_R": float(expectancy(s)) if len(s) else np.nan,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else (np.inf if wins.sum() > 0 else np.nan),
            "sum_R": float(s.sum()) if len(s) else np.nan,
            "best_R": float(s.max()) if len(s) else np.nan,
            "worst_R": float(s.min()) if len(s) else np.nan,
            "std_R": float(s.std(ddof=0)) if len(s) else np.nan,
            "expectancy_std_R": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "expectancy_t_stat": float(expectancy_t_stat(s)) if len(s) > 1 else np.nan,
            "max_drawdown_R": float(max_drawdown(cum)) if len(s) else np.nan,
        }
    )


def grouped_metrics(df: pd.DataFrame, by, min_trades: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(by, dropna=False, observed=False).apply(core_metrics).reset_index()
    out = out[out["trades"] >= min_trades].sort_values(["expectancy_R", "trades"], ascending=[False, False])
    return out.reset_index(drop=True)


def grouped_bootstrap_ci(df: pd.DataFrame, by, min_trades: int = 5, n_boot: int = 2000) -> pd.DataFrame:
    if df.empty or "pnl_R" not in df.columns:
        return pd.DataFrame()

    rows: list[pd.Series] = []
    for key, sub in df.groupby(by, dropna=False, observed=False):
        pnl = pd.to_numeric(sub["pnl_R"], errors="coerce").dropna()
        if len(pnl) < min_trades:
            continue
        row = bootstrap_expectancy_ci(pnl, n_boot=n_boot)
        row[by if isinstance(by, str) else "group"] = key
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    lead = [by] if isinstance(by, str) else ["group"]
    rest = [c for c in out.columns if c not in lead]
    return out[lead + rest].sort_values(["expectancy_R", "trades"], ascending=[False, False]).reset_index(drop=True)


def with_score_buckets(df: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    if df.empty or "score" not in df.columns:
        return df.copy()
    out = df.copy()
    valid = pd.to_numeric(out["score"], errors="coerce").notna()
    out.loc[valid, "score_bucket"] = pd.qcut(pd.to_numeric(out.loc[valid, "score"], errors="coerce"), q=q, duplicates="drop")
    return out


def threshold_sweep(
    df: pd.DataFrame,
    start=None,
    stop=None,
    step: float = 0.25,
) -> pd.DataFrame:
    if df.empty or "score" not in df.columns or "pnl_R" not in df.columns:
        return pd.DataFrame()

    score = pd.to_numeric(df["score"], errors="coerce")
    pnl = pd.to_numeric(df["pnl_R"], errors="coerce")
    tmp = df.copy()
    tmp["score"] = score
    tmp["pnl_R"] = pnl
    tmp = tmp[tmp["score"].notna()]
    if tmp.empty:
        return pd.DataFrame()

    smin = float(np.floor(tmp["score"].min() * 2) / 2) if start is None else start
    smax = float(np.ceil(tmp["score"].max() * 2) / 2) if stop is None else stop
    thresholds = np.arange(smin, smax + step, step)

    rows = []
    for t in thresholds:
        sub = tmp[tmp["score"] >= t]
        pnl_series = pd.to_numeric(sub["pnl_R"], errors="coerce").dropna()
        ci = bootstrap_expectancy_ci(pnl_series, n_boot=1000) if len(pnl_series) else pd.Series(dtype=object)
        rows.append(
            {
                "score_gte": round(float(t), 2),
                "trades": int(len(pnl_series)),
                "win_rate": float((pnl_series > 0).mean()) if len(pnl_series) else np.nan,
                "avg_R": float(pnl_series.mean()) if len(pnl_series) else np.nan,
                "median_R": float(pnl_series.median()) if len(pnl_series) else np.nan,
                "expectancy_R": float(expectancy(pnl_series)) if len(pnl_series) else np.nan,
                "profit_factor": float(pnl_series[pnl_series > 0].sum() / abs(pnl_series[pnl_series < 0].sum())) if len(pnl_series[pnl_series < 0]) and abs(pnl_series[pnl_series < 0].sum()) > 0 else np.nan,
                "ci_low_R": ci.get("ci_low_R", np.nan),
                "ci_high_R": ci.get("ci_high_R", np.nan),
                "ci_excludes_zero": ci.get("ci_excludes_zero", np.nan),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(cum_returns: pd.Series) -> float:
    s = pd.to_numeric(cum_returns, errors="coerce").dropna()
    if s.empty:
        return np.nan
    running_peak = s.cummax()
    return float((s - running_peak).min())


def add_trade_order(df: pd.DataFrame, ts_col: str = "signal_ts_utc") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if ts_col in out.columns:
        out = out.sort_values(ts_col).reset_index(drop=True)
    out["trade_n"] = range(1, len(out) + 1)
    return out


def equity_curve(df: pd.DataFrame, ts_col: str = "signal_ts_utc") -> pd.DataFrame:
    if df.empty or "pnl_R" not in df.columns:
        return pd.DataFrame()
    out = add_trade_order(df, ts_col=ts_col)
    out["pnl_R"] = pd.to_numeric(out["pnl_R"], errors="coerce")
    out = out[out["pnl_R"].notna()].copy()
    if out.empty:
        return out
    out["cum_R"] = out["pnl_R"].cumsum()
    out["running_peak_R"] = out["cum_R"].cummax()
    out["drawdown_R"] = out["cum_R"] - out["running_peak_R"]
    return out


def rolling_expectancy(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    if df.empty or "pnl_R" not in df.columns:
        return pd.DataFrame()
    out = add_trade_order(df)
    pnl = pd.to_numeric(out["pnl_R"], errors="coerce")
    out[f"rolling_avg_R_{window}"] = pnl.rolling(window).mean()
    out[f"rolling_win_rate_{window}"] = (pnl > 0).astype(float).rolling(window).mean()
    out[f"rolling_sum_R_{window}"] = pnl.rolling(window).sum()
    out[f"rolling_std_R_{window}"] = pnl.rolling(window).std(ddof=1)
    out[f"rolling_t_stat_{window}"] = np.where(
        out[f"rolling_std_R_{window}"] > 0,
        (out[f"rolling_avg_R_{window}"] / out[f"rolling_std_R_{window}"]) * np.sqrt(window),
        np.nan,
    )
    return out


# -----------------------------
# MFE / MAE / lifecycle helpers
# -----------------------------
def mfe_reach_table(trades: pd.DataFrame, levels: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.5)) -> pd.DataFrame:
    mfe = pd.to_numeric(trades.get("mfe_R"), errors="coerce").dropna()
    if mfe.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"level_R": level, "pct_reached": float((mfe >= level).mean())} for level in levels]
    )


def mae_reach_table(trades: pd.DataFrame, levels: Sequence[float] = (0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    mae = pd.to_numeric(trades.get("mae_R"), errors="coerce").dropna()
    if mae.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"level_R": level, "pct_reached": float((mae >= level).mean())} for level in levels]
    )


def loser_mfe_summary(
    trades: pd.DataFrame,
    levels: Sequence[float] = (0.25, 0.5, 1.0, 1.5),
) -> pd.DataFrame:
    if trades.empty or "mfe_R" not in trades.columns or "pnl_R" not in trades.columns:
        return pd.DataFrame()

    losers = trades[pd.to_numeric(trades["pnl_R"], errors="coerce") < 0].copy()
    row: dict[str, float | int] = {"losing_trades": int(len(losers))}
    for level in levels:
        row[f"pct_losers_reached_{level:.2f}R".replace(".00", "")] = float((pd.to_numeric(losers["mfe_R"], errors="coerce") >= level).mean()) if len(losers) else np.nan
    return pd.DataFrame([row])


def outcome_path_summary(trades: pd.DataFrame) -> pd.DataFrame:
    needed = {"pnl_R", "mfe_R", "mae_R"}
    if trades.empty or not needed.issubset(trades.columns):
        return pd.DataFrame()

    pnl = pd.to_numeric(trades["pnl_R"], errors="coerce")
    mfe = pd.to_numeric(trades["mfe_R"], errors="coerce")
    mae = pd.to_numeric(trades["mae_R"], errors="coerce")

    rows = {
        "closed_trades": int(pnl.notna().sum()),
        "losers": int((pnl < 0).sum()),
        "winners": int((pnl > 0).sum()),
        "pct_losers_reached_0.5R_mfe": float(((pnl < 0) & (mfe >= 0.5)).mean()),
        "pct_losers_reached_1.0R_mfe": float(((pnl < 0) & (mfe >= 1.0)).mean()),
        "pct_winners_saw_0.5R_mae": float(((pnl > 0) & (mae >= 0.5)).mean()),
        "pct_winners_saw_1.0R_mae": float(((pnl > 0) & (mae >= 1.0)).mean()),
        "avg_giveback_R": float((mfe - pnl).dropna().mean()) if ((mfe - pnl).dropna().shape[0] > 0) else np.nan,
    }
    return pd.DataFrame([rows])


# -----------------------------
# Coverage / schema focus helpers
# -----------------------------
def schema_focus_coverage(
    trades: pd.DataFrame,
    columns: Sequence[str],
    *,
    closed_only: bool = False,
    include_closed_coverage: bool = False,
) -> pd.DataFrame:
    base = _closed(trades) if closed_only else trades
    rows = []
    total_rows = int(len(base))
    closed_base = _closed(trades) if include_closed_coverage else pd.DataFrame()
    closed_total_rows = int(len(closed_base))
    for col in columns:
        exists = col in base.columns
        non_null_rows = int(base[col].notna().sum()) if exists else 0
        row = {
            "field": col,
            "exists": exists,
            "non_null_rows": non_null_rows,
            "coverage_pct": round(non_null_rows / total_rows, 4) if total_rows else 0.0,
        }
        if include_closed_coverage:
            closed_exists = col in closed_base.columns
            closed_non_null_rows = int(closed_base[col].notna().sum()) if closed_exists else 0
            row["closed_non_null_rows"] = closed_non_null_rows
            row["closed_coverage_pct"] = round(closed_non_null_rows / closed_total_rows, 4) if closed_total_rows else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def slippage_field_coverage(
    trades: pd.DataFrame,
    columns: Sequence[str] = ("entry_fill_px", "entry_fill_ts_utc", "slippage_pct", "entry_efficiency_pct"),
) -> pd.DataFrame:
    available = [col for col in columns if col in trades.columns]
    if trades.empty or not available:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "field": available,
            "non_null_rows": [int(trades[col].notna().sum()) for col in available],
            "coverage_pct": [round(100 * trades[col].notna().mean(), 1) for col in available],
        }
    )


# -----------------------------
# Feature / bucket studies
# -----------------------------
def _closed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pnl_R" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["pnl_R"] = pd.to_numeric(out["pnl_R"], errors="coerce")
    return out[out["pnl_R"].notna()].copy()


def _expectancy_proxy(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _group_quality(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty or by not in df.columns:
        return pd.DataFrame()

    out = df.groupby(by, observed=False).agg(
        trades=(by, "size"),
        closed_trades=("pnl_R", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        wins=("pnl_R", lambda s: (pd.to_numeric(s, errors="coerce") > 0).sum()),
        losses=("pnl_R", lambda s: (pd.to_numeric(s, errors="coerce") < 0).sum()),
        avg_pnl_R=("pnl_R", lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()),
        expectancy_proxy=("pnl_R", _expectancy_proxy),
    ).reset_index()

    out["win_rate"] = out["wins"] / out["closed_trades"].replace(0, pd.NA)
    return out


def bucket_feature_study(
    trades: pd.DataFrame,
    feature_col: str,
    *,
    bins: Sequence[float] | None = None,
    quantiles: int | None = None,
    min_closed_trades: int = 1,
) -> pd.DataFrame:
    if trades.empty or feature_col not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()
    df[feature_col] = _to_numeric(df, feature_col)
    df["pnl_R"] = _to_numeric(df, "pnl_R")
    df = df[df[feature_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    bucket_col = f"{feature_col}_bucket"
    if bins is not None:
        df[bucket_col] = pd.cut(df[feature_col], bins=bins, include_lowest=True)
    else:
        q = quantiles if quantiles is not None else 10
        unique_values = df[feature_col].nunique(dropna=True)
        q = max(1, min(q, int(unique_values)))
        df[bucket_col] = pd.qcut(df[feature_col], q=q, duplicates="drop")

    out = _group_quality(df, by=bucket_col)
    if out.empty:
        return out

    out = out[out["closed_trades"] >= min_closed_trades].copy()
    return out.sort_values([bucket_col]).reset_index(drop=True)


def score_bucket_study(
    trades: pd.DataFrame,
    buckets: int = 10,
    min_closed_trades: int = 1,
) -> pd.DataFrame:
    return bucket_feature_study(
        trades,
        "score",
        quantiles=buckets,
        min_closed_trades=min_closed_trades,
    )


def score_threshold_sweep(
    trades: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    if trades.empty or "score" not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()
    df["score"] = _to_numeric(df, "score")
    df["pnl_R"] = _to_numeric(df, "pnl_R")
    df = df[df["score"].notna()].copy()

    rows: list[dict] = []
    for t in sorted(set(float(x) for x in thresholds)):
        sub = df[df["score"] >= t].copy()
        closed = sub[sub["pnl_R"].notna()].copy()
        wins = int((closed["pnl_R"] > 0).sum()) if not closed.empty else 0
        losses = int((closed["pnl_R"] < 0).sum()) if not closed.empty else 0
        ci = bootstrap_expectancy_ci(closed["pnl_R"], n_boot=1000) if len(closed) else pd.Series(dtype=object)
        rows.append(
            {
                "score_threshold": t,
                "trades": int(len(sub)),
                "closed_trades": int(len(closed)),
                "wins": wins,
                "losses": losses,
                "win_rate": float(wins / len(closed)) if len(closed) else None,
                "avg_pnl_R": float(closed["pnl_R"].mean()) if len(closed) else None,
                "expectancy_proxy": float(closed["pnl_R"].mean()) if len(closed) else None,
                "ci_low_R": ci.get("ci_low_R", np.nan),
                "ci_high_R": ci.get("ci_high_R", np.nan),
                "ci_excludes_zero": ci.get("ci_excludes_zero", np.nan),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_hour(trades: pd.DataFrame, min_closed_trades: int = 1) -> pd.DataFrame:
    if trades.empty or "signal_hour_utc" not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()
    df["signal_hour_utc"] = pd.to_numeric(df["signal_hour_utc"], errors="coerce")
    df = df[df["signal_hour_utc"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    out = _group_quality(df, by="signal_hour_utc")
    out = out[out["closed_trades"] >= min_closed_trades].copy()
    out["signal_hour_utc"] = out["signal_hour_utc"].astype(int)
    return out.sort_values("signal_hour_utc").reset_index(drop=True)


def summarize_by_day_of_week(trades: pd.DataFrame, min_closed_trades: int = 1) -> pd.DataFrame:
    day_col = "signal_day_of_week" if "signal_day_of_week" in trades.columns else "signal_dow_utc"
    if trades.empty or day_col not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()

    if day_col == "signal_dow_utc":
        df[day_col] = pd.to_numeric(df[day_col], errors="coerce")
        df = df[df[day_col].notna()].copy()
        if df.empty:
            return pd.DataFrame()
        df["signal_day"] = df[day_col].astype(int).map({i: d for i, d in enumerate(_DAY_ORDER)})
        by = "signal_day"
    else:
        df[day_col] = df[day_col].astype(str).str[:3]
        by = day_col

    out = _group_quality(df, by=by)
    out = out[out["closed_trades"] >= min_closed_trades].copy()
    out[by] = pd.Categorical(out[by], categories=_DAY_ORDER, ordered=True)
    return out.sort_values(by).reset_index(drop=True)


def slippage_bucket_study(
    trades: pd.DataFrame,
    *,
    bucket_count: int = 4,
    min_trades: int = 1,
) -> pd.DataFrame:
    if trades.empty or "slippage_pct" not in trades.columns or "pnl_R" not in trades.columns:
        return pd.DataFrame()

    valid = trades.copy()
    valid["slippage_pct"] = pd.to_numeric(valid["slippage_pct"], errors="coerce")
    valid["pnl_R"] = pd.to_numeric(valid["pnl_R"], errors="coerce")
    valid = valid[valid["slippage_pct"].notna() & valid["pnl_R"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    unique_values = valid["slippage_pct"].nunique()
    if unique_values >= bucket_count:
        valid["slippage_bucket"] = pd.qcut(valid["slippage_pct"], q=min(bucket_count, unique_values), duplicates="drop")
    else:
        valid["slippage_bucket"] = valid["slippage_pct"]
    return grouped_metrics(valid, "slippage_bucket", min_trades=min_trades)


def entry_efficiency_bucket_study(
    trades: pd.DataFrame,
    *,
    bucket_count: int = 4,
    min_trades: int = 1,
) -> pd.DataFrame:
    if trades.empty or "entry_efficiency_pct" not in trades.columns or "pnl_R" not in trades.columns:
        return pd.DataFrame()

    valid = trades.copy()
    valid["entry_efficiency_pct"] = pd.to_numeric(valid["entry_efficiency_pct"], errors="coerce")
    valid["pnl_R"] = pd.to_numeric(valid["pnl_R"], errors="coerce")
    valid = valid[valid["entry_efficiency_pct"].notna() & valid["pnl_R"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    unique_values = valid["entry_efficiency_pct"].nunique()
    if unique_values >= bucket_count:
        valid["entry_efficiency_bucket"] = pd.qcut(valid["entry_efficiency_pct"], q=min(bucket_count, unique_values), duplicates="drop")
    else:
        valid["entry_efficiency_bucket"] = valid["entry_efficiency_pct"]
    return grouped_metrics(valid, "entry_efficiency_bucket", min_trades=min_trades)


def conditional_pair_metrics(
    trades: pd.DataFrame,
    candidate_dims: Sequence[tuple[str, str]],
    *,
    score_bucket_quantiles: int | None = None,
    min_trades: int = 1,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    tmp = trades.copy()
    dims = list(candidate_dims)
    if score_bucket_quantiles is not None and "score" in tmp.columns:
        tmp = with_score_buckets(tmp, q=score_bucket_quantiles)

    combos: list[pd.DataFrame] = []
    for a, b in dims:
        if a not in tmp.columns or b not in tmp.columns:
            continue
        grouped = (
            tmp.groupby([a, b], dropna=False, observed=False)
            .apply(core_metrics)
            .reset_index()
        )
        if "trades" in grouped.columns:
            grouped = grouped[grouped["trades"] >= min_trades].copy()
        grouped["dimension"] = f"{a} x {b}"
        combos.append(grouped)

    if not combos:
        return pd.DataFrame()

    out = pd.concat(combos, ignore_index=True)
    return out.sort_values(["trades", "expectancy_R"], ascending=[False, False]).reset_index(drop=True)


def first_touch_summary_tables(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty or "first_touch" not in trades.columns:
        return {}

    out: dict[str, pd.DataFrame] = {
        "counts": trades["first_touch"].value_counts(dropna=False).rename_axis("first_touch").to_frame("count"),
        "metrics": grouped_metrics(trades, "first_touch", min_trades=1),
    }
    if "outcome_reason" in trades.columns:
        out["outcome_crosstab"] = pd.crosstab(trades["first_touch"], trades["outcome_reason"], dropna=False)
    if "time_in_trade_min" in trades.columns:
        out["time_in_trade"] = (
            trades.groupby("first_touch", dropna=False)["time_in_trade_min"]
            .agg(["count", "mean", "median", "min", "max"])
            .sort_values("count", ascending=False)
        )
    return out


def trade_duration_study(trades: pd.DataFrame, min_trades: int = 1) -> pd.DataFrame:
    if trades.empty or "trade_duration_bucket" not in trades.columns:
        return pd.DataFrame()
    return grouped_metrics(trades, "trade_duration_bucket", min_trades=min_trades)


# -----------------------------
# Summary helpers
# -----------------------------
def build_research_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"message": "No closed V3 trades yet."}])

    row = core_metrics(df).to_dict()

    if "mfe_R" in df.columns:
        mfe = pd.to_numeric(df["mfe_R"], errors="coerce")
        row["pct_reach_0.5R"] = float((mfe >= 0.5).mean())
        row["pct_reach_1.0R"] = float((mfe >= 1.0).mean())
        row["pct_reach_1.5R"] = float((mfe >= 1.5).mean())
        row["pct_reach_2.5R"] = float((mfe >= 2.5).mean())

    if {"pnl_R", "mfe_R"}.issubset(df.columns):
        pnl = pd.to_numeric(df["pnl_R"], errors="coerce")
        mfe = pd.to_numeric(df["mfe_R"], errors="coerce")
        losers = pnl < 0
        row["pct_losers_reached_0.5R"] = float((mfe[losers] >= 0.5).mean()) if losers.any() else np.nan
        row["pct_losers_reached_1.0R"] = float((mfe[losers] >= 1.0).mean()) if losers.any() else np.nan

    ci = bootstrap_expectancy_ci(df["pnl_R"], n_boot=2000)
    row["expectancy_ci_low_R"] = ci.get("ci_low_R", np.nan)
    row["expectancy_ci_high_R"] = ci.get("ci_high_R", np.nan)
    row["expectancy_ci_excludes_zero"] = ci.get("ci_excludes_zero", np.nan)

    eq = equity_curve(df)
    if not eq.empty:
        row["current_cum_R"] = float(eq["cum_R"].iloc[-1])
        row["max_drawdown_R"] = float(eq["drawdown_R"].min())

    if "exit_reason" in df.columns:
        top_exit = df["exit_reason"].value_counts(dropna=False)
        if len(top_exit):
            row["top_exit_reason"] = top_exit.index[0]
            row["top_exit_reason_count"] = int(top_exit.iloc[0])

    if "tier" in df.columns:
        tier_tbl = grouped_metrics(df, "tier", min_trades=1)
        if not tier_tbl.empty:
            best = tier_tbl.iloc[0]
            row["best_tier_by_expectancy"] = best["tier"]
            row["best_tier_expectancy_R"] = best["expectancy_R"]

    if "anchor_regime" in df.columns:
        reg_tbl = grouped_metrics(df, "anchor_regime", min_trades=1)
        if not reg_tbl.empty:
            best = reg_tbl.iloc[0]
            row["best_regime_by_expectancy"] = best["anchor_regime"]
            row["best_regime_expectancy_R"] = best["expectancy_R"]

    if "score" in df.columns:
        sweep = threshold_sweep(df)
        viable = sweep[sweep["trades"] >= max(5, math.ceil(len(df) * 0.1))]
        if not viable.empty:
            best = viable.sort_values("expectancy_R", ascending=False).iloc[0]
            row["best_score_threshold"] = best["score_gte"]
            row["best_score_threshold_expectancy_R"] = best["expectancy_R"]
            row["best_score_threshold_trades"] = int(best["trades"])
            row["best_score_threshold_ci_excludes_zero"] = best.get("ci_excludes_zero", np.nan)

    if "entry_efficiency_pct" in df.columns:
        row["avg_entry_efficiency_pct"] = float(pd.to_numeric(df["entry_efficiency_pct"], errors="coerce").mean())

    if "slippage_pct" in df.columns:
        row["avg_slippage_pct"] = float(pd.to_numeric(df["slippage_pct"], errors="coerce").mean())

    if "gave_back_R" in df.columns:
        row["avg_gave_back_R"] = float(pd.to_numeric(df["gave_back_R"], errors="coerce").mean())

    return pd.DataFrame([row])


# -----------------------------
# Plot helpers
# -----------------------------
def plot_metric_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 4),
):
    if df.empty or x not in df.columns or y not in df.columns:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df[x].astype(str), df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def plot_equity_curve(
    df: pd.DataFrame,
    *,
    ts_col: str = "signal_ts_utc",
    figsize: tuple[int, int] = (10, 4),
):
    eq = equity_curve(df, ts_col=ts_col)
    if eq.empty:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    x = eq[ts_col] if ts_col in eq.columns else eq["trade_n"]
    ax.plot(x, eq["cum_R"])
    ax.set_xlabel(ts_col if ts_col in eq.columns else "trade_n")
    ax.set_ylabel("cum_R")
    ax.set_title("Equity Curve (R)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def plot_drawdown_curve(
    df: pd.DataFrame,
    *,
    ts_col: str = "signal_ts_utc",
    figsize: tuple[int, int] = (10, 4),
):
    eq = equity_curve(df, ts_col=ts_col)
    if eq.empty:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    x = eq[ts_col] if ts_col in eq.columns else eq["trade_n"]
    ax.plot(x, eq["drawdown_R"])
    ax.set_xlabel(ts_col if ts_col in eq.columns else "trade_n")
    ax.set_ylabel("drawdown_R")
    ax.set_title("Drawdown Curve (R)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax
