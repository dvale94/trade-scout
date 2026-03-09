from __future__ import annotations

from typing import Sequence

import pandas as pd


_DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


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
    """Bucket a numeric feature and summarize trade quality by bucket.

    PnL stats are computed over closed trades only (pnl_R not null).
    """
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


def plot_metric_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 4),
):
    """Simple optional plotting helper for notebook use."""
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
