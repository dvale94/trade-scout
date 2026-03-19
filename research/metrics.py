from __future__ import annotations
import pandas as pd


def _coerce_pnl_r(trades: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(trades, pd.Series):
        return pd.to_numeric(trades, errors="coerce")
    if trades.empty or "pnl_R" not in trades.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(trades["pnl_R"], errors="coerce")


def _closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Closed trades are rows where pnl_R is present (exit happened and we computed pnl).
    This excludes still-open watches and NO_ENTRY_FILL if pnl_R wasn't set.
    """
    pnl = _coerce_pnl_r(df)
    if df.empty or pnl.empty:
        return pd.DataFrame()
    out = df.copy()
    out["pnl_R"] = pnl
    return out[out["pnl_R"].notna()].copy()


def summarize_trade_performance(trades: pd.DataFrame) -> dict:
    df = _closed_trades(trades)
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "avg_R": None,
            "median_R": None,
            "expectancy_R": None,
            "avg_win_R": None,
            "avg_loss_R_abs": None,
            "payoff_ratio": None,
            "profit_factor": None,
        }

    pnl = _coerce_pnl_r(df).dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    trades_n = int(len(pnl))
    wins_n = int(len(wins))
    losses_n = int(len(losses))

    win_rate = float(wins_n / trades_n) if trades_n else None
    avg_r = float(pnl.mean()) if trades_n else None
    median_r = float(pnl.median()) if trades_n else None

    avg_win_r = float(wins.mean()) if wins_n else None
    avg_loss_r_abs = float(abs(losses.mean())) if losses_n else None

    expectancy_r = None
    if win_rate is not None:
        expectancy_r = float(
            (win_rate * (avg_win_r or 0.0)) - ((1.0 - win_rate) * (avg_loss_r_abs or 0.0))
        )

    payoff_ratio = None
    if avg_win_r is not None and avg_loss_r_abs not in (None, 0.0):
        payoff_ratio = float(avg_win_r / avg_loss_r_abs)

    gross_profit = float(wins.sum()) if wins_n else 0.0
    gross_loss = float(abs(losses.sum())) if losses_n else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    return {
        "trades": trades_n,
        "wins": wins_n,
        "losses": losses_n,
        "win_rate": win_rate,
        "avg_R": avg_r,
        "median_R": median_r,
        "expectancy_R": expectancy_r,
        "avg_win_R": avg_win_r,
        "avg_loss_R_abs": avg_loss_r_abs,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
    }


def grouped_trade_performance(
    df: pd.DataFrame,
    by,
    min_trades: int = 1,
) -> pd.DataFrame:
    if df.empty or "pnl_R" not in df.columns:
        return pd.DataFrame()

    if isinstance(by, str):
        group_cols = [by]
    else:
        group_cols = list(by)

    if not group_cols or any(col not in df.columns for col in group_cols):
        return pd.DataFrame()

    rows = []
    for group_key, group_df in df.groupby(group_cols, observed=False, dropna=False):
        summary = summarize_trade_performance(group_df)
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

    return (
        pd.DataFrame(rows)
        .sort_values(["expectancy_R", "trades"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )


def core_metrics(trades: pd.DataFrame) -> dict:
    summary = summarize_trade_performance(trades)
    if summary["trades"] == 0:
        return {
            "trade_count": 0,
            "win_rate": None,
            "avg_win_R": None,
            "avg_loss_R": None,
            "expectancy_R": None,
            "profit_factor": None,
        }

    return {
        "trade_count": summary["trades"],
        "win_rate": summary["win_rate"],
        "avg_win_R": summary["avg_win_R"],
        "avg_loss_R": (-summary["avg_loss_R_abs"]) if summary["avg_loss_R_abs"] is not None else None,
        "expectancy_R": summary["expectancy_R"],
        "profit_factor": summary["profit_factor"],
    }


def pretty_print_core(metrics: dict) -> str:
    def pct(x):
        return f"{x*100:.1f}%" if x is not None else "n/a"

    def num(x):
        return f"{x:.3f}" if x is not None else "n/a"

    return (
        f"Trades: {metrics['trade_count']}\n"
        f"Win rate: {pct(metrics['win_rate'])}\n"
        f"Avg win: {num(metrics['avg_win_R'])}R\n"
        f"Avg loss: {num(metrics['avg_loss_R'])}R\n"
        f"Expectancy: {num(metrics['expectancy_R'])}R\n"
        f"Profit factor: {num(metrics['profit_factor'])}\n"
    )


def breakdown_expectancy(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Returns expectancy by group (tier, symbol, anchor_regime, etc.)
    """
    grouped = grouped_trade_performance(trades, by=by, min_trades=1)
    if grouped.empty:
        return grouped
    return grouped.rename(
        columns={
            "avg_R": "avg",
            "median_R": "med",
            "expectancy_R": "expectancy",
        }
    )


def score_buckets(trades: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    if trades.empty or "score" not in trades.columns or "pnl_R" not in trades.columns:
        return pd.DataFrame()

    df = trades[trades["pnl_R"].notna()].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["pnl_R"] = pd.to_numeric(df["pnl_R"], errors="coerce")
    df = df.dropna(subset=["score", "pnl_R"])

    df["score_bucket"] = pd.qcut(df["score"], q=buckets, duplicates="drop")
    return breakdown_expectancy(df, by="score_bucket")


def threshold_sweep(trades: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    if trades.empty or "score" not in trades.columns or "pnl_R" not in trades.columns:
        return pd.DataFrame()

    df = trades[trades["pnl_R"].notna()].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["pnl_R"] = pd.to_numeric(df["pnl_R"], errors="coerce")
    df = df.dropna(subset=["score", "pnl_R"])

    rows = []
    for t in thresholds:
        sub = df[df["score"] >= t]
        if sub.empty:
            rows.append({"threshold": t, "trades": 0, "expectancy_R": None, "avg_R": None})
            continue
        summary = summarize_trade_performance(sub)
        rows.append({
            "threshold": t,
            "trades": int(len(sub)),
            "expectancy_R": summary["expectancy_R"],
            "avg_R": summary["avg_R"],
        })
    return pd.DataFrame(rows).sort_values("threshold")
