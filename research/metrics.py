from __future__ import annotations
import pandas as pd


def _closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Closed trades are rows where pnl_R is present (exit happened and we computed pnl).
    This excludes still-open watches and NO_ENTRY_FILL if pnl_R wasn't set.
    """
    if df.empty or "pnl_R" not in df.columns:
        return pd.DataFrame()
    return df[df["pnl_R"].notna()].copy()


def core_metrics(trades: pd.DataFrame) -> dict:
    df = _closed_trades(trades)
    if df.empty:
        return {
            "trade_count": 0,
            "win_rate": None,
            "avg_win_R": None,
            "avg_loss_R": None,
            "expectancy_R": None,
            "profit_factor": None,
        }

    pnl = df["pnl_R"].astype(float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    trade_count = int(len(pnl))
    win_rate = float(len(wins) / trade_count) if trade_count else None

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative

    loss_rate = 1.0 - win_rate if win_rate is not None else None

    expectancy = None
    if win_rate is not None and loss_rate is not None:
        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_win_R": avg_win if len(wins) else None,
        "avg_loss_R": avg_loss if len(losses) else None,
        "expectancy_R": expectancy,
        "profit_factor": profit_factor,
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
    if trades.empty or "pnl_R" not in trades.columns or by not in trades.columns:
        return pd.DataFrame()

    df = trades[trades["pnl_R"].notna()].copy()
    df["pnl_R"] = df["pnl_R"].astype(float)

    def _exp(g: pd.Series) -> float:
        wins = g[g > 0]
        losses = g[g < 0]
        if len(g) == 0:
            return float("nan")
        win_rate = len(wins) / len(g)
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = abs(losses.mean()) if len(losses) else 0.0
        return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))

    out = (
        df.groupby(by)["pnl_R"]
        .agg(trades="count", avg="mean", med="median", expectancy=_exp)
        .reset_index()
        .sort_values(["expectancy", "trades"], ascending=[False, False])
    )
    return out


def score_buckets(trades: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    if trades.empty or "score" not in trades.columns or "pnl_R" not in trades.columns:
        return pd.DataFrame()

    df = trades[trades["pnl_R"].notna()].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["pnl_R"] = pd.to_numeric(df["pnl_R"], errors="coerce")
    df = df.dropna(subset=["score", "pnl_R"])

    df["score_bucket"] = pd.qcut(df["score"], q=buckets, duplicates="drop")
    return breakdown_expectancy(df, by="score_bucket")