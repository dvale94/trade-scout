from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def hist(df: pd.DataFrame, col: str, bins: int = 40, title: str | None = None):
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None

    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title or f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    return plt.gcf()


def plot_distributions(trades: pd.DataFrame):
    # closed trades only for pnl
    hist(trades, "pnl_R", title="PnL Distribution (R)")
    hist(trades, "mfe_R", title="MFE Distribution (R)")
    hist(trades, "mae_R", title="MAE Distribution (R)")
    hist(trades, "time_in_trade_s", title="Time in trade (seconds)")
    plt.show()