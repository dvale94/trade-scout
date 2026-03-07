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


def equity_curve(trades: pd.DataFrame, title: str = "Equity curve (R)"):
    if trades.empty or "pnl_R" not in trades.columns:
        return None
    pnl = pd.to_numeric(trades["pnl_R"], errors="coerce").dropna()
    if pnl.empty:
        return None
    eq = pnl.cumsum()

    plt.figure()
    plt.plot(eq.values)
    plt.title(title)
    plt.xlabel("trade #")
    plt.ylabel("cumulative R")
    return plt.gcf()


def scatter_mfe_mae(trades: pd.DataFrame):
    if trades.empty:
        return None
    mfe = pd.to_numeric(trades.get("mfe_R"), errors="coerce")
    mae = pd.to_numeric(trades.get("mae_R"), errors="coerce")
    df = pd.DataFrame({"mfe_R": mfe, "mae_R": mae}).dropna()
    if df.empty:
        return None

    plt.figure()
    plt.scatter(df["mae_R"], df["mfe_R"])
    plt.title("MFE vs MAE")
    plt.xlabel("MAE (R)")
    plt.ylabel("MFE (R)")
    return plt.gcf()