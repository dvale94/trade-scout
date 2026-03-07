from __future__ import annotations
import pandas as pd


def edge_diagnostics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    df = trades.copy()
    for c in ["mfe_R", "mae_R", "pnl_R"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    mfe = df["mfe_R"].dropna() if "mfe_R" in df.columns else pd.Series(dtype=float)
    mae = df["mae_R"].dropna() if "mae_R" in df.columns else pd.Series(dtype=float)

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