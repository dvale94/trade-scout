from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ExitPolicyResult:
    name: str
    trade_count: int
    expectancy_R: float | None


def simulate_policy_hold_t2_or_stop(trades: pd.DataFrame) -> pd.Series:
    """
    Current strategy approximation:
      - If MAE >= 1R => STOP (-1R)
      - Else if MFE >= 2.5R => T2 (+2.5R)
      - Else => TIMEOUT exit at whatever pnl_R already computed (fallback to pnl_R)
    """
    df = trades.copy()
    mfe = pd.to_numeric(df.get("mfe_R"), errors="coerce")
    mae = pd.to_numeric(df.get("mae_R"), errors="coerce")
    pnl = pd.to_numeric(df.get("pnl_R"), errors="coerce")

    sim = pnl.copy()

    sim = sim.where(~(mae >= 1.0), other=-1.0)
    sim = sim.where(~((mae < 1.0) & (mfe >= 2.5)), other=2.5)

    return sim


def simulate_policy_full_exit_t1(trades: pd.DataFrame) -> pd.Series:
    """
    Full exit at T1 when possible:
      - If MAE >= 1R before exit => STOP (-1R)
      - Else if MFE >= 1.5R => +1.5R
      - Else fallback to pnl_R
    """
    df = trades.copy()
    mfe = pd.to_numeric(df.get("mfe_R"), errors="coerce")
    mae = pd.to_numeric(df.get("mae_R"), errors="coerce")
    pnl = pd.to_numeric(df.get("pnl_R"), errors="coerce")

    sim = pnl.copy()
    sim = sim.where(~(mae >= 1.0), other=-1.0)
    sim = sim.where(~((mae < 1.0) & (mfe >= 1.5)), other=1.5)
    return sim


def simulate_policy_half_t1_rest_t2_be(trades: pd.DataFrame) -> pd.Series:
    """
    Strategy A:
      - Take 50% at T1 (+0.75R)
      - Move stop to breakeven on remainder once T1 hit
      - Remainder to T2 (+1.25R) if MFE >= 2.5R
      - If after T1 price later would have hit original stop, we assume BE on remainder (0R), but overall keeps the +0.75R.
      - If stop before T1: -1R
    """
    df = trades.copy()
    mfe = pd.to_numeric(df.get("mfe_R"), errors="coerce")
    mae = pd.to_numeric(df.get("mae_R"), errors="coerce")
    pnl = pd.to_numeric(df.get("pnl_R"), errors="coerce")

    sim = pnl.copy()

    # Stop before anything
    sim = sim.where(~(mae >= 1.0), other=-1.0)

    # If we reached T1 (MFE >= 1.5) without stopping, bank +0.75R
    reached_t1 = (mae < 1.0) & (mfe >= 1.5)
    sim = sim.where(~reached_t1, other=0.75)

    # If we also reached T2, add remainder +1.25R (0.5 position * 2.5R)
    reached_t2 = (mae < 1.0) & (mfe >= 2.5)
    sim = sim.where(~reached_t2, other=0.75 + 1.25)

    return sim


def expectancy(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    wins = s[s > 0]
    losses = s[s < 0]
    win_rate = len(wins) / len(s)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))


def compare_exit_policies(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a table of simulated expectancy by policy.
    """
    if trades.empty:
        return pd.DataFrame()

    policies = [
        ("hold_t2_or_stop", simulate_policy_hold_t2_or_stop),
        ("full_exit_t1", simulate_policy_full_exit_t1),
        ("half_t1_rest_t2_be", simulate_policy_half_t1_rest_t2_be),
    ]

    rows = []
    for name, fn in policies:
        sim = fn(trades)
        rows.append({
            "policy": name,
            "trades": int(pd.to_numeric(sim, errors="coerce").dropna().shape[0]),
            "expectancy_R": expectancy(sim),
            "avg_R": float(pd.to_numeric(sim, errors="coerce").mean()),
        })

    return pd.DataFrame(rows).sort_values("expectancy_R", ascending=False)