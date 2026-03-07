from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    alerts_intents: Path
    executions: Path
    outcomes: Path


def default_paths(log_dir: str | Path, market_type: str = "crypto") -> DatasetPaths:
    """
    Matches journal.py naming:
      {market_type}_alerts_intents.csv
      {market_type}_executions.csv
      {market_type}_outcomes.csv
    """
    log_dir = Path(log_dir)
    return DatasetPaths(
        alerts_intents=log_dir / f"{market_type}_alerts_intents.csv",
        executions=log_dir / f"{market_type}_executions.csv",
        outcomes=log_dir / f"{market_type}_outcomes.csv",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Return empty DF with no rows; downstream joins stay safe
        return pd.DataFrame()
    return pd.read_csv(path)


def load_datasets(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intents = _read_csv(paths.alerts_intents)
    execs = _read_csv(paths.executions)
    outcomes = _read_csv(paths.outcomes)
    return intents, execs, outcomes


def build_trades_dataframe(
    intents: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core join:
      alerts_intents JOIN outcomes ON alert_id

    Notes:
    - alerts_intents includes *attempts* (skips/errors). We want the created alerts.
    - outcomes includes exits for watches that entered and ended (STOP/T2/TIMEOUT/NO_ENTRY_FILL).
    """
    if intents.empty and outcomes.empty:
        return pd.DataFrame()

    i = intents.copy()
    o = outcomes.copy()

    # Keep only created alerts in intents
    if "alert_status" in i.columns:
        i = i[i["alert_status"] == "ALERT_CREATED"].copy()

    # Standardize dtypes
    if "alert_id" in i.columns:
        i["alert_id"] = i["alert_id"].astype(str)
    if "alert_id" in o.columns:
        o["alert_id"] = o["alert_id"].astype(str)

    # Merge (left join = keep all created alerts, even if outcome not yet present)
    trades = i.merge(o, on="alert_id", how="left", suffixes=("_intent", "_outcome"))

    # Normalize shared columns that may have been suffixed by the merge
    if "symbol" not in trades.columns:
        if "symbol_intent" in trades.columns:
            trades["symbol"] = trades["symbol_intent"]
        elif "symbol_outcome" in trades.columns:
            trades["symbol"] = trades["symbol_outcome"]

    if "tier" not in trades.columns:
        if "tier_intent" in trades.columns:
            trades["tier"] = trades["tier_intent"]
        elif "tier_outcome" in trades.columns:
            trades["tier"] = trades["tier_outcome"]

    # Ensure booleans not strings
    for bcol in ["t1_touched", "t2_touched", "stop_touched", "entered"]:
        if bcol in trades.columns:
            trades[bcol] = trades[bcol].astype("boolean")

    # Convenience column
    if "time_in_trade_s" in trades.columns:
        trades["time_in_trade_min"] = trades["time_in_trade_s"] / 60.0

    # Keep a practical column subset (but don’t drop others)
    preferred = [
        "alert_id",
        "symbol",
        "tier",
        "score",
        "vol_z",
        "anchor_regime",
        "entry_px",
        "exit_px",
        "exit_reason",
        "pnl_R",
        "mfe_R",
        "mae_R",
        "time_in_trade_s",
        "time_in_trade_min",
        "t1_touched",
        "t2_touched",
        "stop_touched",
        "entered",
    ]
    cols = [c for c in preferred if c in trades.columns] + [c for c in trades.columns if c not in preferred]
    trades = trades[cols]

    return trades


def load_trades(log_dir: str | Path, market_type: str = "crypto") -> pd.DataFrame:
    paths = default_paths(log_dir, market_type=market_type)
    intents, execs, outcomes = load_datasets(paths)
    # execs not merged yet (we’ll use it later)
    return build_trades_dataframe(intents, outcomes)