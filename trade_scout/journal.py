import csv
import os
import json
from datetime import timezone
from typing import Any

from . import config
from .strategy import Candidate

LOG_DIR = config.LOG_DIR

def _path(name: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, name)


def alerts_path() -> str:
    return _path(f"{config.MARKET_TYPE}_alerts.csv")


def outcomes_path() -> str:
    return _path(f"{config.MARKET_TYPE}_outcomes.csv")


def telemetry_path() -> str:
    return _path(f"{config.MARKET_TYPE}_telemetry.csv")


def watches_state_path() -> str:
    return _path(f"{config.MARKET_TYPE}_active_watches.json")


def alerts_intents_path() -> str:
    return _path(f"{config.MARKET_TYPE}_alerts_intents.csv")


def executions_path() -> str:
    return _path(f"{config.MARKET_TYPE}_executions.csv")


def outcomes_v2_path() -> str:
    return _path(f"{config.MARKET_TYPE}_outcomes_v2.csv")


def _write_row(path: str, row: dict[str, Any]):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_alert(candidate: Candidate, computed: dict, tier: str, strategy_profile: str):
    """
    Logs a single alert event with full context.
    candidate: your Candidate dataclass instance
    computed: dict from score_candidate()
    tier: "A_PLUS_RETEST" or "A_CONTINUATION"
    """
    row = {
        "alert_id": candidate.alert_id,
        "market_type": config.MARKET_TYPE,
        "strategy_profile": strategy_profile,
        "timeframe_min": candidate.timeframe_min,
        "timestamp_utc": candidate.ts.astimezone(timezone.utc).isoformat(),
        "symbol": candidate.symbol,
        "tier": tier,
        "score": round(candidate.score, 4),
        "entry_low": round(candidate.entry_low, 6),
        "entry_high": round(candidate.entry_high, 6),
        "stop": round(candidate.stop, 6),
        "t1": round(candidate.t1, 6),
        "t2": round(candidate.t2, 6),
        "R": round(candidate.r, 6),
        "risk_pct": round((candidate.r / candidate.entry_high) * 100.0, 4),

        # context features
        "atr": computed.get("atr"),
        "ma20": computed.get("ma20"),
        "ma50": computed.get("ma50"),
        "ma200": computed.get("ma200"),
        "breakout_level": computed.get("breakout_level"),
        "anchor_symbol": computed.get("anchor_symbol"),
        "anchor_regime": computed.get("anchor_regime"),
        "anchor_close": computed.get("anchor_close"),
        "anchor_ma50": computed.get("anchor_ma50"),
        "anchor_ma200": computed.get("anchor_ma200"),
        "vol_z": computed.get("vol_z"),
        "extended_atr": computed.get("extended_atr"),

        # news hooks (future)
        "news_sentiment": candidate.news_sentiment,
        "news_summary": candidate.news_summary,

        # human-readable reasons
        "reasons": " | ".join(candidate.reasons),
    }
    _write_row(alerts_path(), row)


def log_outcome(outcome_row: dict[str, Any]):
    """
    outcome_row should already contain all fields needed for outcomes.csv
    """
    _write_row(outcomes_path(), outcome_row)


def log_telemetry(row: dict[str, Any]):
    """
    Writes a single aggregated telemetry snapshot (e.g., once per hour).
    Keep this low-frequency to avoid noisy logs.
    """
    _write_row(telemetry_path(), row)


def log_alert_intent(row: dict[str, Any]):
    _write_row(alerts_intents_path(), row)


def log_execution(row: dict[str, Any]):
    _write_row(executions_path(), row)


def log_outcome_v2(row: dict[str, Any]):
    _write_row(outcomes_v2_path(), row)


def save_active_watches(payload: dict):
    with open(watches_state_path(), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_active_watches() -> dict | None:
    path = watches_state_path()
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)    