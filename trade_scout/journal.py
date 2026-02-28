import csv
import os
from datetime import timezone
from typing import Any

ALERTS_CSV = os.getenv("ALERTS_CSV", "alerts.csv")
OUTCOMES_CSV = os.getenv("OUTCOMES_CSV", "outcomes.csv")


def _write_row(path: str, row: dict[str, Any]):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_alert(candidate, computed: dict, tier: str):
    """
    Logs a single alert event with full context.
    candidate: your Candidate dataclass instance
    computed: dict from score_candidate()
    tier: "A_PLUS_RETEST" or "A_CONTINUATION"
    """
    row = {
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
        "vol_z": computed.get("vol_z"),
        "extended_atr": computed.get("extended_atr"),

        # news hooks (future)
        "news_sentiment": candidate.news_sentiment,
        "news_summary": candidate.news_summary,

        # human-readable reasons
        "reasons": " | ".join(candidate.reasons),
    }
    _write_row(ALERTS_CSV, row)


def log_outcome(outcome_row: dict[str, Any]):
    """
    outcome_row should already contain all fields needed for outcomes.csv
    """
    _write_row(OUTCOMES_CSV, outcome_row)