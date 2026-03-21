from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

"""Legacy compatibility surface for the historical Candidate shape."""


@dataclass
class Candidate:
    alert_id: str
    symbol: str
    timeframe_min: int
    score: float
    ts: datetime
    entry_low: float
    entry_high: float
    stop: float
    t1: float
    t2: float
    r: float
    reasons: list[str]
    news_sentiment: float | None = None
    news_summary: str | None = None
