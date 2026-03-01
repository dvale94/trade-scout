from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from . import journal


@dataclass
class Watch:
    alert_id: str
    symbol: str
    timeframe_min: int
    tier: str
    created_ts: datetime

    entry_low: float
    entry_high: float
    stop: float
    t1: float
    t2: float
    R: float

    score: float

    # tracking
    max_favorable: float  # highest price seen since alert
    max_adverse: float    # lowest price seen since alert
    hit_t1: bool = False
    hit_t2: bool = False
    done: bool = False
    done_reason: str | None = None
    done_ts: datetime | None = None

    def as_outcome_row(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "timeframe_min": self.timeframe_min,
            "tier": self.tier,
            "alert_ts_utc": self.created_ts.astimezone(timezone.utc).isoformat(),
            "done_ts_utc": (self.done_ts.astimezone(timezone.utc).isoformat() if self.done_ts else None),
            "done_reason": self.done_reason,
            "score": round(self.score, 4),
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop": self.stop,
            "t1": self.t1,
            "t2": self.t2,
            "R": self.R,
            "max_favorable": self.max_favorable,
            "max_adverse": self.max_adverse,
            "mfe_R": round((self.max_favorable - self.entry_high) / self.R, 4) if self.R > 0 else None,
            "mae_R": round((self.entry_high - self.max_adverse) / self.R, 4) if self.R > 0 else None,
        }


class WatchManager:
    """
    Holds active watches and updates them each bar.
    """
    def __init__(self, timeout_minutes: int = 240, max_active_per_symbol: int = 1):
        self.timeout = timedelta(minutes=timeout_minutes)
        self.max_active_per_symbol = max_active_per_symbol
        self.active: dict[str, list[Watch]] = {}


    def add(self, w: Watch):
        arr = self.active.setdefault(w.symbol, [])
        # limit active watches per symbol
        if len(arr) >= self.max_active_per_symbol:
            return
        arr.append(w)


    def update(self, symbol: str, high: float, low: float, ts: datetime):
        if symbol not in self.active:
            return

        keep: list[Watch] = []

        for w in self.active[symbol]:
            if w.done:
                continue

            # Track excursions
            w.max_favorable = max(w.max_favorable, high)
            w.max_adverse = min(w.max_adverse, low)

            # Stop hit
            if low <= w.stop:
                w.done = True
                w.done_reason = "STOP"
                w.done_ts = ts
                journal.log_outcome(w.as_outcome_row())
                continue

            # Target 1
            if not w.hit_t1 and high >= w.t1:
                w.hit_t1 = True

            # Target 2
            if not w.hit_t2 and high >= w.t2:
                w.hit_t2 = True
                w.done = True
                w.done_reason = "T2"
                w.done_ts = ts
                journal.log_outcome(w.as_outcome_row())
                continue

            # Timeout
            if ts - w.created_ts >= self.timeout:
                w.done = True
                w.done_reason = "TIMEOUT"
                w.done_ts = ts
                journal.log_outcome(w.as_outcome_row())
                continue

            keep.append(w)

        self.active[symbol] = keep
        if not self.active[symbol]:
            del self.active[symbol]