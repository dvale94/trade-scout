from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from . import journal

def _dt_to_iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _dt_from_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


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
    

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "timeframe_min": self.timeframe_min,
            "tier": self.tier,
            "created_ts": _dt_to_iso(self.created_ts),

            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop": self.stop,
            "t1": self.t1,
            "t2": self.t2,
            "R": self.R,
            "score": self.score,

            "max_favorable": self.max_favorable,
            "max_adverse": self.max_adverse,
            "hit_t1": self.hit_t1,
            "hit_t2": self.hit_t2,
        }


    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Watch":
        return Watch(
            alert_id=d["alert_id"],
            symbol=d["symbol"],
            timeframe_min=int(d["timeframe_min"]),
            tier=d["tier"],
            created_ts=_dt_from_iso(d["created_ts"]) or datetime.now(timezone.utc),

            entry_low=float(d["entry_low"]),
            entry_high=float(d["entry_high"]),
            stop=float(d["stop"]),
            t1=float(d["t1"]),
            t2=float(d["t2"]),
            R=float(d["R"]),
            score=float(d["score"]),

            max_favorable=float(d.get("max_favorable", d["entry_high"])),
            max_adverse=float(d.get("max_adverse", d["entry_high"])),
            hit_t1=bool(d.get("hit_t1", False)),
            hit_t2=bool(d.get("hit_t2", False)),
        )


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
        self.save_state()


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

        self.save_state()


    def save_state(self):
        payload = {
            "saved_ts_utc": datetime.now(timezone.utc).isoformat(),
            "active": {sym: [w.to_dict() for w in ws] for sym, ws in self.active.items()}
        }
        journal.save_active_watches(payload)

    def load_state(self):
        payload = journal.load_active_watches()
        if not payload:
            return
        active = payload.get("active", {})
        restored: dict[str, list[Watch]] = {}
        for sym, items in active.items():
            restored[sym] = [Watch.from_dict(x) for x in items]
        self.active = restored