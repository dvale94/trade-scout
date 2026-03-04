from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

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

    # strategy metadata (so outcomes_v2 can join cleanly)
    strategy_profile: str | None = None
    strategy_version: str | None = None
    config_hash: str | None = None
    side: str = "LONG"

    # execution simulation state
    entered: bool = False
    entry_ts: datetime | None = None
    entry_px: float | None = None

    t1_ts: datetime | None = None
    t2_ts: datetime | None = None
    stop_ts: datetime | None = None


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

            "strategy_profile": self.strategy_profile,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "side": self.side,

            "entered": self.entered,
            "entry_ts": _dt_to_iso(self.entry_ts),
            "entry_px": self.entry_px,

            "t1_ts": _dt_to_iso(self.t1_ts),
            "t2_ts": _dt_to_iso(self.t2_ts),
            "stop_ts": _dt_to_iso(self.stop_ts),
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

            strategy_profile=d.get("strategy_profile"),
            strategy_version=d.get("strategy_version"),
            config_hash=d.get("config_hash"),
            side=d.get("side", "LONG"),

            entered=bool(d.get("entered", False)),
            entry_ts=_dt_from_iso(d.get("entry_ts")),
            entry_px=(float(d["entry_px"]) if d.get("entry_px") is not None else None),

            t1_ts=_dt_from_iso(d.get("t1_ts")),
            t2_ts=_dt_from_iso(d.get("t2_ts")),
            stop_ts=_dt_from_iso(d.get("stop_ts")),
        )
    

    def _exec_row(self, exec_type: str, ts: datetime, px: float, px_source: str) -> dict[str, Any]:
        latency_ms = None
        try:
            latency_ms = int((ts - self.created_ts).total_seconds() * 1000)
        except Exception:
            pass

        return {
            "exec_id": str(uuid4()),
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "side": self.side,
            "exec_type": exec_type,
            "exec_ts_utc": ts.astimezone(timezone.utc).isoformat(),
            "exec_px": float(px),
            "exec_px_source": px_source,
            "qty": None,  # placeholder for now
            "fee_estimate": None,
            "slippage_bps_estimate": None,
            "latency_ms": latency_ms,
        }


    def as_outcome_v2_row(self) -> dict[str, Any]:
        # Touch flags
        t1_touched = self.t1_ts is not None
        t2_touched = self.t2_ts is not None
        stop_touched = self.stop_ts is not None

        # Time metrics
        def _dt_diff_s(a: datetime | None, b: datetime | None) -> int | None:
            if not a or not b:
                return None
            return int((b - a).total_seconds())

        entry_ts = self.entry_ts
        exit_ts = self.done_ts

        time_in_trade_s = _dt_diff_s(entry_ts, exit_ts) if self.entered else None

        # R-based PnL
        pnl_R = None
        if self.entered and self.entry_px is not None and self.done_reason and self.R > 0:
            if self.done_reason == "STOP":
                pnl_R = -1.0
            elif self.done_reason == "T2":
                pnl_R = 2.5
            elif self.done_reason == "TIMEOUT":
                # conservative: assume flat at entry (until we add close-based timeout fills)
                pnl_R = 0.0

        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "side": self.side,

            "strategy_profile": self.strategy_profile,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,

            "entry_ts_utc": _dt_to_iso(self.entry_ts),
            "entry_px": self.entry_px,

            "exit_ts_utc": _dt_to_iso(self.done_ts),
            "exit_px": None,  # we’ll fill with actual exit px once we store it (see update logic below)
            "exit_reason": self.done_reason,

            "pnl_R": pnl_R,
            "pnl_usd_est": None,

            "t1_touched": t1_touched,
            "t1_ts_utc": _dt_to_iso(self.t1_ts),
            "t2_touched": t2_touched,
            "t2_ts_utc": _dt_to_iso(self.t2_ts),
            "stop_touched": stop_touched,
            "stop_ts_utc": _dt_to_iso(self.stop_ts),

            "mfe_R": round((self.max_favorable - self.entry_high) / self.R, 4) if self.R > 0 else None,
            "mae_R": round((self.entry_high - self.max_adverse) / self.R, 4) if self.R > 0 else None,

            "time_in_trade_s": time_in_trade_s,
            "time_to_t1_s": _dt_diff_s(self.entry_ts, self.t1_ts) if self.entered else None,
            "time_to_t2_s": _dt_diff_s(self.entry_ts, self.t2_ts) if self.entered else None,
            "time_to_stop_s": _dt_diff_s(self.entry_ts, self.stop_ts) if self.entered else None,

            "max_spread_bps_seen": None,
            "avg_spread_bps_seen": None,

            # useful to distinguish “watch existed” vs “trade existed”
            "entered": self.entered,
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
        self.save_state()


    def update(self, symbol: str, high: float, low: float, close: float, ts: datetime):
        if symbol not in self.active:
            return

        keep: list[Watch] = []

        for w in self.active[symbol]:
            if w.done:
                continue

            # Track excursions
            w.max_favorable = max(w.max_favorable, high)
            w.max_adverse = min(w.max_adverse, low)

            # ENTRY simulation (only once)
            if not w.entered:
                zone_touched = (low <= w.entry_high) and (high >= w.entry_low)
                if zone_touched:
                    w.entered = True
                    w.entry_ts = ts
                    w.entry_px = float(w.entry_high)  # conservative worst fill in zone
                    journal.log_execution(w._exec_row("ENTRY", ts, w.entry_px, "zone_touch_entry_high"))

            if not w.entered:
                if ts - w.created_ts >= self.timeout:
                    w.done = True
                    w.done_reason = "NO_ENTRY_FILL"
                    w.done_ts = ts
                    # No execution rows (because no fills happened)
                    journal.log_outcome_v2(w.as_outcome_v2_row())
                    continue

                keep.append(w)
                continue

            # STOP (worst-case priority)
            if low <= w.stop:
                w.done = True
                w.done_reason = "STOP"
                w.done_ts = ts
                w.stop_ts = ts

                exit_px = float(w.stop)
                journal.log_execution(w._exec_row("EXIT_STOP", ts, exit_px, "bar_touch_stop"))
                out = w.as_outcome_v2_row()
                out["exit_px"] = exit_px
                journal.log_outcome_v2(out)
                continue

            # T1 touch (log execution once)
            if not w.hit_t1 and high >= w.t1:
                w.hit_t1 = True
                w.t1_ts = ts
                journal.log_execution(w._exec_row("EXIT_T1", ts, float(w.t1), "bar_touch_t1"))

            # T2 = final exit
            if not w.hit_t2 and high >= w.t2:
                w.hit_t2 = True
                w.t2_ts = ts
                w.done = True
                w.done_reason = "T2"
                w.done_ts = ts

                exit_px = float(w.t2)
                journal.log_execution(w._exec_row("EXIT_T2", ts, exit_px, "bar_touch_t2"))
                out = w.as_outcome_v2_row()
                out["exit_px"] = exit_px
                journal.log_outcome_v2(out)
                continue

            # TIMEOUT (entered but no stop/t2)
            if ts - w.created_ts >= self.timeout:
                w.done = True
                w.done_reason = "TIMEOUT"
                w.done_ts = ts

                exit_px = float(close)  # best available from bar stream
                journal.log_execution(w._exec_row("EXIT_TIMEOUT", ts, exit_px, "bar_close_timeout"))
                out = w.as_outcome_v2_row()
                out["exit_px"] = exit_px
                journal.log_outcome_v2(out)
                continue

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