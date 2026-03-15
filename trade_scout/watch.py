from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

from . import journal
from .journal_rows import ExecutionRow, OutcomeRow
from .watch_state import WatchLifecycleState, assert_valid_watch_lifecycle, lifecycle_state_for_watch
from .watch_transitions import WatchBar, WatchDecision, WatchTransitionType, decide_watch_bar_transition

def _dt_to_iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _dt_from_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _dt_diff_s(a: datetime | None, b: datetime | None) -> int | None:
    if not a or not b:
        return None
    return int((b - a).total_seconds())


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

    # strategy metadata (so outcomes can join cleanly)
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

    @property
    def lifecycle_state(self) -> WatchLifecycleState:
        return lifecycle_state_for_watch(self)


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

            "done": self.done,
            "done_reason": self.done_reason,
            "done_ts": _dt_to_iso(self.done_ts),
        }


    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Watch":
        watch = Watch(
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

            done=bool(d.get("done", False)),
            done_reason=d.get("done_reason"),
            done_ts=_dt_from_iso(d.get("done_ts")),
        )
        assert_valid_watch_lifecycle(watch)
        return watch
    

    def _exec_row(self, exec_type: str, ts: datetime, px: float, px_source: str) -> ExecutionRow:
        latency_ms = None
        try:
            exec_ts = ts.astimezone(timezone.utc)
            created_ts = self.created_ts.astimezone(timezone.utc)

            exec_ts = exec_ts.replace(microsecond=(exec_ts.microsecond // 1000) * 1000)
            created_ts = created_ts.replace(microsecond=(created_ts.microsecond // 1000) * 1000)

            latency_ms = int((exec_ts - created_ts).total_seconds() * 1000)
            if -1000 < latency_ms < 0:
                latency_ms = 0
        except Exception:
            pass

        entry_fill_px = self.entry_px
        entry_fill_ts = self.entry_ts
        if exec_type == "ENTRY":
            entry_fill_px = float(px)
            entry_fill_ts = ts

        slippage_pct = None
        if entry_fill_px is not None and self.entry_high > 0:
            slippage_pct = ((entry_fill_px - self.entry_high) / self.entry_high) * 100.0

        return ExecutionRow(
            exec_id=str(uuid4()),
            alert_id=self.alert_id,
            symbol=self.symbol,
            side=self.side,
            exec_type=exec_type,
            exec_ts_utc=ts.astimezone(timezone.utc).isoformat(),
            exec_px=float(px),
            exec_px_source=px_source,
            entry_fill_px=entry_fill_px,
            entry_fill_ts_utc=_dt_to_iso(entry_fill_ts),
            slippage_pct=round(slippage_pct, 6) if slippage_pct is not None else None,
            qty=None,
            fee_estimate=None,
            slippage_bps_estimate=None,
            latency_ms=latency_ms,
        )


    def _first_touch_info(self) -> tuple[str, str | None]:
        events: list[tuple[int, str, datetime]] = []
        if self.stop_ts is not None:
            events.append((0, "STOP", self.stop_ts))
        if self.t1_ts is not None:
            events.append((1, "T1", self.t1_ts))
        if self.t2_ts is not None:
            events.append((2, "T2", self.t2_ts))
        if self.done_reason == "TIMEOUT" and self.done_ts is not None:
            events.append((3, "TIMEOUT", self.done_ts))

        if not events:
            return "NONE", None

        _, label, ts = min(events, key=lambda item: (item[2], item[0]))
        return label, _dt_to_iso(ts)


    def _mfe_mae_metrics(self) -> dict[str, float | None]:
        if self.R <= 0:
            return {"mfe_r": None, "mae_r": None, "mfe_px": None, "mae_px": None}

        entry_ref = self.entry_px if self.entry_px is not None else self.entry_high
        if entry_ref is None:
            return {"mfe_r": None, "mae_r": None, "mfe_px": None, "mae_px": None}

        mfe_r = round((self.max_favorable - entry_ref) / self.R, 4)
        mae_r = round((entry_ref - self.max_adverse) / self.R, 4)
        return {
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "mfe_px": round(self.max_favorable, 6),
            "mae_px": round(self.max_adverse, 6),
        }


    def as_outcome_row(self) -> OutcomeRow:
        # Touch flags
        t1_touched = self.t1_ts is not None
        t2_touched = self.t2_ts is not None
        stop_touched = self.stop_ts is not None

        entry_ts = self.entry_ts
        exit_ts = self.done_ts

        time_in_trade_s = _dt_diff_s(entry_ts, exit_ts) if self.entered else None
        first_touch, first_touch_ts_utc = self._first_touch_info()
        mfe_mae = self._mfe_mae_metrics()

        # R-based PnL
        pnl_R = None
        if self.entered and self.entry_px is not None and self.done_reason and self.R > 0:
            if self.done_reason == "STOP":
                pnl_R = -1.0
            elif self.done_reason == "T2":
                pnl_R = 2.5
            elif self.done_reason == "TIMEOUT":
                # TIMEOUT pnl_R is overridden at finalize using exit_px
                pnl_R = 0.0

        return OutcomeRow(
            alert_id=self.alert_id,
            symbol=self.symbol,
            side=self.side,
            strategy_profile=self.strategy_profile,
            strategy_version=self.strategy_version,
            config_hash=self.config_hash,
            entry_ts_utc=_dt_to_iso(self.entry_ts),
            entry_px=self.entry_px,
            exit_ts_utc=_dt_to_iso(self.done_ts),
            exit_px=None,
            exit_reason=self.done_reason,
            pnl_R=pnl_R,
            pnl_usd_est=None,
            t1_touched=t1_touched,
            t1_ts_utc=_dt_to_iso(self.t1_ts),
            t2_touched=t2_touched,
            t2_ts_utc=_dt_to_iso(self.t2_ts),
            stop_touched=stop_touched,
            stop_ts_utc=_dt_to_iso(self.stop_ts),
            first_touch=first_touch,
            first_touch_ts_utc=first_touch_ts_utc,
            mfe_R=mfe_mae["mfe_r"],
            mae_R=mfe_mae["mae_r"],
            mfe_r=mfe_mae["mfe_r"],
            mae_r=mfe_mae["mae_r"],
            mfe_px=mfe_mae["mfe_px"],
            mae_px=mfe_mae["mae_px"],
            time_in_trade_s=time_in_trade_s,
            time_to_t1_s=_dt_diff_s(self.entry_ts, self.t1_ts) if self.entered else None,
            time_to_t2_s=_dt_diff_s(self.entry_ts, self.t2_ts) if self.entered else None,
            time_to_stop_s=_dt_diff_s(self.entry_ts, self.stop_ts) if self.entered else None,
            max_spread_bps_seen=None,
            avg_spread_bps_seen=None,
            entered=self.entered,
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
        assert_valid_watch_lifecycle(w)
        arr = self.active.setdefault(w.symbol, [])
        # limit active watches per symbol
        if len(arr) >= self.max_active_per_symbol:
            return
        arr.append(w)
        self.save_state()


    def update(self, symbol: str, high: float, low: float, close: float, ts: datetime):
        if symbol not in self.active:
            return

        bar = WatchBar(high=high, low=low, close=close, ts=ts)
        keep: list[Watch] = []

        for w in self.active[symbol]:
            assert_valid_watch_lifecycle(w)
            if w.done:
                continue

            decision = decide_watch_bar_transition(w, bar, self.timeout)
            self._apply_decision(w, decision)
            assert_valid_watch_lifecycle(w)
            if decision.keep_active:
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

    def _apply_decision(self, w: Watch, decision: WatchDecision) -> None:
        w.max_favorable = decision.max_favorable
        w.max_adverse = decision.max_adverse

        for transition in decision.transitions:
            if transition.transition_type == WatchTransitionType.ENTER:
                self._apply_enter_transition(w, transition)
                continue
            if transition.transition_type == WatchTransitionType.EXPIRE_PENDING:
                self._apply_expire_pending_transition(w, transition)
                continue
            if transition.transition_type == WatchTransitionType.TOUCH_T1:
                self._apply_touch_t1_transition(w, transition)
                continue
            if transition.transition_type == WatchTransitionType.EXIT_STOP:
                self._apply_exit_stop_transition(w, transition)
                continue
            if transition.transition_type == WatchTransitionType.EXIT_T2:
                self._apply_exit_t2_transition(w, transition)
                continue
            if transition.transition_type == WatchTransitionType.EXIT_TIMEOUT:
                self._apply_exit_timeout_transition(w, transition)
                continue
            raise ValueError(f"Unknown watch transition type: {transition.transition_type}")

    def _apply_enter_transition(self, w: Watch, transition) -> None:
        if w.lifecycle_state != WatchLifecycleState.PENDING_ENTRY:
            raise ValueError("watch cannot enter twice")
        w.entered = True
        w.entry_ts = transition.ts
        w.entry_px = float(w.entry_high)
        journal.log_execution(w._exec_row("ENTRY", transition.ts, w.entry_px, transition.px_source or "zone_touch_entry_high"))

    def _apply_expire_pending_transition(self, w: Watch, transition) -> None:
        if w.lifecycle_state != WatchLifecycleState.PENDING_ENTRY:
            raise ValueError("only pending watches can expire without entry")
        w.done = True
        w.done_reason = "NO_ENTRY_FILL"
        w.done_ts = transition.ts
        journal.log_outcome(w.as_outcome_row())

    def _apply_touch_t1_transition(self, w: Watch, transition) -> None:
        if w.lifecycle_state != WatchLifecycleState.ENTERED:
            raise ValueError("watch must be entered before T1")
        if w.hit_t1:
            raise ValueError("watch cannot touch T1 twice")
        w.hit_t1 = True
        w.t1_ts = transition.ts
        journal.log_execution(w._exec_row("EXIT_T1", transition.ts, float(w.t1), transition.px_source or "bar_touch_t1"))

    def _apply_exit_stop_transition(self, w: Watch, transition) -> None:
        if w.lifecycle_state != WatchLifecycleState.ENTERED:
            raise ValueError("watch cannot stop out before entry")
        w.done = True
        w.done_reason = "STOP"
        w.done_ts = transition.ts
        w.stop_ts = transition.ts

        exit_px = float(w.stop)
        journal.log_execution(w._exec_row("EXIT_STOP", transition.ts, exit_px, transition.px_source or "bar_touch_stop"))
        out = replace(w.as_outcome_row(), exit_px=exit_px)
        journal.log_outcome(out)

    def _apply_exit_t2_transition(self, w: Watch, transition) -> None:
        if w.lifecycle_state != WatchLifecycleState.ENTERED:
            raise ValueError("watch cannot exit T2 before entry")
        if w.hit_t2:
            raise ValueError("watch cannot exit T2 twice")
        w.hit_t2 = True
        w.t2_ts = transition.ts
        w.done = True
        w.done_reason = "T2"
        w.done_ts = transition.ts

        exit_px = float(w.t2)
        journal.log_execution(w._exec_row("EXIT_T2", transition.ts, exit_px, transition.px_source or "bar_touch_t2"))
        out = replace(w.as_outcome_row(), exit_px=exit_px)
        journal.log_outcome(out)

    def _apply_exit_timeout_transition(self, w: Watch, transition) -> None:
        if w.lifecycle_state != WatchLifecycleState.ENTERED:
            raise ValueError("watch cannot timeout before entry")
        w.done = True
        w.done_reason = "TIMEOUT"
        w.done_ts = transition.ts

        exit_px = float(transition.px)
        journal.log_execution(w._exec_row("EXIT_TIMEOUT", transition.ts, exit_px, transition.px_source or "bar_close_timeout"))
        out = replace(w.as_outcome_row(), exit_px=exit_px)
        if w.entry_px is not None and w.R > 0:
            out = replace(out, pnl_R=round((exit_px - w.entry_px) / w.R, 4))
        journal.log_outcome(out)
