from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .watch_state import WatchLifecycleState, lifecycle_state_for_watch


class WatchTransitionType(str, Enum):
    ENTER = "ENTER"
    EXPIRE_PENDING = "EXPIRE_PENDING"
    TOUCH_T1 = "TOUCH_T1"
    EXIT_STOP = "EXIT_STOP"
    EXIT_T2 = "EXIT_T2"
    EXIT_TIMEOUT = "EXIT_TIMEOUT"


@dataclass(frozen=True)
class WatchBar:
    high: float
    low: float
    close: float
    ts: datetime


@dataclass(frozen=True)
class WatchTransition:
    transition_type: WatchTransitionType
    ts: datetime
    px: float | None = None
    px_source: str | None = None


@dataclass(frozen=True)
class WatchDecision:
    state_before: WatchLifecycleState
    state_after: WatchLifecycleState
    max_favorable: float
    max_adverse: float
    transitions: tuple[WatchTransition, ...]
    keep_active: bool


def decide_watch_bar_transition(watch, bar: WatchBar, timeout: timedelta) -> WatchDecision:
    state_before = lifecycle_state_for_watch(watch)
    max_favorable = max(watch.max_favorable, bar.high)
    max_adverse = min(watch.max_adverse, bar.low)

    if state_before.is_terminal:
        return WatchDecision(
            state_before=state_before,
            state_after=state_before,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
            transitions=(),
            keep_active=False,
        )

    transitions: list[WatchTransition] = []
    entered = watch.entered
    hit_t1 = watch.hit_t1
    hit_t2 = watch.hit_t2
    timed_out = (bar.ts - watch.created_ts) >= timeout

    if state_before == WatchLifecycleState.PENDING_ENTRY:
        zone_touched = (bar.low <= watch.entry_high) and (bar.high >= watch.entry_low)
        if zone_touched:
            entered = True
            transitions.append(
                WatchTransition(
                    transition_type=WatchTransitionType.ENTER,
                    ts=bar.ts,
                    px=float(watch.entry_high),
                    px_source="zone_touch_entry_high",
                )
            )

        if not entered:
            if timed_out:
                transitions.append(
                    WatchTransition(
                        transition_type=WatchTransitionType.EXPIRE_PENDING,
                        ts=bar.ts,
                    )
                )
                state_after = WatchLifecycleState.EXPIRED
                keep_active = False
            else:
                state_after = WatchLifecycleState.PENDING_ENTRY
                keep_active = True
            return WatchDecision(
                state_before=state_before,
                state_after=state_after,
                max_favorable=max_favorable,
                max_adverse=max_adverse,
                transitions=tuple(transitions),
                keep_active=keep_active,
            )

    if bar.low <= watch.stop:
        transitions.append(
            WatchTransition(
                transition_type=WatchTransitionType.EXIT_STOP,
                ts=bar.ts,
                px=float(watch.stop),
                px_source="bar_touch_stop",
            )
        )
        return WatchDecision(
            state_before=state_before,
            state_after=WatchLifecycleState.EXITED,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
            transitions=tuple(transitions),
            keep_active=False,
        )

    if not hit_t1 and bar.high >= watch.t1:
        hit_t1 = True
        transitions.append(
            WatchTransition(
                transition_type=WatchTransitionType.TOUCH_T1,
                ts=bar.ts,
                px=float(watch.t1),
                px_source="bar_touch_t1",
            )
        )

    if not hit_t2 and bar.high >= watch.t2:
        transitions.append(
            WatchTransition(
                transition_type=WatchTransitionType.EXIT_T2,
                ts=bar.ts,
                px=float(watch.t2),
                px_source="bar_touch_t2",
            )
        )
        return WatchDecision(
            state_before=state_before,
            state_after=WatchLifecycleState.EXITED,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
            transitions=tuple(transitions),
            keep_active=False,
        )

    if timed_out:
        transitions.append(
            WatchTransition(
                transition_type=WatchTransitionType.EXIT_TIMEOUT,
                ts=bar.ts,
                px=float(bar.close),
                px_source="bar_close_timeout",
            )
        )
        return WatchDecision(
            state_before=state_before,
            state_after=WatchLifecycleState.EXITED,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
            transitions=tuple(transitions),
            keep_active=False,
        )

    return WatchDecision(
        state_before=state_before,
        state_after=WatchLifecycleState.ENTERED,
        max_favorable=max_favorable,
        max_adverse=max_adverse,
        transitions=tuple(transitions),
        keep_active=True,
    )
