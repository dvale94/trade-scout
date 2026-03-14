from __future__ import annotations

from enum import Enum


class WatchLifecycleState(str, Enum):
    PENDING_ENTRY = "PENDING_ENTRY"
    ENTERED = "ENTERED"
    EXITED = "EXITED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in {self.EXITED, self.EXPIRED}


def lifecycle_state_for_watch(watch) -> WatchLifecycleState:
    if watch.done:
        if watch.done_reason == "NO_ENTRY_FILL":
            return WatchLifecycleState.EXPIRED
        return WatchLifecycleState.EXITED
    if watch.entered:
        return WatchLifecycleState.ENTERED
    return WatchLifecycleState.PENDING_ENTRY


def assert_valid_watch_lifecycle(watch) -> None:
    state = lifecycle_state_for_watch(watch)
    if state == WatchLifecycleState.PENDING_ENTRY:
        if watch.done or watch.entered:
            raise ValueError("pending watch cannot be done or entered")
    elif state == WatchLifecycleState.ENTERED:
        if not watch.entered or watch.done:
            raise ValueError("entered watch must be entered and not done")
        if watch.entry_ts is None or watch.entry_px is None:
            raise ValueError("entered watch must have entry timestamp and price")
    elif state == WatchLifecycleState.EXITED:
        if not watch.done or watch.done_reason in {None, "NO_ENTRY_FILL"}:
            raise ValueError("exited watch must be done with an exit reason")
        if not watch.entered:
            raise ValueError("exited watch cannot exit before entering")
        if watch.done_ts is None:
            raise ValueError("exited watch must have done timestamp")
    elif state == WatchLifecycleState.EXPIRED:
        if not watch.done or watch.done_reason != "NO_ENTRY_FILL":
            raise ValueError("expired watch must be done with NO_ENTRY_FILL")
        if watch.entered:
            raise ValueError("expired watch cannot also be entered")
        if watch.done_ts is None:
            raise ValueError("expired watch must have done timestamp")
