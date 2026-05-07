from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field


_RUN_LOCK = threading.Lock()
_RUN_ACTIVE = False

HARD_FAILURE_REASONS = {
    "HTTP_429",
    "CONNECTION_LIMIT_EXCEEDED",
    "AUTH_FAILURE",
}


def try_acquire_run_guard() -> bool:
    global _RUN_ACTIVE
    with _RUN_LOCK:
        if _RUN_ACTIVE:
            return False
        _RUN_ACTIVE = True
        return True


def release_run_guard() -> None:
    global _RUN_ACTIVE
    with _RUN_LOCK:
        _RUN_ACTIVE = False


def websocket_backoff_delay(
    attempt: int,
    *,
    base: float = 5.0,
    cap: float = 300.0,
    jitter_ratio: float = 0.25,
) -> float:
    normalized_attempt = max(1, int(attempt))
    delay = min(cap, base * (2 ** (normalized_attempt - 1)))
    jitter = random.uniform(0.0, delay * max(jitter_ratio, 0.0))
    return delay + jitter


def classify_websocket_failure(
    error: Exception | str | None,
    *,
    immediate_disconnect: bool = False,
) -> str:
    text = str(error or "").strip().lower()
    if "429" in text or "http 429" in text:
        return "HTTP_429"
    if "connection limit exceeded" in text:
        return "CONNECTION_LIMIT_EXCEEDED"
    if any(
        token in text
        for token in (
            "auth failed",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "not authenticated",
        )
    ):
        return "AUTH_FAILURE"
    if immediate_disconnect:
        return "IMMEDIATE_DISCONNECT"
    return "UNEXPECTED_DISCONNECT"


@dataclass
class WebsocketCircuitBreaker:
    initial_delay_seconds: float = 5.0
    max_delay_seconds: float = 300.0
    cooldown_seconds: float = 600.0
    failure_window_seconds: float = 180.0
    max_failures_in_window: int = 5
    immediate_disconnect_window_seconds: float = 120.0
    immediate_disconnect_threshold: int = 3
    reconnect_attempt: int = 0
    cooldown_until_monotonic: float = 0.0
    recent_failure_times: list[float] = field(default_factory=list)
    recent_immediate_disconnect_times: list[float] = field(default_factory=list)

    def record_success(self) -> None:
        self.reconnect_attempt = 0
        self.cooldown_until_monotonic = 0.0
        self.recent_failure_times.clear()
        self.recent_immediate_disconnect_times.clear()

    def is_cooling_down(self, now_monotonic: float) -> bool:
        return now_monotonic < self.cooldown_until_monotonic

    def cooldown_remaining_seconds(self, now_monotonic: float) -> float:
        return max(0.0, self.cooldown_until_monotonic - now_monotonic)

    def record_failure(
        self,
        *,
        reason: str,
        now_monotonic: float,
    ) -> tuple[float, bool]:
        self.reconnect_attempt += 1
        self._prune_recent_failures(now_monotonic)
        self.recent_failure_times.append(now_monotonic)

        if reason == "IMMEDIATE_DISCONNECT":
            self._prune_recent_immediate_disconnects(now_monotonic)
            self.recent_immediate_disconnect_times.append(now_monotonic)

        if self._should_trigger_cooldown(reason):
            self.cooldown_until_monotonic = max(
                self.cooldown_until_monotonic,
                now_monotonic + self.cooldown_seconds,
            )
            return self.cooldown_remaining_seconds(now_monotonic), True

        return (
            websocket_backoff_delay(
                self.reconnect_attempt,
                base=self.initial_delay_seconds,
                cap=self.max_delay_seconds,
            ),
            False,
        )

    def _should_trigger_cooldown(self, reason: str) -> bool:
        if reason in HARD_FAILURE_REASONS:
            return True
        if len(self.recent_failure_times) >= self.max_failures_in_window:
            return True
        if len(self.recent_immediate_disconnect_times) >= self.immediate_disconnect_threshold:
            return True
        return False

    def _prune_recent_failures(self, now_monotonic: float) -> None:
        cutoff = now_monotonic - self.failure_window_seconds
        self.recent_failure_times = [ts for ts in self.recent_failure_times if ts >= cutoff]

    def _prune_recent_immediate_disconnects(self, now_monotonic: float) -> None:
        cutoff = now_monotonic - self.immediate_disconnect_window_seconds
        self.recent_immediate_disconnect_times = [
            ts for ts in self.recent_immediate_disconnect_times if ts >= cutoff
        ]
