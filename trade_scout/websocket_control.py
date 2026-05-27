from __future__ import annotations

import asyncio
import inspect
import random
import threading
from dataclasses import dataclass, field
from typing import Any


_RUN_LOCK = threading.Lock()
_RUN_ACTIVE = False

WEBSOCKET_STATE_STOPPED = "STOPPED"
WEBSOCKET_STATE_STARTING = "STARTING"
WEBSOCKET_STATE_RUNNING = "RUNNING"
WEBSOCKET_STATE_COOLDOWN = "COOLDOWN"
WEBSOCKET_STATE_FAILED = "FAILED"

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
    if "429" in text or "http 429" in text or "server rejected websocket connection" in text:
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


def format_websocket_reason(reason: str) -> str:
    return str(reason).strip().lower()


def stop_stream_client(stream: Any) -> bool:
    if stream is None:
        return False

    attempted = False
    for method_name in ("stop", "stop_ws", "close"):
        method = getattr(stream, method_name, None)
        if not callable(method):
            continue
        attempted = True
        result = method()
        if inspect.isawaitable(result):
            asyncio.run(result)
    return attempted


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


@dataclass
class WebsocketLifecycleGuard:
    circuit_breaker: WebsocketCircuitBreaker = field(default_factory=WebsocketCircuitBreaker)
    state: str = WEBSOCKET_STATE_STOPPED
    _client: Any | None = field(default=None, init=False, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def try_mark_starting(self) -> bool:
        with self._state_lock:
            if self.state in {
                WEBSOCKET_STATE_STARTING,
                WEBSOCKET_STATE_RUNNING,
                WEBSOCKET_STATE_COOLDOWN,
            }:
                return False
            self.state = WEBSOCKET_STATE_STARTING
            return True

    def bind_client(self, client: Any) -> None:
        with self._state_lock:
            self._client = client

    def mark_running(self) -> None:
        with self._state_lock:
            self.state = WEBSOCKET_STATE_RUNNING

    def mark_stopped(self) -> None:
        with self._state_lock:
            self.state = WEBSOCKET_STATE_STOPPED
            self._client = None

    def is_cooling_down(self, now_monotonic: float) -> bool:
        cooling = self.circuit_breaker.is_cooling_down(now_monotonic)
        with self._state_lock:
            if cooling:
                self.state = WEBSOCKET_STATE_COOLDOWN
            elif self.state == WEBSOCKET_STATE_COOLDOWN:
                self.state = WEBSOCKET_STATE_STOPPED
        return cooling

    def cooldown_remaining_seconds(self, now_monotonic: float) -> float:
        return self.circuit_breaker.cooldown_remaining_seconds(now_monotonic)

    def current_state(self) -> str:
        with self._state_lock:
            return self.state

    def stop_existing_client(self) -> bool:
        with self._state_lock:
            client = self._client
            self._client = None
        if client is None:
            return False
        stop_stream_client(client)
        return True

    def record_success(self) -> None:
        self.circuit_breaker.record_success()
        with self._state_lock:
            self.state = WEBSOCKET_STATE_RUNNING

    def record_failure(
        self,
        *,
        reason: str,
        now_monotonic: float,
    ) -> tuple[float, bool]:
        delay, cooldown_triggered = self.circuit_breaker.record_failure(
            reason=reason,
            now_monotonic=now_monotonic,
        )
        with self._state_lock:
            self.state = WEBSOCKET_STATE_COOLDOWN if cooldown_triggered else WEBSOCKET_STATE_FAILED
        return delay, cooldown_triggered
