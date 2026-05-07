from __future__ import annotations

import unittest

from trade_scout.websocket_control import (
    WebsocketCircuitBreaker,
    classify_websocket_failure,
    release_run_guard,
    try_acquire_run_guard,
    websocket_backoff_delay,
)


class WebsocketControlTest(unittest.TestCase):
    def tearDown(self) -> None:
        release_run_guard()

    def test_websocket_backoff_delay_respects_initial_delay_and_cap(self) -> None:
        attempt_one = websocket_backoff_delay(1, base=5.0, cap=300.0, jitter_ratio=0.0)
        attempt_three = websocket_backoff_delay(3, base=5.0, cap=300.0, jitter_ratio=0.0)
        attempt_large = websocket_backoff_delay(20, base=5.0, cap=300.0, jitter_ratio=0.0)

        self.assertEqual(5.0, attempt_one)
        self.assertEqual(20.0, attempt_three)
        self.assertEqual(300.0, attempt_large)

    def test_http_429_triggers_cooldown(self) -> None:
        breaker = WebsocketCircuitBreaker(cooldown_seconds=600.0)

        delay, cooldown_triggered = breaker.record_failure(
            reason=classify_websocket_failure("HTTP 429 too many requests"),
            now_monotonic=100.0,
        )

        self.assertTrue(cooldown_triggered)
        self.assertEqual(600.0, delay)
        self.assertTrue(breaker.is_cooling_down(101.0))

    def test_connection_limit_and_auth_failure_trigger_cooldown(self) -> None:
        limit_breaker = WebsocketCircuitBreaker(cooldown_seconds=600.0)
        auth_breaker = WebsocketCircuitBreaker(cooldown_seconds=600.0)

        _, limit_cooldown = limit_breaker.record_failure(
            reason=classify_websocket_failure("connection limit exceeded"),
            now_monotonic=50.0,
        )
        _, auth_cooldown = auth_breaker.record_failure(
            reason=classify_websocket_failure("auth failed"),
            now_monotonic=50.0,
        )

        self.assertTrue(limit_cooldown)
        self.assertTrue(auth_cooldown)

    def test_repeated_immediate_disconnects_trigger_cooldown(self) -> None:
        breaker = WebsocketCircuitBreaker(
            cooldown_seconds=600.0,
            immediate_disconnect_threshold=3,
            immediate_disconnect_window_seconds=120.0,
        )

        breaker.record_failure(reason="IMMEDIATE_DISCONNECT", now_monotonic=10.0)
        breaker.record_failure(reason="IMMEDIATE_DISCONNECT", now_monotonic=20.0)
        delay, cooldown_triggered = breaker.record_failure(
            reason="IMMEDIATE_DISCONNECT",
            now_monotonic=30.0,
        )

        self.assertTrue(cooldown_triggered)
        self.assertEqual(600.0, delay)

    def test_duplicate_run_guard_skips_second_start(self) -> None:
        self.assertTrue(try_acquire_run_guard())
        self.assertFalse(try_acquire_run_guard())
