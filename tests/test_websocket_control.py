from __future__ import annotations

import unittest

from trade_scout.websocket_control import (
    WebsocketCircuitBreaker,
    WebsocketLifecycleGuard,
    classify_websocket_failure,
    format_websocket_reason,
    release_run_guard,
    stop_stream_client,
    try_acquire_run_guard,
    websocket_backoff_delay,
    WEBSOCKET_STATE_COOLDOWN,
    WEBSOCKET_STATE_FAILED,
    WEBSOCKET_STATE_RUNNING,
    WEBSOCKET_STATE_STARTING,
)


class _FakeStream:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.close_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


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

    def test_server_rejected_websocket_connection_classifies_as_hard_failure(self) -> None:
        self.assertEqual(
            "HTTP_429",
            classify_websocket_failure("server rejected WebSocket connection"),
        )

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

    def test_transient_disconnect_uses_bounded_exponential_backoff(self) -> None:
        breaker = WebsocketCircuitBreaker(
            initial_delay_seconds=5.0,
            max_delay_seconds=300.0,
        )

        first_delay, first_cooldown = breaker.record_failure(
            reason="UNEXPECTED_DISCONNECT",
            now_monotonic=10.0,
        )
        second_delay, second_cooldown = breaker.record_failure(
            reason="UNEXPECTED_DISCONNECT",
            now_monotonic=20.0,
        )

        self.assertFalse(first_cooldown)
        self.assertFalse(second_cooldown)
        self.assertGreaterEqual(first_delay, 5.0)
        self.assertLessEqual(first_delay, 300.0 * 1.25)
        self.assertGreater(second_delay, first_delay)

    def test_lifecycle_guard_skips_duplicate_start_while_active(self) -> None:
        lifecycle = WebsocketLifecycleGuard()

        self.assertTrue(lifecycle.try_mark_starting())
        self.assertEqual(WEBSOCKET_STATE_STARTING, lifecycle.current_state())
        lifecycle.mark_running()

        self.assertFalse(lifecycle.try_mark_starting())
        self.assertEqual(WEBSOCKET_STATE_RUNNING, lifecycle.current_state())

    def test_lifecycle_guard_stop_existing_client_before_reconnect(self) -> None:
        lifecycle = WebsocketLifecycleGuard()
        stream = _FakeStream()
        lifecycle.bind_client(stream)

        stopped = lifecycle.stop_existing_client()

        self.assertTrue(stopped)
        self.assertEqual(1, stream.stop_calls)
        self.assertEqual(1, stream.close_calls)

    def test_lifecycle_guard_connection_limit_exceeded_triggers_cooldown(self) -> None:
        lifecycle = WebsocketLifecycleGuard()

        delay, cooldown_triggered = lifecycle.record_failure(
            reason=classify_websocket_failure("connection limit exceeded"),
            now_monotonic=100.0,
        )

        self.assertTrue(cooldown_triggered)
        self.assertEqual(WEBSOCKET_STATE_COOLDOWN, lifecycle.current_state())
        self.assertEqual(600.0, delay)

    def test_lifecycle_guard_http_429_triggers_cooldown(self) -> None:
        lifecycle = WebsocketLifecycleGuard()

        _, cooldown_triggered = lifecycle.record_failure(
            reason=classify_websocket_failure("HTTP 429 too many requests"),
            now_monotonic=100.0,
        )

        self.assertTrue(cooldown_triggered)
        self.assertEqual(WEBSOCKET_STATE_COOLDOWN, lifecycle.current_state())

    def test_lifecycle_guard_auth_failed_triggers_cooldown(self) -> None:
        lifecycle = WebsocketLifecycleGuard()

        _, cooldown_triggered = lifecycle.record_failure(
            reason=classify_websocket_failure("auth failed"),
            now_monotonic=100.0,
        )

        self.assertTrue(cooldown_triggered)
        self.assertEqual(WEBSOCKET_STATE_COOLDOWN, lifecycle.current_state())

    def test_lifecycle_guard_repeated_connection_limit_errors_stay_in_cooldown(self) -> None:
        lifecycle = WebsocketLifecycleGuard()
        lifecycle.record_failure(
            reason=classify_websocket_failure("connection limit exceeded"),
            now_monotonic=100.0,
        )

        self.assertTrue(lifecycle.is_cooling_down(150.0))
        self.assertEqual(WEBSOCKET_STATE_COOLDOWN, lifecycle.current_state())
        self.assertFalse(lifecycle.try_mark_starting())

    def test_record_success_resets_failed_state_and_attempt_count(self) -> None:
        lifecycle = WebsocketLifecycleGuard()
        lifecycle.record_failure(
            reason="UNEXPECTED_DISCONNECT",
            now_monotonic=100.0,
        )

        self.assertEqual(WEBSOCKET_STATE_FAILED, lifecycle.current_state())
        self.assertGreater(lifecycle.circuit_breaker.reconnect_attempt, 0)

        lifecycle.record_success()

        self.assertEqual(WEBSOCKET_STATE_RUNNING, lifecycle.current_state())
        self.assertEqual(0, lifecycle.circuit_breaker.reconnect_attempt)

    def test_stop_stream_client_handles_missing_methods(self) -> None:
        self.assertFalse(stop_stream_client(object()))

    def test_format_websocket_reason_for_logs(self) -> None:
        self.assertEqual("connection_limit_exceeded", format_websocket_reason("CONNECTION_LIMIT_EXCEEDED"))
