from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from trade_scout import main
from trade_scout.websocket_control import release_run_guard


class _FakeAnchorTracker:
    def __init__(self) -> None:
        self.anchor_symbol = "BTC/USD"


class _FakeWatchManager:
    def __init__(self, *args, **kwargs) -> None:
        self.active = {}

    def load_state(self) -> None:
        return None


class _FakeTelemetryTracker:
    def __init__(self, *args, **kwargs) -> None:
        return None


class _FakeExecutionWorker:
    def __init__(self, *args, **kwargs) -> None:
        self.run_forever_called = False

    def run_forever(self) -> None:
        self.run_forever_called = True


class _FakeBarProcessor:
    def __init__(self, *args, **kwargs) -> None:
        self.on_bar = lambda *a, **k: None


class _FakeStream:
    def __init__(self) -> None:
        self.subscribed_symbols: list[str] = []
        self.stop_calls = 0
        self.close_calls = 0

    def subscribe_bars(self, callback, symbol: str) -> None:
        self.subscribed_symbols.append(symbol)

    def run(self) -> None:
        raise ValueError("connection limit exceeded")

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _FakeThread:
    def __init__(self, *, target=None, name=None, daemon=None) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


class MainWebsocketReliabilityTest(unittest.TestCase):
    def tearDown(self) -> None:
        release_run_guard()

    def test_websocket_cooldown_does_not_stop_execution_worker_startup(self) -> None:
        profile = SimpleNamespace(
            name="test_profile",
            ma_mid=5,
            atr_period=5,
            breakout_lookback=5,
            persist_bars=2,
            lookback_bars=20,
            timeframe_min=5,
            seed_minutes=60,
            cooldown_min=5,
            watch_timeout_min=240,
        )
        fake_stream = _FakeStream()
        fake_thread = _FakeThread(target=lambda: None, name="worker", daemon=True)
        scheduler_started = {"value": False}

        def _fake_sleep(seconds: float) -> None:
            raise KeyboardInterrupt()

        with ExitStack() as stack:
            stack.enter_context(patch.dict(main.strategy_profiles.PROFILES, {"test_profile": profile}, clear=False))
            stack.enter_context(patch.object(main.config, "STRATEGY_PROFILE", "test_profile"))
            stack.enter_context(patch.object(main.config, "SYMBOLS", ["ETH/USD"]))
            stack.enter_context(patch.object(main.config, "ENABLED_STRATEGY_IDS", ["fake"]))
            stack.enter_context(patch.object(main.config, "STRATEGY_PARAMS", {}))
            stack.enter_context(patch.object(main.config, "EXECUTION_DB_PATH", "test.db"))
            stack.enter_context(patch.object(main.config, "EXECUTION_POLL_INTERVAL_SECONDS", 1.0))
            stack.enter_context(patch.object(main.config, "EXECUTION_BATCH_SIZE", 10))
            stack.enter_context(patch.object(main.config, "EXECUTION_MAX_ALERT_AGE_SECONDS", 3600))
            stack.enter_context(patch.object(main.config, "EXECUTION_WORKER_ID", "exec-worker"))
            stack.enter_context(patch.object(main.config, "EXECUTION_ENABLED", True))
            stack.enter_context(patch.object(main, "compute_config_hash", return_value="config123"))
            stack.enter_context(
                patch.object(main.seed, "seed_history", return_value={"ETH/USD": [], "BTC/USD": []})
            )
            stack.enter_context(patch.object(main.anchors, "AnchorTracker", _FakeAnchorTracker))
            stack.enter_context(patch.object(main.watch, "WatchManager", _FakeWatchManager))
            stack.enter_context(patch.object(main, "TelemetryTracker", _FakeTelemetryTracker))
            stack.enter_context(patch.object(main, "ExecutionWorker", _FakeExecutionWorker))
            stack.enter_context(patch.object(main, "BarProcessor", _FakeBarProcessor))
            stack.enter_context(
                patch.object(main.threading, "Thread", side_effect=lambda **kwargs: fake_thread)
            )
            stack.enter_context(
                patch.object(
                    main.scheduler_tasks,
                    "start_scheduler",
                    side_effect=lambda: scheduler_started.__setitem__("value", True),
                )
            )
            stack.enter_context(patch.object(main.market, "make_stream", return_value=fake_stream))
            stack.enter_context(
                patch.object(
                    main,
                    "load_enabled_strategies",
                    return_value=[SimpleNamespace(strategy_id="fake", strategy_version="v1")],
                )
            )
            stack.enter_context(
                patch.object(
                    main,
                    "build_strategy_metadata",
                    return_value=SimpleNamespace(strategy_id="fake", strategy_version="v1"),
                )
            )
            stack.enter_context(patch.object(main.time, "sleep", side_effect=_fake_sleep))

            with self.assertRaises(KeyboardInterrupt):
                main.run()

        self.assertTrue(scheduler_started["value"])
        self.assertTrue(fake_thread.started)
        self.assertGreaterEqual(fake_stream.stop_calls, 1)
        self.assertGreaterEqual(fake_stream.close_calls, 1)
