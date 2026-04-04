from __future__ import annotations

import asyncio
import sys
import types
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.Response = object
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    class _FakeTimestamp:
        def __init__(self, value):
            self.value = value

        def to_pydatetime(self):
            if isinstance(self.value, datetime):
                return self.value
            text = str(self.value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text)

    class _FakeILoc:
        def __init__(self, df):
            self.df = df

        def __getitem__(self, item):
            if isinstance(item, slice):
                return _FakeDataFrame(self.df.rows[item])
            raise TypeError("unsupported iloc access")

    class _FakeDataFrame:
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        @property
        def iloc(self):
            return _FakeILoc(self)

        def reset_index(self, drop=False):
            return self

    def _fake_dataframe(rows):
        return _FakeDataFrame(rows)

    def _fake_concat(frames, ignore_index=False):
        rows = []
        for frame in frames:
            rows.extend(frame.rows)
        return _FakeDataFrame(rows)

    def _fake_to_datetime(value, utc=False):
        return _FakeTimestamp(value)

    pandas_stub.DataFrame = _fake_dataframe
    pandas_stub.concat = _fake_concat
    pandas_stub.to_datetime = _fake_to_datetime
    pandas_stub.Timestamp = lambda value: _FakeTimestamp(value)
    sys.modules["pandas"] = pandas_stub

alert_pipeline_stub = types.ModuleType("trade_scout.alert_pipeline")
runtime_context_stub = types.ModuleType("trade_scout.runtime_context")


class _NoopAlertPipeline:
    def __init__(self, context):
        self.context = context

    def process(self, **kwargs) -> None:
        return None


alert_pipeline_stub.AlertPipeline = _NoopAlertPipeline
runtime_context_stub.RuntimeContext = object
sys.modules["trade_scout.alert_pipeline"] = alert_pipeline_stub
sys.modules["trade_scout.runtime_context"] = runtime_context_stub

from trade_scout.bar_processor import BarProcessor


class _FakeAnchor:
    def __init__(self, *, is_anchor: bool = False):
        self._is_anchor = is_anchor

    def is_anchor(self, symbol: str) -> bool:
        return self._is_anchor

    def update_from_computed(self, computed, ts) -> None:
        return None

    def inject_into_computed(self, computed):
        return dict(computed)


class _FakeTelemetry:
    def record_bar(self, *, is_anchor: bool) -> None:
        return None

    def record_score(self, score: float) -> None:
        return None

    def record_insufficient_history(self) -> None:
        return None

    def record_indicator_missing(self) -> None:
        return None

    def record_below_score(self) -> None:
        return None

    def record_tier_selected(self, tier: str) -> None:
        return None

    def maybe_flush(self, now) -> None:
        return None


class _FakeWatchManager:
    def update(self, symbol: str, high: float, low: float, close: float, ts) -> None:
        return None


class _FakeStrategy:
    def evaluate_candidate(self, df, profile):
        return SimpleNamespace(
            score=1.0,
            computed={"ma200": 100.0, "atr": 1.0},
            tier=None,
            reasons=[],
        )


class BarProcessorExecutionTriggerTest(unittest.TestCase):
    def _make_context(self, *, anchor_is_anchor: bool = False, include_df: bool = True, min_needed: int = 1):
        existing_df = sys.modules["pandas"].DataFrame(
            [
                {
                    "ts": "2026-04-04T12:00:00+00:00",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.5,
                    "volume": 1000.0,
                }
            ]
        )
        return SimpleNamespace(
            reconnect_attempt=0,
            anchor=_FakeAnchor(is_anchor=anchor_is_anchor),
            telemetry=_FakeTelemetry(),
            watch_mgr=_FakeWatchManager(),
            data={"BTC/USD": existing_df} if include_df else {},
            max_keep=100,
            min_needed=min_needed,
            strategies=[SimpleNamespace(strategy=_FakeStrategy(), metadata=SimpleNamespace(strategy_id="s1"))],
            profile=SimpleNamespace(),
            cooldown=SimpleNamespace(mark=lambda *args, **kwargs: None),
        )

    @staticmethod
    def _make_bar(close: float = 123.45):
        return SimpleNamespace(
            symbol="BTC/USD",
            timestamp="2026-04-04T12:01:00+00:00",
            open=122.0,
            high=124.0,
            low=121.5,
            close=close,
            volume=2500.0,
        )

    def test_on_bar_calls_execution_worker_trigger_processing_with_symbol_close_map(self) -> None:
        context = self._make_context()
        execution_worker = MagicMock()
        execution_worker.process_position_triggers.return_value = 0

        processor = BarProcessor(context, execution_worker=execution_worker)
        asyncio.run(processor.on_bar(self._make_bar(close=123.45)))

        execution_worker.process_position_triggers.assert_called_once_with(
            price_by_symbol={"BTC/USD": 123.45}
        )

    def test_on_bar_does_not_fail_when_execution_worker_is_none(self) -> None:
        context = self._make_context()

        processor = BarProcessor(context, execution_worker=None)
        asyncio.run(processor.on_bar(self._make_bar(close=123.45)))

        self.assertEqual(2, len(context.data["BTC/USD"]))

    def test_on_bar_catches_trigger_processing_exceptions(self) -> None:
        context = self._make_context()
        execution_worker = MagicMock()
        execution_worker.process_position_triggers.side_effect = RuntimeError("tp trigger boom")

        processor = BarProcessor(context, execution_worker=execution_worker)
        asyncio.run(processor.on_bar(self._make_bar(close=123.45)))

        self.assertEqual(2, len(context.data["BTC/USD"]))
        execution_worker.process_position_triggers.assert_called_once_with(
            price_by_symbol={"BTC/USD": 123.45}
        )

    def test_trigger_processing_runs_when_df_is_none(self) -> None:
        context = self._make_context(include_df=False)
        execution_worker = MagicMock()
        execution_worker.process_position_triggers.return_value = 0

        processor = BarProcessor(context, execution_worker=execution_worker)
        asyncio.run(processor.on_bar(self._make_bar(close=123.45)))

        execution_worker.process_position_triggers.assert_called_once_with(
            price_by_symbol={"BTC/USD": 123.45}
        )

    def test_trigger_processing_runs_when_history_is_insufficient(self) -> None:
        context = self._make_context(min_needed=5)
        execution_worker = MagicMock()
        execution_worker.process_position_triggers.return_value = 0

        processor = BarProcessor(context, execution_worker=execution_worker)
        asyncio.run(processor.on_bar(self._make_bar(close=123.45)))

        execution_worker.process_position_triggers.assert_called_once_with(
            price_by_symbol={"BTC/USD": 123.45}
        )

    def test_trigger_processing_runs_on_anchor_symbol_early_return(self) -> None:
        context = self._make_context(anchor_is_anchor=True)
        execution_worker = MagicMock()
        execution_worker.process_position_triggers.return_value = 0

        processor = BarProcessor(context, execution_worker=execution_worker)
        with patch("trade_scout.bar_processor.config.SYMBOLS", []):
            asyncio.run(processor.on_bar(self._make_bar(close=123.45)))

        execution_worker.process_position_triggers.assert_called_once_with(
            price_by_symbol={"BTC/USD": 123.45}
        )
