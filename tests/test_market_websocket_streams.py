from __future__ import annotations

import asyncio
import unittest

from alpaca.data.enums import DataFeed

from trade_scout.market import GuardedCryptoDataStream, GuardedStockDataStream


async def _noop_bar_handler(bar) -> None:
    return None


class MarketWebsocketStreamTest(unittest.TestCase):
    def _make_crypto_stream(self) -> GuardedCryptoDataStream:
        return GuardedCryptoDataStream("key", "secret")

    def _make_stock_stream(self) -> GuardedStockDataStream:
        return GuardedStockDataStream("key", "secret", feed=DataFeed.IEX)

    def test_guarded_crypto_stream_preserves_subscription_and_run_order(self) -> None:
        stream = self._make_crypto_stream()
        stream.subscribe_bars(_noop_bar_handler, "BTC/USD")
        calls: list[str] = []

        async def _start_ws() -> None:
            calls.append("start_ws")

        async def _send_subscribe_msg() -> None:
            calls.append("send_subscribe")

        async def _consume() -> None:
            calls.append("consume")

        stream._start_ws = _start_ws
        stream._send_subscribe_msg = _send_subscribe_msg
        stream._consume = _consume

        stream.run()

        self.assertEqual(["start_ws", "send_subscribe", "consume"], calls)
        self.assertIs(_noop_bar_handler, stream._handlers["bars"]["BTC/USD"])
        self.assertFalse(stream._running)

    def test_guarded_stock_stream_preserves_subscription_and_run_order(self) -> None:
        stream = self._make_stock_stream()
        stream.subscribe_bars(_noop_bar_handler, "AAPL")
        calls: list[str] = []

        async def _start_ws() -> None:
            calls.append("start_ws")

        async def _send_subscribe_msg() -> None:
            calls.append("send_subscribe")

        async def _consume() -> None:
            calls.append("consume")

        stream._start_ws = _start_ws
        stream._send_subscribe_msg = _send_subscribe_msg
        stream._consume = _consume

        stream.run()

        self.assertEqual(["start_ws", "send_subscribe", "consume"], calls)
        self.assertIs(_noop_bar_handler, stream._handlers["bars"]["AAPL"])
        self.assertFalse(stream._running)

    def test_guarded_stream_raises_hard_failure_without_internal_retry_loop(self) -> None:
        stream = self._make_crypto_stream()
        stream.subscribe_bars(_noop_bar_handler, "BTC/USD")
        start_calls = {"count": 0}

        async def _start_ws() -> None:
            start_calls["count"] += 1
            raise ValueError("connection limit exceeded")

        async def _send_subscribe_msg() -> None:
            raise AssertionError("subscribe should not run after failed start")

        stream._start_ws = _start_ws
        stream._send_subscribe_msg = _send_subscribe_msg

        with self.assertRaisesRegex(ValueError, "connection limit exceeded"):
            asyncio.run(stream._run_forever())

        self.assertEqual(1, start_calls["count"])
        self.assertFalse(stream._running)

    def test_guarded_stream_resets_running_state_when_consume_raises(self) -> None:
        stream = self._make_stock_stream()
        stream.subscribe_bars(_noop_bar_handler, "AAPL")

        async def _start_ws() -> None:
            return None

        async def _send_subscribe_msg() -> None:
            return None

        async def _consume() -> None:
            raise RuntimeError("consume boom")

        stream._start_ws = _start_ws
        stream._send_subscribe_msg = _send_subscribe_msg
        stream._consume = _consume

        with self.assertRaisesRegex(RuntimeError, "consume boom"):
            asyncio.run(stream._run_forever())

        self.assertFalse(stream._running)

