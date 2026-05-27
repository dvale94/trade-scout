from __future__ import annotations

import asyncio
import queue
import unittest

from trade_scout.market import _SingleOwnerStreamMixin


class _FakeSingleOwnerStream(_SingleOwnerStreamMixin):
    def __init__(self) -> None:
        self._handlers = {"bars": {"BTC/USD": object()}, "cancelErrors": {}, "corrections": {}}
        self._stop_stream_queue = queue.Queue()
        self._should_run = True
        self._running = False
        self._loop = None
        self.start_calls = 0
        self.subscribe_calls = 0
        self.consume_calls = 0

    async def _start_ws(self) -> None:
        self.start_calls += 1
        raise ValueError("connection limit exceeded")

    async def _send_subscribe_msg(self) -> None:
        self.subscribe_calls += 1

    async def _consume(self) -> None:
        self.consume_calls += 1


class MarketWebsocketStreamTest(unittest.TestCase):
    def test_single_owner_stream_raises_hard_failure_without_internal_retry_loop(self) -> None:
        stream = _FakeSingleOwnerStream()

        with self.assertRaisesRegex(ValueError, "connection limit exceeded"):
            asyncio.run(stream._run_forever())

        self.assertEqual(1, stream.start_calls)
        self.assertEqual(0, stream.subscribe_calls)
        self.assertEqual(0, stream.consume_calls)
        self.assertFalse(stream._running)

