from __future__ import annotations

import asyncio

from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.enums import DataFeed
from . import config


class _SingleOwnerStreamMixin:
    async def _run_forever(self) -> None:
        """Run one websocket lifecycle and let Trade Scout own retries/cooldowns."""
        self._loop = asyncio.get_running_loop()
        while not any(
            v
            for k, v in self._handlers.items()
            if k not in ("cancelErrors", "corrections")
        ):
            if not self._stop_stream_queue.empty():
                self._stop_stream_queue.get(timeout=1)
                return
            await asyncio.sleep(0)

        self._should_run = True
        self._running = False
        if not self._should_run:
            return

        await self._start_ws()
        await self._send_subscribe_msg()
        self._running = True
        try:
            await self._consume()
        finally:
            self._running = False


class GuardedCryptoDataStream(_SingleOwnerStreamMixin, CryptoDataStream):
    pass


class GuardedStockDataStream(_SingleOwnerStreamMixin, StockDataStream):
    pass


def make_stream():
    if config.MARKET_TYPE == "crypto":
        return GuardedCryptoDataStream(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    if config.MARKET_TYPE == "stocks":
        feed = DataFeed.IEX if config.ALPACA_STOCK_FEED == "iex" else DataFeed.SIP
        return GuardedStockDataStream(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, feed=feed)

    raise ValueError(f"Unknown MARKET_TYPE: {config.MARKET_TYPE}")


def anchor_symbol() -> str:
    if config.MARKET_TYPE == "crypto":
        return "BTC/USD"
    return "SPY"


def classify_anchor_regime(computed: dict) -> str | None:
    """
    Generic regime classification using computed indicators.
    Works for BTC, SPY, etc.
    """
    close = computed.get("close")
    ma50 = computed.get("ma50")
    ma200 = computed.get("ma200")
    if close is None or ma200 is None:
        return None
    if close < ma200:
        return "bear"
    if ma50 is not None and close > ma200 and close > ma50:
        return "bull"
    return "chop"
