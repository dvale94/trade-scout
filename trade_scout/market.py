from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.enums import DataFeed
from . import config


def make_stream():
    if config.MARKET_TYPE == "crypto":
        return CryptoDataStream(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    if config.MARKET_TYPE == "stocks":
        feed = DataFeed.IEX if config.ALPACA_STOCK_FEED == "iex" else DataFeed.SIP
        return StockDataStream(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, feed=feed)

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