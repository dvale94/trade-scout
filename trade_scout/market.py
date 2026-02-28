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