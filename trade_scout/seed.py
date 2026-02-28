from datetime import datetime, timezone, timedelta
import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest

from . import config


def seed_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=config.SEED_MINUTES)

    if config.MARKET_TYPE == "crypto":
        client = CryptoHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(config.BAR_TIMEFRAME_MIN, TimeFrameUnit.Minute),
            start=start,
            end=end,
        )
        resp = client.get_crypto_bars(req).df  # MultiIndex: (symbol, timestamp)

    elif config.MARKET_TYPE == "stocks":
        client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(config.BAR_TIMEFRAME_MIN, TimeFrameUnit.Minute),
            start=start,
            end=end,
        )
        resp = client.get_stock_bars(req).df  # MultiIndex: (symbol, timestamp)

    else:
        raise ValueError(f"Unknown MARKET_TYPE: {config.MARKET_TYPE}")

    out: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        sym_u = sym.strip().upper()
        if resp is None or len(resp) == 0:
            out[sym_u] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            continue

        try:
            sdf = resp.loc[sym_u].reset_index()
        except Exception:
            out[sym_u] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            continue

        sdf = sdf.rename(columns={"timestamp": "ts"})
        sdf = sdf[["ts", "open", "high", "low", "close", "volume"]].copy()
        sdf["ts"] = pd.to_datetime(sdf["ts"], utc=True)

        out[sym_u] = sdf.tail(config.LOOKBACK_BARS).reset_index(drop=True)

    return out