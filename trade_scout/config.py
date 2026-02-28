import os
from dotenv import load_dotenv

load_dotenv()

# Core mode switch
MARKET_TYPE = os.getenv("MARKET_TYPE", "crypto").strip().lower()  # "crypto" or "stocks"

# Alpaca auth
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Stocks feed (only relevant for MARKET_TYPE=stocks)
ALPACA_STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip().lower()  # "iex" or "sip"

# Notifications
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Universe
SYMBOLS = [s.strip().upper() for s in os.getenv(
    "SYMBOLS",
    "BTC/USD,ETH/USD,SOL/USD" if MARKET_TYPE == "crypto" else "SPY,QQQ,AAPL,MSFT,NVDA"
).split(",") if s.strip()]

# Bars / history
BAR_TIMEFRAME_MIN = int(os.getenv("BAR_TIMEFRAME_MIN", "5"))
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "500"))
SEED_MINUTES = int(os.getenv("SEED_MINUTES", "1440"))  # default 24h

# Stops & entry zone
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_K = float(os.getenv("ATR_K", "2.0"))
ENTRY_BUFFER_PCT = float(os.getenv("ENTRY_BUFFER_PCT", "0.0015"))
SWING_LOW_LOOKBACK = int(os.getenv("SWING_LOW_LOOKBACK", "20"))

# Trend & breakout scoring
PERSIST_BARS = int(os.getenv("PERSIST_BARS", "2"))
MA_FAST = int(os.getenv("MA_FAST", "20"))
MA_MID = int(os.getenv("MA_MID", "50"))
MA_SLOW = int(os.getenv("MA_SLOW", "200"))

BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "30"))
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "30"))
VOLUME_Z = float(os.getenv("VOLUME_Z", "1.0"))
EXTENDED_ATR_MAX = float(os.getenv("EXTENDED_ATR_MAX", "1.5"))

# Tier thresholds
A_PLUS_THRESHOLD = float(os.getenv("A_PLUS_THRESHOLD", "8.5"))
A_THRESHOLD = float(os.getenv("A_THRESHOLD", "7.5"))

# Retest definition
RETEST_ATR_BAND = float(os.getenv("RETEST_ATR_BAND", "0.8"))
RETEST_HOLD_BARS = int(os.getenv("RETEST_HOLD_BARS", "2"))

# Alerting
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))

# Outcome tracking
WATCH_TIMEOUT_MIN = int(os.getenv("WATCH_TIMEOUT_MIN", "240"))