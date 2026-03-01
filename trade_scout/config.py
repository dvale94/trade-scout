import os
from dotenv import load_dotenv

load_dotenv()

# Which market are we running?
MARKET_TYPE = os.getenv("MARKET_TYPE", "crypto").strip().lower()  # crypto | stocks

# Which tuned strategy profile should we use?
STRATEGY_PROFILE = os.getenv("STRATEGY_PROFILE", f"{MARKET_TYPE}_v1").strip()

# Alpaca auth
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Stocks-only setting
ALPACA_STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip().lower()  # iex | sip

# Notifications
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Universe
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]

# Logging directory
LOG_DIR = os.getenv("LOG_DIR", "logs").strip()