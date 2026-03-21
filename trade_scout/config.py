import json
import os
from dotenv import load_dotenv

load_dotenv()


def _csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _json_object_env(name: str) -> dict[str, dict[str, object]]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError(f"{name} must map strategy IDs to JSON objects")
    return parsed

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

# Strategy loading
ENABLED_STRATEGY_IDS = _csv_env("ENABLED_STRATEGY_IDS") or ["long_breakout_continuation"]
STRATEGY_PARAMS = {
    "long_breakout_continuation": {
        "strategy_version": "v3_research_step2",
    }
}
STRATEGY_PARAMS.update(_json_object_env("STRATEGY_PARAMS_JSON"))
