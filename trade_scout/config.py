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


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid float") from exc

# Which market are we running?
MARKET_TYPE = os.getenv("MARKET_TYPE", "crypto").strip().lower()  # crypto | stocks

# Which tuned strategy profile should we use?
STRATEGY_PROFILE = os.getenv("STRATEGY_PROFILE", f"{MARKET_TYPE}_v1").strip()

# Alpaca auth
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()

# Stocks-only setting
ALPACA_STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip().lower()  # iex | sip

# Notifications
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Universe
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]

# Logging directory
LOG_DIR = os.getenv("LOG_DIR", "logs").strip()

# Execution subsystem
EXECUTION_ENABLED = os.getenv("EXECUTION_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
EXECUTION_DB_PATH = os.getenv("EXECUTION_DB_PATH", "./execution.db").strip()
EXECUTION_POLL_INTERVAL_SECONDS = float(os.getenv("EXECUTION_POLL_INTERVAL_SECONDS", "5").strip())
EXECUTION_MAX_ALERT_AGE_SECONDS = int(os.getenv("EXECUTION_MAX_ALERT_AGE_SECONDS", "300").strip())
EXECUTION_BATCH_SIZE = int(os.getenv("EXECUTION_BATCH_SIZE", "10").strip())
EXECUTION_WORKER_ID = os.getenv("EXECUTION_WORKER_ID", "execution-worker-1").strip()
EXECUTION_ORDER_QTY = _float_env("EXECUTION_ORDER_QTY", "1")

# Strategy loading
ENABLED_STRATEGY_IDS = _csv_env("ENABLED_STRATEGY_IDS") or ["long_breakout_continuation"]
STRATEGY_PARAMS = {
    "long_breakout_continuation": {
        "strategy_version": "v3_research_step2",
    }
}
STRATEGY_PARAMS.update(_json_object_env("STRATEGY_PARAMS_JSON"))
