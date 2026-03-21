from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .telemetry import TelemetryTracker
from . import config
from . import watch
from . import anchors
from .cooldown import Cooldown
from .strategies.base import Strategy
from .strategies.types import StrategyMetadata


def compute_config_hash(profile) -> str:
    payload = {
        "market_type": config.MARKET_TYPE,
        "strategy_profile": config.STRATEGY_PROFILE,
        "alpaca_stock_feed": config.ALPACA_STOCK_FEED,
        "symbols": list(config.SYMBOLS),
        "profile": profile.__dict__,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@dataclass
class RuntimeContext:
    profile: object
    strategy: Strategy
    strategy_metadata: StrategyMetadata
    strategy_version: str
    config_hash: str
    min_needed: int
    max_keep: int
    data: dict[str, pd.DataFrame]
    anchor: anchors.AnchorTracker
    cooldown: Cooldown
    watch_mgr: watch.WatchManager
    telemetry: TelemetryTracker
    reconnect_attempt: int = 0
