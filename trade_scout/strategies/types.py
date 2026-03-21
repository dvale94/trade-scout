from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Union


JournalScalar = Optional[Union[str, int, float, bool]]
ComputedValues = Mapping[str, object]


def _require_nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} is required")


@dataclass(frozen=True)
class StrategyMetadata:
    strategy_id: str
    strategy_version: str
    setup_family: str
    strategy_params_json: str | None = None
    strategy_params_hash: str | None = None
    journal_metadata: Mapping[str, JournalScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.strategy_id, "strategy_id")
        _require_nonempty(self.strategy_version, "strategy_version")
        _require_nonempty(self.setup_family, "setup_family")


@dataclass(frozen=True)
class SetupCandidate:
    score: float
    reasons: tuple[str, ...]
    computed: ComputedValues
    tier: str | None = None


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    side: str
    timeframe_min: int
    entry_low: float
    entry_high: float
    stop_px: float
    t1_px: float
    t2_px: float
    stop_distance: float

    def __post_init__(self) -> None:
        _require_nonempty(self.symbol, "symbol")
        _require_nonempty(self.side, "side")
