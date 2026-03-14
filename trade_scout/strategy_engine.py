from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from . import strategy
from .strategy_profiles import StrategyProfile


class StrategyEngine(Protocol):
    """Small runtime-facing strategy contract for future pluggable implementations."""

    name: str

    def score_candidate(self, df: pd.DataFrame, profile: StrategyProfile) -> tuple[float, list[str], dict]:
        ...

    def pick_tier(
        self,
        score: float,
        df: pd.DataFrame,
        profile: StrategyProfile,
        computed: dict,
    ) -> str | None:
        ...

    def build_trade_candidate(
        self,
        symbol: str,
        profile: StrategyProfile,
        computed: dict,
        reasons: list[str],
        score: float,
    ) -> strategy.Candidate | None:
        ...


@dataclass(frozen=True)
class CurrentStrategyAdapter:
    """Adapter over the current concrete strategy module used by the live runtime."""

    name: str = "current_strategy"

    def score_candidate(self, df: pd.DataFrame, profile: StrategyProfile) -> tuple[float, list[str], dict]:
        return strategy.score_candidate(df, profile)

    def pick_tier(
        self,
        score: float,
        df: pd.DataFrame,
        profile: StrategyProfile,
        computed: dict,
    ) -> str | None:
        return strategy.pick_tier(score, df, profile, computed)

    def build_trade_candidate(
        self,
        symbol: str,
        profile: StrategyProfile,
        computed: dict,
        reasons: list[str],
        score: float,
    ) -> strategy.Candidate | None:
        return strategy.compute_entry_stop_targets(symbol, profile, computed, reasons, score)
