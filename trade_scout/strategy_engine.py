from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .strategy import Candidate
from .strategies.base import Strategy
from .strategies.long_breakout_continuation import LongBreakoutContinuationStrategy
from .strategy_profiles import StrategyProfile


@dataclass(frozen=True)
class CurrentStrategyAdapter:
    """Temporary runtime bridge from the legacy orchestration surface to the canonical strategy."""

    name: str = "current_strategy"
    strategy: Strategy = field(default_factory=LongBreakoutContinuationStrategy)

    def score_candidate(self, df: pd.DataFrame, profile: StrategyProfile) -> tuple[float, list[str], dict]:
        return self.strategy.score_candidate(df, profile)

    def pick_tier(
        self,
        score: float,
        df: pd.DataFrame,
        profile: StrategyProfile,
        computed: dict,
    ) -> str | None:
        return self.strategy.pick_tier(score, df, profile, computed)

    def build_trade_candidate(
        self,
        symbol: str,
        profile: StrategyProfile,
        computed: dict,
        reasons: list[str],
        score: float,
    ) -> Candidate | None:
        return self.strategy.build_trade_candidate(
            symbol=symbol,
            profile=profile,
            computed=computed,
            reasons=reasons,
            score=score,
        )
