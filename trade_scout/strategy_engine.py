from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .strategy import Candidate
from .strategies.base import Strategy
from .strategies.types import SetupCandidate
from .strategies.long_breakout_continuation import LongBreakoutContinuationStrategy
from .strategy_profiles import StrategyProfile


@dataclass(frozen=True)
class CurrentStrategyAdapter:
    """Temporary runtime bridge from the legacy orchestration surface to the canonical strategy."""

    name: str = "current_strategy"
    strategy: Strategy = field(default_factory=LongBreakoutContinuationStrategy)

    def score_candidate(self, df: pd.DataFrame, profile: StrategyProfile) -> tuple[float, list[str], dict]:
        candidate = self.strategy.evaluate_candidate(df, profile)
        return candidate.score, list(candidate.reasons), dict(candidate.computed)

    def pick_tier(
        self,
        score: float,
        df: pd.DataFrame,
        profile: StrategyProfile,
        computed: dict,
    ) -> str | None:
        return self.strategy.evaluate_candidate(df, profile).tier

    def build_trade_candidate(
        self,
        symbol: str,
        profile: StrategyProfile,
        computed: dict,
        reasons: list[str],
        score: float,
    ) -> Candidate | None:
        setup_candidate = SetupCandidate(
            score=score,
            reasons=tuple(reasons),
            computed=dict(computed),
            tier=None,
        )
        trade_plan = self.strategy.build_trade_plan(symbol, profile, setup_candidate)
        if trade_plan is None:
            return None
        from datetime import datetime, timezone
        from uuid import uuid4

        return Candidate(
            alert_id=str(uuid4()),
            symbol=symbol,
            timeframe_min=trade_plan.timeframe_min,
            score=float(setup_candidate.score),
            ts=datetime.now(timezone.utc),
            entry_low=float(trade_plan.entry_low),
            entry_high=float(trade_plan.entry_high),
            stop=float(trade_plan.stop_px),
            t1=float(trade_plan.t1_px),
            t2=float(trade_plan.t2_px),
            r=float(trade_plan.stop_distance),
            reasons=list(setup_candidate.reasons),
            news_sentiment=None,
            news_summary=None,
        )
