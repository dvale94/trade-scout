from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from trade_scout.strategy_profiles import StrategyProfile

from .types import SetupCandidate, StrategyMetadata, TradePlan

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    @property
    def strategy_id(self) -> str:
        ...

    @property
    def strategy_version(self) -> str:
        ...

    @property
    def setup_family(self) -> str:
        ...

    def evaluate_candidate(
        self,
        symbol: str,
        df: "pd.DataFrame",
        profile: StrategyProfile,
    ) -> SetupCandidate:
        ...

    def build_trade_plan(
        self,
        symbol: str,
        profile: StrategyProfile,
        candidate: SetupCandidate,
    ) -> TradePlan | None:
        ...

    def metadata_for_journaling(self) -> StrategyMetadata:
        ...
