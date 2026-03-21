from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from trade_scout.strategy_profiles import StrategyProfile

from .types import SetupCandidate, StrategyMetadata, TradePlan

if TYPE_CHECKING:
    import pandas as pd
    from trade_scout.strategy import Candidate


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

    # Temporary compatibility surface for the current runtime orchestration.
    def score_candidate(self, df: "pd.DataFrame", profile: StrategyProfile) -> tuple[float, list[str], dict]:
        ...

    def pick_tier(
        self,
        score: float,
        df: "pd.DataFrame",
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
    ) -> "Candidate | None":
        ...
