from __future__ import annotations

import sys
import types
import unittest

sys.modules.setdefault("numpy", types.SimpleNamespace(isnan=lambda value: False))
sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=object, Series=object))

from trade_scout.strategies import (
    LongBreakoutContinuationStrategy,
    SetupCandidate,
    Strategy,
    load_enabled_strategies,
    StrategyMetadata,
    TradePlan,
)
from trade_scout.strategy_profiles import PROFILES


class _StubStrategy:
    @property
    def strategy_id(self) -> str:
        return "breakout"

    @property
    def strategy_version(self) -> str:
        return "v1"

    @property
    def setup_family(self) -> str:
        return "continuation"

    def evaluate_candidate(self, df, profile) -> SetupCandidate:
        return SetupCandidate(
            score=7.5,
            reasons=("Close > MA50",),
            computed={"close": 101.0, "atr": 2.0},
            tier="A_CONTINUATION",
        )

    def build_trade_plan(self, symbol, profile, candidate) -> TradePlan | None:
        return TradePlan(
            symbol=symbol,
            side="LONG",
            timeframe_min=profile.timeframe_min,
            entry_low=100.0,
            entry_high=101.0,
            stop_px=95.0,
            t1_px=110.0,
            t2_px=116.0,
            stop_distance=6.0,
        )

    def metadata_for_journaling(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            setup_family=self.setup_family,
        )


class StrategyContractTest(unittest.TestCase):
    def test_types_instantiate_with_current_runtime_shape(self):
        metadata = StrategyMetadata(
            strategy_id="breakout",
            strategy_version="v1",
            setup_family="continuation",
            journal_metadata={"entry_model": "zone_buffer"},
        )
        candidate = SetupCandidate(
            score=7.8,
            reasons=("Close > MA50", "Close > MA200"),
            computed={"close": 101.0, "atr": 2.4, "ma50": 98.5},
            tier="A_CONTINUATION",
        )
        plan = TradePlan(
            symbol="BTC/USD",
            side="LONG",
            timeframe_min=5,
            entry_low=100.0,
            entry_high=101.0,
            stop_px=95.0,
            t1_px=110.0,
            t2_px=116.0,
            stop_distance=6.0,
        )

        self.assertEqual(metadata.strategy_id, "breakout")
        self.assertEqual(candidate.computed["atr"], 2.4)
        self.assertEqual(plan.stop_distance, 6.0)

    def test_strategy_protocol_can_be_implemented_without_runtime_wiring(self):
        strategy = _StubStrategy()
        profile = PROFILES["crypto_v1"]

        self.assertIsInstance(strategy, Strategy)

        candidate = strategy.evaluate_candidate(None, profile)
        plan = strategy.build_trade_plan("BTC/USD", profile, candidate)
        metadata = strategy.metadata_for_journaling()

        self.assertEqual(candidate.tier, "A_CONTINUATION")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.timeframe_min, profile.timeframe_min)
        self.assertEqual(metadata.setup_family, "continuation")

    def test_current_strategy_instantiates_and_preserves_current_shape(self):
        strategy = LongBreakoutContinuationStrategy()
        metadata = strategy.metadata_for_journaling()

        self.assertEqual(strategy.strategy_id, "long_breakout_continuation")
        self.assertEqual(metadata.strategy_version, "v1")
        self.assertEqual(metadata.setup_family, "breakout_continuation")

    def test_registry_loads_default_strategy_shape(self):
        strategies = load_enabled_strategies(
            ["long_breakout_continuation"],
            {"long_breakout_continuation": {"strategy_version": "v3_research_step2"}},
        )

        self.assertEqual(len(strategies), 1)
        self.assertIsInstance(strategies[0], LongBreakoutContinuationStrategy)
        self.assertEqual(strategies[0].metadata_for_journaling().strategy_version, "v3_research_step2")


if __name__ == "__main__":
    unittest.main()
