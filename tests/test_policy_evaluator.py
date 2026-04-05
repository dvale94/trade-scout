from __future__ import annotations

import json
import unittest

from trade_scout.journal_rows import AlertContextRow, AlertIntentRow
from trade_scout.policy.evaluator import evaluate_policy
from trade_scout.policy.models import PolicyConfig


class PolicyEvaluatorTest(unittest.TestCase):
    def test_all_criteria_omitted_allows(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(),
            self._policy_config(filters={}),
            policy_decision_id="decision-1",
            decision_ts_utc="2026-04-05T12:00:00+00:00",
        )

        self.assertEqual("ALLOW", decision.decision)
        self.assertEqual(["all configured policy criteria passed"], decision.reasons)
        self.assertEqual([], decision.criteria_results)

    def test_allowed_setup_families_passes(self) -> None:
        decision = evaluate_policy(
            self._intent(setup_family="breakout_continuation"),
            self._context(),
            self._policy_config(filters={"allowed_setup_families": ["breakout_continuation"]}),
        )

        self.assertEqual("ALLOW", decision.decision)
        self.assertTrue(decision.criteria_results[0].passed)

    def test_allowed_setup_families_blocks(self) -> None:
        decision = evaluate_policy(
            self._intent(setup_family="retest"),
            self._context(),
            self._policy_config(filters={"allowed_setup_families": ["breakout_continuation"]}),
        )

        self.assertEqual("BLOCK", decision.decision)
        self.assertIn("setup_family 'retest' not allowed", decision.reasons)

    def test_allowed_anchor_regimes_passes(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(anchor_regime="bull"),
            self._policy_config(filters={"allowed_anchor_regimes": ["bull"]}),
        )

        self.assertEqual("ALLOW", decision.decision)
        self.assertTrue(decision.criteria_results[0].passed)

    def test_allowed_anchor_regimes_blocks(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(anchor_regime="bear"),
            self._policy_config(filters={"allowed_anchor_regimes": ["bull"]}),
        )

        self.assertEqual("BLOCK", decision.decision)
        self.assertIn("anchor_regime 'bear' not allowed", decision.reasons)

    def test_min_score_bucket_passes(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(score_bucket="8_PLUS"),
            self._policy_config(filters={"min_score_bucket": "7_TO_7_9"}),
        )

        self.assertEqual("ALLOW", decision.decision)
        self.assertTrue(decision.criteria_results[0].passed)

    def test_min_score_bucket_blocks(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(score_bucket="6_TO_6_9"),
            self._policy_config(filters={"min_score_bucket": "7_TO_7_9"}),
        )

        self.assertEqual("BLOCK", decision.decision)
        self.assertIn(
            "score_bucket '6_TO_6_9' below required minimum '7_TO_7_9'",
            decision.reasons,
        )

    def test_allowed_symbols_passes(self) -> None:
        decision = evaluate_policy(
            self._intent(symbol="AAPL"),
            self._context(),
            self._policy_config(filters={"allowed_symbols": ["AAPL", "MSFT"]}),
        )

        self.assertEqual("ALLOW", decision.decision)
        self.assertTrue(decision.criteria_results[0].passed)

    def test_allowed_symbols_blocks(self) -> None:
        decision = evaluate_policy(
            self._intent(symbol="DOGE/USD"),
            self._context(),
            self._policy_config(filters={"allowed_symbols": ["AAPL", "MSFT"]}),
        )

        self.assertEqual("BLOCK", decision.decision)
        self.assertIn("symbol 'DOGE/USD' not allowed", decision.reasons)

    def test_multiple_criteria_with_one_failure_blocks(self) -> None:
        decision = evaluate_policy(
            self._intent(symbol="AAPL", setup_family="breakout_continuation"),
            self._context(anchor_regime="bear", score_bucket="8_PLUS"),
            self._policy_config(
                filters={
                    "allowed_symbols": ["AAPL"],
                    "allowed_setup_families": ["breakout_continuation"],
                    "allowed_anchor_regimes": ["bull"],
                    "min_score_bucket": "7_TO_7_9",
                }
            ),
        )

        self.assertEqual("BLOCK", decision.decision)
        self.assertEqual(4, len(decision.criteria_results))
        self.assertIn("anchor_regime 'bear' not allowed", decision.reasons)

    def test_multiple_criteria_all_passing_allow(self) -> None:
        decision = evaluate_policy(
            self._intent(symbol="AAPL", setup_family="breakout_continuation"),
            self._context(anchor_regime="bull", score_bucket="8_PLUS"),
            self._policy_config(
                filters={
                    "allowed_symbols": ["AAPL"],
                    "allowed_setup_families": ["breakout_continuation"],
                    "allowed_anchor_regimes": ["bull"],
                    "min_score_bucket": "7_TO_7_9",
                }
            ),
        )

        self.assertEqual("ALLOW", decision.decision)
        self.assertEqual(4, len(decision.criteria_results))
        self.assertEqual(["all configured policy criteria passed"], decision.reasons)

    def test_reasons_are_populated_meaningfully(self) -> None:
        decision = evaluate_policy(
            self._intent(setup_family="retest"),
            self._context(anchor_regime="bear"),
            self._policy_config(
                filters={
                    "allowed_setup_families": ["breakout_continuation"],
                    "allowed_anchor_regimes": ["bull"],
                }
            ),
        )

        self.assertIn("setup_family 'retest' not allowed", decision.reasons)
        self.assertIn("anchor_regime 'bear' not allowed", decision.reasons)

    def test_execution_candidate_matches_decision(self) -> None:
        allow_decision = evaluate_policy(
            self._intent(),
            self._context(),
            self._policy_config(filters={}),
        )
        block_decision = evaluate_policy(
            self._intent(symbol="DOGE/USD"),
            self._context(),
            self._policy_config(filters={"allowed_symbols": ["AAPL"]}),
        )

        self.assertTrue(allow_decision.execution_candidate)
        self.assertFalse(block_decision.execution_candidate)

    def test_policy_decision_to_row_serializes_evaluator_output(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(),
            self._policy_config(filters={"allowed_symbols": ["AAPL"]}),
            policy_decision_id="decision-row",
            decision_ts_utc="2026-04-05T12:00:00+00:00",
        )

        row = decision.to_row("policy-1")

        self.assertEqual("decision-row", row.policy_decision_id)
        self.assertEqual("policy-1", row.policy_id)
        self.assertEqual(1, row.execution_candidate)
        self.assertEqual(["all configured policy criteria passed"], json.loads(row.decision_reasons_json))
        self.assertEqual("AAPL", json.loads(row.feature_snapshot_json)["symbol"])
        self.assertEqual("allowed_symbols", json.loads(row.criteria_results_json)[0]["criterion_name"])

    def test_missing_required_feature_raises_clear_value_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "policy evaluation requires alert_intent.setup_family",
        ):
            evaluate_policy(
                self._intent(setup_family=None),
                self._context(),
                self._policy_config(filters={}),
            )

    def test_unknown_score_bucket_blocks_explicitly(self) -> None:
        decision = evaluate_policy(
            self._intent(),
            self._context(score_bucket="MEDIUM"),
            self._policy_config(filters={"min_score_bucket": "7_TO_7_9"}),
        )

        self.assertEqual("BLOCK", decision.decision)
        self.assertIn("score_bucket 'MEDIUM' is unknown", decision.reasons)

    def _policy_config(self, *, filters: dict[str, object]) -> PolicyConfig:
        return PolicyConfig(
            policy_name="policy-v1",
            policy_version="1.0.0",
            policy_hash="hash-1",
            description="shadow policy",
            filters=filters,
        )

    def _intent(
        self,
        *,
        alert_id: str = "alert-1",
        symbol: str = "AAPL",
        setup_family: str | None = "breakout_continuation",
    ) -> AlertIntentRow:
        return AlertIntentRow(
            alert_id=alert_id,
            symbol=symbol,
            side="LONG",
            market_type="stocks",
            strategy_profile="swing",
            strategy_version="1.0.0",
            config_hash="cfg-123",
            tier="A",
            signal_ts_utc="2026-04-05T12:00:00+00:00",
            timeframe_min=5,
            signal_px=100.0,
            signal_px_type="close",
            alert_status=None,
            skip_reason=None,
            error_message=None,
            entry_low=99.5,
            entry_high=100.5,
            stop_px=95.0,
            t1_px=110.0,
            t2_px=115.0,
            timeout_seconds=300,
            stop_distance=5.0,
            t1_R=2.0,
            t2_R=3.0,
            risk_model="fixed_risk",
            risk_usd=10.0,
            qty_model="notional",
            qty=1.0,
            reasons=None,
            strategy_id="long_breakout_continuation",
            setup_family=setup_family,
            strategy_params_json="{}",
            strategy_params_hash="params-123",
        )

    def _context(
        self,
        *,
        alert_id: str = "alert-1",
        anchor_regime: str | None = "bull",
        score: float | None = 8.2,
        score_bucket: str | None = "8_PLUS",
    ) -> AlertContextRow:
        return AlertContextRow(
            alert_id=alert_id,
            market_type="stocks",
            strategy_profile="swing",
            strategy_version="1.0.0",
            config_hash="cfg-123",
            signal_ts_utc="2026-04-05T12:00:00+00:00",
            signal_hour_utc=12,
            signal_day_of_week="SAT",
            bar_start_ts_utc="2026-04-05T11:55:00+00:00",
            bar_end_ts_utc="2026-04-05T12:00:00+00:00",
            score=score,
            atr=1.2,
            ma20=99.0,
            ma50=98.0,
            ma200=95.0,
            breakout_level=100.0,
            breakout_distance_atr=0.5,
            ma20_distance_atr=0.4,
            ma50_distance_atr=0.8,
            ma200_distance_atr=1.5,
            ma20_slope_pct=0.2,
            ma50_slope_pct=0.3,
            range_pct=1.0,
            close_location_in_bar=0.9,
            entry_zone_width_pct=0.3,
            anchor_symbol="SPY",
            anchor_regime=anchor_regime,
            anchor_close=500.0,
            anchor_ma50=495.0,
            anchor_ma200=480.0,
            vol_z=0.1,
            extended_atr=0.2,
            market_regime="trend",
            volatility_regime="normal",
            anchor_trend_1h="up",
            anchor_trend_4h="up",
            anchor_trend_1d="up",
            score_bucket=score_bucket,
            strategy_id="long_breakout_continuation",
            setup_family="breakout_continuation",
            strategy_params_json="{}",
            strategy_params_hash="params-123",
        )
