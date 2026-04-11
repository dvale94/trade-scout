from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trade_scout.scoring import RuntimeShadowScorer
from trade_scout.scoring.buckets import shadow_score_bucket
from trade_scout.scoring.features import (
    RUNTIME_SHADOW_SCORE_FEATURE_COLUMNS,
    build_shadow_score_features,
    validate_runtime_feature_keys,
)
from trade_scout.scoring.models import RuntimeShadowScoreConfig
from trade_scout.scoring.registry import active_shadow_score_config


class RuntimeShadowScoringTest(unittest.TestCase):
    def test_active_runtime_candidate_is_expected_r_only(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SHADOW_SCORE_RECIPE_MODE": "expected_r_only",
                "SHADOW_SCORE_RECIPE_VERSION": "score_recipe_expected_r_v1",
            },
        ):
            config = active_shadow_score_config()

        self.assertEqual("expected_r_only", config.score_recipe_mode)
        self.assertEqual("score_recipe_expected_r_v1", config.score_recipe_version)

    def test_active_config_parses_expected_r_coefficients_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SHADOW_SCORE_EXPECTED_R_INTERCEPT": "-0.5",
                "SHADOW_SCORE_EXPECTED_R_WEIGHTS_JSON": '{"score": 0.1, "risk_pct": 2.0}',
            },
        ):
            config = active_shadow_score_config()

        self.assertEqual(-0.5, config.expected_r_intercept)
        self.assertEqual({"score": 0.1, "risk_pct": 2.0}, config.expected_r_weights)

    def test_unsupported_runtime_recipe_mode_is_rejected_by_config(self) -> None:
        with patch.dict("os.environ", {"SHADOW_SCORE_RECIPE_MODE": "blended"}):
            with self.assertRaisesRegex(ValueError, "unsupported SHADOW_SCORE_RECIPE_MODE"):
                active_shadow_score_config()

    def test_unsupported_runtime_recipe_mode_is_rejected_by_scorer(self) -> None:
        config = RuntimeShadowScoreConfig(
            score_model_version="shadow_score_v1_blended",
            feature_set_version="score_features_v1",
            score_recipe_version="score_recipe_blended_v1",
            score_recipe_mode="blended",
        )

        with self.assertRaisesRegex(ValueError, "unsupported runtime shadow score recipe mode"):
            RuntimeShadowScorer(config)

    def test_expected_r_only_output_shape_with_missing_artifact_is_null_safe(self) -> None:
        scorer = RuntimeShadowScorer(
            RuntimeShadowScoreConfig(
                score_model_version="shadow_score_v1_expected_r_only",
                feature_set_version="score_features_v1",
                score_recipe_version="score_recipe_expected_r_v1",
                score_recipe_mode="expected_r_only",
            )
        )

        output = scorer.score(self._intent(), self._context())

        self.assertEqual("shadow_score_v1_expected_r_only", output.score_model_version)
        self.assertEqual("UNCONFIGURED", output.shadow_score_status)
        self.assertIsNone(output.pred_expected_r)
        self.assertIsNone(output.score_shadow)
        self.assertIsNone(output.score_shadow_bucket)

    def test_disabled_output_is_distinct_from_unconfigured(self) -> None:
        scorer = RuntimeShadowScorer(
            RuntimeShadowScoreConfig(
                score_model_version="shadow_score_disabled",
                feature_set_version="score_features_v1",
                score_recipe_version="score_recipe_disabled",
                score_recipe_mode="disabled",
                enabled=False,
            )
        )

        output = scorer.score(self._intent(), self._context())

        self.assertEqual("DISABLED", output.shadow_score_status)
        self.assertIsNone(output.score_shadow)

    def test_expected_r_only_computes_from_configured_weights(self) -> None:
        scorer = RuntimeShadowScorer(
            RuntimeShadowScoreConfig(
                score_model_version="shadow_score_v1_expected_r_only",
                feature_set_version="score_features_v1",
                score_recipe_version="score_recipe_expected_r_v1",
                score_recipe_mode="expected_r_only",
                expected_r_intercept=-1.0,
                expected_r_weights={"score": 0.2, "risk_pct": 10.0},
            )
        )

        output = scorer.score(self._intent(), self._context())

        self.assertAlmostEqual(1.234059, output.pred_expected_r)
        self.assertEqual(output.pred_expected_r, output.score_shadow)
        self.assertEqual("SCORED", output.shadow_score_status)
        self.assertEqual("HIGH_POSITIVE", output.score_shadow_bucket)

    def test_shadow_score_buckets_are_explicit_and_stable(self) -> None:
        cases = [
            (None, None),
            (-0.01, "NEGATIVE"),
            (0.0, "LOW_POSITIVE"),
            (0.2499, "LOW_POSITIVE"),
            (0.25, "MEDIUM_POSITIVE"),
            (0.7499, "MEDIUM_POSITIVE"),
            (0.75, "HIGH_POSITIVE"),
        ]

        for score, expected_bucket in cases:
            with self.subTest(score=score):
                self.assertEqual(expected_bucket, shadow_score_bucket(score))

    def test_feature_builder_uses_alert_time_fields_only_and_handles_missing(self) -> None:
        features = build_shadow_score_features(self._intent(risk_usd=None), self._context(anchor_regime=None))

        self.assertEqual(8.2, features["score"])
        self.assertEqual("BTC/USD", features["symbol"])
        self.assertIsNone(features["risk_usd"])
        self.assertIsNone(features["anchor_regime"])
        self.assertAlmostEqual((101.0 - 95.0) / 101.0, features["risk_pct"])
        self.assertEqual(set(RUNTIME_SHADOW_SCORE_FEATURE_COLUMNS), set(features))

    def test_runtime_feature_parity_guard_detects_drift(self) -> None:
        features = build_shadow_score_features(self._intent(), self._context())
        features.pop("score")

        with self.assertRaisesRegex(ValueError, "runtime shadow score feature drift"):
            validate_runtime_feature_keys(features)

    def _intent(self, **overrides):
        values = {
            "symbol": "BTC/USD",
            "side": "LONG",
            "market_type": "crypto",
            "strategy_profile": "crypto_v1",
            "strategy_id": "long_breakout_continuation",
            "strategy_version": "v1",
            "setup_family": "breakout_continuation",
            "tier": "A",
            "signal_px": 100.5,
            "signal_px_type": "bar_close",
            "entry_low": 100.0,
            "entry_high": 101.0,
            "stop_px": 95.0,
            "t1_px": 110.0,
            "t2_px": 115.0,
            "timeframe_min": 5,
            "timeout_seconds": 14400,
            "stop_distance": 6.0,
            "t1_R": 1.5,
            "t2_R": 2.5,
            "risk_model": "fixed_usd",
            "risk_usd": None,
            "qty_model": None,
            "qty": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _context(self, **overrides):
        values = {
            "score": 8.2,
            "signal_hour_utc": 12,
            "atr": 1.2,
            "ma20": 99.0,
            "ma50": 98.0,
            "ma200": 95.0,
            "breakout_level": 100.0,
            "breakout_distance_atr": 0.5,
            "ma20_distance_atr": 0.4,
            "ma50_distance_atr": 0.8,
            "ma200_distance_atr": 1.5,
            "ma20_slope_pct": 0.2,
            "ma50_slope_pct": 0.3,
            "range_pct": 1.0,
            "close_location_in_bar": 0.9,
            "entry_zone_width_pct": (101.0 - 100.0) / 101.0,
            "anchor_symbol": "SPY",
            "anchor_regime": "bull",
            "anchor_close": 500.0,
            "anchor_ma50": 495.0,
            "anchor_ma200": 480.0,
            "vol_z": 0.1,
            "extended_atr": 0.2,
            "market_regime": "BULL",
            "volatility_regime": "NORMAL",
            "anchor_trend_1h": "UP",
            "anchor_trend_4h": "UP",
            "anchor_trend_1d": "UP",
        }
        values.update(overrides)
        return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
