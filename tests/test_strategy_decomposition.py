from __future__ import annotations

import unittest

import pandas as pd

from research.strategy_decomposition import (
    assign_setup_subtypes,
    derive_quantile_buckets,
    safe_col,
    summarize_exit_efficiency,
    summarize_subtype_expectancy,
)


class StrategyDecompositionTest(unittest.TestCase):
    def test_safe_col_returns_default_series_when_column_missing(self) -> None:
        df = pd.DataFrame(index=["a", "b"])

        result = safe_col(df, "missing", default="unknown")

        self.assertEqual(["a", "b"], list(result.index))
        self.assertEqual(["unknown", "unknown"], list(result))

    def test_assign_setup_subtypes_adds_columns_without_outcomes(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "alert_id": "alert-1",
                    "breakout_distance_atr": 0.2,
                    "ma20_distance_atr": 0.2,
                    "ma50_distance_atr": 0.8,
                    "ma20": 101,
                    "ma50": 100,
                    "ma200": 98,
                    "ma20_slope_pct": 0.2,
                    "ma50_slope_pct": 0.1,
                    "vol_z": 1.2,
                    "extended_atr": 1.1,
                    "anchor_regime": "bull",
                }
            ]
        )

        result = assign_setup_subtypes(trades)

        for column in (
            "breakout_quality",
            "pullback_structure",
            "volatility_context",
            "trend_context",
            "liquidity_volume_context",
            "setup_family",
        ):
            self.assertIn(column, result.columns)
        self.assertEqual("tight", result.iloc[0]["breakout_quality"])
        self.assertEqual("trend_continuation_shallow_pullback", result.iloc[0]["setup_family"])

    def test_missing_columns_do_not_crash_and_default_to_unknown(self) -> None:
        trades = pd.DataFrame([{"alert_id": "alert-1"}])

        result = assign_setup_subtypes(trades)

        self.assertEqual("unknown", result.iloc[0]["breakout_quality"])
        self.assertEqual("unknown", result.iloc[0]["trend_context"])
        self.assertEqual("unclassified", result.iloc[0]["setup_family"])

    def test_summaries_return_grouped_rows(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "alert_id": "alert-1",
                    "anchor_regime": "bull",
                    "breakout_distance_atr": 0.2,
                    "ma20_distance_atr": 0.2,
                    "ma50_distance_atr": 0.7,
                    "pnl_R": 1.5,
                    "mfe_R": 2.0,
                    "mae_R": -0.5,
                },
                {
                    "alert_id": "alert-2",
                    "anchor_regime": "bull",
                    "breakout_distance_atr": 1.1,
                    "ma20_distance_atr": 0.7,
                    "ma50_distance_atr": 1.5,
                    "pnl_R": -1.0,
                    "mfe_R": 1.0,
                    "mae_R": -1.2,
                },
            ]
        )

        expectancy = summarize_subtype_expectancy(trades, ["anchor_regime"])
        efficiency = summarize_exit_efficiency(trades, ["anchor_regime"])

        self.assertEqual(1, len(expectancy))
        self.assertEqual("bull", expectancy.iloc[0]["anchor_regime"])
        self.assertEqual(2, expectancy.iloc[0]["n"])
        self.assertAlmostEqual(0.5, expectancy.iloc[0]["win_rate"])
        self.assertAlmostEqual(0.25, expectancy.iloc[0]["avg_realized_r"])
        self.assertAlmostEqual(1.5, expectancy.iloc[0]["avg_mfe"])
        self.assertAlmostEqual(-0.85, expectancy.iloc[0]["avg_mae"])

        self.assertEqual(1, len(efficiency))
        self.assertEqual(2, efficiency.iloc[0]["n"])
        self.assertAlmostEqual(-0.125, efficiency.iloc[0]["mean_exit_efficiency"])
        self.assertAlmostEqual(-0.125, efficiency.iloc[0]["median_exit_efficiency"])

    def test_quantile_bucket_helper_produces_balanced_terciles(self) -> None:
        trades = pd.DataFrame({"breakout_distance_atr_num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})

        buckets = derive_quantile_buckets(
            trades,
            "breakout_distance_atr_num",
            labels=["tight", "clean", "extended"],
        )

        counts = buckets.value_counts().to_dict()
        self.assertEqual(2, counts["tight"])
        self.assertEqual(2, counts["clean"])
        self.assertEqual(2, counts["extended"])

    def test_assign_setup_subtypes_supports_quantile_mode(self) -> None:
        trades = pd.DataFrame(
            [
                {"alert_id": "a1", "breakout_distance_atr": 0.1, "ma20_distance_atr": 0.1, "vol_z": -1.0},
                {"alert_id": "a2", "breakout_distance_atr": 0.2, "ma20_distance_atr": 0.2, "vol_z": -0.4},
                {"alert_id": "a3", "breakout_distance_atr": 0.3, "ma20_distance_atr": 0.3, "vol_z": 0.0},
                {"alert_id": "a4", "breakout_distance_atr": 0.4, "ma20_distance_atr": 0.4, "vol_z": 0.4},
                {"alert_id": "a5", "breakout_distance_atr": 0.5, "ma20_distance_atr": 0.5, "vol_z": 1.0},
                {"alert_id": "a6", "breakout_distance_atr": 0.6, "ma20_distance_atr": 0.6, "vol_z": 1.2},
            ]
        )

        result = assign_setup_subtypes(trades, decomposition_mode="quantile")

        self.assertEqual(["tight", "tight", "clean", "clean", "extended", "extended"], list(result["breakout_quality"]))
        self.assertEqual(["shallow", "shallow", "moderate", "moderate", "deep", "deep"], list(result["pullback_structure"]))
        self.assertEqual(
            [
                "thin_relative_volume",
                "thin_relative_volume",
                "normal_relative_volume",
                "normal_relative_volume",
                "high_relative_volume",
                "high_relative_volume",
            ],
            list(result["liquidity_volume_context"]),
        )


if __name__ == "__main__":
    unittest.main()
