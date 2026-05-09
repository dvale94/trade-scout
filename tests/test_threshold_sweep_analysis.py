from __future__ import annotations

import unittest

import pandas as pd

from research.threshold_sweep_analysis import summarize_threshold_sweep


class ThresholdSweepAnalysisTest(unittest.TestCase):
    def test_gte_threshold_sweep_computes_expected_metrics(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_distance_atr": 0.1, "pnl_R": -1.0, "mfe_R": 0.2, "mae_R": 1.0},
                {"breakout_distance_atr": 0.4, "pnl_R": 0.5, "mfe_R": 1.0, "mae_R": 0.5},
                {"breakout_distance_atr": 0.8, "pnl_R": 1.0, "mfe_R": 1.5, "mae_R": 0.2},
            ]
        )

        summary = summarize_threshold_sweep(trades, "breakout_distance_atr", [0.4], direction="gte")
        row = summary.iloc[0]

        self.assertEqual(2, row["n"])
        self.assertAlmostEqual(2.0 / 3.0, row["coverage_pct"])
        self.assertAlmostEqual(1.0, row["win_rate"])
        self.assertAlmostEqual(0.75, row["avg_realized_r"])
        self.assertAlmostEqual(0.75, row["median_realized_r"])
        self.assertAlmostEqual(1.25, row["avg_mfe"])
        self.assertAlmostEqual(0.35, row["avg_mae"])

    def test_lte_threshold_sweep_computes_expected_metrics(self) -> None:
        trades = pd.DataFrame(
            [
                {"extended_atr": 1.0, "pnl_R": -1.0, "mfe_R": 0.2, "mae_R": 1.0},
                {"extended_atr": 1.5, "pnl_R": 0.5, "mfe_R": 1.0, "mae_R": 0.5},
                {"extended_atr": 2.0, "pnl_R": 1.0, "mfe_R": 1.5, "mae_R": 0.2},
            ]
        )

        summary = summarize_threshold_sweep(trades, "extended_atr", [1.5], direction="lte")
        row = summary.iloc[0]

        self.assertEqual(2, row["n"])
        self.assertAlmostEqual(2.0 / 3.0, row["coverage_pct"])
        self.assertAlmostEqual(0.5, row["win_rate"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"alert_id": "a1"}])

        summary = summarize_threshold_sweep(trades, "breakout_distance_atr", [0.5], direction="gte")
        row = summary.iloc[0]

        self.assertEqual(0, row["n"])
        self.assertAlmostEqual(0.0, row["coverage_pct"])
        self.assertIsNone(row["avg_realized_r"])
        self.assertIsNone(row["avg_mfe"])


if __name__ == "__main__":
    unittest.main()
