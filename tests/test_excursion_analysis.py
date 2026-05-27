from __future__ import annotations

import unittest

import pandas as pd

from research.excursion_analysis import summarize_capture_efficiency, summarize_excursion_thresholds


class ExcursionAnalysisTest(unittest.TestCase):
    def test_threshold_percentages_compute_correctly(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_quality": "tight", "mfe_R": 0.2},
                {"breakout_quality": "tight", "mfe_R": 0.5},
                {"breakout_quality": "tight", "mfe_R": 1.2},
                {"breakout_quality": "tight", "mfe_R": 2.1},
            ]
        )

        summary = summarize_excursion_thresholds(trades, ["breakout_quality"])
        row = summary.iloc[0]

        self.assertEqual(4, row["n"])
        self.assertAlmostEqual(0.75, row["pct_reached_0_25R"])
        self.assertAlmostEqual(0.75, row["pct_reached_0_5R"])
        self.assertAlmostEqual(0.5, row["pct_reached_1R"])
        self.assertAlmostEqual(0.25, row["pct_reached_1_5R"])
        self.assertAlmostEqual(0.25, row["pct_reached_2R"])

    def test_capture_efficiency_is_divide_by_zero_safe(self) -> None:
        trades = pd.DataFrame(
            [
                {"setup_family": "a", "pnl_R": 1.0, "mfe_R": 2.0},
                {"setup_family": "a", "pnl_R": -1.0, "mfe_R": 0.0},
                {"setup_family": "a", "pnl_R": 0.5, "mfe_R": 1.0},
            ]
        )

        summary = summarize_capture_efficiency(trades, ["setup_family"])
        row = summary.iloc[0]

        self.assertEqual(3, row["n"])
        self.assertAlmostEqual(0.5, row["median_capture_ratio"])
        self.assertAlmostEqual(0.5, row["mean_capture_ratio"])
        self.assertAlmostEqual(0.5, row["p25_capture_ratio"])
        self.assertAlmostEqual(0.5, row["p75_capture_ratio"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"alert_id": "a1"}])

        thresholds = summarize_excursion_thresholds(trades, ["breakout_quality"])
        capture = summarize_capture_efficiency(trades, ["setup_family"])

        self.assertEqual(1, len(thresholds))
        self.assertEqual("unknown", thresholds.iloc[0]["breakout_quality"])
        self.assertIsNone(thresholds.iloc[0]["pct_reached_1R"])

        self.assertEqual(1, len(capture))
        self.assertEqual("unknown", capture.iloc[0]["setup_family"])
        self.assertTrue(pd.isna(capture.iloc[0]["mean_capture_ratio"]))


if __name__ == "__main__":
    unittest.main()
