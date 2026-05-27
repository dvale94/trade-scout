from __future__ import annotations

import unittest

import pandas as pd

from research.band_grid_analysis import summarize_band_grid


class BandGridAnalysisTest(unittest.TestCase):
    def test_band_grid_computes_expected_metrics(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_distance_atr": 0.2, "ma20_distance_atr": 1.0, "pnl_R": -1.0, "mfe_R": 0.2, "mae_R": 1.0},
                {"breakout_distance_atr": 0.5, "ma20_distance_atr": 1.5, "pnl_R": 0.5, "mfe_R": 1.0, "mae_R": 0.5},
                {"breakout_distance_atr": 0.7, "ma20_distance_atr": 1.7, "pnl_R": 1.0, "mfe_R": 1.5, "mae_R": 0.2},
            ]
        )

        summary = summarize_band_grid(
            trades,
            "breakout_distance_atr",
            [0.0, 0.35, 0.8],
            "ma20_distance_atr",
            [0.0, 1.25, 2.0],
        )

        row = summary[
            summary["feature_a_low"].eq(0.35)
            & summary["feature_a_high"].eq(0.8)
            & summary["feature_b_low"].eq(1.25)
            & summary["feature_b_high"].eq(2.0)
        ].iloc[0]

        self.assertEqual(2, row["n"])
        self.assertAlmostEqual(2.0 / 3.0, row["coverage_pct"])
        self.assertAlmostEqual(1.0, row["win_rate"])
        self.assertAlmostEqual(0.75, row["avg_realized_r"])
        self.assertAlmostEqual(1.25, row["avg_mfe"])
        self.assertAlmostEqual(0.35, row["avg_mae"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"alert_id": "a1"}])

        summary = summarize_band_grid(
            trades,
            "breakout_distance_atr",
            [0.0, 0.5, 1.0],
            "ma20_distance_atr",
            [0.0, 1.0, 2.0],
        )
        row = summary.iloc[0]

        self.assertEqual(0, row["n"])
        self.assertAlmostEqual(0.0, row["coverage_pct"])
        self.assertIsNone(row["avg_realized_r"])
        self.assertIsNone(row["avg_mfe"])


if __name__ == "__main__":
    unittest.main()
