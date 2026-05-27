from __future__ import annotations

import unittest

import pandas as pd

from research.candidate_band_analysis import (
    OUTSIDE_CANDIDATE_BAND,
    assign_candidate_bands,
    summarize_candidate_bands,
)


class CandidateBandAnalysisTest(unittest.TestCase):
    def test_assign_candidate_bands_marks_expected_band(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_distance_atr": 0.4, "ma20_distance_atr": 1.7},
                {"breakout_distance_atr": 0.6, "ma20_distance_atr": 1.2},
                {"breakout_distance_atr": 0.2, "ma20_distance_atr": 2.4},
            ]
        )

        result = assign_candidate_bands(trades, None)

        self.assertEqual("clean_mid_pullback", result.iloc[0]["candidate_band"])
        self.assertEqual("clean_upper_shallow", result.iloc[1]["candidate_band"])
        self.assertEqual(OUTSIDE_CANDIDATE_BAND, result.iloc[2]["candidate_band"])

    def test_summarize_candidate_bands_returns_expected_metrics(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_distance_atr": 0.4, "ma20_distance_atr": 1.7, "pnl_R": 1.0, "mfe_R": 1.4, "mae_R": 0.5},
                {"breakout_distance_atr": 0.45, "ma20_distance_atr": 1.8, "pnl_R": -1.0, "mfe_R": 0.6, "mae_R": 1.1},
            ]
        )

        summary = summarize_candidate_bands(trades)
        row = summary[summary["candidate_band"] == "clean_mid_pullback"].iloc[0]

        self.assertEqual(2, row["n"])
        self.assertAlmostEqual(0.5, row["win_rate"])
        self.assertAlmostEqual(0.0, row["avg_realized_r"])
        self.assertAlmostEqual(0.0, row["median_realized_r"])
        self.assertAlmostEqual(1.0, row["avg_mfe"])
        self.assertAlmostEqual(0.8, row["avg_mae"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"alert_id": "a1"}])

        result = assign_candidate_bands(trades, None)
        summary = summarize_candidate_bands(result, ["symbol"])

        self.assertEqual(OUTSIDE_CANDIDATE_BAND, result.iloc[0]["candidate_band"])
        self.assertEqual(1, len(summary))
        self.assertEqual("unknown", summary.iloc[0]["symbol"])
        self.assertIsNone(summary.iloc[0]["avg_realized_r"])


if __name__ == "__main__":
    unittest.main()
