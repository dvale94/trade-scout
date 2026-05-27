from __future__ import annotations

import unittest

import pandas as pd

from research.cohort_analysis import compare_two_cohorts, summarize_feature_distributions


class CohortAnalysisTest(unittest.TestCase):
    def test_summarize_feature_distributions_returns_expected_stats(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_quality": "clean", "score": 8.0},
                {"breakout_quality": "clean", "score": 10.0},
                {"breakout_quality": "tight", "score": 6.0},
                {"breakout_quality": "tight", "score": 8.0},
            ]
        )

        summary = summarize_feature_distributions(trades, "breakout_quality", ["score"])
        clean_row = summary[summary["breakout_quality"] == "clean"].iloc[0]

        self.assertEqual("score", clean_row["feature"])
        self.assertEqual(2, clean_row["count"])
        self.assertAlmostEqual(9.0, clean_row["mean"])
        self.assertAlmostEqual(9.0, clean_row["median"])
        self.assertAlmostEqual(1.41421356, clean_row["std"], places=6)
        self.assertAlmostEqual(8.5, clean_row["p25"])
        self.assertAlmostEqual(9.5, clean_row["p75"])

    def test_compare_two_cohorts_returns_deltas(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_quality": "clean", "score": 8.0},
                {"breakout_quality": "clean", "score": 10.0},
                {"breakout_quality": "tight", "score": 6.0},
                {"breakout_quality": "tight", "score": 8.0},
            ]
        )

        comparison = compare_two_cohorts(trades, "breakout_quality", "clean", "tight", ["score"])
        row = comparison.iloc[0]

        self.assertEqual("score", row["feature"])
        self.assertAlmostEqual(9.0, row["cohort_a_mean"])
        self.assertAlmostEqual(7.0, row["cohort_b_mean"])
        self.assertAlmostEqual(2.0, row["mean_delta"])
        self.assertAlmostEqual(1.41421356, row["standardized_delta"], places=6)
        self.assertAlmostEqual(0.0, row["overlap_proxy"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"alert_id": "a1"}])

        summary = summarize_feature_distributions(trades, "breakout_quality", ["score"])
        comparison = compare_two_cohorts(trades, "breakout_quality", "clean", "tight", ["score"])

        self.assertEqual(1, len(summary))
        self.assertEqual("unknown", summary.iloc[0]["breakout_quality"])
        self.assertIsNone(summary.iloc[0]["mean"])

        self.assertEqual(1, len(comparison))
        self.assertEqual("score", comparison.iloc[0]["feature"])
        self.assertTrue(pd.isna(comparison.iloc[0]["standardized_delta"]))


if __name__ == "__main__":
    unittest.main()
