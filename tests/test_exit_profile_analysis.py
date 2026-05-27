from __future__ import annotations

import unittest

import pandas as pd

from research.exit_profile_analysis import simulate_exit_profiles


class ExitProfileAnalysisTest(unittest.TestCase):
    def test_tp_full_profile_uses_mfe_threshold(self) -> None:
        trades = pd.DataFrame(
            [
                {"breakout_quality": "clean", "pnl_R": -1.0, "mfe_R": 1.2},
                {"breakout_quality": "clean", "pnl_R": 0.2, "mfe_R": 0.4},
            ]
        )

        summary = simulate_exit_profiles(trades, ["breakout_quality"], profiles=["tp_1R_full"])
        row = summary.iloc[0]

        self.assertEqual("tp_1R_full", row["profile"])
        self.assertEqual(2, row["n"])
        self.assertAlmostEqual(0.6, row["avg_simulated_r"])
        self.assertAlmostEqual(0.6, row["median_simulated_r"])
        self.assertAlmostEqual(1.0, row["win_rate"])
        self.assertAlmostEqual(1.0, row["avg_delta_vs_baseline"])

    def test_half_at_1r_rest_breakeven_profile_works(self) -> None:
        trades = pd.DataFrame(
            [
                {"setup_family": "family_a", "pnl_R": -1.0, "mfe_R": 1.1},
                {"setup_family": "family_a", "pnl_R": 0.2, "mfe_R": 0.3},
            ]
        )

        summary = simulate_exit_profiles(
            trades,
            ["setup_family"],
            profiles=["half_at_1R_rest_breakeven"],
        )
        row = summary.iloc[0]

        self.assertEqual("half_at_1R_rest_breakeven", row["profile"])
        self.assertAlmostEqual(0.35, row["avg_simulated_r"])
        self.assertAlmostEqual(0.35, row["median_simulated_r"])
        self.assertAlmostEqual(1.0, row["win_rate"])
        self.assertAlmostEqual(0.75, row["avg_delta_vs_baseline"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"alert_id": "a1"}])

        summary = simulate_exit_profiles(trades, ["breakout_quality"], profiles=["baseline_realized", "tp_1R_full"])

        self.assertEqual(2, len(summary))
        self.assertEqual("unknown", summary.iloc[0]["breakout_quality"])
        self.assertTrue(pd.isna(summary.iloc[0]["avg_simulated_r"]))
        self.assertTrue(pd.isna(summary.iloc[1]["avg_delta_vs_baseline"]))


if __name__ == "__main__":
    unittest.main()
