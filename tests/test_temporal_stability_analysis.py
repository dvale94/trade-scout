from __future__ import annotations

import unittest

import pandas as pd

from research.temporal_stability_analysis import (
    summarize_cumulative_performance,
    summarize_rolling_performance,
    summarize_temporal_performance,
)


class TemporalStabilityAnalysisTest(unittest.TestCase):
    def test_summarize_temporal_performance_groups_by_week(self) -> None:
        trades = pd.DataFrame(
            [
                {"candidate_band": "band_a", "signal_ts_utc": "2026-05-01T00:00:00+00:00", "pnl_R": 1.0, "mfe_R": 1.5, "mae_R": 0.5},
                {"candidate_band": "band_a", "signal_ts_utc": "2026-05-02T00:00:00+00:00", "pnl_R": -1.0, "mfe_R": 0.3, "mae_R": 1.0},
            ]
        )

        summary = summarize_temporal_performance(trades, "candidate_band", "signal_ts_utc", freq="W")
        row = summary.iloc[0]

        self.assertEqual("band_a", row["candidate_band"])
        self.assertEqual(2, row["n"])
        self.assertAlmostEqual(0.5, row["win_rate"])
        self.assertAlmostEqual(0.0, row["avg_realized_r"])

    def test_summarize_rolling_performance_uses_trade_count_windows(self) -> None:
        trades = pd.DataFrame(
            [
                {"candidate_band": "band_a", "signal_ts_utc": "2026-05-01T00:00:00+00:00", "pnl_R": 1.0},
                {"candidate_band": "band_a", "signal_ts_utc": "2026-05-02T00:00:00+00:00", "pnl_R": -1.0},
                {"candidate_band": "band_a", "signal_ts_utc": "2026-05-03T00:00:00+00:00", "pnl_R": 0.5},
            ]
        )

        summary = summarize_rolling_performance(trades, "candidate_band", "signal_ts_utc", window_size=2)
        last_row = summary.iloc[-1]

        self.assertEqual(2, last_row["n"])
        self.assertEqual(3, last_row["window_end_trade_n"])
        self.assertAlmostEqual(0.5, last_row["win_rate"])
        self.assertAlmostEqual(-0.25, last_row["avg_realized_r"])

    def test_missing_columns_do_not_crash(self) -> None:
        trades = pd.DataFrame([{"candidate_band": "band_a"}])

        temporal = summarize_temporal_performance(trades, "candidate_band", "signal_ts_utc", freq="M")
        rolling = summarize_rolling_performance(trades, "candidate_band", "signal_ts_utc", window_size=2)
        cumulative = summarize_cumulative_performance(trades, "candidate_band", "signal_ts_utc")

        self.assertTrue(temporal.empty)
        self.assertTrue(rolling.empty)
        self.assertTrue(cumulative.empty)


if __name__ == "__main__":
    unittest.main()
