from __future__ import annotations

import unittest

import pandas as pd

from research.scoring.contracts import DEFAULT_SCORING_CONTRACT, available_feature_columns
from research.scoring.dataset import build_scoring_modeling_dataset
from research.scoring.evaluation import evaluate_current_score, evaluate_prediction_buckets
from research.scoring.predictions import SCORE_OUTPUT_COLUMNS, build_prediction_output
from research.scoring.splits import time_train_test_split


class ScoringResearchTest(unittest.TestCase):
    def test_feature_contract_excludes_outcome_leakage_columns(self):
        frame = pd.DataFrame(
            {
                "score": [7.0],
                "symbol": ["BTC/USD"],
                "pnl_R": [1.2],
                "exit_reason": ["T2"],
            }
        )

        features = available_feature_columns(frame, DEFAULT_SCORING_CONTRACT)

        self.assertIn("score", features)
        self.assertIn("symbol", features)
        self.assertNotIn("pnl_R", features)
        self.assertNotIn("exit_reason", features)

    def test_time_train_test_split_sorts_without_random_shuffle(self):
        frame = pd.DataFrame(
            {
                "alert_id": ["c", "a", "b", "d"],
                "signal_ts_utc": [
                    "2026-01-03T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                ],
            }
        )

        split = time_train_test_split(frame, test_size=0.5, min_train_rows=1, min_test_rows=1)

        self.assertEqual(split.train["alert_id"].tolist(), ["a", "b"])
        self.assertEqual(split.test["alert_id"].tolist(), ["c", "d"])
        self.assertEqual(split.train_rows, 2)
        self.assertEqual(split.test_rows, 2)

    def test_prediction_output_uses_versioned_schema(self):
        frame = pd.DataFrame(
            {
                "alert_id": ["a1", "a2"],
                "signal_ts_utc": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            }
        )

        out = build_prediction_output(
            frame,
            score_model_version="shadow_score_v1",
            feature_set_version="score_features_v1",
            pred_win_prob=pd.Series([0.55, 0.65]),
            pred_expected_r=pd.Series([0.1, 0.4]),
        )

        self.assertEqual(tuple(out.columns), SCORE_OUTPUT_COLUMNS)
        self.assertEqual(out["score_model_version"].unique().tolist(), ["shadow_score_v1"])
        self.assertTrue(out["score_shadow"].notna().all())

    def test_score_evaluation_returns_bucket_summary(self):
        frame = pd.DataFrame(
            {
                "score": [5.0, 6.0, 7.0, 8.0],
                "score_shadow": [0.1, 0.2, 0.3, 0.4],
                "pnl_R": [-1.0, -0.2, 0.5, 1.5],
            }
        )

        baseline = evaluate_current_score(frame, buckets=2)
        shadow = evaluate_prediction_buckets(frame, buckets=2)

        self.assertFalse(baseline["bucket_summary"].empty)
        self.assertFalse(shadow["bucket_summary"].empty)
        self.assertIn("win_rate", baseline["bucket_summary"].columns)


if __name__ == "__main__":
    unittest.main()
