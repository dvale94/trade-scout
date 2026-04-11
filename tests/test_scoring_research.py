from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from research.scoring.contracts import DEFAULT_SCORING_CONTRACT, ScoringContract, available_feature_columns
from research.scoring.dataset import (
    _build_scoring_modeling_dataset_from_source,
    audit_scoring_dataset_export,
    build_scoring_modeling_dataset_with_audit,
)
from research.scoring.diagnostics import (
    categorical_cardinality_summary,
    feature_missingness_summary,
    numeric_constant_feature_summary,
    train_test_schema_check,
)
from research.scoring.evaluation import (
    evaluate_current_score,
    evaluate_prediction_buckets,
    segmented_score_evaluation,
)
from research.scoring.predictions import SCORE_OUTPUT_COLUMNS, build_prediction_output
from research.scoring.score_recipes import (
    BLENDED_RECIPE,
    EXPECTED_R_ONLY_RECIPE,
    WIN_PROB_ONLY_RECIPE,
    compose_shadow_score,
)
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
            score_recipe=BLENDED_RECIPE,
            pred_win_prob=pd.Series([0.55, 0.65]),
            pred_expected_r=pd.Series([0.1, 0.4]),
        )

        self.assertEqual(tuple(out.columns), SCORE_OUTPUT_COLUMNS)
        self.assertEqual(out["score_model_version"].unique().tolist(), ["shadow_score_v1"])
        self.assertEqual(out["score_recipe_version"].unique().tolist(), ["score_recipe_blended_v1"])
        self.assertTrue(out["score_shadow"].notna().all())

    def test_prediction_output_preserves_timezone_aware_signal_timestamp(self):
        frame = pd.DataFrame(
            {
                "alert_id": ["a1"],
                "signal_ts_utc": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            }
        )

        out = build_prediction_output(
            frame,
            score_model_version="shadow_score_v1",
            feature_set_version="score_features_v1",
            pred_expected_r=pd.Series([0.2]),
        )

        merged = frame.merge(out, on=["alert_id", "signal_ts_utc"], how="left", validate="1:1")
        self.assertEqual(len(merged), 1)
        self.assertEqual(str(out["signal_ts_utc"].dtype), str(frame["signal_ts_utc"].dtype))

    def test_score_recipe_composition_modes_are_explicit(self):
        expected_r = pd.Series([0.2, 0.4])
        win_prob = pd.Series([0.6, 0.8])

        self.assertEqual(
            compose_shadow_score(
                pred_expected_r=expected_r,
                pred_win_prob=win_prob,
                recipe=EXPECTED_R_ONLY_RECIPE,
            ).tolist(),
            [0.2, 0.4],
        )
        self.assertEqual(
            compose_shadow_score(
                pred_expected_r=expected_r,
                pred_win_prob=win_prob,
                recipe=WIN_PROB_ONLY_RECIPE,
            ).tolist(),
            [0.6, 0.8],
        )
        blended = compose_shadow_score(
            pred_expected_r=expected_r,
            pred_win_prob=win_prob,
            recipe=BLENDED_RECIPE,
        )
        self.assertEqual(blended.round(4).tolist(), [0.3, 0.7])

    def test_dataset_export_audit_separates_leakage_and_unknown_columns(self):
        source = pd.DataFrame(
            {
                "alert_id": ["a1"],
                "score": [7.0],
                "pnl_R": [1.0],
                "exit_reason": ["T2"],
                "surprise_unapproved": ["nope"],
            }
        )
        exported = source[["alert_id", "score", "pnl_R"]].copy()

        audit = audit_scoring_dataset_export(source, exported)

        self.assertIn("exit_reason", audit["columns_dropped_for_leakage"])
        self.assertIn("surprise_unapproved", audit["columns_dropped_unknown_unapproved"])

    def test_derived_features_are_exported_and_numeric_coerced(self):
        contract = ScoringContract(
            feature_set_version="test_features_v1",
            metadata_columns=("alert_id", "signal_ts_utc"),
            numeric_feature_columns=("score",),
            categorical_feature_columns=(),
            derived_feature_columns=("derived_signal_strength",),
            target_columns=("pnl_R", "profitable"),
            leakage_columns=("exit_reason",),
            diagnostic_only_columns=(),
        )
        source = pd.DataFrame(
            {
                "alert_id": ["a1"],
                "signal_ts_utc": ["2026-01-01T00:00:00Z"],
                "score": ["7.5"],
                "derived_signal_strength": ["0.42"],
                "pnl_R": ["1.2"],
                "exit_reason": ["T2"],
            }
        )

        dataset = _build_scoring_modeling_dataset_from_source(source, contract=contract)
        constants = numeric_constant_feature_summary(dataset, contract)

        self.assertIn("derived_signal_strength", dataset.columns)
        self.assertEqual(float(dataset["derived_signal_strength"].iloc[0]), 0.42)
        self.assertIn("derived_signal_strength", available_feature_columns(dataset, contract))
        self.assertIn("derived_signal_strength", constants["feature"].tolist())
        self.assertNotIn("exit_reason", dataset.columns)

    def test_dataset_with_audit_loads_source_once(self):
        source = pd.DataFrame(
            {
                "alert_id": ["a1"],
                "signal_ts_utc": ["2026-01-01T00:00:00Z"],
                "score": [7.5],
                "pnl_R": [1.2],
            }
        )

        with patch("research.scoring.dataset._load_authoritative_trade_surface", return_value=source) as loader:
            dataset, audit = build_scoring_modeling_dataset_with_audit()

        self.assertEqual(loader.call_count, 1)
        self.assertEqual(dataset["alert_id"].tolist(), ["a1"])
        self.assertIn("score", audit["exported_columns"])

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

    def test_diagnostics_and_segmented_evaluation_return_notebook_ready_frames(self):
        frame = pd.DataFrame(
            {
                "score": [1.0, 1.0, 1.0],
                "symbol": ["BTC/USD", "ETH/USD", "BTC/USD"],
                "pnl_R": [1.0, -1.0, 0.5],
                "score_shadow": [0.8, 0.2, 0.7],
            }
        )

        missing = feature_missingness_summary(frame)
        constants = numeric_constant_feature_summary(frame)
        cardinality = categorical_cardinality_summary(frame)
        segments = segmented_score_evaluation(frame, segment_col="symbol")
        schema = train_test_schema_check(frame.iloc[:2], frame.iloc[2:])

        self.assertFalse(missing.empty)
        self.assertTrue(constants.loc[constants["feature"].eq("score"), "is_constant"].iloc[0])
        self.assertFalse(cardinality.empty)
        self.assertFalse(segments.empty)
        self.assertIn("schema_match", schema.columns)


if __name__ == "__main__":
    unittest.main()
