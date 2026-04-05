from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.Response = object
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    requests_stub.delete = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

from research.policy_analysis import (
    counts_by_decision,
    flatten_policy_criteria_results,
    flatten_policy_decision_reasons,
    flatten_policy_feature_snapshot,
    join_ready_policy_decisions,
    load_policy_decisions,
    top_failed_criteria,
)
from trade_scout.execution.queue import get_connection, init_db, utc_now_iso
from trade_scout.policy.models import PolicyDecisionRow, PolicyDefinitionRow
from trade_scout.policy.repository import insert_policy_decision, insert_policy_definition


class PolicyAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "execution.db")
        init_db(self.db_path)
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(
                conn,
                PolicyDefinitionRow(
                    policy_id="policy-1",
                    policy_name="execution_policy",
                    policy_version="1.0.0",
                    policy_hash="hash-1",
                    status="SHADOW_ONLY",
                    config_json="{}",
                    description="shadow policy",
                    created_ts_utc=utc_now_iso(),
                ),
            )
            insert_policy_decision(conn, self._decision_row_one())
            insert_policy_decision(conn, self._decision_row_two())
        finally:
            conn.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_policy_decisions_returns_expected_rows(self) -> None:
        df = load_policy_decisions(self.db_path)

        self.assertEqual(2, len(df))
        self.assertIn("alert_id", df.columns)
        self.assertIn("decision_reasons_json", df.columns)

    def test_flatten_feature_snapshot_includes_expected_columns_and_values(self) -> None:
        df = load_policy_decisions(self.db_path)
        features = flatten_policy_feature_snapshot(df)

        self.assertIn("symbol", features.columns)
        self.assertIn("setup_family", features.columns)
        row = features[features["alert_id"] == "alert-1"].iloc[0]
        self.assertEqual("AAPL", row["symbol"])
        self.assertEqual("breakout_continuation", row["setup_family"])

    def test_flatten_criteria_results_produces_long_form_rows(self) -> None:
        df = load_policy_decisions(self.db_path)
        criteria = flatten_policy_criteria_results(df)

        self.assertEqual(3, len(criteria))
        self.assertIn("criterion_name", criteria.columns)
        self.assertIn("allowed_symbols", set(criteria["criterion_name"]))

    def test_flatten_reasons_produces_long_form_rows(self) -> None:
        df = load_policy_decisions(self.db_path)
        reasons = flatten_policy_decision_reasons(df)

        self.assertEqual(3, len(reasons))
        self.assertIn("reason", reasons.columns)
        self.assertIn("symbol 'DOGE/USD' not allowed", set(reasons["reason"]))

    def test_summary_counts_by_decision_are_correct(self) -> None:
        df = load_policy_decisions(self.db_path)
        summary = counts_by_decision(df)

        counts = dict(zip(summary["decision"], summary["count"]))
        self.assertEqual(1, counts["ALLOW"])
        self.assertEqual(1, counts["BLOCK"])

    def test_summary_of_failed_criteria_is_correct(self) -> None:
        df = load_policy_decisions(self.db_path)
        failed = top_failed_criteria(df)

        self.assertEqual("allowed_symbols", failed.iloc[0]["criterion_name"])
        self.assertEqual(1, failed.iloc[0]["count"])

    def test_join_ready_output_by_alert_id_is_deterministic(self) -> None:
        df = load_policy_decisions(self.db_path)
        joined = join_ready_policy_decisions(df)

        self.assertEqual(["alert-1", "alert-2"], list(joined["alert_id"]))
        self.assertIn("score_bucket", joined.columns)
        self.assertEqual("BLOCK", joined[joined["alert_id"] == "alert-2"].iloc[0]["decision"])

    def _decision_row_one(self) -> PolicyDecisionRow:
        return PolicyDecisionRow(
            policy_decision_id="decision-1",
            alert_id="alert-1",
            policy_id="policy-1",
            policy_name="execution_policy",
            policy_version="1.0.0",
            policy_hash="hash-1",
            mode="SHADOW",
            decision="ALLOW",
            decision_ts_utc="2026-04-05T12:00:00+00:00",
            decision_reasons_json='["all configured policy criteria passed"]',
            criteria_results_json=(
                '[{"criterion_name":"allowed_symbols","passed":true,'
                '"actual_value":"AAPL","expected_value":["AAPL","MSFT"]}]'
            ),
            feature_snapshot_json=(
                '{"alert_id":"alert-1","symbol":"AAPL","setup_family":"breakout_continuation",'
                '"anchor_regime":"bull","score":8.2,"score_bucket":"8_PLUS"}'
            ),
            execution_candidate=1,
            request_id=None,
        )

    def _decision_row_two(self) -> PolicyDecisionRow:
        return PolicyDecisionRow(
            policy_decision_id="decision-2",
            alert_id="alert-2",
            policy_id="policy-1",
            policy_name="execution_policy",
            policy_version="1.0.0",
            policy_hash="hash-1",
            mode="SHADOW",
            decision="BLOCK",
            decision_ts_utc="2026-04-05T12:05:00+00:00",
            decision_reasons_json='["symbol \'DOGE/USD\' not allowed","anchor_regime \'bear\' not allowed"]',
            criteria_results_json=(
                '[{"criterion_name":"allowed_symbols","passed":false,'
                '"actual_value":"DOGE/USD","expected_value":["AAPL","MSFT"]},'
                '{"criterion_name":"allowed_anchor_regimes","passed":false,'
                '"actual_value":"bear","expected_value":["bull"]}]'
            ),
            feature_snapshot_json=(
                '{"alert_id":"alert-2","symbol":"DOGE/USD","setup_family":"retest",'
                '"anchor_regime":"bear","score":6.1,"score_bucket":"6_TO_6_9"}'
            ),
            execution_candidate=0,
            request_id=None,
        )
