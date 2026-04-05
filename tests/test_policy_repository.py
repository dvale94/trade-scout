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

from trade_scout.execution.queue import get_connection, init_db, utc_now_iso
from trade_scout.policy.models import (
    PolicyActivationRow,
    PolicyDecisionRow,
    PolicyDefinitionRow,
)
from trade_scout.policy.repository import (
    activate_policy,
    get_active_policy,
    get_policy_decision,
    get_policy_definition,
    insert_policy_decision,
    insert_policy_definition,
)


class PolicyRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "execution.db")
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_insert_policy_definition_creates_new_row(self) -> None:
        conn = get_connection(self.db_path)
        try:
            inserted = insert_policy_definition(conn, self._policy_definition_row())
        finally:
            conn.close()

        self.assertEqual("policy-1", inserted.policy_id)
        self.assertEqual("execution_policy", inserted.policy_name)

    def test_duplicate_policy_definition_insert_returns_existing_row(self) -> None:
        conn = get_connection(self.db_path)
        try:
            first = insert_policy_definition(conn, self._policy_definition_row(policy_id="policy-1"))
            second = insert_policy_definition(conn, self._policy_definition_row(policy_id="policy-2"))
            row_count = conn.execute("SELECT COUNT(*) AS n FROM policy_definitions").fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual("policy-1", first.policy_id)
        self.assertEqual("policy-1", second.policy_id)
        self.assertEqual(1, row_count)

    def test_get_policy_definition_returns_row(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row())
            row = get_policy_definition(conn, "execution_policy", "1.0.0")
        finally:
            conn.close()

        self.assertIsNotNone(row)
        self.assertEqual("policy-1", row.policy_id)

    def test_insert_policy_decision_creates_row(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row())
            inserted = insert_policy_decision(conn, self._policy_decision_row())
        finally:
            conn.close()

        self.assertEqual("decision-1", inserted.policy_decision_id)
        self.assertEqual("ALLOW", inserted.decision)

    def test_duplicate_policy_decision_insert_is_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row())
            first = insert_policy_decision(conn, self._policy_decision_row(policy_decision_id="decision-1"))
            second = insert_policy_decision(conn, self._policy_decision_row(policy_decision_id="decision-2"))
            row_count = conn.execute("SELECT COUNT(*) AS n FROM policy_decisions").fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual("decision-1", first.policy_decision_id)
        self.assertEqual("decision-1", second.policy_decision_id)
        self.assertEqual(1, row_count)

    def test_get_policy_decision_returns_row(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row())
            insert_policy_decision(conn, self._policy_decision_row())
            row = get_policy_decision(conn, "alert-1", "execution_policy", "1.0.0")
        finally:
            conn.close()

        self.assertIsNotNone(row)
        self.assertEqual("decision-1", row.policy_decision_id)

    def test_activate_policy_creates_activation(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row())
            activation = activate_policy(conn, self._activation_row())
        finally:
            conn.close()

        self.assertEqual("activation-1", activation.activation_id)
        self.assertIsNone(activation.effective_to_ts_utc)

    def test_activating_new_policy_closes_previous_one(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row(policy_id="policy-1", policy_version="1.0.0"))
            insert_policy_definition(conn, self._policy_definition_row(policy_id="policy-2", policy_version="1.1.0"))
            activate_policy(
                conn,
                self._activation_row(
                    activation_id="activation-1",
                    policy_id="policy-1",
                    policy_version="1.0.0",
                    effective_from_ts_utc="2026-04-05T12:00:00+00:00",
                ),
            )
            activate_policy(
                conn,
                self._activation_row(
                    activation_id="activation-2",
                    policy_id="policy-2",
                    policy_version="1.1.0",
                    effective_from_ts_utc="2026-04-05T13:00:00+00:00",
                ),
            )
            previous = conn.execute(
                """
                SELECT *
                FROM policy_activation_history
                WHERE activation_id = ?
                """,
                ("activation-1",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("2026-04-05T13:00:00+00:00", previous["effective_to_ts_utc"])

    def test_get_active_policy_returns_correct_policy(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row(policy_id="policy-1", policy_version="1.0.0"))
            insert_policy_definition(conn, self._policy_definition_row(policy_id="policy-2", policy_version="1.1.0"))
            activate_policy(
                conn,
                self._activation_row(
                    activation_id="activation-1",
                    policy_id="policy-1",
                    policy_version="1.0.0",
                    effective_from_ts_utc="2026-04-05T12:00:00+00:00",
                ),
            )
            activate_policy(
                conn,
                self._activation_row(
                    activation_id="activation-2",
                    policy_id="policy-2",
                    policy_version="1.1.0",
                    effective_from_ts_utc="2026-04-05T13:00:00+00:00",
                ),
            )
            active = get_active_policy(
                conn,
                "execution_policy",
                "2026-04-05T13:30:00+00:00",
                mode="SHADOW",
            )
        finally:
            conn.close()

        self.assertIsNotNone(active)
        self.assertEqual("policy-2", active.policy_id)

    def test_get_active_policy_returns_none_when_no_active_policy(self) -> None:
        conn = get_connection(self.db_path)
        try:
            active = get_active_policy(
                conn,
                "execution_policy",
                "2026-04-05T13:30:00+00:00",
                mode="SHADOW",
            )
        finally:
            conn.close()

        self.assertIsNone(active)

    def test_repeated_activation_call_is_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_policy_definition(conn, self._policy_definition_row())
            first = activate_policy(conn, self._activation_row())
            second = activate_policy(conn, self._activation_row(activation_id="activation-2"))
            row_count = conn.execute(
                "SELECT COUNT(*) AS n FROM policy_activation_history"
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual("activation-1", first.activation_id)
        self.assertEqual("activation-1", second.activation_id)
        self.assertEqual(1, row_count)

    def _policy_definition_row(
        self,
        *,
        policy_id: str = "policy-1",
        policy_version: str = "1.0.0",
    ) -> PolicyDefinitionRow:
        return PolicyDefinitionRow(
            policy_id=policy_id,
            policy_name="execution_policy",
            policy_version=policy_version,
            policy_hash=f"hash-{policy_version}",
            status="SHADOW_ONLY",
            config_json="{}",
            description="shadow policy",
            created_ts_utc=utc_now_iso(),
        )

    def _policy_decision_row(
        self,
        *,
        policy_decision_id: str = "decision-1",
    ) -> PolicyDecisionRow:
        return PolicyDecisionRow(
            policy_decision_id=policy_decision_id,
            alert_id="alert-1",
            policy_id="policy-1",
            policy_name="execution_policy",
            policy_version="1.0.0",
            policy_hash="hash-1.0.0",
            mode="SHADOW",
            decision="ALLOW",
            decision_ts_utc=utc_now_iso(),
            decision_reasons_json='["all configured policy criteria passed"]',
            criteria_results_json="[]",
            feature_snapshot_json='{"alert_id":"alert-1"}',
            execution_candidate=1,
            request_id=None,
        )

    def _activation_row(
        self,
        *,
        activation_id: str = "activation-1",
        policy_id: str = "policy-1",
        policy_version: str = "1.0.0",
        effective_from_ts_utc: str = "2026-04-05T12:00:00+00:00",
    ) -> PolicyActivationRow:
        return PolicyActivationRow(
            activation_id=activation_id,
            policy_id=policy_id,
            policy_name="execution_policy",
            policy_version=policy_version,
            mode="SHADOW",
            effective_from_ts_utc=effective_from_ts_utc,
            effective_to_ts_utc=None,
        )
