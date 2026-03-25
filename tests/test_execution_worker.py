from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.Response = object
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

from trade_scout.execution.models import BrokerOrderResult, ExecutionRequest, ExecutionStatus
from trade_scout.execution.queue import (
    get_connection,
    get_paper_order_by_request_id_and_role,
    init_db,
    insert_request,
    utc_now_iso,
)
from trade_scout.execution.worker import ExecutionWorker, ExecutionWorkerConfig


class ExecutionWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "execution.db")
        init_db(self.db_path)
        self.worker = ExecutionWorker(
            ExecutionWorkerConfig(
                db_path=self.db_path,
                max_alert_age_seconds=3600,
                worker_id="test-worker",
            ),
            alert_lookup=self._alert_lookup,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_init_db_creates_trade_plan_and_normalized_order_tables(self) -> None:
        conn = get_connection(self.db_path)
        try:
            table_names = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            self.assertIn("paper_trade_plans", table_names)
            self.assertIn("paper_orders", table_names)

            plan_columns = self._table_columns(conn, "paper_trade_plans")
            order_columns = self._table_columns(conn, "paper_orders")
        finally:
            conn.close()

        self.assertIn("plan_id", plan_columns)
        self.assertIn("request_id", plan_columns)
        self.assertIn("entry_type", plan_columns)
        self.assertIn("plan_status", plan_columns)

        self.assertIn("paper_order_id", order_columns)
        self.assertIn("plan_id", order_columns)
        self.assertIn("client_order_id", order_columns)
        self.assertIn("order_role", order_columns)
        self.assertIn("order_status", order_columns)

    def test_run_once_creates_one_trade_plan_and_entry_order(self) -> None:
        self._insert_pending_request(request_id="req-1", alert_id="alert-1")

        with patch(
            "trade_scout.execution.worker.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.worker.submit_validated_order",
            return_value=self._broker_result("broker-1"),
        ):
            attempted = self.worker.run_once()

        self.assertEqual(1, attempted)

        conn = get_connection(self.db_path)
        try:
            plan_rows = conn.execute("SELECT * FROM paper_trade_plans").fetchall()
            order_rows = conn.execute("SELECT * FROM paper_orders").fetchall()
            request_row = conn.execute(
                "SELECT * FROM execution_requests WHERE request_id = ?",
                ("req-1",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, len(plan_rows))
        self.assertEqual(1, len(order_rows))

        plan_row = plan_rows[0]
        order_row = order_rows[0]

        self.assertEqual("req-1", plan_row["request_id"])
        self.assertEqual("alert-1", plan_row["alert_id"])
        self.assertEqual("AAPL", plan_row["symbol"])
        self.assertEqual("LONG", plan_row["side"])
        self.assertEqual("swing", plan_row["strategy_profile"])
        self.assertEqual("long_breakout_continuation", plan_row["strategy_id"])
        self.assertEqual("1.0.0", plan_row["strategy_version"])
        self.assertEqual("MARKET", plan_row["entry_type"])
        self.assertEqual("CREATED", plan_row["plan_status"])
        self.assertEqual(101.5, plan_row["entry_low"])
        self.assertEqual(95.0, plan_row["stop_px"])
        self.assertEqual(25.0, plan_row["intended_notional"])

        self.assertEqual(plan_row["plan_id"], order_row["plan_id"])
        self.assertEqual("req-1", order_row["request_id"])
        self.assertEqual("alert-1", order_row["alert_id"])
        self.assertEqual("req-1", order_row["client_order_id"])
        self.assertEqual("ENTRY", order_row["order_role"])
        self.assertEqual("SUBMITTED", order_row["order_status"])
        self.assertEqual("broker-1", order_row["broker_order_id"])
        self.assertEqual("alpaca_paper", order_row["broker"])
        self.assertEqual("submitted", order_row["broker_status_raw"])

        self.assertEqual("SUBMITTED", request_row["status"])
        self.assertEqual("broker-1", request_row["broker_order_id"])

    def test_insert_request_preserves_requested_ts_utc(self) -> None:
        requested_ts_utc = "2026-03-25T15:45:30+00:00"

        conn = get_connection(self.db_path)
        try:
            insert_request(
                conn,
                ExecutionRequest(
                    request_id="req-ts",
                    alert_id="alert-ts",
                    requested_ts_utc=requested_ts_utc,
                    status=ExecutionStatus.PENDING,
                ),
            )
            stored_row = conn.execute(
                "SELECT * FROM execution_requests WHERE request_id = ?",
                ("req-ts",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(requested_ts_utc, stored_row["requested_ts_utc"])

    def test_reprocessing_same_validated_request_does_not_duplicate_plan_or_order(self) -> None:
        self._insert_pending_request(request_id="req-2", alert_id="alert-2")

        with patch(
            "trade_scout.execution.worker.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.worker.submit_validated_order",
            return_value=self._broker_result("broker-2"),
        ):
            self.worker.run_once()

        conn = get_connection(self.db_path)
        try:
            conn.execute(
                "UPDATE execution_requests SET status = ? WHERE request_id = ?",
                (ExecutionStatus.VALIDATED.value, "req-2"),
            )
            conn.commit()
            request_row = conn.execute(
                "SELECT * FROM execution_requests WHERE request_id = ?",
                ("req-2",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.find_existing_broker_order",
                return_value=None,
            ), patch(
                "trade_scout.execution.worker.submit_validated_order",
                side_effect=AssertionError("submit_validated_order should not be called"),
            ):
                self.worker.process_validated_request(conn, request_row)

            plan_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_trade_plans WHERE request_id = ?",
                ("req-2",),
            ).fetchone()["n"]
            order_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE client_order_id = ?",
                ("req-2",),
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(1, plan_count)
        self.assertEqual(1, order_count)

    def test_remote_recovery_still_persists_normalized_order_by_client_order_id(self) -> None:
        self._insert_pending_request(request_id="req-3", alert_id="alert-3")

        with patch(
            "trade_scout.execution.worker.find_existing_broker_order",
            return_value=self._broker_result("broker-3"),
        ), patch(
            "trade_scout.execution.worker.submit_validated_order",
            side_effect=AssertionError("submit_validated_order should not be called"),
        ):
            attempted = self.worker.run_once()

        self.assertEqual(1, attempted)

        conn = get_connection(self.db_path)
        try:
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE client_order_id = ?",
                ("req-3",),
            ).fetchone()
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-3",),
            ).fetchone()
            request_row = conn.execute(
                "SELECT * FROM execution_requests WHERE request_id = ?",
                ("req-3",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(plan_row)
        self.assertIsNotNone(order_row)
        self.assertEqual(plan_row["plan_id"], order_row["plan_id"])
        self.assertEqual("req-3", order_row["client_order_id"])
        self.assertEqual("broker-3", order_row["broker_order_id"])
        self.assertEqual("SUBMITTED", request_row["status"])

    def test_role_aware_lookup_returns_only_entry_order(self) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO paper_orders (
                    paper_order_id,
                    plan_id,
                    request_id,
                    alert_id,
                    broker,
                    broker_order_id,
                    client_order_id,
                    parent_paper_order_id,
                    order_role,
                    symbol,
                    side,
                    order_type,
                    time_in_force,
                    qty,
                    notional,
                    submitted_ts_utc,
                    accepted_ts_utc,
                    broker_status_raw,
                    order_status,
                    last_broker_update_ts_utc,
                    raw_response_json,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "po-entry",
                    "plan-1",
                    "req-role",
                    "alert-role",
                    "alpaca_paper",
                    "broker-entry",
                    "req-role",
                    None,
                    "ENTRY",
                    "AAPL",
                    "LONG",
                    "market",
                    "gtc",
                    None,
                    25.0,
                    "2026-03-25T14:00:00+00:00",
                    None,
                    "submitted",
                    "SUBMITTED",
                    "2026-03-25T14:00:00+00:00",
                    '{"id":"broker-entry"}',
                    "1",
                ),
            )
            conn.execute(
                """
                INSERT INTO paper_orders (
                    paper_order_id,
                    plan_id,
                    request_id,
                    alert_id,
                    broker,
                    broker_order_id,
                    client_order_id,
                    parent_paper_order_id,
                    order_role,
                    symbol,
                    side,
                    order_type,
                    time_in_force,
                    qty,
                    notional,
                    submitted_ts_utc,
                    accepted_ts_utc,
                    broker_status_raw,
                    order_status,
                    last_broker_update_ts_utc,
                    raw_response_json,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "po-stop",
                    "plan-1",
                    "req-role",
                    "alert-role",
                    "alpaca_paper",
                    "broker-stop",
                    "req-role-stop",
                    "po-entry",
                    "STOP",
                    "AAPL",
                    "SELL",
                    "stop",
                    "gtc",
                    None,
                    None,
                    "2026-03-25T14:01:00+00:00",
                    None,
                    "new",
                    "SUBMITTED",
                    "2026-03-25T14:01:00+00:00",
                    '{"id":"broker-stop"}',
                    "1",
                ),
            )
            conn.commit()

            entry_order = get_paper_order_by_request_id_and_role(conn, "req-role", "ENTRY")
            stop_order = get_paper_order_by_request_id_and_role(conn, "req-role", "STOP")
        finally:
            conn.close()

        self.assertEqual("po-entry", entry_order["paper_order_id"])
        self.assertEqual("broker-entry", entry_order["broker_order_id"])
        self.assertEqual("po-stop", stop_order["paper_order_id"])
        self.assertEqual("broker-stop", stop_order["broker_order_id"])

    def _insert_pending_request(self, *, request_id: str, alert_id: str) -> None:
        conn = get_connection(self.db_path)
        try:
            insert_request(
                conn,
                ExecutionRequest(
                    request_id=request_id,
                    alert_id=alert_id,
                    requested_ts_utc=utc_now_iso(),
                    status=ExecutionStatus.PENDING,
                ),
            )
        finally:
            conn.close()

    def _alert_lookup(self, alert_id: str):
        created_ts_utc = utc_now_iso()
        return {
            "alert_id": alert_id,
            "created_ts_utc": created_ts_utc,
            "signal_ts_utc": created_ts_utc,
            "symbol": "AAPL",
            "side": "LONG",
            "market_type": "stocks",
            "strategy_profile": "swing",
            "strategy_version": "1.0.0",
            "config_hash": "cfg-123",
            "strategy_id": "long_breakout_continuation",
            "setup_family": "breakout_continuation",
            "strategy_params_json": "{\"foo\": 1}",
            "strategy_params_hash": "params-123",
            "tier": "A",
            "signal_px": "102.25",
            "entry_low": "101.5",
            "entry_high": "102.0",
            "stop_px": "95.0",
            "t1_px": "110.0",
            "t2_px": "115.0",
            "timeout_seconds": "300",
            "risk_model": "fixed_risk",
            "risk_usd": "10",
            "qty_model": "notional",
            "qty": "0.5",
        }

    @staticmethod
    def _broker_result(broker_order_id: str) -> BrokerOrderResult:
        return BrokerOrderResult(
            broker="alpaca_paper",
            broker_order_id=broker_order_id,
            submitted_ts_utc="2026-03-25T14:00:05+00:00",
            raw_response_json='{"id":"%s"}' % broker_order_id,
            broker_status_raw="submitted",
            accepted_ts_utc="2026-03-25T14:00:06+00:00",
        )

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
        return {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }


if __name__ == "__main__":
    unittest.main()
