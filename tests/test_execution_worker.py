from __future__ import annotations

import json
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
    requests_stub.delete = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

from trade_scout.execution.models import BrokerOrderResult, ExecutionRequest, ExecutionStatus
from trade_scout.execution.queue import (
    get_connection,
    get_current_paper_order_by_plan_id_and_role,
    get_open_paper_orders_for_sync,
    get_paper_fill_by_order_id_and_role,
    get_paper_order_by_plan_id_and_role,
    get_paper_order_by_request_id_and_role,
    init_db,
    insert_request,
    reconcile_paper_order_from_broker,
    utc_now_iso,
)
from trade_scout.execution.worker import (
    ExecutionWorker,
    ExecutionWorkerConfig,
    derive_entry_status_from_order_status,
    normalize_broker_order_status,
)


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
            self.assertIn("paper_fills", table_names)
            self.assertIn("paper_trade_outcomes", table_names)

            plan_columns = self._table_columns(conn, "paper_trade_plans")
            order_columns = self._table_columns(conn, "paper_orders")
            fill_columns = self._table_columns(conn, "paper_fills")
            outcome_columns = self._table_columns(conn, "paper_trade_outcomes")
        finally:
            conn.close()

        self.assertIn("plan_id", plan_columns)
        self.assertIn("request_id", plan_columns)
        self.assertIn("entry_type", plan_columns)
        self.assertIn("plan_status", plan_columns)
        self.assertIn("entry_status", plan_columns)
        self.assertIn("entry_filled_qty", plan_columns)
        self.assertIn("entry_avg_fill_px", plan_columns)
        self.assertIn("entry_fill_ts_utc", plan_columns)
        self.assertIn("position_status", plan_columns)
        self.assertIn("protective_stop_order_id", plan_columns)
        self.assertIn("tp1_order_id", plan_columns)
        self.assertIn("tp1_filled_qty", plan_columns)
        self.assertIn("tp1_avg_fill_px", plan_columns)
        self.assertIn("tp1_fill_ts_utc", plan_columns)
        self.assertIn("remaining_qty", plan_columns)
        self.assertIn("partial_exit_status", plan_columns)
        self.assertIn("closed_ts_utc", plan_columns)

        self.assertIn("paper_order_id", order_columns)
        self.assertIn("plan_id", order_columns)
        self.assertIn("client_order_id", order_columns)
        self.assertIn("order_role", order_columns)
        self.assertIn("order_status", order_columns)
        self.assertIn("replaced_from_paper_order_id", order_columns)
        self.assertIn("superseded_by_paper_order_id", order_columns)

        self.assertIn("paper_fill_id", fill_columns)
        self.assertIn("paper_order_id", fill_columns)
        self.assertIn("broker_order_id", fill_columns)
        self.assertIn("fill_role", fill_columns)
        self.assertIn("fill_ts_utc", fill_columns)
        self.assertIn("fill_px", fill_columns)
        self.assertIn("fill_qty", fill_columns)

        self.assertIn("outcome_id", outcome_columns)
        self.assertIn("plan_id", outcome_columns)
        self.assertIn("exit_reason", outcome_columns)
        self.assertIn("gross_pnl", outcome_columns)
        self.assertIn("realized_r", outcome_columns)

    def test_normalize_broker_order_status_maps_expected_values(self) -> None:
        self.assertEqual("SUBMITTED", normalize_broker_order_status("submitted"))
        self.assertEqual("SUBMITTED", normalize_broker_order_status("pending_new"))
        self.assertEqual("ACCEPTED", normalize_broker_order_status("new"))
        self.assertEqual("ACCEPTED", normalize_broker_order_status("accepted"))
        self.assertEqual("PARTIALLY_FILLED", normalize_broker_order_status("partially_filled"))
        self.assertEqual("FILLED", normalize_broker_order_status("filled"))
        self.assertEqual("CANCELED", normalize_broker_order_status("canceled"))
        self.assertEqual("REJECTED", normalize_broker_order_status("rejected"))
        self.assertEqual("EXPIRED", normalize_broker_order_status("expired"))
        self.assertEqual("SUBMITTED", normalize_broker_order_status("unknown_status"))

    def test_derive_entry_status_from_order_status_maps_fill_states(self) -> None:
        self.assertEqual("PARTIALLY_FILLED", derive_entry_status_from_order_status("PARTIALLY_FILLED"))
        self.assertEqual("FILLED", derive_entry_status_from_order_status("FILLED"))
        self.assertIsNone(derive_entry_status_from_order_status("ACCEPTED"))

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
        self.assertEqual("PENDING", plan_row["entry_status"])
        self.assertIsNone(plan_row["entry_filled_qty"])
        self.assertEqual("PENDING_ENTRY", plan_row["position_status"])
        self.assertEqual("NONE", plan_row["partial_exit_status"])

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

    def test_get_open_paper_orders_for_sync_excludes_terminal_statuses(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-open-1",
                request_id="req-open-1",
                client_order_id="req-open-1",
                broker_order_id="broker-open-1",
                order_status="SUBMITTED",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-open-2",
                request_id="req-open-2",
                client_order_id="req-open-2",
                broker_order_id="broker-open-2",
                order_status="ACCEPTED",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-filled",
                request_id="req-filled",
                client_order_id="req-filled",
                broker_order_id="broker-filled",
                order_status="FILLED",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-canceled",
                request_id="req-canceled",
                client_order_id="req-canceled",
                broker_order_id="broker-canceled",
                order_status="CANCELED",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-replaced",
                request_id="req-replaced",
                client_order_id="req-replaced:STOP",
                broker_order_id="broker-replaced",
                order_status="REPLACED",
                order_role="STOP",
            )

            rows = get_open_paper_orders_for_sync(conn, limit=10)
        finally:
            conn.close()

        self.assertEqual(["po-open-1", "po-open-2"], [row["paper_order_id"] for row in rows])

    def test_reconcile_paper_order_from_broker_updates_filled_state(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-fill",
                request_id="req-fill",
                client_order_id="req-fill",
                broker_order_id="broker-fill",
                order_status="SUBMITTED",
            )

            reconciled = reconcile_paper_order_from_broker(
                conn,
                paper_order_id="po-fill",
                broker_snapshot=self._broker_result(
                    "broker-fill",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T14:10:00+00:00",
                    raw_response_json='{"id":"broker-fill","status":"filled"}',
                ),
                order_status="FILLED",
            )
        finally:
            conn.close()

        self.assertEqual("filled", reconciled["broker_status_raw"])
        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertEqual("2026-03-25T14:10:00+00:00", reconciled["filled_ts_utc"])
        self.assertEqual('{"id":"broker-fill","status":"filled"}', reconciled["raw_response_json"])

    def test_reconcile_paper_order_from_broker_updates_canceled_state(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-cancel",
                request_id="req-cancel",
                client_order_id="req-cancel",
                broker_order_id="broker-cancel",
                order_status="ACCEPTED",
            )

            reconciled = reconcile_paper_order_from_broker(
                conn,
                paper_order_id="po-cancel",
                broker_snapshot=self._broker_result(
                    "broker-cancel",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-25T14:11:00+00:00",
                    raw_response_json='{"id":"broker-cancel","status":"canceled"}',
                ),
                order_status="CANCELED",
            )
        finally:
            conn.close()

        self.assertEqual("canceled", reconciled["broker_status_raw"])
        self.assertEqual("CANCELED", reconciled["order_status"])
        self.assertEqual("2026-03-25T14:11:00+00:00", reconciled["canceled_ts_utc"])
        self.assertEqual('{"id":"broker-cancel","status":"canceled"}', reconciled["raw_response_json"])

    def test_get_paper_order_by_plan_id_and_role_returns_expected_stop_or_entry_row(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-plan-entry",
                request_id="req-plan-role",
                client_order_id="req-plan-role",
                broker_order_id="broker-plan-entry",
                order_status="SUBMITTED",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-plan-stop",
                request_id="req-plan-role",
                client_order_id="req-plan-role-stop",
                broker_order_id="broker-plan-stop",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-plan-entry",
            )
            entry_row = get_paper_order_by_plan_id_and_role(conn, "plan-req-plan-role", "ENTRY")
            stop_row = get_paper_order_by_plan_id_and_role(conn, "plan-req-plan-role", "STOP")
        finally:
            conn.close()

        self.assertEqual("po-plan-entry", entry_row["paper_order_id"])
        self.assertEqual("po-plan-stop", stop_row["paper_order_id"])

    def test_reconcile_preserves_existing_accepted_timestamp_when_snapshot_omits_it(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-accepted-preserve",
                request_id="req-accepted-preserve",
                client_order_id="req-accepted-preserve",
                broker_order_id="broker-accepted-preserve",
                order_status="ACCEPTED",
            )
            reconciled = reconcile_paper_order_from_broker(
                conn,
                paper_order_id="po-accepted-preserve",
                broker_snapshot=self._broker_result(
                    "broker-accepted-preserve",
                    broker_status_raw="accepted",
                    accepted_ts_utc=None,
                    raw_response_json='{"id":"broker-accepted-preserve","status":"accepted"}',
                ),
                order_status="ACCEPTED",
            )
        finally:
            conn.close()

        self.assertEqual("2026-03-25T14:00:01+00:00", reconciled["accepted_ts_utc"])

    def test_reconcile_preserves_existing_filled_timestamp_when_snapshot_omits_it(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-filled-preserve",
                request_id="req-filled-preserve",
                client_order_id="req-filled-preserve",
                broker_order_id="broker-filled-preserve",
                order_status="FILLED",
                filled_ts_utc="2026-03-25T14:30:00+00:00",
            )
            reconciled = reconcile_paper_order_from_broker(
                conn,
                paper_order_id="po-filled-preserve",
                broker_snapshot=self._broker_result(
                    "broker-filled-preserve",
                    broker_status_raw="filled",
                    filled_ts_utc=None,
                    raw_response_json='{"id":"broker-filled-preserve","status":"filled"}',
                ),
                order_status="FILLED",
            )
        finally:
            conn.close()

        self.assertEqual("2026-03-25T14:30:00+00:00", reconciled["filled_ts_utc"])

    def test_reconcile_stores_new_accepted_timestamp_when_snapshot_provides_it(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-accepted-replace",
                request_id="req-accepted-replace",
                client_order_id="req-accepted-replace",
                broker_order_id="broker-accepted-replace",
                order_status="SUBMITTED",
                accepted_ts_utc=None,
            )
            reconciled = reconcile_paper_order_from_broker(
                conn,
                paper_order_id="po-accepted-replace",
                broker_snapshot=self._broker_result(
                    "broker-accepted-replace",
                    broker_status_raw="accepted",
                    accepted_ts_utc="2026-03-25T14:31:00+00:00",
                    raw_response_json='{"id":"broker-accepted-replace","status":"accepted"}',
                ),
                order_status="ACCEPTED",
            )
        finally:
            conn.close()

        self.assertEqual("2026-03-25T14:31:00+00:00", reconciled["accepted_ts_utc"])

    def test_sync_is_idempotent_when_broker_snapshot_is_unchanged(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-idem",
                request_id="req-idem",
                client_order_id="req-idem",
                broker_order_id="broker-idem",
                order_status="SUBMITTED",
            )
            self._insert_trade_plan_row(conn, request_id="req-idem")

            snapshot = self._broker_result(
                "broker-idem",
                broker_status_raw="filled",
                filled_ts_utc="2026-03-25T14:12:00+00:00",
                filled_qty=0.75,
                filled_avg_price=101.25,
                raw_response_json='{"id":"broker-idem","status":"filled"}',
                last_broker_update_ts_utc="2026-03-25T14:12:01+00:00",
            )
            row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-idem",),
            ).fetchone()
            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result("broker-stop-idem", broker_status_raw="submitted"),
            ), patch(
                "trade_scout.execution.worker.submit_limit_order",
                return_value=self._broker_result("broker-tp1-idem", broker_status_raw="submitted"),
            ):
                first = self.worker.reconcile_paper_order(conn, row, snapshot)
                second_row = conn.execute(
                    "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                    ("po-idem",),
                ).fetchone()
                second = self.worker.reconcile_paper_order(conn, second_row, snapshot)
            row_count = conn.execute("SELECT COUNT(*) AS n FROM paper_orders").fetchone()["n"]
            fill_count = conn.execute("SELECT COUNT(*) AS n FROM paper_fills").fetchone()["n"]
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-idem",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(3, row_count)
        self.assertEqual(1, fill_count)
        self.assertEqual(dict(first), dict(second))
        self.assertEqual("FILLED", plan_row["entry_status"])
        self.assertEqual(0.75, plan_row["entry_filled_qty"])
        self.assertEqual(101.25, plan_row["entry_avg_fill_px"])
        self.assertEqual("OPEN", plan_row["position_status"])
        self.assertIsNotNone(plan_row["protective_stop_order_id"])
        self.assertIsNotNone(plan_row["tp1_order_id"])
        self.assertEqual(0.75, plan_row["remaining_qty"])

    def test_filled_order_ingests_one_entry_fill_and_updates_plan_summary(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-fill-sync")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-fill-sync",
                request_id="req-fill-sync",
                client_order_id="req-fill-sync",
                broker_order_id="broker-fill-sync",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-fill-sync",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result("broker-stop-fill-sync", broker_status_raw="submitted"),
            ), patch(
                "trade_scout.execution.worker.submit_limit_order",
                return_value=self._broker_result("broker-tp1-fill-sync", broker_status_raw="submitted"),
            ):
                reconciled = self.worker.reconcile_paper_order(
                    conn,
                    order_row,
                    self._broker_result(
                        "broker-fill-sync",
                        broker_status_raw="filled",
                        filled_ts_utc="2026-03-25T14:40:00+00:00",
                        filled_qty=0.8,
                        filled_avg_price=103.5,
                        raw_response_json='{"id":"broker-fill-sync","status":"filled"}',
                    ),
                )
            fill_row = get_paper_fill_by_order_id_and_role(conn, "po-fill-sync", "ENTRY")
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-fill-sync",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertIsNotNone(fill_row)
        self.assertEqual("po-fill-sync:ENTRY", fill_row["paper_fill_id"])
        self.assertEqual("ENTRY", fill_row["fill_role"])
        self.assertEqual(0.8, fill_row["fill_qty"])
        self.assertEqual(103.5, fill_row["fill_px"])
        self.assertEqual("FILLED", plan_row["entry_status"])
        self.assertEqual(0.8, plan_row["entry_filled_qty"])
        self.assertEqual(103.5, plan_row["entry_avg_fill_px"])
        self.assertEqual("2026-03-25T14:40:00+00:00", plan_row["entry_fill_ts_utc"])
        self.assertEqual("OPEN", plan_row["position_status"])
        self.assertIsNotNone(plan_row["protective_stop_order_id"])
        self.assertIsNotNone(plan_row["tp1_order_id"])
        self.assertEqual(0.8, plan_row["remaining_qty"])

    def test_filled_entry_creates_one_protective_stop_and_marks_plan_open(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop", stop_px=95.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-entry-stop",
                request_id="req-stop",
                client_order_id="req-stop",
                broker_order_id="broker-entry-stop",
                order_status="SUBMITTED",
            )
            entry_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-entry-stop",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result("broker-stop-1", broker_status_raw="submitted"),
            ) as submit_stop_mock, patch(
                "trade_scout.execution.worker.submit_limit_order",
                return_value=self._broker_result("broker-tp1-1", broker_status_raw="submitted"),
            ) as submit_tp1_mock:
                self.worker.reconcile_paper_order(
                    conn,
                    entry_row,
                    self._broker_result(
                        "broker-entry-stop",
                        broker_status_raw="filled",
                        filled_ts_utc="2026-03-26T14:00:00+00:00",
                        filled_qty=1.0,
                        filled_avg_price=100.0,
                    ),
                )
                second_entry_row = conn.execute(
                    "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                    ("po-entry-stop",),
                ).fetchone()
                self.worker.reconcile_paper_order(
                    conn,
                    second_entry_row,
                    self._broker_result(
                        "broker-entry-stop",
                        broker_status_raw="filled",
                        filled_ts_utc="2026-03-26T14:00:00+00:00",
                        filled_qty=1.0,
                        filled_avg_price=100.0,
                    ),
                )

            stop_orders = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-stop", "STOP"),
            ).fetchall()
            tp1_orders = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-stop", "TP1"),
            ).fetchall()
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-stop",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, submit_stop_mock.call_count)
        self.assertEqual(1, submit_tp1_mock.call_count)
        self.assertEqual(1, len(stop_orders))
        self.assertEqual(1, len(tp1_orders))
        self.assertEqual("OPEN", plan_row["position_status"])
        self.assertEqual(stop_orders[0]["paper_order_id"], plan_row["protective_stop_order_id"])
        self.assertEqual(tp1_orders[0]["paper_order_id"], plan_row["tp1_order_id"])
        self.assertEqual(1.0, plan_row["remaining_qty"])

    def test_existing_stop_row_wins_over_plan_pointer_and_realigns_pointer(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-pointer-align",
                entry_status="FILLED",
                entry_filled_qty=1.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T10:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="stale-pointer",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-entry-pointer-align",
                request_id="req-pointer-align",
                client_order_id="req-pointer-align",
                broker_order_id="broker-entry-pointer-align",
                order_status="FILLED",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-pointer-align",
                request_id="req-pointer-align",
                client_order_id="req-pointer-align:STOP",
                broker_order_id="broker-stop-pointer-align",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-pointer-align",
                side="SHORT",
                stop_price=95.0,
            )
            entry_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-entry-pointer-align",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                side_effect=AssertionError("submit_stop_order should not be called"),
            ), patch(
                "trade_scout.execution.worker.submit_limit_order",
                side_effect=AssertionError("submit_limit_order should not be called"),
            ):
                self.worker.ensure_protective_stop_for_entry_fill(conn, entry_row)

            stop_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-pointer-align", "STOP"),
            ).fetchone()["n"]
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-pointer-align",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, stop_count)
        self.assertEqual("po-stop-pointer-align", plan_row["protective_stop_order_id"])

    def test_stale_plan_pointer_does_not_block_stop_recovery_or_placement(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-stale-pointer",
                entry_status="FILLED",
                entry_filled_qty=1.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T10:05:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="stale-pointer-only",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-entry-stale-pointer",
                request_id="req-stale-pointer",
                client_order_id="req-stale-pointer",
                broker_order_id="broker-entry-stale-pointer",
                order_status="FILLED",
            )
            entry_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-entry-stale-pointer",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.find_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result("broker-stop-stale-pointer", broker_status_raw="submitted"),
            ) as submit_stop_mock:
                self.worker.ensure_protective_stop_for_entry_fill(conn, entry_row)
                refreshed_entry_row = conn.execute(
                    "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                    ("po-entry-stale-pointer",),
                ).fetchone()
                self.worker.ensure_protective_stop_for_entry_fill(conn, refreshed_entry_row)

            stop_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-stale-pointer", "STOP"),
            ).fetchall()
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-stale-pointer",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, submit_stop_mock.call_count)
        self.assertEqual(1, len(stop_rows))
        self.assertEqual(stop_rows[0]["paper_order_id"], plan_row["protective_stop_order_id"])

    def test_partially_filled_entry_does_not_create_stop(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-no-stop")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-entry-no-stop",
                request_id="req-no-stop",
                client_order_id="req-no-stop",
                broker_order_id="broker-entry-no-stop",
                order_status="SUBMITTED",
            )
            entry_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-entry-no-stop",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                side_effect=AssertionError("submit_stop_order should not be called"),
            ), patch(
                "trade_scout.execution.worker.submit_limit_order",
                side_effect=AssertionError("submit_limit_order should not be called"),
            ):
                self.worker.reconcile_paper_order(
                    conn,
                    entry_row,
                    self._broker_result(
                        "broker-entry-no-stop",
                        broker_status_raw="partially_filled",
                        filled_ts_utc=None,
                        filled_qty=0.3,
                        filled_avg_price=100.0,
                        last_broker_update_ts_utc="2026-03-26T14:05:00+00:00",
                    ),
                )

            stop_orders = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-no-stop", "STOP"),
            ).fetchone()["n"]
            tp1_orders = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-no-stop", "TP1"),
            ).fetchone()["n"]
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-no-stop",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(0, stop_orders)
        self.assertEqual("PENDING_ENTRY", plan_row["position_status"])
        self.assertEqual(0, tp1_orders)

    def test_partial_fill_ingests_or_updates_entry_fill_and_plan_summary(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-partial")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-partial",
                request_id="req-partial",
                client_order_id="req-partial",
                broker_order_id="broker-partial",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-partial",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-partial",
                    broker_status_raw="partially_filled",
                    filled_ts_utc="2026-03-25T14:41:00+00:00",
                    filled_qty=0.25,
                    filled_avg_price=102.0,
                ),
            )
            fill_row = get_paper_fill_by_order_id_and_role(conn, "po-partial", "ENTRY")
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-partial",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(fill_row)
        self.assertEqual(0.25, fill_row["fill_qty"])
        self.assertEqual(102.0, fill_row["fill_px"])
        self.assertEqual("PARTIALLY_FILLED", plan_row["entry_status"])
        self.assertEqual(0.25, plan_row["entry_filled_qty"])
        self.assertEqual(102.0, plan_row["entry_avg_fill_px"])
        self.assertEqual("PENDING_ENTRY", plan_row["position_status"])

    def test_partial_fill_timestamp_preserves_first_fallback_timestamp(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-partial-ts")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-partial-ts",
                request_id="req-partial-ts",
                client_order_id="req-partial-ts",
                broker_order_id="broker-partial-ts",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-partial-ts",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-partial-ts",
                    broker_status_raw="partially_filled",
                    filled_ts_utc=None,
                    filled_qty=0.1,
                    filled_avg_price=100.5,
                    last_broker_update_ts_utc="2026-03-25T14:50:00+00:00",
                ),
            )
            second_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-partial-ts",),
            ).fetchone()
            self.worker.reconcile_paper_order(
                conn,
                second_order_row,
                self._broker_result(
                    "broker-partial-ts",
                    broker_status_raw="partially_filled",
                    filled_ts_utc=None,
                    filled_qty=0.2,
                    filled_avg_price=100.75,
                    last_broker_update_ts_utc="2026-03-25T14:55:00+00:00",
                ),
            )

            fill_row = get_paper_fill_by_order_id_and_role(conn, "po-partial-ts", "ENTRY")
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-partial-ts",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("2026-03-25T14:50:00+00:00", fill_row["fill_ts_utc"])
        self.assertEqual("2026-03-25T14:50:00+00:00", plan_row["entry_fill_ts_utc"])

    def test_explicit_filled_timestamp_replaces_partial_fallback_timestamp(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-fill-ts")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-fill-ts",
                request_id="req-fill-ts",
                client_order_id="req-fill-ts",
                broker_order_id="broker-fill-ts",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-fill-ts",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-fill-ts",
                    broker_status_raw="partially_filled",
                    filled_ts_utc=None,
                    filled_qty=0.15,
                    filled_avg_price=101.0,
                    last_broker_update_ts_utc="2026-03-25T15:00:00+00:00",
                ),
            )
            second_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-fill-ts",),
            ).fetchone()
            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result("broker-stop-fill-ts", broker_status_raw="submitted"),
            ), patch(
                "trade_scout.execution.worker.submit_limit_order",
                return_value=self._broker_result("broker-tp1-fill-ts", broker_status_raw="submitted"),
            ):
                self.worker.reconcile_paper_order(
                    conn,
                    second_order_row,
                    self._broker_result(
                        "broker-fill-ts",
                        broker_status_raw="filled",
                        filled_ts_utc="2026-03-25T15:05:00+00:00",
                        filled_qty=0.5,
                        filled_avg_price=101.5,
                        last_broker_update_ts_utc="2026-03-25T15:06:00+00:00",
                    ),
                )

            fill_row = get_paper_fill_by_order_id_and_role(conn, "po-fill-ts", "ENTRY")
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-fill-ts",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("2026-03-25T15:05:00+00:00", fill_row["fill_ts_utc"])
        self.assertEqual("2026-03-25T15:05:00+00:00", plan_row["entry_fill_ts_utc"])

    def test_skip_fill_ingestion_when_snapshot_lacks_required_fill_data(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-skip-fill")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-skip-fill",
                request_id="req-skip-fill",
                client_order_id="req-skip-fill",
                broker_order_id="broker-skip-fill",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-skip-fill",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-skip-fill",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T14:42:00+00:00",
                    filled_qty=None,
                    filled_avg_price=104.0,
                ),
            )
            fill_row = get_paper_fill_by_order_id_and_role(conn, "po-skip-fill", "ENTRY")
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-skip-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNone(fill_row)
        self.assertEqual("PENDING", plan_row["entry_status"])
        self.assertIsNone(plan_row["entry_filled_qty"])

    def test_tp1_fill_replaces_broker_stop_to_breakeven_remaining_qty(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-tp1-fill",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-tp1-fill",
                tp1_order_id="po-tp1-fill",
                remaining_qty=10.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-tp1-fill",
                request_id="req-tp1-fill",
                client_order_id="req-tp1-fill:STOP",
                broker_order_id="broker-stop-tp1-fill",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-tp1-fill",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-fill",
                request_id="req-tp1-fill",
                client_order_id="req-tp1-fill:TP1",
                broker_order_id="broker-tp1-fill",
                order_status="SUBMITTED",
                order_role="TP1",
                parent_paper_order_id="po-entry-tp1-fill",
                side="SHORT",
                qty=5.0,
                limit_price=110.0,
                reduce_only=1,
            )
            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-fill",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                return_value=self._broker_result(
                    "broker-stop-tp1-fill",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-26T14:10:01+00:00",
                ),
            ) as cancel_mock, patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result(
                    "broker-stop-tp1-fill-r2",
                    broker_status_raw="submitted",
                ),
            ) as submit_stop_mock:
                reconciled = self.worker.reconcile_paper_order(
                    conn,
                    tp1_row,
                    self._broker_result(
                        "broker-tp1-fill",
                        broker_status_raw="filled",
                        filled_ts_utc="2026-03-26T14:10:00+00:00",
                        filled_qty=5.0,
                        filled_avg_price=110.0,
                    ),
                )
            tp1_fill = get_paper_fill_by_order_id_and_role(conn, "po-tp1-fill", "TP1")
            old_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-tp1-fill",),
            ).fetchone()
            replacement_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE client_order_id = ?",
                ("req-tp1-fill:STOP:R2",),
            ).fetchone()
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-tp1-fill",),
            ).fetchone()
            outcome_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_trade_outcomes WHERE plan_id = ?",
                ("plan-req-tp1-fill",),
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertIsNotNone(tp1_fill)
        self.assertEqual("TP1", tp1_fill["fill_role"])
        self.assertEqual(5.0, tp1_fill["fill_qty"])
        self.assertEqual(110.0, tp1_fill["fill_px"])
        self.assertEqual(1, cancel_mock.call_count)
        self.assertEqual(1, submit_stop_mock.call_count)
        self.assertEqual("po-tp1-fill", plan_row["tp1_order_id"])
        self.assertEqual(5.0, plan_row["tp1_filled_qty"])
        self.assertEqual(110.0, plan_row["tp1_avg_fill_px"])
        self.assertEqual("2026-03-26T14:10:00+00:00", plan_row["tp1_fill_ts_utc"])
        self.assertEqual(5.0, plan_row["remaining_qty"])
        self.assertEqual("TP1_PARTIAL", plan_row["partial_exit_status"])
        self.assertEqual("OPEN", plan_row["position_status"])
        self.assertEqual("REPLACED", old_stop_row["order_status"])
        self.assertEqual(replacement_stop_row["paper_order_id"], old_stop_row["superseded_by_paper_order_id"])
        self.assertEqual("po-stop-tp1-fill", replacement_stop_row["replaced_from_paper_order_id"])
        self.assertEqual(5.0, replacement_stop_row["qty"])
        self.assertEqual(100.05, replacement_stop_row["stop_price"])
        self.assertEqual(replacement_stop_row["paper_order_id"], plan_row["protective_stop_order_id"])
        self.assertEqual(0, outcome_count)

    def test_tp1_filled_snapshot_is_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-tp1-idem",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-tp1-idem",
                tp1_order_id="po-tp1-idem",
                remaining_qty=10.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-tp1-idem",
                request_id="req-tp1-idem",
                client_order_id="req-tp1-idem:STOP",
                broker_order_id="broker-stop-tp1-idem",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-tp1-idem",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-idem",
                request_id="req-tp1-idem",
                client_order_id="req-tp1-idem:TP1",
                broker_order_id="broker-tp1-idem",
                order_status="SUBMITTED",
                order_role="TP1",
                parent_paper_order_id="po-entry-tp1-idem",
                side="SHORT",
                qty=5.0,
                limit_price=110.0,
                reduce_only=1,
            )
            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-idem",),
            ).fetchone()
            snapshot = self._broker_result(
                "broker-tp1-idem",
                broker_status_raw="filled",
                filled_ts_utc="2026-03-26T14:12:00+00:00",
                filled_qty=5.0,
                filled_avg_price=110.0,
            )
            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                return_value=self._broker_result(
                    "broker-stop-tp1-idem",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-26T14:12:01+00:00",
                ),
            ) as cancel_mock, patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result(
                    "broker-stop-tp1-idem-r2",
                    broker_status_raw="submitted",
                ),
            ) as submit_stop_mock:
                self.worker.reconcile_paper_order(conn, tp1_row, snapshot)
                refreshed_tp1_row = conn.execute(
                    "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                    ("po-tp1-idem",),
                ).fetchone()
                self.worker.reconcile_paper_order(conn, refreshed_tp1_row, snapshot)
            tp1_fill_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_fills WHERE paper_order_id = ? AND fill_role = ?",
                ("po-tp1-idem", "TP1"),
            ).fetchone()["n"]
            tp1_order_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-tp1-idem", "TP1"),
            ).fetchone()["n"]
            stop_order_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ? ORDER BY submitted_ts_utc ASC",
                ("plan-req-tp1-idem", "STOP"),
            ).fetchall()
            outcome_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_trade_outcomes WHERE plan_id = ?",
                ("plan-req-tp1-idem",),
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(1, cancel_mock.call_count)
        self.assertEqual(1, submit_stop_mock.call_count)
        self.assertEqual(1, tp1_fill_count)
        self.assertEqual(1, tp1_order_count)
        self.assertEqual(2, len(stop_order_rows))
        self.assertEqual("REPLACED", stop_order_rows[0]["order_status"])
        self.assertEqual("SUBMITTED", stop_order_rows[1]["order_status"])
        self.assertEqual(100.05, stop_order_rows[1]["stop_price"])
        self.assertEqual(0, outcome_count)

    def test_cancel_success_and_replacement_failure_leaves_truthful_unprotected_state(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-stop-fail",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-fail",
                remaining_qty=5.0,
                partial_exit_status="TP1_PARTIAL",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-fail",
                request_id="req-stop-fail",
                client_order_id="req-stop-fail:STOP",
                broker_order_id="broker-stop-fail",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-stop-fail",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
            )

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                return_value=self._broker_result(
                    "broker-stop-fail",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-26T14:10:01+00:00",
                ),
            ), patch(
                "trade_scout.execution.worker.submit_stop_order",
                side_effect=RuntimeError("replacement submit failed"),
            ), patch(
                "trade_scout.execution.worker.find_broker_order_by_client_order_id",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "replacement submit failed"):
                    self.worker.maybe_move_stop_to_breakeven(conn, "plan-req-stop-fail", 5.0)

            old_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-fail",),
            ).fetchone()
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-stop-fail",),
            ).fetchone()
            current_stop_row = get_current_paper_order_by_plan_id_and_role(
                conn,
                "plan-req-stop-fail",
                "STOP",
            )

            with patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result(
                    "broker-stop-fail-r2",
                    broker_status_raw="submitted",
                ),
            ), patch(
                "trade_scout.execution.worker.find_broker_order_by_client_order_id",
                return_value=None,
            ):
                self.worker.maybe_move_stop_to_breakeven(conn, "plan-req-stop-fail", 5.0)

            replacement_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE client_order_id = ?",
                ("req-stop-fail:STOP:R2",),
            ).fetchone()
            refreshed_plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-stop-fail",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("CANCELED", old_stop_row["order_status"])
        self.assertEqual("canceled_for_replace", old_stop_row["status_reason"])
        self.assertIsNone(plan_row["protective_stop_order_id"])
        self.assertIsNone(current_stop_row)
        self.assertIsNotNone(replacement_stop_row)
        self.assertEqual(100.05, replacement_stop_row["stop_price"])
        self.assertEqual(replacement_stop_row["paper_order_id"], refreshed_plan_row["protective_stop_order_id"])

    def test_current_stop_selection_ignores_canceled_stop_after_replace_cancel(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-stop-canceled",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id=None,
                remaining_qty=5.0,
                partial_exit_status="TP1_PARTIAL",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-canceled",
                request_id="req-stop-canceled",
                client_order_id="req-stop-canceled:STOP",
                broker_order_id="broker-stop-canceled",
                order_status="CANCELED",
                order_role="STOP",
                parent_paper_order_id="po-entry-stop-canceled",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
            )

            current_stop_row = get_current_paper_order_by_plan_id_and_role(
                conn,
                "plan-req-stop-canceled",
                "STOP",
            )
        finally:
            conn.close()

        self.assertIsNone(current_stop_row)

    def test_no_stop_replace_when_current_stop_already_matches_remaining_qty_and_breakeven(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-stop-aligned",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-aligned-r2",
                remaining_qty=5.0,
                partial_exit_status="TP1_PARTIAL",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-aligned-old",
                request_id="req-stop-aligned",
                client_order_id="req-stop-aligned:STOP",
                broker_order_id="broker-stop-aligned-old",
                order_status="REPLACED",
                order_role="STOP",
                parent_paper_order_id="po-entry-stop-aligned",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
                superseded_by_paper_order_id="po-stop-aligned-r2",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-aligned-r2",
                request_id="req-stop-aligned",
                client_order_id="req-stop-aligned:STOP:R2",
                broker_order_id="broker-stop-aligned-r2",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-stop-aligned",
                side="SHORT",
                qty=5.0,
                stop_price=100.05,
                replaced_from_paper_order_id="po-stop-aligned-old",
            )

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                side_effect=AssertionError("cancel_broker_order should not be called"),
            ), patch(
                "trade_scout.execution.worker.submit_stop_order",
                side_effect=AssertionError("submit_stop_order should not be called"),
            ):
                self.worker.maybe_move_stop_to_breakeven(conn, "plan-req-stop-aligned", 5.0)

            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-stop-aligned",),
            ).fetchone()
            stop_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-stop-aligned", "STOP"),
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual("po-stop-aligned-r2", plan_row["protective_stop_order_id"])
        self.assertEqual(2, stop_count)

    def test_stale_stop_pointer_realigns_to_existing_current_replacement(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-stop-recover",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-recover-old",
                remaining_qty=5.0,
                partial_exit_status="TP1_PARTIAL",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-recover-old",
                request_id="req-stop-recover",
                client_order_id="req-stop-recover:STOP",
                broker_order_id="broker-stop-recover-old",
                order_status="REPLACED",
                order_role="STOP",
                parent_paper_order_id="po-entry-stop-recover",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
                superseded_by_paper_order_id="po-stop-recover-r2",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-recover-r2",
                request_id="req-stop-recover",
                client_order_id="req-stop-recover:STOP:R2",
                broker_order_id="broker-stop-recover-r2",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-stop-recover",
                side="SHORT",
                qty=5.0,
                stop_price=100.05,
                replaced_from_paper_order_id="po-stop-recover-old",
            )

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                side_effect=AssertionError("cancel_broker_order should not be called"),
            ), patch(
                "trade_scout.execution.worker.submit_stop_order",
                side_effect=AssertionError("submit_stop_order should not be called"),
            ):
                self.worker.maybe_move_stop_to_breakeven(conn, "plan-req-stop-recover", 5.0)

            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-stop-recover",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("po-stop-recover-r2", plan_row["protective_stop_order_id"])

    def test_no_breakeven_replace_before_tp1_partial_exit(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-no-breakeven",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-no-breakeven",
                remaining_qty=10.0,
                partial_exit_status="NONE",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-no-breakeven",
                request_id="req-no-breakeven",
                client_order_id="req-no-breakeven:STOP",
                broker_order_id="broker-stop-no-breakeven",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-no-breakeven",
                side="SHORT",
                qty=10.0,
                stop_price=95.0,
            )

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                side_effect=AssertionError("cancel_broker_order should not be called"),
            ), patch(
                "trade_scout.execution.worker.submit_stop_order",
                side_effect=AssertionError("submit_stop_order should not be called"),
            ):
                self.worker.maybe_move_stop_to_breakeven(
                    conn,
                    "plan-req-no-breakeven",
                    10.0,
                )

            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-no-breakeven",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(95.0, stop_row["stop_price"])

    def test_short_trade_tp1_fill_moves_stop_to_breakeven(self) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO paper_trade_plans (
                    plan_id,
                    request_id,
                    alert_id,
                    created_ts_utc,
                    symbol,
                    side,
                    entry_type,
                    stop_px,
                    t1_px,
                    plan_rules_json,
                    plan_status,
                    entry_status,
                    entry_filled_qty,
                    entry_avg_fill_px,
                    entry_fill_ts_utc,
                    position_status,
                    protective_stop_order_id,
                    tp1_order_id,
                    tp1_filled_qty,
                    tp1_avg_fill_px,
                    tp1_fill_ts_utc,
                    remaining_qty,
                    partial_exit_status,
                    closed_ts_utc,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "plan-req-short-breakeven",
                    "req-short-breakeven",
                    "alert-req-short-breakeven",
                    "2026-03-25T14:00:00+00:00",
                    "AAPL",
                    "SHORT",
                    "MARKET",
                    105.0,
                    90.0,
                    "{}",
                    "CREATED",
                    "FILLED",
                    10.0,
                    100.0,
                    "2026-03-26T14:00:00+00:00",
                    "OPEN",
                    "po-stop-short-breakeven",
                    "po-tp1-short-breakeven",
                    5.0,
                    90.0,
                    "2026-03-26T14:05:00+00:00",
                    5.0,
                    "TP1_PARTIAL",
                    None,
                    "1",
                ),
            )
            conn.commit()
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-short-breakeven",
                request_id="req-short-breakeven",
                client_order_id="req-short-breakeven:STOP",
                broker_order_id="broker-stop-short-breakeven",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-short-breakeven",
                side="LONG",
                qty=10.0,
                stop_price=105.0,
            )

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                return_value=self._broker_result(
                    "broker-stop-short-breakeven",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-26T14:10:01+00:00",
                ),
            ), patch(
                "trade_scout.execution.worker.submit_stop_order",
                return_value=self._broker_result(
                    "broker-stop-short-breakeven-r2",
                    broker_status_raw="submitted",
                ),
            ):
                self.worker.maybe_move_stop_to_breakeven(
                    conn,
                    "plan-req-short-breakeven",
                    5.0,
                )

            replacement_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE client_order_id = ?",
                ("req-short-breakeven:STOP:R2",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(5.0, replacement_stop_row["qty"])
        self.assertEqual(99.95, replacement_stop_row["stop_price"])

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

    def test_run_once_syncs_existing_open_orders_after_request_processing(self) -> None:
        self._insert_pending_request(request_id="req-4", alert_id="alert-4")
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-sync")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-sync",
                request_id="req-sync",
                client_order_id="req-sync",
                broker_order_id="broker-sync",
                order_status="SUBMITTED",
                broker_status_raw="submitted",
            )
        finally:
            conn.close()

        with patch(
            "trade_scout.execution.worker.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.worker.submit_validated_order",
            return_value=self._broker_result("broker-4", broker_status_raw="submitted"),
        ), patch(
            "trade_scout.execution.worker.submit_stop_order",
            return_value=self._broker_result("broker-stop-sync", broker_status_raw="submitted"),
        ), patch(
            "trade_scout.execution.worker.submit_limit_order",
            return_value=self._broker_result("broker-tp1-sync", broker_status_raw="submitted"),
        ), patch(
            "trade_scout.execution.worker.fetch_broker_order_by_id",
            side_effect=lambda broker_order_id: (
                self._broker_result(
                    broker_order_id,
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T14:20:00+00:00",
                    filled_qty=0.5,
                    filled_avg_price=105.0,
                    raw_response_json='{"id":"%s","status":"filled"}' % broker_order_id,
                    last_broker_update_ts_utc="2026-03-25T14:20:01+00:00",
                )
                if broker_order_id == "broker-sync"
                else self._broker_result(broker_order_id, broker_status_raw="submitted")
            ),
        ):
            attempted = self.worker.run_once()

        self.assertEqual(1, attempted)

        conn = get_connection(self.db_path)
        try:
            synced_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-sync",),
            ).fetchone()
            synced_fill = get_paper_fill_by_order_id_and_role(conn, "po-sync", "ENTRY")
            synced_plan = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-sync",),
            ).fetchone()
            synced_stop_order = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-sync", "STOP"),
            ).fetchone()
            synced_tp1_order = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-sync", "TP1"),
            ).fetchone()
            new_request_order = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ?",
                ("req-4",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("FILLED", synced_row["order_status"])
        self.assertEqual("filled", synced_row["broker_status_raw"])
        self.assertEqual("2026-03-25T14:20:00+00:00", synced_row["filled_ts_utc"])
        self.assertIsNotNone(synced_fill)
        self.assertEqual(0.5, synced_fill["fill_qty"])
        self.assertEqual(105.0, synced_fill["fill_px"])
        self.assertEqual("FILLED", synced_plan["entry_status"])
        self.assertEqual("OPEN", synced_plan["position_status"])
        self.assertIsNotNone(synced_stop_order)
        self.assertIsNotNone(synced_tp1_order)
        self.assertIsNotNone(new_request_order)

    def test_stop_fill_creates_outcome_and_closes_plan(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-close",
                entry_status="FILLED",
                entry_filled_qty=1.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-close",
                request_id="req-close",
                client_order_id="req-close:STOP",
                broker_order_id="broker-stop-close",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-close",
                side="SHORT",
                stop_price=95.0,
            )
            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-close",),
            ).fetchone()

            reconciled = self.worker.reconcile_paper_order(
                conn,
                stop_row,
                self._broker_result(
                    "broker-stop-close",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-26T14:10:00+00:00",
                    filled_qty=1.0,
                    filled_avg_price=95.0,
                ),
            )
            stop_fill = get_paper_fill_by_order_id_and_role(conn, "po-stop-close", "STOP")
            outcome_row = conn.execute(
                "SELECT * FROM paper_trade_outcomes WHERE plan_id = ?",
                ("plan-req-close",),
            ).fetchone()
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-close",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertIsNotNone(stop_fill)
        self.assertEqual("STOP", stop_fill["fill_role"])
        self.assertIsNotNone(outcome_row)
        self.assertEqual("STOP", outcome_row["exit_reason"])
        self.assertEqual(1, outcome_row["closed"])
        self.assertEqual(1.0, outcome_row["exit_qty"])
        self.assertEqual(95.0, outcome_row["exit_avg_px"])
        self.assertEqual(-5.0, outcome_row["gross_pnl"])
        self.assertEqual(-1.0, outcome_row["realized_r"])
        self.assertEqual("CLOSED", plan_row["position_status"])
        self.assertEqual("2026-03-26T14:10:00+00:00", plan_row["closed_ts_utc"])
        summary = json.loads(outcome_row["raw_summary_json"])
        self.assertEqual(["po-stop-close:STOP"], summary["exit_fill_ids"])
        self.assertEqual(1.0, summary["total_exit_qty"])
        self.assertEqual(95.0, summary["weighted_exit_avg_px"])
        self.assertEqual(-5.0, summary["realized_gross_pnl"])
        self.assertEqual(1, len(summary["exit_legs"]))

    def test_tp1_and_stop_fills_aggregate_into_final_outcome(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-close-multi",
                entry_status="FILLED",
                entry_filled_qty=10.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                tp1_order_id="po-tp1-close-multi",
                tp1_filled_qty=5.0,
                tp1_avg_fill_px=110.0,
                tp1_fill_ts_utc="2026-03-26T14:05:00+00:00",
                remaining_qty=5.0,
                partial_exit_status="TP1_PARTIAL",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-entry-close-multi",
                request_id="req-close-multi",
                client_order_id="req-close-multi",
                broker_order_id="broker-entry-close-multi",
                order_status="FILLED",
                qty=10.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-close-multi",
                request_id="req-close-multi",
                client_order_id="req-close-multi:TP1",
                broker_order_id="broker-tp1-close-multi",
                order_status="FILLED",
                order_role="TP1",
                parent_paper_order_id="po-entry-close-multi",
                side="SHORT",
                qty=5.0,
                limit_price=110.0,
                reduce_only=1,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-close-multi",
                request_id="req-close-multi",
                client_order_id="req-close-multi:STOP",
                broker_order_id="broker-stop-close-multi",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-close-multi",
                side="SHORT",
                qty=5.0,
                stop_price=95.0,
            )
            conn.execute(
                """
                INSERT INTO paper_fills (
                    paper_fill_id,
                    plan_id,
                    request_id,
                    alert_id,
                    paper_order_id,
                    broker,
                    broker_order_id,
                    broker_fill_id,
                    fill_role,
                    symbol,
                    side,
                    fill_ts_utc,
                    fill_px,
                    fill_qty,
                    gross_notional,
                    fee,
                    liquidity,
                    fill_sequence,
                    raw_fill_json,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "po-entry-close-multi:ENTRY",
                    "plan-req-close-multi",
                    "req-close-multi",
                    "alert-req-close-multi",
                    "po-entry-close-multi",
                    "alpaca_paper",
                    "broker-entry-close-multi",
                    None,
                    "ENTRY",
                    "AAPL",
                    "LONG",
                    "2026-03-26T14:00:00+00:00",
                    100.0,
                    10.0,
                    1000.0,
                    None,
                    None,
                    1,
                    '{"id":"broker-entry-close-multi"}',
                    "1",
                ),
            )
            conn.execute(
                """
                INSERT INTO paper_fills (
                    paper_fill_id,
                    plan_id,
                    request_id,
                    alert_id,
                    paper_order_id,
                    broker,
                    broker_order_id,
                    broker_fill_id,
                    fill_role,
                    symbol,
                    side,
                    fill_ts_utc,
                    fill_px,
                    fill_qty,
                    gross_notional,
                    fee,
                    liquidity,
                    fill_sequence,
                    raw_fill_json,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "po-tp1-close-multi:TP1",
                    "plan-req-close-multi",
                    "req-close-multi",
                    "alert-req-close-multi",
                    "po-tp1-close-multi",
                    "alpaca_paper",
                    "broker-tp1-close-multi",
                    None,
                    "TP1",
                    "AAPL",
                    "SHORT",
                    "2026-03-26T14:05:00+00:00",
                    110.0,
                    5.0,
                    550.0,
                    None,
                    None,
                    1,
                    '{"id":"broker-tp1-close-multi"}',
                    "1",
                ),
            )
            conn.commit()
            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-close-multi",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                stop_row,
                self._broker_result(
                    "broker-stop-close-multi",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-26T14:10:00+00:00",
                    filled_qty=5.0,
                    filled_avg_price=95.0,
                ),
            )
            outcome_row = conn.execute(
                "SELECT * FROM paper_trade_outcomes WHERE plan_id = ?",
                ("plan-req-close-multi",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(outcome_row)
        self.assertEqual(10.0, outcome_row["entry_qty"])
        self.assertEqual(10.0, outcome_row["exit_qty"])
        self.assertEqual(102.5, outcome_row["exit_avg_px"])
        self.assertEqual(25.0, outcome_row["gross_pnl"])
        self.assertEqual(0.5, outcome_row["realized_r"])
        self.assertEqual("2026-03-26T14:10:00+00:00", outcome_row["exit_fill_ts_utc"])
        summary = json.loads(outcome_row["raw_summary_json"])
        self.assertEqual(
            ["po-tp1-close-multi:TP1", "po-stop-close-multi:STOP"],
            summary["exit_fill_ids"],
        )
        self.assertEqual(2, len(summary["exit_legs"]))
        self.assertEqual(10.0, summary["total_exit_qty"])
        self.assertEqual(102.5, summary["weighted_exit_avg_px"])
        self.assertEqual(25.0, summary["realized_gross_pnl"])

    def test_stop_filled_snapshot_is_outcome_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-close-idem",
                entry_status="FILLED",
                entry_filled_qty=1.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-close-idem",
                request_id="req-close-idem",
                client_order_id="req-close-idem:STOP",
                broker_order_id="broker-stop-close-idem",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-close-idem",
                side="SHORT",
                stop_price=95.0,
            )
            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-close-idem",),
            ).fetchone()
            snapshot = self._broker_result(
                "broker-stop-close-idem",
                broker_status_raw="filled",
                filled_ts_utc="2026-03-26T14:15:00+00:00",
                filled_qty=1.0,
                filled_avg_price=95.0,
            )
            self.worker.reconcile_paper_order(conn, stop_row, snapshot)
            second_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-close-idem",),
            ).fetchone()
            self.worker.reconcile_paper_order(conn, second_stop_row, snapshot)
            outcome_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_trade_outcomes WHERE plan_id = ?",
                ("plan-req-close-idem",),
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(1, outcome_count)

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
    def _broker_result(
        broker_order_id: str,
        *,
        broker_status_raw: str = "submitted",
        accepted_ts_utc: str = "2026-03-25T14:00:06+00:00",
        canceled_ts_utc: str | None = None,
        expired_ts_utc: str | None = None,
        rejected_ts_utc: str | None = None,
        filled_ts_utc: str | None = None,
        filled_qty: float | None = None,
        filled_avg_price: float | None = None,
        status_reason: str | None = None,
        raw_response_json: str | None = None,
        last_broker_update_ts_utc: str = "2026-03-25T14:00:07+00:00",
    ) -> BrokerOrderResult:
        return BrokerOrderResult(
            broker="alpaca_paper",
            broker_order_id=broker_order_id,
            submitted_ts_utc="2026-03-25T14:00:05+00:00",
            raw_response_json=raw_response_json or '{"id":"%s"}' % broker_order_id,
            broker_status_raw=broker_status_raw,
            accepted_ts_utc=accepted_ts_utc,
            canceled_ts_utc=canceled_ts_utc,
            expired_ts_utc=expired_ts_utc,
            rejected_ts_utc=rejected_ts_utc,
            filled_ts_utc=filled_ts_utc,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            status_reason=status_reason,
            last_broker_update_ts_utc=last_broker_update_ts_utc,
        )

    @staticmethod
    def _insert_trade_plan_row(
        conn,
        *,
        request_id: str,
        entry_status: str = "PENDING",
        entry_filled_qty: float | None = None,
        entry_avg_fill_px: float | None = None,
        entry_fill_ts_utc: str | None = None,
        position_status: str = "PENDING_ENTRY",
        protective_stop_order_id: str | None = None,
        tp1_order_id: str | None = None,
        tp1_filled_qty: float | None = None,
        tp1_avg_fill_px: float | None = None,
        tp1_fill_ts_utc: str | None = None,
        remaining_qty: float | None = None,
        partial_exit_status: str = "NONE",
        closed_ts_utc: str | None = None,
        stop_px: float | None = 95.0,
        t1_px: float | None = 110.0,
    ) -> None:
        conn.execute(
            """
            INSERT INTO paper_trade_plans (
                plan_id,
                request_id,
                alert_id,
                created_ts_utc,
                symbol,
                side,
                entry_type,
                stop_px,
                t1_px,
                plan_rules_json,
                plan_status,
                entry_status,
                entry_filled_qty,
                entry_avg_fill_px,
                entry_fill_ts_utc,
                position_status,
                protective_stop_order_id,
                tp1_order_id,
                tp1_filled_qty,
                tp1_avg_fill_px,
                tp1_fill_ts_utc,
                remaining_qty,
                partial_exit_status,
                closed_ts_utc,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "plan-" + request_id,
                request_id,
                "alert-" + request_id,
                "2026-03-25T14:00:00+00:00",
                "AAPL",
                "LONG",
                "MARKET",
                stop_px,
                t1_px,
                "{}",
                "CREATED",
                entry_status,
                entry_filled_qty,
                entry_avg_fill_px,
                entry_fill_ts_utc,
                position_status,
                protective_stop_order_id,
                tp1_order_id,
                tp1_filled_qty,
                tp1_avg_fill_px,
                tp1_fill_ts_utc,
                remaining_qty,
                partial_exit_status,
                closed_ts_utc,
                "1",
            ),
        )
        conn.commit()

    @staticmethod
    def _insert_paper_order_row(
        conn,
        *,
        paper_order_id: str,
        request_id: str,
        client_order_id: str,
        broker_order_id: str,
        order_status: str,
        broker_status_raw: str = "submitted",
        accepted_ts_utc: str | None = "2026-03-25T14:00:01+00:00",
        filled_ts_utc: str | None = None,
        order_role: str = "ENTRY",
        parent_paper_order_id: str | None = None,
        replaced_from_paper_order_id: str | None = None,
        superseded_by_paper_order_id: str | None = None,
        side: str = "LONG",
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        reduce_only: int = 0,
    ) -> None:
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
                replaced_from_paper_order_id,
                superseded_by_paper_order_id,
                order_role,
                symbol,
                side,
                order_type,
                time_in_force,
                qty,
                notional,
                limit_price,
                stop_price,
                reduce_only,
                submitted_ts_utc,
                accepted_ts_utc,
                filled_ts_utc,
                broker_status_raw,
                order_status,
                last_broker_update_ts_utc,
                raw_response_json,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_order_id,
                "plan-" + request_id,
                request_id,
                "alert-" + request_id,
                "alpaca_paper",
                broker_order_id,
                client_order_id,
                parent_paper_order_id,
                replaced_from_paper_order_id,
                superseded_by_paper_order_id,
                order_role,
                "AAPL",
                side,
                "limit" if order_role == "TP1" else "market",
                "gtc",
                qty,
                25.0,
                limit_price,
                stop_price,
                reduce_only,
                "2026-03-25T14:00:00+00:00",
                accepted_ts_utc,
                filled_ts_utc,
                broker_status_raw,
                order_status,
                "2026-03-25T14:00:02+00:00",
                '{"id":"%s","status":"%s"}' % (broker_order_id, broker_status_raw),
                "1",
            ),
        )
        conn.commit()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
        return {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }


if __name__ == "__main__":
    unittest.main()
