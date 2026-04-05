from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
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

from trade_scout.execution.models import (
    BrokerOrderResult,
    ExecutionRequest,
    ExecutionStatus,
    PaperPositionActionStatus,
    PaperPositionStatus,
)
from trade_scout.execution.queue import (
    get_connection,
    get_open_paper_orders_for_sync,
    get_paper_fill_by_order_id_and_role,
    get_paper_order_by_plan_id_and_role,
    get_paper_order_by_request_id_and_role,
    init_db,
    insert_request,
    reconcile_paper_order_from_broker,
    utc_now_iso,
)
from trade_scout.execution.reconciler import (
    derive_entry_status_from_order_status,
    normalize_broker_order_status,
)
from trade_scout.execution.worker import (
    ExecutionWorker,
    ExecutionWorkerConfig,
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
            self.assertIn("paper_positions", table_names)

            plan_columns = self._table_columns(conn, "paper_trade_plans")
            order_columns = self._table_columns(conn, "paper_orders")
            fill_columns = self._table_columns(conn, "paper_fills")
            position_columns = self._table_columns(conn, "paper_positions")
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

        self.assertIn("paper_order_id", order_columns)
        self.assertIn("plan_id", order_columns)
        self.assertIn("client_order_id", order_columns)
        self.assertIn("parent_paper_order_id", order_columns)
        self.assertIn("order_role", order_columns)
        self.assertIn("order_status", order_columns)

        self.assertIn("paper_fill_id", fill_columns)
        self.assertIn("paper_order_id", fill_columns)
        self.assertIn("broker_order_id", fill_columns)
        self.assertIn("fill_role", fill_columns)
        self.assertIn("fill_ts_utc", fill_columns)
        self.assertIn("fill_px", fill_columns)
        self.assertIn("fill_qty", fill_columns)

        self.assertIn("position_id", position_columns)
        self.assertIn("plan_id", position_columns)
        self.assertIn("position_status", position_columns)
        self.assertIn("entry_order_id", position_columns)
        self.assertIn("entry_client_order_id", position_columns)
        self.assertIn("entry_filled_qty", position_columns)
        self.assertIn("remaining_qty", position_columns)
        self.assertIn("updated_ts_utc", position_columns)
        self.assertIn("pending_action_status", position_columns)
        self.assertIn("active_stop_paper_order_id", position_columns)
        self.assertIn("stop_qty", position_columns)
        self.assertIn("stop_price", position_columns)
        self.assertIn("stop_limit_price", position_columns)
        self.assertIn("tp1_triggered_ts_utc", position_columns)
        self.assertIn("tp2_triggered_ts_utc", position_columns)
        self.assertIn("active_tp1_paper_order_id", position_columns)
        self.assertIn("active_tp2_paper_order_id", position_columns)
        self.assertIn("tp1_target_qty", position_columns)
        self.assertIn("tp2_target_qty", position_columns)
        self.assertIn("tp1_filled_qty", position_columns)
        self.assertIn("tp2_filled_qty", position_columns)
        self.assertIn("last_action_error", position_columns)

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
            "trade_scout.execution.service.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.service.submit_validated_order",
            return_value=self._broker_result("broker-1"),
        ):
            attempted = self.worker.run_once()

        self.assertEqual(1, attempted)

        conn = get_connection(self.db_path)
        try:
            plan_rows = conn.execute("SELECT * FROM paper_trade_plans").fetchall()
            order_rows = conn.execute("SELECT * FROM paper_orders").fetchall()
            position_rows = conn.execute("SELECT * FROM paper_positions").fetchall()
            request_row = conn.execute(
                "SELECT * FROM execution_requests WHERE request_id = ?",
                ("req-1",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, len(plan_rows))
        self.assertEqual(1, len(order_rows))
        self.assertEqual(1, len(position_rows))

        plan_row = plan_rows[0]
        order_row = order_rows[0]
        position_row = position_rows[0]

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
        self.assertEqual(plan_row["plan_id"], position_row["plan_id"])
        self.assertEqual("PENDING_ENTRY", position_row["position_status"])
        self.assertEqual(0.0, position_row["entry_filled_qty"])
        self.assertEqual(0.0, position_row["remaining_qty"])
        self.assertEqual("NONE", position_row["pending_action_status"])
        self.assertIsNone(position_row["opened_ts_utc"])

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
            "trade_scout.execution.service.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.service.submit_validated_order",
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
                "trade_scout.execution.service.find_existing_broker_order",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_validated_order",
                side_effect=AssertionError("submit_validated_order should not be called"),
            ):
                self.worker.process_validated_request(conn, request_row)

            plan_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_trade_plans WHERE request_id = ?",
                ("req-2",),
            ).fetchone()["n"]
            position_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_positions WHERE request_id = ?",
                ("req-2",),
            ).fetchone()["n"]
            order_count = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_orders WHERE client_order_id = ?",
                ("req-2",),
            ).fetchone()["n"]
        finally:
            conn.close()

        self.assertEqual(1, plan_count)
        self.assertEqual(1, position_count)
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
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-idem",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, row_count)
        self.assertEqual(1, fill_count)
        self.assertEqual(dict(first), dict(second))
        self.assertEqual("FILLED", plan_row["entry_status"])
        self.assertEqual(0.75, plan_row["entry_filled_qty"])
        self.assertEqual(101.25, plan_row["entry_avg_fill_px"])
        self.assertEqual("OPEN", position_row["position_status"])
        self.assertEqual(0.75, position_row["entry_filled_qty"])
        self.assertEqual(0.75, position_row["remaining_qty"])
        self.assertEqual(101.25, position_row["entry_avg_fill_px"])
        self.assertEqual("PLACE_INITIAL_STOP", position_row["pending_action_status"])

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
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
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
        self.assertEqual("OPEN", position_row["position_status"])
        self.assertEqual("po-fill-sync", position_row["entry_order_id"])
        self.assertEqual("req-fill-sync", position_row["entry_client_order_id"])
        self.assertEqual(0.8, position_row["entry_filled_qty"])
        self.assertEqual(0.8, position_row["remaining_qty"])
        self.assertEqual(103.5, position_row["entry_avg_fill_px"])
        self.assertEqual("2026-03-25T14:40:00+00:00", position_row["opened_ts_utc"])
        self.assertEqual("PLACE_INITIAL_STOP", position_row["pending_action_status"])

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
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
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
        self.assertEqual("OPEN", position_row["position_status"])
        self.assertEqual(0.25, position_row["entry_filled_qty"])
        self.assertEqual(0.25, position_row["remaining_qty"])
        self.assertEqual(102.0, position_row["entry_avg_fill_px"])
        self.assertEqual("PLACE_INITIAL_STOP", position_row["pending_action_status"])

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
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-skip-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNone(fill_row)
        self.assertEqual("PENDING", plan_row["entry_status"])
        self.assertIsNone(plan_row["entry_filled_qty"])
        self.assertEqual("PENDING_ENTRY", position_row["position_status"])
        self.assertEqual(0.0, position_row["entry_filled_qty"])
        self.assertEqual(0.0, position_row["remaining_qty"])
        self.assertEqual("NONE", position_row["pending_action_status"])

    def test_position_remaining_qty_comes_from_actual_fill_not_intended_qty(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-actual-fill")
            conn.execute(
                """
                UPDATE paper_trade_plans
                SET intended_qty = ?
                WHERE request_id = ?
                """,
                (5.0, "req-actual-fill"),
            )
            conn.commit()
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-actual-fill",
                request_id="req-actual-fill",
                client_order_id="req-actual-fill",
                broker_order_id="broker-actual-fill",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-actual-fill",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-actual-fill",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T16:00:00+00:00",
                    filled_qty=0.33,
                    filled_avg_price=106.25,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-actual-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(0.33, position_row["entry_filled_qty"])
        self.assertEqual(0.33, position_row["remaining_qty"])
        self.assertNotEqual(5.0, position_row["remaining_qty"])
        self.assertEqual("PLACE_INITIAL_STOP", position_row["pending_action_status"])
        self.assertAlmostEqual(0.165, position_row["tp1_target_qty"])
        self.assertAlmostEqual(0.165, position_row["tp2_target_qty"])

    def test_open_position_initializes_exit_allocation_from_actual_fill_truth(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-alloc-open")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-alloc-open",
                request_id="req-alloc-open",
                client_order_id="req-alloc-open",
                broker_order_id="broker-alloc-open",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-alloc-open",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-alloc-open",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T18:00:00+00:00",
                    filled_qty=0.75,
                    filled_avg_price=101.25,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-alloc-open",),
            ).fetchone()
        finally:
            conn.close()

        self.assertAlmostEqual(0.375, position_row["tp1_target_qty"])
        self.assertAlmostEqual(0.375, position_row["tp2_target_qty"])
        self.assertAlmostEqual(
            position_row["entry_filled_qty"],
            position_row["tp1_target_qty"] + position_row["tp2_target_qty"],
        )

    def test_repeated_reconciliation_does_not_reinitialize_exit_allocation(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-alloc-idem")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-alloc-idem",
                request_id="req-alloc-idem",
                client_order_id="req-alloc-idem",
                broker_order_id="broker-alloc-idem",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-alloc-idem",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                order_row,
                self._broker_result(
                    "broker-alloc-idem",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T18:10:00+00:00",
                    filled_qty=0.8,
                    filled_avg_price=100.0,
                ),
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                (0.3, 0.5, "req-alloc-idem"),
            )
            conn.commit()

            second_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-alloc-idem",),
            ).fetchone()
            self.worker.reconcile_paper_order(
                conn,
                second_order_row,
                self._broker_result(
                    "broker-alloc-idem",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T18:10:00+00:00",
                    filled_qty=0.8,
                    filled_avg_price=100.0,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-alloc-idem",),
            ).fetchone()
        finally:
            conn.close()

        self.assertAlmostEqual(0.3, position_row["tp1_target_qty"])
        self.assertAlmostEqual(0.5, position_row["tp2_target_qty"])

    def test_process_position_actions_places_initial_stop_and_links_position(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-place", stop_px=94.5)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, entry_avg_fill_px = ?,
                    opened_ts_utc = ?, pending_action_status = ?, entry_order_id = ?, entry_client_order_id = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.6,
                    0.6,
                    101.0,
                    "2026-03-25T16:20:00+00:00",
                    "PLACE_INITIAL_STOP",
                    "po-entry-stop-place",
                    "req-stop-place",
                    "req-stop-place",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                return_value=self._broker_result("broker-stop-place", broker_status_raw="new"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_order = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-stop-place", "STOP"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-place",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertIsNotNone(stop_order)
        self.assertEqual("STOP", stop_order["order_role"])
        self.assertEqual("stop:plan-req-stop-place", stop_order["client_order_id"])
        self.assertEqual(0.6, stop_order["qty"])
        self.assertEqual(94.5, stop_order["stop_price"])
        self.assertEqual(94.5, stop_order["limit_price"])
        self.assertEqual("po-entry-stop-place", stop_order["parent_paper_order_id"])
        self.assertEqual(stop_order["paper_order_id"], position_row["active_stop_paper_order_id"])
        self.assertEqual(0.6, position_row["stop_qty"])
        self.assertEqual(94.5, position_row["stop_price"])
        self.assertEqual(94.5, position_row["stop_limit_price"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["last_action_error"])

    def test_process_position_actions_is_idempotent_for_existing_local_stop(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-local", stop_px=93.0)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.4, 0.4, "PLACE_INITIAL_STOP", "req-stop-local"),
            )
            conn.commit()
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-local",
                request_id="req-stop-local",
                client_order_id="stop:plan-req-stop-local",
                broker_order_id="broker-stop-local",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-local",
                order_type="stop_limit",
                qty=0.4,
                notional=None,
                limit_price=93.0,
                stop_price=93.0,
                broker_status_raw="new",
            )

            with patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                side_effect=AssertionError("submit_protective_stop_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-stop-local", "STOP"),
            ).fetchall()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-local",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(1, len(stop_rows))
        self.assertEqual("po-stop-local", position_row["active_stop_paper_order_id"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_process_position_actions_recovers_remote_stop_by_deterministic_client_order_id(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-remote", stop_px=92.25)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.55, 0.55, "PLACE_INITIAL_STOP", "req-stop-remote"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=self._broker_result("broker-stop-remote", broker_status_raw="accepted"),
            ), patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                side_effect=AssertionError("submit_protective_stop_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_order = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-stop-remote", "STOP"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-remote",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertIsNotNone(stop_order)
        self.assertEqual("stop:plan-req-stop-remote", stop_order["client_order_id"])
        self.assertEqual("broker-stop-remote", stop_order["broker_order_id"])
        self.assertEqual("STOP", stop_order["order_role"])
        self.assertEqual(stop_order["paper_order_id"], position_row["active_stop_paper_order_id"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_process_position_actions_rearms_stop_after_tp1_for_remaining_qty(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-rearm", stop_px=91.5)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?,
                    pending_action_status = ?, stop_qty = ?, stop_price = ?, stop_limit_price = ?, entry_order_id = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.4,
                    0.8,
                    0.4,
                    "REARM_STOP_AFTER_TP1",
                    0.8,
                    91.5,
                    91.5,
                    "po-entry-rearm",
                    "req-stop-rearm",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                return_value=self._broker_result("broker-stop-rearm", broker_status_raw="new"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_order = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-stop-rearm", "STOP"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-rearm",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertIsNotNone(stop_order)
        self.assertEqual("STOP", stop_order["order_role"])
        self.assertEqual("stop:rearm:tp1:plan-req-stop-rearm", stop_order["client_order_id"])
        self.assertEqual(0.4, stop_order["qty"])
        self.assertEqual(91.5, stop_order["stop_price"])
        self.assertEqual(91.5, stop_order["limit_price"])
        self.assertEqual(stop_order["paper_order_id"], position_row["active_stop_paper_order_id"])
        self.assertEqual(0.4, position_row["stop_qty"])
        self.assertEqual(91.5, position_row["stop_price"])
        self.assertEqual(91.5, position_row["stop_limit_price"])
        self.assertEqual("PARTIAL", position_row["position_status"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["last_action_error"])

    def test_rearmed_stop_local_reuse_is_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-rearm-local", stop_px=90.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-rearm-local",
                request_id="req-stop-rearm-local",
                client_order_id="stop:rearm:tp1:plan-req-stop-rearm-local",
                broker_order_id="broker-stop-rearm-local",
                order_status="SUBMITTED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.35,
                notional=None,
                limit_price=90.0,
                stop_price=90.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?,
                    pending_action_status = ?
                WHERE request_id = ?
                """,
                ("PARTIAL", 0.35, 0.7, 0.35, "REARM_STOP_AFTER_TP1", "req-stop-rearm-local"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                side_effect=AssertionError("submit_protective_stop_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND client_order_id = ?",
                ("req-stop-rearm-local", "stop:rearm:tp1:plan-req-stop-rearm-local"),
            ).fetchall()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-rearm-local",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(1, len(stop_rows))
        self.assertEqual("po-stop-rearm-local", position_row["active_stop_paper_order_id"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_rearmed_stop_remote_recovery_by_deterministic_client_order_id(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-rearm-remote", stop_px=89.75)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?,
                    pending_action_status = ?
                WHERE request_id = ?
                """,
                ("PARTIAL", 0.25, 0.5, 0.25, "REARM_STOP_AFTER_TP1", "req-stop-rearm-remote"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=self._broker_result("broker-stop-rearm-remote", broker_status_raw="accepted"),
            ), patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                side_effect=AssertionError("submit_protective_stop_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_order = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND client_order_id = ?",
                ("req-stop-rearm-remote", "stop:rearm:tp1:plan-req-stop-rearm-remote"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-rearm-remote",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertIsNotNone(stop_order)
        self.assertEqual("broker-stop-rearm-remote", stop_order["broker_order_id"])
        self.assertEqual("STOP", stop_order["order_role"])
        self.assertEqual(stop_order["paper_order_id"], position_row["active_stop_paper_order_id"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_rearm_stop_processing_does_not_duplicate_stop_or_submit_tp2(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-stop-rearm-repeat", stop_px=88.0)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?,
                    pending_action_status = ?
                WHERE request_id = ?
                """,
                ("PARTIAL", 0.2, 0.4, 0.2, "REARM_STOP_AFTER_TP1", "req-stop-rearm-repeat"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                return_value=self._broker_result("broker-stop-rearm-repeat", broker_status_raw="new"),
            ):
                self.worker.service.process_position_actions(conn, limit=10)
                self.worker.service.process_position_actions(conn, limit=10)

            stop_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-stop-rearm-repeat", "STOP"),
            ).fetchall()
            roles = [
                row["order_role"]
                for row in conn.execute(
                    "SELECT order_role FROM paper_orders WHERE request_id = ? ORDER BY order_role",
                    ("req-stop-rearm-repeat",),
                ).fetchall()
            ]
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-stop-rearm-repeat",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, len(stop_rows))
        self.assertEqual(["STOP"], roles)
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_tp1_trigger_records_timestamp_and_schedules_cancel_stop_for_tp1(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1", stop_px=92.0)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.7, 0.7, "NONE", "req-tp1"),
            )
            conn.commit()

            processed = self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 110.5},
                now_utc=datetime(2026, 3, 25, 17, 0, tzinfo=timezone.utc),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual("2026-03-25T17:00:00+00:00", position_row["tp1_triggered_ts_utc"])
        self.assertIsNone(position_row["tp2_triggered_ts_utc"])
        self.assertEqual(PaperPositionActionStatus.CANCEL_STOP_FOR_TP1.value, position_row["pending_action_status"])

    def test_tp1_trigger_is_recorded_only_once_across_repeated_evaluations(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-once")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.7, 0.7, "NONE", "req-tp1-once"),
            )
            conn.commit()

            self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 111.0},
                now_utc=datetime(2026, 3, 25, 17, 1, tzinfo=timezone.utc),
            )
            first_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-once",),
            ).fetchone()
            self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 112.0},
                now_utc=datetime(2026, 3, 25, 17, 2, tzinfo=timezone.utc),
            )
            second_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-once",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(first_row["tp1_triggered_ts_utc"], second_row["tp1_triggered_ts_utc"])
        self.assertEqual(PaperPositionActionStatus.CANCEL_STOP_FOR_TP1.value, second_row["pending_action_status"])

    def test_tp2_is_not_recorded_before_tp1_in_default_sequencing(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp-seq")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.5, 0.5, "NONE", "req-tp-seq"),
            )
            conn.commit()

            self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 116.0},
                now_utc=datetime(2026, 3, 25, 17, 3, tzinfo=timezone.utc),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp-seq",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(position_row["tp1_triggered_ts_utc"])
        self.assertIsNone(position_row["tp2_triggered_ts_utc"])
        self.assertEqual(PaperPositionActionStatus.CANCEL_STOP_FOR_TP1.value, position_row["pending_action_status"])

    def test_tp2_records_after_tp1_and_schedules_cancel_stop_for_tp2(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?,
                    tp1_triggered_ts_utc = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.45, 0.45, "NONE", "2026-03-25T17:04:00+00:00", "req-tp2"),
            )
            conn.commit()

            processed = self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 116.0},
                now_utc=datetime(2026, 3, 25, 17, 5, tzinfo=timezone.utc),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual("2026-03-25T17:04:00+00:00", position_row["tp1_triggered_ts_utc"])
        self.assertEqual("2026-03-25T17:05:00+00:00", position_row["tp2_triggered_ts_utc"])
        self.assertEqual(PaperPositionActionStatus.CANCEL_STOP_FOR_TP2.value, position_row["pending_action_status"])

    def test_positions_with_pending_actions_are_not_retriggered(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-trigger-pending")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.6, 0.6, "PLACE_INITIAL_STOP", "req-trigger-pending"),
            )
            conn.commit()

            processed = self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 111.0},
                now_utc=datetime(2026, 3, 25, 17, 6, tzinfo=timezone.utc),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-trigger-pending",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(0, processed)
        self.assertIsNone(position_row["tp1_triggered_ts_utc"])
        self.assertEqual("PLACE_INITIAL_STOP", position_row["pending_action_status"])

    def test_closed_error_and_zero_remaining_positions_are_ignored_for_triggers(self) -> None:
        conn = get_connection(self.db_path)
        try:
            for request_id, status, remaining_qty in (
                ("req-trigger-closed", "CLOSED", 0.5),
                ("req-trigger-error", "ERROR", 0.5),
                ("req-trigger-flat", "OPEN", 0.0),
            ):
                self._insert_trade_plan_row(conn, request_id=request_id)
                conn.execute(
                    """
                    UPDATE paper_positions
                    SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                    WHERE request_id = ?
                    """,
                    (status, remaining_qty, remaining_qty, "NONE", request_id),
                )
            conn.commit()

            processed = self.worker.service.process_position_triggers(
                conn,
                price_by_symbol={"AAPL": 111.0},
                now_utc=datetime(2026, 3, 25, 17, 7, tzinfo=timezone.utc),
            )
            rows = conn.execute(
                """
                SELECT request_id, tp1_triggered_ts_utc, pending_action_status
                FROM paper_positions
                WHERE request_id IN (?, ?, ?)
                ORDER BY request_id
                """,
                ("req-trigger-closed", "req-trigger-error", "req-trigger-flat"),
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(0, processed)
        for row in rows:
            self.assertIsNone(row["tp1_triggered_ts_utc"])
            self.assertEqual("NONE", row["pending_action_status"])

    def test_trigger_evaluation_does_not_submit_tp_orders(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-no-tp-submit")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.75, 0.75, "NONE", "req-no-tp-submit"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.submit_protective_stop_order",
                side_effect=AssertionError("submit_protective_stop_order should not be called"),
            ), patch(
                "trade_scout.execution.service.submit_validated_order",
                side_effect=AssertionError("submit_validated_order should not be called"),
            ):
                processed = self.worker.service.process_position_triggers(
                    conn,
                    price_by_symbol={"AAPL": 111.0},
                    now_utc=datetime(2026, 3, 25, 17, 8, tzinfo=timezone.utc),
                )

            order_roles = [
                row["order_role"]
                for row in conn.execute(
                    "SELECT order_role FROM paper_orders WHERE request_id = ? ORDER BY order_role",
                    ("req-no-tp-submit",),
                ).fetchall()
            ]
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-no-tp-submit",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual([], order_roles)
        self.assertEqual(PaperPositionActionStatus.CANCEL_STOP_FOR_TP1.value, position_row["pending_action_status"])

    def test_tp1_action_cancels_stop_and_submits_tp1_order(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-action", stop_px=92.0, t1_px=110.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-tp1-action",
                request_id="req-tp1-action",
                client_order_id="stop:plan-req-tp1-action",
                broker_order_id="broker-stop-tp1-action",
                order_status="ACCEPTED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.8,
                notional=None,
                limit_price=92.0,
                stop_price=92.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_triggered_ts_utc = ?,
                    pending_action_status = ?, active_stop_paper_order_id = ?, stop_qty = ?, stop_price = ?, stop_limit_price = ?,
                    tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.8,
                    0.8,
                    "2026-03-25T17:10:00+00:00",
                    "CANCEL_STOP_FOR_TP1",
                    "po-stop-tp1-action",
                    0.8,
                    92.0,
                    92.0,
                    0.4,
                    0.4,
                    "req-tp1-action",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.cancel_broker_order",
                return_value=self._broker_result(
                    "broker-stop-tp1-action",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-25T17:11:00+00:00",
                ),
            ) as cancel_mock, patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp1-action", broker_status_raw="new"),
            ) as tp1_submit_mock:
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-tp1-action",),
            ).fetchone()
            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp1-action", "TP1"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-action",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(1, cancel_mock.call_count)
        self.assertEqual(1, tp1_submit_mock.call_count)
        self.assertEqual("CANCELED", stop_row["order_status"])
        self.assertEqual("TP1", tp1_row["order_role"])
        self.assertEqual("tp1:plan-req-tp1-action", tp1_row["client_order_id"])
        self.assertEqual(0.4, tp1_row["qty"])
        self.assertEqual(110.0, tp1_row["limit_price"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["active_stop_paper_order_id"])
        self.assertEqual(tp1_row["paper_order_id"], position_row["active_tp1_paper_order_id"])
        self.assertEqual(0.4, position_row["tp1_target_qty"])
        self.assertEqual(0.4, position_row["tp2_target_qty"])

    def test_local_stop_already_canceled_still_advances_to_tp1_submission(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-stop-canceled")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-canceled",
                request_id="req-tp1-stop-canceled",
                client_order_id="stop:plan-req-tp1-stop-canceled",
                broker_order_id="broker-stop-canceled",
                order_status="CANCELED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.5,
                notional=None,
                limit_price=95.0,
                stop_price=95.0,
                broker_status_raw="canceled",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_triggered_ts_utc = ?,
                    pending_action_status = ?, active_stop_paper_order_id = ?, tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.5,
                    0.5,
                    "2026-03-25T17:12:00+00:00",
                    "CANCEL_STOP_FOR_TP1",
                    "po-stop-canceled",
                    0.25,
                    0.25,
                    "req-tp1-stop-canceled",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.cancel_broker_order",
                side_effect=AssertionError("cancel_broker_order should not be called"),
            ), patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp1-stop-canceled", broker_status_raw="new"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-stop-canceled",),
            ).fetchone()
            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp1-stop-canceled", "TP1"),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])
        self.assertIsNotNone(tp1_row)
        self.assertEqual(0.25, tp1_row["qty"])

    def test_local_tp1_reuse_is_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-local-reuse")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-local-reuse",
                request_id="req-tp1-local-reuse",
                client_order_id="tp1:plan-req-tp1-local-reuse",
                broker_order_id="broker-tp1-local-reuse",
                order_status="SUBMITTED",
                order_role="TP1",
                order_type="limit",
                qty=0.6,
                notional=None,
                limit_price=110.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?,
                    tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.6, 0.6, "SUBMIT_TP1", 0.3, 0.3, "req-tp1-local-reuse"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.submit_take_profit_order",
                side_effect=AssertionError("submit_take_profit_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            tp1_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp1-local-reuse", "TP1"),
            ).fetchall()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-local-reuse",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(1, len(tp1_rows))
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])
        self.assertEqual("po-tp1-local-reuse", position_row["active_tp1_paper_order_id"])

    def test_remote_tp1_recovery_by_deterministic_client_order_id(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-remote")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?,
                    tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.55, 0.55, "SUBMIT_TP1", 0.275, 0.275, "req-tp1-remote"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=self._broker_result("broker-tp1-remote", broker_status_raw="accepted"),
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                side_effect=AssertionError("submit_take_profit_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp1-remote", "TP1"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-remote",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual("tp1:plan-req-tp1-remote", tp1_row["client_order_id"])
        self.assertEqual("broker-tp1-remote", tp1_row["broker_order_id"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])
        self.assertEqual(0.275, tp1_row["qty"])

    def test_tp1_submission_uses_tp1_target_qty_not_full_remaining_qty(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-target-qty")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?,
                    tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.8, 0.8, "SUBMIT_TP1", 0.3, 0.5, "req-tp1-target-qty"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp1-target-qty", broker_status_raw="new"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp1-target-qty", "TP1"),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(0.3, tp1_row["qty"])

    def test_repeated_tp1_action_processing_does_not_duplicate_tp1_orders(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-repeat")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-repeat",
                request_id="req-tp1-repeat",
                client_order_id="stop:plan-req-tp1-repeat",
                broker_order_id="broker-stop-repeat",
                order_status="CANCELED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.45,
                notional=None,
                limit_price=95.0,
                stop_price=95.0,
                broker_status_raw="canceled",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_triggered_ts_utc = ?,
                    pending_action_status = ?, active_stop_paper_order_id = ?, tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.45,
                    0.45,
                    "2026-03-25T17:13:00+00:00",
                    "CANCEL_STOP_FOR_TP1",
                    "po-stop-repeat",
                    0.225,
                    0.225,
                    "req-tp1-repeat",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp1-repeat", broker_status_raw="new"),
            ):
                self.worker.service.process_position_actions(conn, limit=10)
                self.worker.service.process_position_actions(conn, limit=10)

            tp1_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp1-repeat", "TP1"),
            ).fetchall()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-repeat",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, len(tp1_rows))
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["active_stop_paper_order_id"])

    def test_tp1_action_does_not_submit_tp2_or_rearm_stop(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-no-tp2")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-no-tp2",
                request_id="req-tp1-no-tp2",
                client_order_id="stop:plan-req-tp1-no-tp2",
                broker_order_id="broker-stop-no-tp2",
                order_status="CANCELED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.5,
                notional=None,
                limit_price=95.0,
                stop_price=95.0,
                broker_status_raw="canceled",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?,
                    tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.5, 0.5, "SUBMIT_TP1", 0.25, 0.25, "req-tp1-no-tp2"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp1-no-tp2", broker_status_raw="new"),
            ):
                self.worker.service.process_position_actions(conn, limit=10)

            roles = [
                row["order_role"]
                for row in conn.execute(
                    "SELECT order_role FROM paper_orders WHERE request_id = ? ORDER BY order_role",
                    ("req-tp1-no-tp2",),
                ).fetchall()
            ]
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-no-tp2",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(["STOP", "TP1"], roles)
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])

    def test_tp2_action_cancels_stop_and_submits_tp2_order(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-action", stop_px=92.0, t2_px=115.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-tp2-action",
                request_id="req-tp2-action",
                client_order_id="stop:rearm:tp1:plan-req-tp2-action",
                broker_order_id="broker-stop-tp2-action",
                order_status="ACCEPTED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.4,
                notional=None,
                limit_price=92.0,
                stop_price=92.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?, tp2_triggered_ts_utc = ?,
                    pending_action_status = ?, active_stop_paper_order_id = ?, stop_qty = ?, stop_price = ?, stop_limit_price = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.4,
                    0.8,
                    0.4,
                    "2026-03-25T20:00:00+00:00",
                    "CANCEL_STOP_FOR_TP2",
                    "po-stop-tp2-action",
                    0.4,
                    92.0,
                    92.0,
                    "req-tp2-action",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.cancel_broker_order",
                return_value=self._broker_result(
                    "broker-stop-tp2-action",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-25T20:01:00+00:00",
                ),
            ) as cancel_mock, patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp2-action", broker_status_raw="new"),
            ) as tp2_submit_mock:
                processed = self.worker.service.process_position_actions(conn, limit=10)

            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-tp2-action",),
            ).fetchone()
            tp2_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp2-action", "TP2"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-action",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(1, cancel_mock.call_count)
        self.assertEqual(1, tp2_submit_mock.call_count)
        self.assertEqual("CANCELED", stop_row["order_status"])
        self.assertEqual("TP2", tp2_row["order_role"])
        self.assertEqual("tp2:plan-req-tp2-action", tp2_row["client_order_id"])
        self.assertEqual(0.4, tp2_row["qty"])
        self.assertEqual(115.0, tp2_row["limit_price"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["active_stop_paper_order_id"])
        self.assertEqual(tp2_row["paper_order_id"], position_row["active_tp2_paper_order_id"])

    def test_local_stop_already_terminal_still_advances_to_tp2_submission(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-stop-canceled", t2_px=115.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-tp2-canceled",
                request_id="req-tp2-stop-canceled",
                client_order_id="stop:rearm:tp1:plan-req-tp2-stop-canceled",
                broker_order_id="broker-stop-tp2-canceled",
                order_status="CANCELED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.25,
                notional=None,
                limit_price=95.0,
                stop_price=95.0,
                broker_status_raw="canceled",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?,
                    tp2_triggered_ts_utc = ?, pending_action_status = ?, active_stop_paper_order_id = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.25,
                    0.5,
                    0.25,
                    "2026-03-25T20:02:00+00:00",
                    "CANCEL_STOP_FOR_TP2",
                    "po-stop-tp2-canceled",
                    "req-tp2-stop-canceled",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.cancel_broker_order",
                side_effect=AssertionError("cancel_broker_order should not be called"),
            ), patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp2-stop-canceled", broker_status_raw="new"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-stop-canceled",),
            ).fetchone()
            tp2_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp2-stop-canceled", "TP2"),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, position_row["pending_action_status"])
        self.assertIsNotNone(tp2_row)
        self.assertEqual(0.25, tp2_row["qty"])

    def test_local_tp2_reuse_is_idempotent(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-local-reuse", t2_px=115.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-local-reuse",
                request_id="req-tp2-local-reuse",
                client_order_id="tp2:plan-req-tp2-local-reuse",
                broker_order_id="broker-tp2-local-reuse",
                order_status="SUBMITTED",
                order_role="TP2",
                order_type="limit",
                qty=0.3,
                notional=None,
                limit_price=115.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("PARTIAL", 0.3, 0.6, 0.3, "SUBMIT_TP2", "req-tp2-local-reuse"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.submit_take_profit_order",
                side_effect=AssertionError("submit_take_profit_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            tp2_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp2-local-reuse", "TP2"),
            ).fetchall()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-local-reuse",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual(1, len(tp2_rows))
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, position_row["pending_action_status"])
        self.assertEqual("po-tp2-local-reuse", position_row["active_tp2_paper_order_id"])

    def test_remote_tp2_recovery_by_deterministic_client_order_id(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-remote", t2_px=116.0)
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("PARTIAL", 0.275, 0.55, 0.275, "SUBMIT_TP2", "req-tp2-remote"),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=self._broker_result("broker-tp2-remote", broker_status_raw="accepted"),
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                side_effect=AssertionError("submit_take_profit_order should not be called"),
            ):
                processed = self.worker.service.process_position_actions(conn, limit=10)

            tp2_row = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp2-remote", "TP2"),
            ).fetchone()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-remote",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, processed)
        self.assertEqual("tp2:plan-req-tp2-remote", tp2_row["client_order_id"])
        self.assertEqual("broker-tp2-remote", tp2_row["broker_order_id"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, position_row["pending_action_status"])
        self.assertEqual(0.275, tp2_row["qty"])

    def test_repeated_tp2_action_processing_does_not_duplicate_tp2_orders(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-repeat", t2_px=116.0)
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-tp2-repeat",
                request_id="req-tp2-repeat",
                client_order_id="stop:rearm:tp1:plan-req-tp2-repeat",
                broker_order_id="broker-stop-tp2-repeat",
                order_status="CANCELED",
                order_role="STOP",
                order_type="stop_limit",
                qty=0.225,
                notional=None,
                limit_price=95.0,
                stop_price=95.0,
                broker_status_raw="canceled",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, tp1_filled_qty = ?, tp2_triggered_ts_utc = ?,
                    pending_action_status = ?, active_stop_paper_order_id = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.225,
                    0.45,
                    0.225,
                    "2026-03-25T20:03:00+00:00",
                    "CANCEL_STOP_FOR_TP2",
                    "po-stop-tp2-repeat",
                    "req-tp2-repeat",
                ),
            )
            conn.commit()

            with patch(
                "trade_scout.execution.service.find_existing_broker_order_by_client_order_id",
                return_value=None,
            ), patch(
                "trade_scout.execution.service.submit_take_profit_order",
                return_value=self._broker_result("broker-tp2-repeat", broker_status_raw="new"),
            ):
                self.worker.service.process_position_actions(conn, limit=10)
                self.worker.service.process_position_actions(conn, limit=10)

            tp2_rows = conn.execute(
                "SELECT * FROM paper_orders WHERE request_id = ? AND order_role = ?",
                ("req-tp2-repeat", "TP2"),
            ).fetchall()
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-repeat",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(1, len(tp2_rows))
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["active_stop_paper_order_id"])

    def test_tp2_fill_updates_position_truth_and_closes_position(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-fill")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-fill",
                request_id="req-tp2-fill",
                client_order_id="tp2:plan-req-tp2-fill",
                broker_order_id="broker-tp2-fill",
                order_status="SUBMITTED",
                order_role="TP2",
                order_type="limit",
                qty=0.4,
                notional=None,
                limit_price=115.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp2_paper_order_id = ?, tp1_filled_qty = ?, tp2_filled_qty = ?, active_stop_paper_order_id = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.8,
                    0.4,
                    "WAIT_TP2_FILL",
                    "po-tp2-fill",
                    0.4,
                    0.0,
                    "po-stop-stale",
                    "req-tp2-fill",
                ),
            )
            conn.commit()
            tp2_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-fill",),
            ).fetchone()

            reconciled = self.worker.reconcile_paper_order(
                conn,
                tp2_order_row,
                self._broker_result(
                    "broker-tp2-fill",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T20:10:00+00:00",
                    filled_qty=0.4,
                    filled_avg_price=115.0,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertEqual(0.4, position_row["tp2_filled_qty"])
        self.assertEqual(0.0, position_row["remaining_qty"])
        self.assertEqual(PaperPositionStatus.CLOSED.value, position_row["position_status"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["active_tp2_paper_order_id"])
        self.assertIsNone(position_row["active_stop_paper_order_id"])
        self.assertEqual("2026-03-25T20:10:00+00:00", position_row["closed_ts_utc"])

    def test_tp2_partial_fill_recomputes_remaining_qty_from_total_fill_truth(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-partial")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-partial",
                request_id="req-tp2-partial",
                client_order_id="tp2:plan-req-tp2-partial",
                broker_order_id="broker-tp2-partial",
                order_status="SUBMITTED",
                order_role="TP2",
                order_type="limit",
                qty=0.4,
                notional=None,
                limit_price=115.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp2_paper_order_id = ?, tp1_filled_qty = ?, tp2_filled_qty = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.8,
                    0.4,
                    "WAIT_TP2_FILL",
                    "po-tp2-partial",
                    0.4,
                    0.0,
                    "req-tp2-partial",
                ),
            )
            conn.commit()
            tp2_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-partial",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                tp2_order_row,
                self._broker_result(
                    "broker-tp2-partial",
                    broker_status_raw="partially_filled",
                    filled_ts_utc="2026-03-25T20:12:00+00:00",
                    filled_qty=0.1,
                    filled_avg_price=115.0,
                ),
            )
            partial_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-partial",),
            ).fetchone()

            second_tp2_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-partial",),
            ).fetchone()
            self.worker.reconcile_paper_order(
                conn,
                second_tp2_order_row,
                self._broker_result(
                    "broker-tp2-partial",
                    broker_status_raw="partially_filled",
                    filled_ts_utc="2026-03-25T20:12:00+00:00",
                    filled_qty=0.1,
                    filled_avg_price=115.0,
                ),
            )
            repeated_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-partial",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(0.1, partial_row["tp2_filled_qty"])
        self.assertEqual(0.3, partial_row["remaining_qty"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, partial_row["pending_action_status"])
        self.assertEqual("po-tp2-partial", partial_row["active_tp2_paper_order_id"])
        self.assertEqual(PaperPositionStatus.PARTIAL.value, partial_row["position_status"])
        self.assertEqual(0.1, repeated_row["tp2_filled_qty"])
        self.assertEqual(0.3, repeated_row["remaining_qty"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, repeated_row["pending_action_status"])

    def test_wait_tp2_fill_remains_when_tp2_has_no_fill_yet(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-wait")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-wait",
                request_id="req-tp2-wait",
                client_order_id="tp2:plan-req-tp2-wait",
                broker_order_id="broker-tp2-wait",
                order_status="SUBMITTED",
                order_role="TP2",
                order_type="limit",
                qty=0.2,
                notional=None,
                limit_price=115.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp2_paper_order_id = ?, tp1_filled_qty = ?, tp2_filled_qty = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.8,
                    0.2,
                    "WAIT_TP2_FILL",
                    "po-tp2-wait",
                    0.6,
                    0.0,
                    "req-tp2-wait",
                ),
            )
            conn.commit()
            tp2_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-wait",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                tp2_order_row,
                self._broker_result(
                    "broker-tp2-wait",
                    broker_status_raw="accepted",
                    filled_qty=None,
                    filled_avg_price=None,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-wait",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(PaperPositionActionStatus.WAIT_TP2_FILL.value, position_row["pending_action_status"])
        self.assertEqual(PaperPositionStatus.PARTIAL.value, position_row["position_status"])
        self.assertEqual(0.2, position_row["remaining_qty"])
        self.assertEqual("po-tp2-wait", position_row["active_tp2_paper_order_id"])

    def test_tp2_terminal_no_fill_clears_active_link_without_closing(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp2-no-fill")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-no-fill",
                request_id="req-tp2-no-fill",
                client_order_id="tp2:plan-req-tp2-no-fill",
                broker_order_id="broker-tp2-no-fill",
                order_status="SUBMITTED",
                order_role="TP2",
                order_type="limit",
                qty=0.2,
                notional=None,
                limit_price=115.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp2_paper_order_id = ?, tp1_filled_qty = ?, tp2_filled_qty = ?
                WHERE request_id = ?
                """,
                (
                    "PARTIAL",
                    0.8,
                    0.2,
                    "WAIT_TP2_FILL",
                    "po-tp2-no-fill",
                    0.6,
                    0.0,
                    "req-tp2-no-fill",
                ),
            )
            conn.commit()
            tp2_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-no-fill",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                tp2_order_row,
                self._broker_result(
                    "broker-tp2-no-fill",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-25T20:15:00+00:00",
                    filled_qty=0.0,
                    filled_avg_price=None,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp2-no-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNone(position_row["active_tp2_paper_order_id"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])
        self.assertEqual(PaperPositionStatus.PARTIAL.value, position_row["position_status"])

    def test_tp1_fill_updates_position_truth_and_schedules_rearm(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-fill")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-fill",
                request_id="req-tp1-fill",
                client_order_id="tp1:plan-req-tp1-fill",
                broker_order_id="broker-tp1-fill",
                order_status="SUBMITTED",
                order_role="TP1",
                order_type="limit",
                qty=0.4,
                notional=None,
                limit_price=110.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp1_paper_order_id = ?, tp1_target_qty = ?, tp2_target_qty = ?, tp1_filled_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.8,
                    0.8,
                    "WAIT_TP1_FILL",
                    "po-tp1-fill",
                    0.4,
                    0.4,
                    0.0,
                    "req-tp1-fill",
                ),
            )
            conn.commit()
            tp1_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-fill",),
            ).fetchone()

            reconciled = self.worker.reconcile_paper_order(
                conn,
                tp1_order_row,
                self._broker_result(
                    "broker-tp1-fill",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-25T19:00:00+00:00",
                    filled_qty=0.4,
                    filled_avg_price=110.0,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertEqual(0.4, position_row["tp1_filled_qty"])
        self.assertEqual(0.4, position_row["remaining_qty"])
        self.assertEqual("PARTIAL", position_row["position_status"])
        self.assertEqual(PaperPositionActionStatus.REARM_STOP_AFTER_TP1.value, position_row["pending_action_status"])
        self.assertIsNone(position_row["active_tp1_paper_order_id"])

    def test_tp1_partial_fill_recomputes_remaining_qty_from_total_fill_truth(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-partial")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-partial",
                request_id="req-tp1-partial",
                client_order_id="tp1:plan-req-tp1-partial",
                broker_order_id="broker-tp1-partial",
                order_status="SUBMITTED",
                order_role="TP1",
                order_type="limit",
                qty=0.4,
                notional=None,
                limit_price=110.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp1_paper_order_id = ?, tp1_target_qty = ?, tp2_target_qty = ?, tp1_filled_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.8,
                    0.8,
                    "WAIT_TP1_FILL",
                    "po-tp1-partial",
                    0.4,
                    0.4,
                    0.0,
                    "req-tp1-partial",
                ),
            )
            conn.commit()
            tp1_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-partial",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                tp1_order_row,
                self._broker_result(
                    "broker-tp1-partial",
                    broker_status_raw="partially_filled",
                    filled_ts_utc="2026-03-25T19:10:00+00:00",
                    filled_qty=0.2,
                    filled_avg_price=110.0,
                ),
            )
            partial_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-partial",),
            ).fetchone()

            second_tp1_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-partial",),
            ).fetchone()
            self.worker.reconcile_paper_order(
                conn,
                second_tp1_order_row,
                self._broker_result(
                    "broker-tp1-partial",
                    broker_status_raw="partially_filled",
                    filled_ts_utc="2026-03-25T19:10:00+00:00",
                    filled_qty=0.2,
                    filled_avg_price=110.0,
                ),
            )
            repeated_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-partial",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(0.2, partial_row["tp1_filled_qty"])
        self.assertEqual(0.6, partial_row["remaining_qty"])
        self.assertEqual("PARTIAL", partial_row["position_status"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, partial_row["pending_action_status"])
        self.assertEqual("po-tp1-partial", partial_row["active_tp1_paper_order_id"])
        self.assertEqual(0.2, repeated_row["tp1_filled_qty"])
        self.assertEqual(0.6, repeated_row["remaining_qty"])
        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, repeated_row["pending_action_status"])
        self.assertEqual("po-tp1-partial", repeated_row["active_tp1_paper_order_id"])

    def test_wait_tp1_fill_remains_when_tp1_has_no_fill_yet(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-wait")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-wait",
                request_id="req-tp1-wait",
                client_order_id="tp1:plan-req-tp1-wait",
                broker_order_id="broker-tp1-wait",
                order_status="SUBMITTED",
                order_role="TP1",
                order_type="limit",
                qty=0.4,
                notional=None,
                limit_price=110.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp1_paper_order_id = ?, tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.8,
                    0.8,
                    "WAIT_TP1_FILL",
                    "po-tp1-wait",
                    0.4,
                    0.4,
                    "req-tp1-wait",
                ),
            )
            conn.commit()
            tp1_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-wait",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                tp1_order_row,
                self._broker_result(
                    "broker-tp1-wait",
                    broker_status_raw="accepted",
                    filled_qty=None,
                    filled_avg_price=None,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-wait",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(PaperPositionActionStatus.WAIT_TP1_FILL.value, position_row["pending_action_status"])
        self.assertEqual("OPEN", position_row["position_status"])
        self.assertEqual(0.8, position_row["remaining_qty"])

    def test_tp1_terminal_no_fill_clears_active_link_without_rearm(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-tp1-no-fill")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-no-fill",
                request_id="req-tp1-no-fill",
                client_order_id="tp1:plan-req-tp1-no-fill",
                broker_order_id="broker-tp1-no-fill",
                order_status="SUBMITTED",
                order_role="TP1",
                order_type="limit",
                qty=0.4,
                notional=None,
                limit_price=110.0,
                broker_status_raw="new",
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, entry_filled_qty = ?, remaining_qty = ?, pending_action_status = ?,
                    active_tp1_paper_order_id = ?, tp1_target_qty = ?, tp2_target_qty = ?
                WHERE request_id = ?
                """,
                (
                    "OPEN",
                    0.8,
                    0.8,
                    "WAIT_TP1_FILL",
                    "po-tp1-no-fill",
                    0.4,
                    0.4,
                    "req-tp1-no-fill",
                ),
            )
            conn.commit()
            tp1_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-no-fill",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                tp1_order_row,
                self._broker_result(
                    "broker-tp1-no-fill",
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-25T19:20:00+00:00",
                    filled_qty=0.0,
                    filled_avg_price=None,
                ),
            )
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-tp1-no-fill",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNone(position_row["active_tp1_paper_order_id"])
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_repeated_reconciliation_does_not_reset_pending_action_scaffolding(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-action-idem")
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-action-idem",
                request_id="req-action-idem",
                client_order_id="req-action-idem",
                broker_order_id="broker-action-idem",
                order_status="SUBMITTED",
            )
            order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-action-idem",),
            ).fetchone()
            snapshot = self._broker_result(
                "broker-action-idem",
                broker_status_raw="filled",
                filled_ts_utc="2026-03-25T16:10:00+00:00",
                filled_qty=0.4,
                filled_avg_price=107.0,
            )

            self.worker.reconcile_paper_order(conn, order_row, snapshot)
            conn.execute(
                """
                UPDATE paper_positions
                SET pending_action_status = ?, last_action_error = ?
                WHERE request_id = ?
                """,
                ("PLACE_INITIAL_STOP", "deferred", "req-action-idem"),
            )
            conn.commit()

            second_order_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-action-idem",),
            ).fetchone()
            self.worker.reconcile_paper_order(conn, second_order_row, snapshot)
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-action-idem",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("PLACE_INITIAL_STOP", position_row["pending_action_status"])
        self.assertEqual("deferred", position_row["last_action_error"])

    def test_repository_fetches_positions_with_pending_actions(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-pending-action")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", "PLACE_INITIAL_STOP", "req-pending-action"),
            )
            conn.commit()
            rows = self.worker.repository.get_positions_with_pending_actions(conn, limit=10)
        finally:
            conn.close()

        request_ids = [row["request_id"] for row in rows]
        self.assertIn("req-pending-action", request_ids)

    def test_repository_fetches_positions_eligible_for_trigger_evaluation(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(conn, request_id="req-trigger-eligible")
            conn.execute(
                """
                UPDATE paper_positions
                SET position_status = ?, remaining_qty = ?, entry_filled_qty = ?, pending_action_status = ?
                WHERE request_id = ?
                """,
                ("OPEN", 0.9, 0.9, "NONE", "req-trigger-eligible"),
            )
            conn.commit()
            rows = self.worker.repository.get_positions_eligible_for_trigger_evaluation(
                conn,
                limit=10,
            )
        finally:
            conn.close()

        request_ids = [row["request_id"] for row in rows]
        self.assertIn("req-trigger-eligible", request_ids)

    def test_phase_c_still_does_not_create_tp_orders(self) -> None:
        self._insert_pending_request(request_id="req-no-tp", alert_id="alert-no-tp")

        with patch(
            "trade_scout.execution.service.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.service.submit_validated_order",
            return_value=self._broker_result("broker-no-tp"),
        ):
            self.worker.run_once()

        conn = get_connection(self.db_path)
        try:
            roles = [
                row["order_role"]
                for row in conn.execute(
                    "SELECT order_role FROM paper_orders WHERE request_id = ? ORDER BY order_role",
                    ("req-no-tp",),
                ).fetchall()
            ]
            position_row = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
                ("req-no-tp",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(["ENTRY"], roles)
        self.assertEqual(PaperPositionActionStatus.NONE.value, position_row["pending_action_status"])

    def test_remote_recovery_still_persists_normalized_order_by_client_order_id(self) -> None:
        self._insert_pending_request(request_id="req-3", alert_id="alert-3")

        with patch(
            "trade_scout.execution.service.find_existing_broker_order",
            return_value=self._broker_result("broker-3"),
        ), patch(
            "trade_scout.execution.service.submit_validated_order",
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
            "trade_scout.execution.service.find_existing_broker_order",
            return_value=None,
        ), patch(
            "trade_scout.execution.service.submit_validated_order",
            return_value=self._broker_result("broker-4", broker_status_raw="submitted"),
        ), patch(
            "trade_scout.execution.reconciler.fetch_broker_order_by_id",
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
            synced_position = conn.execute(
                "SELECT * FROM paper_positions WHERE request_id = ?",
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
            synced_tp2_order = conn.execute(
                "SELECT * FROM paper_orders WHERE plan_id = ? AND order_role = ?",
                ("plan-req-sync", "TP2"),
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
        self.assertEqual("OPEN", synced_position["position_status"])
        self.assertEqual("PLACE_INITIAL_STOP", synced_position["pending_action_status"])
        self.assertIsNone(synced_stop_order)
        self.assertIsNone(synced_tp1_order)
        self.assertIsNone(synced_tp2_order)
        self.assertIsNotNone(new_request_order)

    @unittest.skip("legacy paper_trade_outcomes contract no longer matches current execution architecture")
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
                remaining_qty=1.0,
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
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-close",
                request_id="req-close",
                client_order_id="req-close:TP1",
                broker_order_id="broker-tp1-close",
                order_status="ACCEPTED",
                order_role="TP1",
                parent_paper_order_id="po-entry-close",
                side="SHORT",
                qty=0.5,
                limit_price=110.0,
                reduce_only=1,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-close",
                request_id="req-close",
                client_order_id="req-close:TP2",
                broker_order_id="broker-tp2-close",
                order_status="ACCEPTED",
                order_role="TP2",
                parent_paper_order_id="po-entry-close",
                side="SHORT",
                qty=0.5,
                limit_price=115.0,
                reduce_only=1,
            )
            stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-close",),
            ).fetchone()

            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                side_effect=lambda broker_order_id: self._broker_result(
                    broker_order_id,
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-26T14:10:01+00:00",
                ),
            ) as cancel_mock:
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
            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-close",),
            ).fetchone()
            tp2_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-close",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("FILLED", reconciled["order_status"])
        self.assertIsNotNone(stop_fill)
        self.assertEqual("STOP", stop_fill["fill_role"])
        self.assertEqual(2, cancel_mock.call_count)
        self.assertIsNotNone(outcome_row)
        self.assertEqual("STOP", outcome_row["exit_reason"])
        self.assertEqual(1, outcome_row["closed"])
        self.assertEqual(1.0, outcome_row["exit_qty"])
        self.assertEqual(95.0, outcome_row["exit_avg_px"])
        self.assertEqual(-5.0, outcome_row["gross_pnl"])
        self.assertEqual(-1.0, outcome_row["realized_r"])
        self.assertEqual("CANCELED", tp1_row["order_status"])
        self.assertEqual("CANCELED", tp2_row["order_status"])
        self.assertEqual("CLOSED", plan_row["position_status"])
        self.assertEqual(0.0, plan_row["remaining_qty"])
        self.assertEqual("2026-03-26T14:10:00+00:00", plan_row["closed_ts_utc"])
        summary = json.loads(outcome_row["raw_summary_json"])
        self.assertEqual(["po-stop-close:STOP"], summary["exit_fill_ids"])
        self.assertEqual(1.0, summary["total_exit_qty"])
        self.assertEqual(95.0, summary["weighted_exit_avg_px"])
        self.assertEqual(-5.0, summary["realized_gross_pnl"])
        self.assertEqual(1, len(summary["exit_legs"]))

    @unittest.skip("legacy paper_trade_outcomes contract no longer matches current execution architecture")
    def test_replacement_stop_fill_uses_actual_closing_stop_for_outcome(self) -> None:
        conn = get_connection(self.db_path)
        try:
            self._insert_trade_plan_row(
                conn,
                request_id="req-close-stop-replaced",
                entry_status="FILLED",
                entry_filled_qty=5.0,
                entry_avg_fill_px=100.0,
                entry_fill_ts_utc="2026-03-26T14:00:00+00:00",
                position_status="OPEN",
                protective_stop_order_id="po-stop-close-stop-replaced-r2",
                remaining_qty=5.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-entry-close-stop-replaced",
                request_id="req-close-stop-replaced",
                client_order_id="req-close-stop-replaced",
                broker_order_id="broker-entry-close-stop-replaced",
                order_status="FILLED",
                qty=5.0,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-close-stop-replaced",
                request_id="req-close-stop-replaced",
                client_order_id="req-close-stop-replaced:STOP",
                broker_order_id="broker-stop-close-stop-replaced",
                order_status="REPLACED",
                order_role="STOP",
                parent_paper_order_id="po-entry-close-stop-replaced",
                side="SHORT",
                qty=5.0,
                stop_price=95.0,
                superseded_by_paper_order_id="po-stop-close-stop-replaced-r2",
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-stop-close-stop-replaced-r2",
                request_id="req-close-stop-replaced",
                client_order_id="req-close-stop-replaced:STOP:R2",
                broker_order_id="broker-stop-close-stop-replaced-r2",
                order_status="SUBMITTED",
                order_role="STOP",
                parent_paper_order_id="po-entry-close-stop-replaced",
                replaced_from_paper_order_id="po-stop-close-stop-replaced",
                side="SHORT",
                qty=5.0,
                stop_price=100.05,
            )
            current_stop_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-stop-close-stop-replaced-r2",),
            ).fetchone()

            self.worker.reconcile_paper_order(
                conn,
                current_stop_row,
                self._broker_result(
                    "broker-stop-close-stop-replaced-r2",
                    broker_status_raw="filled",
                    filled_ts_utc="2026-03-26T14:10:00+00:00",
                    filled_qty=5.0,
                    filled_avg_price=100.05,
                ),
            )
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-close-stop-replaced",),
            ).fetchone()
            outcome_row = conn.execute(
                "SELECT * FROM paper_trade_outcomes WHERE plan_id = ?",
                ("plan-req-close-stop-replaced",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual("CLOSED", plan_row["position_status"])
        self.assertIsNotNone(outcome_row)
        self.assertEqual("STOP", outcome_row["exit_reason"])
        summary = json.loads(outcome_row["raw_summary_json"])
        self.assertEqual(
            ["po-stop-close-stop-replaced-r2:STOP"],
            summary["exit_fill_ids"],
        )
        self.assertEqual("STOP", summary["exit_legs"][0]["role"])
        self.assertEqual(5.0, summary["exit_legs"][0]["qty"])
        self.assertEqual(100.05, summary["exit_legs"][0]["px"])

    @unittest.skip("legacy paper_trade_outcomes contract no longer matches current execution architecture")
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

    @unittest.skip("legacy paper_trade_outcomes contract no longer matches current execution architecture")
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
                remaining_qty=1.0,
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
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp1-close-idem",
                request_id="req-close-idem",
                client_order_id="req-close-idem:TP1",
                broker_order_id="broker-tp1-close-idem",
                order_status="ACCEPTED",
                order_role="TP1",
                parent_paper_order_id="po-entry-close-idem",
                side="SHORT",
                qty=0.5,
                limit_price=110.0,
                reduce_only=1,
            )
            self._insert_paper_order_row(
                conn,
                paper_order_id="po-tp2-close-idem",
                request_id="req-close-idem",
                client_order_id="req-close-idem:TP2",
                broker_order_id="broker-tp2-close-idem",
                order_status="ACCEPTED",
                order_role="TP2",
                parent_paper_order_id="po-entry-close-idem",
                side="SHORT",
                qty=0.5,
                limit_price=115.0,
                reduce_only=1,
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
            with patch(
                "trade_scout.execution.worker.cancel_broker_order",
                side_effect=lambda broker_order_id: self._broker_result(
                    broker_order_id,
                    broker_status_raw="canceled",
                    canceled_ts_utc="2026-03-26T14:15:01+00:00",
                ),
            ) as cancel_mock:
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
            plan_row = conn.execute(
                "SELECT * FROM paper_trade_plans WHERE request_id = ?",
                ("req-close-idem",),
            ).fetchone()
            tp1_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp1-close-idem",),
            ).fetchone()
            tp2_row = conn.execute(
                "SELECT * FROM paper_orders WHERE paper_order_id = ?",
                ("po-tp2-close-idem",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(2, cancel_mock.call_count)
        self.assertEqual(1, outcome_count)
        self.assertEqual("CANCELED", tp1_row["order_status"])
        self.assertEqual("CANCELED", tp2_row["order_status"])
        self.assertEqual(0.0, plan_row["remaining_qty"])

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
        market_type: str | None = None,
        entry_status: str = "PENDING",
        stop_px: float | None = 95.0,
        t1_px: float | None = 110.0,
        t2_px: float | None = 115.0,
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
                market_type,
                strategy_profile,
                strategy_version,
                config_hash,
                strategy_id,
                setup_family,
                strategy_params_json,
                strategy_params_hash,
                signal_ts_utc,
                tier,
                entry_type,
                entry_ref_px,
                entry_low,
                entry_high,
                stop_px,
                t1_px,
                t2_px,
                timeout_seconds,
                timeout_ts_utc,
                risk_model,
                risk_usd,
                qty_model,
                intended_qty,
                intended_notional,
                plan_rules_json,
                plan_status,
                entry_status,
                entry_filled_qty,
                entry_avg_fill_px,
                entry_fill_ts_utc,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "plan-" + request_id,
                request_id,
                "alert-" + request_id,
                "2026-03-25T14:00:00+00:00",
                "AAPL",
                "LONG",
                market_type,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "MARKET",
                None,
                None,
                None,
                stop_px,
                t1_px,
                t2_px,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "{}",
                "CREATED",
                entry_status,
                None,
                None,
                None,
                "1",
            ),
        )
        conn.execute(
            """
            INSERT INTO paper_positions (
                position_id,
                plan_id,
                request_id,
                alert_id,
                symbol,
                side,
                position_status,
                entry_order_id,
                entry_client_order_id,
                entry_filled_qty,
                entry_avg_fill_px,
                opened_ts_utc,
                closed_ts_utc,
                remaining_qty,
                pending_action_status,
                active_stop_paper_order_id,
                stop_qty,
                stop_price,
                stop_limit_price,
                tp1_triggered_ts_utc,
                tp2_triggered_ts_utc,
                active_tp1_paper_order_id,
                active_tp2_paper_order_id,
                tp1_target_qty,
                tp2_target_qty,
                tp1_filled_qty,
                tp2_filled_qty,
                last_action_error,
                created_ts_utc,
                updated_ts_utc,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "position-" + request_id,
                "plan-" + request_id,
                request_id,
                "alert-" + request_id,
                "AAPL",
                "LONG",
                "PENDING_ENTRY",
                None,
                None,
                0.0,
                None,
                None,
                None,
                0.0,
                "NONE",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0.0,
                0.0,
                None,
                "2026-03-25T14:00:00+00:00",
                "2026-03-25T14:00:00+00:00",
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
        order_role: str = "ENTRY",
        parent_paper_order_id: str | None = None,
        order_type: str = "market",
        qty: float | None = None,
        notional: float | None = 25.0,
        limit_price: float | None = None,
        stop_price: float | None = None,
        broker_status_raw: str = "submitted",
        accepted_ts_utc: str | None = "2026-03-25T14:00:01+00:00",
        filled_ts_utc: str | None = None,
        side: str = "LONG",
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
                order_role,
                symbol,
                side,
                order_type,
                time_in_force,
                qty,
                notional,
                limit_price,
                stop_price,
                trail_price,
                trail_percent,
                reduce_only,
                submitted_ts_utc,
                accepted_ts_utc,
                canceled_ts_utc,
                expired_ts_utc,
                rejected_ts_utc,
                filled_ts_utc,
                broker_status_raw,
                order_status,
                status_reason,
                last_broker_update_ts_utc,
                raw_response_json,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                order_role,
                "AAPL",
                side,
                order_type,
                "gtc",
                qty,
                notional,
                limit_price,
                stop_price,
                None,
                None,
                reduce_only,
                "2026-03-25T14:00:00+00:00",
                accepted_ts_utc,
                None,
                None,
                None,
                filled_ts_utc,
                broker_status_raw,
                order_status,
                None,
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
