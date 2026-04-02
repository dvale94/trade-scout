from __future__ import annotations

import sqlite3

from trade_scout.execution import queue
from trade_scout.execution.models import BrokerOrderResult


class ExecutionRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def init_db(self) -> None:
        queue.init_db(self.db_path)

    def get_connection(self) -> sqlite3.Connection:
        return queue.get_connection(self.db_path)

    def fetch_pending(self, conn: sqlite3.Connection, *, limit: int = 10) -> list[sqlite3.Row]:
        return queue.fetch_pending(conn, limit=limit)

    def fetch_validated(self, conn: sqlite3.Connection, *, limit: int = 10) -> list[sqlite3.Row]:
        return queue.fetch_validated(conn, limit=limit)

    def get_request_by_id(self, conn: sqlite3.Connection, request_id: str) -> sqlite3.Row | None:
        return queue.get_request_by_id(conn, request_id)

    def claim_request(self, conn: sqlite3.Connection, request_id: str, worker_id: str) -> bool:
        return queue.claim_request(conn, request_id, worker_id)

    def mark_validated(self, conn: sqlite3.Connection, request_id: str) -> None:
        queue.mark_validated(conn, request_id)

    def mark_rejected(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        *,
        code: str,
        reason: str,
    ) -> None:
        queue.mark_rejected(conn, request_id, code, reason)

    def mark_expired(self, conn: sqlite3.Connection, request_id: str, *, reason: str) -> None:
        queue.mark_expired(conn, request_id, reason)

    def mark_submitted(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        *,
        broker: str,
        broker_order_id: str,
        submitted_ts_utc: str,
    ) -> None:
        queue.mark_submitted(
            conn,
            request_id,
            broker=broker,
            broker_order_id=broker_order_id,
            submitted_ts_utc=submitted_ts_utc,
        )

    def mark_failed(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        *,
        code: str,
        message: str,
    ) -> None:
        queue.mark_failed(conn, request_id, code, message)

    def get_open_paper_orders_for_sync(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        return queue.get_open_paper_orders_for_sync(conn, limit=limit)

    def get_paper_trade_plan_by_request_id(
        self,
        conn: sqlite3.Connection,
        request_id: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_trade_plan_by_request_id(conn, request_id)

    def get_paper_trade_plan_by_plan_id(
        self,
        conn: sqlite3.Connection,
        plan_id: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_trade_plan_by_plan_id(conn, plan_id)

    def get_paper_position_by_plan_id(
        self,
        conn: sqlite3.Connection,
        plan_id: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_position_by_plan_id(conn, plan_id)

    def get_paper_position_by_request_id(
        self,
        conn: sqlite3.Connection,
        request_id: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_position_by_request_id(conn, request_id)

    def get_positions_with_pending_actions(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        return queue.get_positions_with_pending_actions(conn, limit=limit)

    def get_positions_eligible_for_trigger_evaluation(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        return queue.get_positions_eligible_for_trigger_evaluation(conn, limit=limit)

    def create_paper_trade_plan(self, conn: sqlite3.Connection, **kwargs) -> sqlite3.Row:
        return queue.create_paper_trade_plan(conn, **kwargs)

    def ensure_paper_position(self, conn: sqlite3.Connection, **kwargs) -> sqlite3.Row:
        return queue.ensure_paper_position(conn, **kwargs)

    def get_paper_order_by_request_id_and_role(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        order_role: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_order_by_request_id_and_role(conn, request_id, order_role)

    def get_paper_order_by_plan_id_and_role(
        self,
        conn: sqlite3.Connection,
        plan_id: str,
        order_role: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_order_by_plan_id_and_role(conn, plan_id, order_role)

    def get_paper_order_by_client_order_id(
        self,
        conn: sqlite3.Connection,
        client_order_id: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_order_by_client_order_id(conn, client_order_id)

    def insert_paper_order(self, conn: sqlite3.Connection, **kwargs) -> sqlite3.Row:
        return queue.insert_paper_order(conn, **kwargs)

    def reconcile_paper_order_from_broker(
        self,
        conn: sqlite3.Connection,
        *,
        paper_order_id: str,
        broker_snapshot: BrokerOrderResult,
        order_status: str,
    ) -> sqlite3.Row | None:
        return queue.reconcile_paper_order_from_broker(
            conn,
            paper_order_id=paper_order_id,
            broker_snapshot=broker_snapshot,
            order_status=order_status,
        )

    def get_paper_fill_by_order_id_and_role(
        self,
        conn: sqlite3.Connection,
        paper_order_id: str,
        fill_role: str,
    ) -> sqlite3.Row | None:
        return queue.get_paper_fill_by_order_id_and_role(conn, paper_order_id, fill_role)

    def upsert_entry_fill_from_broker_snapshot(
        self,
        conn: sqlite3.Connection,
        *,
        paper_order_row: sqlite3.Row,
        broker_snapshot: BrokerOrderResult,
        entry_status: str,
    ) -> sqlite3.Row | None:
        return queue.upsert_entry_fill_from_broker_snapshot(
            conn,
            paper_order_row=paper_order_row,
            broker_snapshot=broker_snapshot,
            entry_status=entry_status,
        )

    def update_trade_plan_entry_summary(
        self,
        conn: sqlite3.Connection,
        *,
        plan_id: str,
        entry_status: str,
        entry_filled_qty: float,
        entry_avg_fill_px: float,
        entry_fill_ts_utc: str,
    ) -> sqlite3.Row | None:
        return queue.update_trade_plan_entry_summary(
            conn,
            plan_id=plan_id,
            entry_status=entry_status,
            entry_filled_qty=entry_filled_qty,
            entry_avg_fill_px=entry_avg_fill_px,
            entry_fill_ts_utc=entry_fill_ts_utc,
        )

    def update_paper_position_from_entry_fill(
        self,
        conn: sqlite3.Connection,
        **kwargs,
    ) -> sqlite3.Row | None:
        return queue.update_paper_position_from_entry_fill(conn, **kwargs)

    def update_paper_position_pending_action(
        self,
        conn: sqlite3.Connection,
        *,
        plan_id: str,
        pending_action_status: str,
    ) -> sqlite3.Row | None:
        return queue.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=pending_action_status,
        )

    def update_paper_position_stop_state(
        self,
        conn: sqlite3.Connection,
        **kwargs,
    ) -> sqlite3.Row | None:
        return queue.update_paper_position_stop_state(conn, **kwargs)

    def record_paper_position_trigger(
        self,
        conn: sqlite3.Connection,
        *,
        plan_id: str,
        trigger_name: str,
        triggered_ts_utc: str,
    ) -> sqlite3.Row | None:
        return queue.record_paper_position_trigger(
            conn,
            plan_id=plan_id,
            trigger_name=trigger_name,
            triggered_ts_utc=triggered_ts_utc,
        )

    def update_paper_position_action_error(
        self,
        conn: sqlite3.Connection,
        *,
        plan_id: str,
        last_action_error: str | None,
    ) -> sqlite3.Row | None:
        return queue.update_paper_position_action_error(
            conn,
            plan_id=plan_id,
            last_action_error=last_action_error,
        )
