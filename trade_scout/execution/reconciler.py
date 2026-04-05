from __future__ import annotations

import json
from typing import Any, Callable

from trade_scout.execution.broker import fetch_broker_order_by_id
from trade_scout.execution.models import (
    BrokerOrderResult,
    PaperPositionActionStatus,
    PaperPositionStatus,
    PaperOrderRole,
    PaperOrderStatus,
    PaperTradePlanEntryStatus,
)
from trade_scout.execution.repository import ExecutionRepository


def normalize_broker_order_status(raw_status: Any) -> str:
    normalized = str(raw_status or "").strip().lower()
    if normalized in {"submitted", "pending_new"}:
        return PaperOrderStatus.SUBMITTED.value
    if normalized in {"new", "accepted", "pending_replace", "accepted_for_bidding"}:
        return PaperOrderStatus.ACCEPTED.value
    if normalized == "partially_filled":
        return PaperOrderStatus.PARTIALLY_FILLED.value
    if normalized == "filled":
        return PaperOrderStatus.FILLED.value
    if normalized in {"canceled", "cancelled"}:
        return PaperOrderStatus.CANCELED.value
    if normalized in {"rejected", "suspended", "stopped"}:
        return PaperOrderStatus.REJECTED.value
    if normalized in {"expired", "done_for_day"}:
        return PaperOrderStatus.EXPIRED.value
    return PaperOrderStatus.SUBMITTED.value


def derive_entry_status_from_order_status(order_status: str) -> str | None:
    if order_status == PaperOrderStatus.PARTIALLY_FILLED.value:
        return PaperTradePlanEntryStatus.PARTIALLY_FILLED.value
    if order_status == PaperOrderStatus.FILLED.value:
        return PaperTradePlanEntryStatus.FILLED.value
    return None


class ExecutionReconciler:
    def __init__(
        self,
        repository: ExecutionRepository,
        *,
        broker_order_fetcher: Callable[[str], BrokerOrderResult | None] | None = None,
    ):
        self.repository = repository
        self.broker_order_fetcher = broker_order_fetcher

    def sync_open_paper_orders(self, conn, *, limit: int) -> int:
        synced_count = 0
        open_orders = self.repository.get_open_paper_orders_for_sync(conn, limit=limit)
        for paper_order_row in open_orders:
            paper_order_id = str(paper_order_row["paper_order_id"])
            broker_order_id = self._coerce_optional_str(paper_order_row["broker_order_id"])
            if not broker_order_id:
                print(
                    f"[execution-worker] skipped sync paper_order_id={paper_order_id} "
                    "reason=missing broker_order_id"
                )
                continue

            try:
                broker_snapshot = self._get_broker_order_fetcher()(broker_order_id)
                if broker_snapshot is None:
                    print(
                        f"[execution-worker] skipped sync paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id} reason=broker_order_not_found"
                    )
                    continue

                reconciled = self.reconcile_paper_order(conn, paper_order_row, broker_snapshot)
                synced_count += 1
                if self._has_material_order_state_change(paper_order_row, reconciled):
                    print(
                        f"[execution-worker] synced paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id} order_status={reconciled['order_status']}"
                    )
            except Exception as exc:
                print(
                    f"[execution-worker] sync failed paper_order_id={paper_order_id} "
                    f"broker_order_id={broker_order_id} error={exc}"
                )
        return synced_count

    def reconcile_paper_order(self, conn, paper_order_row, broker_snapshot):
        normalized_status = normalize_broker_order_status(broker_snapshot.broker_status_raw)
        reconciled = self.repository.reconcile_paper_order_from_broker(
            conn,
            paper_order_id=str(paper_order_row["paper_order_id"]),
            broker_snapshot=broker_snapshot,
            order_status=normalized_status,
        )
        if reconciled is None:
            raise RuntimeError("paper order reconcile did not persist")
        self.ingest_entry_fill_if_present(conn, reconciled, broker_snapshot)
        self.apply_tp1_fill_progress_if_present(conn, reconciled, broker_snapshot)
        self.apply_tp2_fill_progress_if_present(conn, reconciled, broker_snapshot)
        return reconciled

    def ingest_entry_fill_if_present(self, conn, paper_order_row, broker_snapshot) -> None:
        if str(paper_order_row["order_role"]) != PaperOrderRole.ENTRY.value:
            return

        entry_status = derive_entry_status_from_order_status(str(paper_order_row["order_status"]))
        if entry_status is None:
            return

        fill_row = self.repository.upsert_entry_fill_from_broker_snapshot(
            conn,
            paper_order_row=paper_order_row,
            broker_snapshot=broker_snapshot,
            entry_status=entry_status,
        )
        if fill_row is None:
            return

        self.repository.update_paper_position_from_entry_fill(
            conn,
            plan_id=str(paper_order_row["plan_id"]),
            request_id=str(paper_order_row["request_id"]),
            alert_id=str(paper_order_row["alert_id"]),
            symbol=str(paper_order_row["symbol"]),
            side=str(paper_order_row["side"]),
            entry_order_id=str(paper_order_row["paper_order_id"]),
            entry_client_order_id=self._coerce_optional_str(paper_order_row["client_order_id"]),
            entry_filled_qty=float(fill_row["fill_qty"]),
            entry_avg_fill_px=float(fill_row["fill_px"]),
            opened_ts_utc=str(fill_row["fill_ts_utc"]),
        )
        self.repository.initialize_paper_position_exit_allocation(
            conn,
            plan_id=str(paper_order_row["plan_id"]),
            entry_filled_qty=float(fill_row["fill_qty"]),
        )
        self.initialize_post_entry_action_state(
            conn,
            plan_id=str(paper_order_row["plan_id"]),
        )

        print(
            f"[execution-worker] ingested fill paper_order_id={paper_order_row['paper_order_id']} "
            f"paper_fill_id={fill_row['paper_fill_id']} entry_status={entry_status}"
        )

    def initialize_post_entry_action_state(self, conn, *, plan_id: str):
        position_row = self.repository.get_paper_position_by_plan_id(conn, plan_id)
        if position_row is None:
            return None
        if str(position_row["position_status"]) not in {"OPEN", "PARTIAL"}:
            return position_row
        if str(position_row["pending_action_status"]) != PaperPositionActionStatus.NONE.value:
            return position_row
        return self.repository.update_paper_position_pending_action(
            conn,
            plan_id=plan_id,
            pending_action_status=PaperPositionActionStatus.PLACE_INITIAL_STOP.value,
        )

    def apply_tp1_fill_progress_if_present(self, conn, paper_order_row, broker_snapshot) -> None:
        if str(paper_order_row["order_role"]) != PaperOrderRole.TP1.value:
            return

        tp1_filled_qty = broker_snapshot.filled_qty
        if tp1_filled_qty is None:
            return

        tp1_order_terminal = str(paper_order_row["order_status"]) in {
            PaperOrderStatus.FILLED.value,
            PaperOrderStatus.CANCELED.value,
            PaperOrderStatus.REJECTED.value,
            PaperOrderStatus.EXPIRED.value,
        }
        position_row = self.repository.apply_tp1_fill_to_position(
            conn,
            plan_id=str(paper_order_row["plan_id"]),
            tp1_filled_qty=float(tp1_filled_qty),
            tp1_order_terminal=tp1_order_terminal,
        )
        if position_row is None:
            return

        print(
            f"[execution-worker] reconciled tp1 fill plan_id={paper_order_row['plan_id']} "
            f"tp1_filled_qty={position_row['tp1_filled_qty']} remaining_qty={position_row['remaining_qty']}"
        )

    def apply_tp2_fill_progress_if_present(self, conn, paper_order_row, broker_snapshot) -> None:
        if str(paper_order_row["order_role"]) != PaperOrderRole.TP2.value:
            return

        tp2_filled_qty = broker_snapshot.filled_qty
        if tp2_filled_qty is None:
            return

        tp2_order_terminal = str(paper_order_row["order_status"]) in {
            PaperOrderStatus.FILLED.value,
            PaperOrderStatus.CANCELED.value,
            PaperOrderStatus.REJECTED.value,
            PaperOrderStatus.EXPIRED.value,
        }
        position_row = self.repository.apply_tp2_fill_to_position(
            conn,
            plan_id=str(paper_order_row["plan_id"]),
            tp2_filled_qty=float(tp2_filled_qty),
            tp2_order_terminal=tp2_order_terminal,
            closed_ts_utc=self._coerce_optional_str(broker_snapshot.filled_ts_utc),
        )
        if position_row is None:
            return

        print(
            f"[execution-worker] reconciled tp2 fill plan_id={paper_order_row['plan_id']} "
            f"tp2_filled_qty={position_row['tp2_filled_qty']} remaining_qty={position_row['remaining_qty']} "
            f"position_status={position_row['position_status']}"
        )

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _has_material_order_state_change(cls, previous_row, current_row) -> bool:
        return cls._material_order_state_snapshot(previous_row) != cls._material_order_state_snapshot(
            current_row
        )

    @staticmethod
    def _material_order_state_snapshot(row) -> tuple[Any, ...]:
        raw_payload = ExecutionReconciler._parse_raw_response_json(row["raw_response_json"])
        return (
            row["broker_order_id"],
            row["order_status"],
            row["broker_status_raw"],
            row["accepted_ts_utc"],
            row["canceled_ts_utc"],
            row["expired_ts_utc"],
            row["rejected_ts_utc"],
            row["filled_ts_utc"],
            row["status_reason"],
            raw_payload.get("filled_qty"),
            raw_payload.get("filled_avg_price"),
        )

    @staticmethod
    def _parse_raw_response_json(raw_response_json: Any) -> dict[str, Any]:
        if raw_response_json is None:
            return {}
        try:
            parsed = json.loads(str(raw_response_json))
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _get_broker_order_fetcher(self) -> Callable[[str], BrokerOrderResult | None]:
        return self.broker_order_fetcher or fetch_broker_order_by_id
