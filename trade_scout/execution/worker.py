from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional

from trade_scout import config
from trade_scout.execution.broker import (
    fetch_broker_order_by_id,
    find_broker_order_by_client_order_id,
    find_existing_broker_order,
    submit_stop_order,
    submit_validated_order,
)
from trade_scout.execution.policy import validate_execution_request
from trade_scout.execution.queue import (
    claim_request,
    create_trade_outcome_from_stop_fill,
    create_paper_trade_plan,
    fetch_pending,
    fetch_validated,
    get_connection,
    get_open_paper_orders_for_sync,
    get_paper_fill_by_order_id_and_role,
    get_paper_order_by_plan_id_and_role,
    get_paper_order_by_request_id_and_role,
    get_paper_trade_plan_by_request_id,
    get_trade_outcome_by_plan_id,
    get_request_by_id,
    init_db,
    insert_paper_order,
    mark_expired,
    mark_failed,
    mark_rejected,
    mark_submitted,
    mark_validated,
    reconcile_paper_order_from_broker,
    update_trade_plan_position_closed,
    update_trade_plan_position_open,
    upsert_paper_fill_from_broker_snapshot,
    upsert_entry_fill_from_broker_snapshot,
)
from trade_scout.execution.models import (
    PaperFillRole,
    PaperOrderRole,
    PaperOrderStatus,
    PaperTradePlanEntryStatus,
)


@dataclass(frozen=True)
class ExecutionWorkerConfig:
    db_path: str
    poll_interval_seconds: float = 5.0
    batch_size: int = 10
    sync_batch_size: int = 25
    max_alert_age_seconds: int = 300
    worker_id: str = "execution-worker-1"
    enabled: bool = True


class ExecutionWorker:
    def __init__(
        self,
        config: ExecutionWorkerConfig,
        *,
        alert_lookup: Callable[[str], Optional[Mapping[str, Any]]],
    ):
        self.config = config
        self.alert_lookup = alert_lookup

    def run_forever(self) -> None:
        if not self.config.enabled:
            return

        while True:
            try:
                self.run_once()
            except Exception as exc:
                print(f"[execution-worker] unexpected error: {exc}")
            time.sleep(self.config.poll_interval_seconds)

    def run_once(self) -> int:
        init_db(self.config.db_path)
        conn = get_connection(self.config.db_path)
        attempted_count = 0
        try:
            pending_rows = fetch_pending(conn, limit=self.config.batch_size)
            for pending_row in pending_rows:
                request_id = self._get_request_id(pending_row)
                if not claim_request(conn, request_id, self.config.worker_id):
                    continue

                attempted_count += 1
                print(
                    f"[execution-worker] claimed request_id={request_id} "
                    f"worker_id={self.config.worker_id}"
                )

                claimed_row = get_request_by_id(conn, request_id)
                if claimed_row is None:
                    mark_failed(
                        conn,
                        request_id,
                        code="REQUEST_NOT_FOUND",
                        message="claimed request could not be reloaded",
                    )
                    print(
                        f"[execution-worker] failed request_id={request_id} "
                        "error=claimed request could not be reloaded"
                    )
                    continue
                self.process_claimed_request(conn, claimed_row)

            remaining = max(0, self.config.batch_size - attempted_count)
            if remaining > 0:
                validated_rows = fetch_validated(conn, limit=remaining)
                for validated_row in validated_rows:
                    attempted_count += 1
                    self.process_validated_request(conn, validated_row)

            self.sync_open_paper_orders(conn, limit=self.config.sync_batch_size)
        finally:
            conn.close()
        return attempted_count

    def process_claimed_request(self, conn, request_row) -> None:
        request_id = self._get_request_id(request_row)
        alert_id = self._get_alert_id(request_row)
        status = str(request_row["status"])

        if status != "CLAIMED":
            print(
                f"[execution-worker] warning request_id={request_id} "
                f"expected_status=CLAIMED actual_status={status}"
            )
            return

        try:
            alert_row = self.alert_lookup(alert_id)
            validation = validate_execution_request(
                request_row,
                alert_row,
                max_alert_age_seconds=self.config.max_alert_age_seconds,
            )

            if validation.is_expired:
                reason = validation.rejection_reason or "execution request expired"
                mark_expired(conn, request_id, reason)
                print(f"[execution-worker] expired request_id={request_id} reason={reason}")
                return

            if not validation.is_valid:
                code = validation.rejection_code or "VALIDATION_REJECTED"
                reason = validation.rejection_reason or "execution request rejected"
                mark_rejected(conn, request_id, code, reason)
                print(
                    f"[execution-worker] rejected request_id={request_id} "
                    f"code={code} reason={reason}"
                )
                return

            mark_validated(conn, request_id)
            print(f"[execution-worker] validated request_id={request_id}")
            validated_row = get_request_by_id(conn, request_id)
            if validated_row is None:
                mark_failed(
                    conn,
                    request_id,
                    code="REQUEST_NOT_FOUND",
                    message="validated request could not be reloaded",
                )
                print(
                    f"[execution-worker] failed request_id={request_id} "
                    "error=validated request could not be reloaded"
                )
                return
            self.process_validated_request(conn, validated_row)
        except Exception as exc:
            mark_failed(conn, request_id, code="PROCESSING_ERROR", message=str(exc))
            print(f"[execution-worker] failed request_id={request_id} error={exc}")

    def sync_open_paper_orders(self, conn, *, limit: int) -> int:
        synced_count = 0
        open_orders = get_open_paper_orders_for_sync(conn, limit=limit)
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
                broker_snapshot = fetch_broker_order_by_id(broker_order_id)
                if broker_snapshot is None:
                    print(
                        f"[execution-worker] skipped sync paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id} reason=broker_order_not_found"
                    )
                    continue

                reconciled = self.reconcile_paper_order(conn, paper_order_row, broker_snapshot)
                synced_count += 1
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
        reconciled = reconcile_paper_order_from_broker(
            conn,
            paper_order_id=str(paper_order_row["paper_order_id"]),
            broker_snapshot=broker_snapshot,
            order_status=normalized_status,
        )
        if reconciled is None:
            raise RuntimeError("paper order reconcile did not persist")
        self.ingest_order_fill_if_present(conn, reconciled, broker_snapshot)
        self.handle_reconciled_order_lifecycle(conn, reconciled)
        return reconciled

    def ingest_order_fill_if_present(self, conn, paper_order_row, broker_snapshot) -> None:
        order_role = str(paper_order_row["order_role"])
        order_status = str(paper_order_row["order_status"])
        if order_role == PaperOrderRole.ENTRY.value:
            entry_status = derive_entry_status_from_order_status(order_status)
            if entry_status is None:
                return
            fill_row = upsert_entry_fill_from_broker_snapshot(
                conn,
                paper_order_row=paper_order_row,
                broker_snapshot=broker_snapshot,
                entry_status=entry_status,
            )
        elif order_role == PaperOrderRole.STOP.value and order_status == PaperOrderStatus.FILLED.value:
            fill_row = upsert_paper_fill_from_broker_snapshot(
                conn,
                paper_order_row=paper_order_row,
                broker_snapshot=broker_snapshot,
                fill_role=PaperFillRole.STOP.value,
            )
        else:
            return

        if fill_row is None:
            return

        print(
            f"[execution-worker] ingested fill paper_order_id={paper_order_row['paper_order_id']} "
            f"paper_fill_id={fill_row['paper_fill_id']} order_role={order_role}"
        )

    def handle_reconciled_order_lifecycle(self, conn, paper_order_row) -> None:
        order_role = str(paper_order_row["order_role"])
        order_status = str(paper_order_row["order_status"])
        if order_role == PaperOrderRole.ENTRY.value and order_status == PaperOrderStatus.FILLED.value:
            self.ensure_protective_stop_for_entry_fill(conn, paper_order_row)
            return
        if order_role == PaperOrderRole.STOP.value and order_status == PaperOrderStatus.FILLED.value:
            self.handle_closed_trade_from_stop_fill(conn, paper_order_row)

    def ensure_protective_stop_for_entry_fill(self, conn, entry_order_row) -> None:
        plan_row = self._get_plan_row_by_id(conn, str(entry_order_row["plan_id"]))
        if plan_row is None:
            return

        # The actual STOP row is stronger local truth than the advisory plan pointer.
        existing_stop_order = get_paper_order_by_plan_id_and_role(
            conn,
            str(plan_row["plan_id"]),
            PaperOrderRole.STOP.value,
        )
        if existing_stop_order is not None:
            update_trade_plan_position_open(
                conn,
                plan_id=str(plan_row["plan_id"]),
                protective_stop_order_id=str(existing_stop_order["paper_order_id"]),
            )
            return

        stop_price = self._coerce_float(plan_row["stop_px"])
        qty = self._coerce_float(plan_row["entry_filled_qty"])
        if stop_price is None or qty is None or qty <= 0:
            return

        client_order_id = self._protective_stop_client_order_id(str(plan_row["request_id"]))
        stop_side = opposite_trade_side(str(plan_row["side"]))
        try:
            recovered = find_broker_order_by_client_order_id(client_order_id)
        except Exception:
            recovered = None
        if recovered is not None:
            stop_order = insert_paper_order(
                conn,
                plan_id=str(plan_row["plan_id"]),
                request_id=str(plan_row["request_id"]),
                alert_id=str(plan_row["alert_id"]),
                broker=recovered.broker,
                client_order_id=client_order_id,
                symbol=str(plan_row["symbol"]),
                side=stop_side,
                qty=qty,
                notional=None,
                order_type="stop",
                time_in_force="gtc",
                order_role=PaperOrderRole.STOP.value,
                parent_paper_order_id=str(entry_order_row["paper_order_id"]),
                stop_price=stop_price,
                result=recovered,
            )
            update_trade_plan_position_open(
                conn,
                plan_id=str(plan_row["plan_id"]),
                protective_stop_order_id=str(stop_order["paper_order_id"]),
            )
            print(
                f"[execution-worker] recovered protective stop plan_id={plan_row['plan_id']} "
                f"paper_order_id={stop_order['paper_order_id']}"
            )
            return

        try:
            result = submit_stop_order(
                client_order_id=client_order_id,
                symbol=str(plan_row["symbol"]),
                side=stop_side,
                qty=qty,
                stop_price=stop_price,
            )
        except Exception as exc:
            recovered = find_broker_order_by_client_order_id(client_order_id)
            if recovered is None:
                raise exc
            result = recovered

        stop_order = insert_paper_order(
            conn,
            plan_id=str(plan_row["plan_id"]),
            request_id=str(plan_row["request_id"]),
            alert_id=str(plan_row["alert_id"]),
            broker=result.broker,
            client_order_id=client_order_id,
            symbol=str(plan_row["symbol"]),
            side=stop_side,
            qty=qty,
            notional=None,
            order_type="stop",
            time_in_force="gtc",
            order_role=PaperOrderRole.STOP.value,
            parent_paper_order_id=str(entry_order_row["paper_order_id"]),
            stop_price=stop_price,
            result=result,
        )
        update_trade_plan_position_open(
            conn,
            plan_id=str(plan_row["plan_id"]),
            protective_stop_order_id=str(stop_order["paper_order_id"]),
        )
        print(
            f"[execution-worker] submitted protective stop plan_id={plan_row['plan_id']} "
            f"paper_order_id={stop_order['paper_order_id']}"
        )

    def handle_closed_trade_from_stop_fill(self, conn, stop_order_row) -> None:
        stop_fill_row = get_paper_fill_by_order_id_and_role(
            conn,
            str(stop_order_row["paper_order_id"]),
            PaperFillRole.STOP.value,
        )
        if stop_fill_row is None:
            return

        plan_row = update_trade_plan_position_closed(
            conn,
            plan_id=str(stop_order_row["plan_id"]),
            closed_ts_utc=str(stop_fill_row["fill_ts_utc"]),
        )
        if plan_row is None:
            return

        if get_trade_outcome_by_plan_id(conn, str(plan_row["plan_id"])) is not None:
            return

        outcome_row = create_trade_outcome_from_stop_fill(
            conn,
            plan_row=plan_row,
            stop_fill_row=stop_fill_row,
        )
        print(
            f"[execution-worker] created outcome plan_id={plan_row['plan_id']} "
            f"outcome_id={outcome_row['outcome_id']}"
        )

    def process_validated_request(self, conn, request_row) -> None:
        request_id = self._get_request_id(request_row)
        alert_id = self._get_alert_id(request_row)
        status = str(request_row["status"])

        if status != "VALIDATED":
            print(
                f"[execution-worker] warning request_id={request_id} "
                f"expected_status=VALIDATED actual_status={status}"
            )
            return

        alert_row: Mapping[str, Any] | None = None
        trade_plan = None
        try:
            alert_row = self._require_alert_row(alert_id)
            trade_plan = self._get_or_create_trade_plan(
                conn,
                request_id=request_id,
                alert_id=alert_id,
                alert_row=alert_row,
            )

            existing_order = get_paper_order_by_request_id_and_role(
                conn, request_id, PaperOrderRole.ENTRY.value
            )
            if existing_order is not None:
                self._mark_submitted_from_local_order(conn, request_id, existing_order)
                print(
                    f"[execution-worker] reused local paper order request_id={request_id} "
                    f"broker_order_id={existing_order['broker_order_id']}"
                )
                return

            recovered_remote_order = find_existing_broker_order(request_id)
            if recovered_remote_order is not None:
                local_order = self._persist_or_reuse_broker_result(
                    conn,
                    trade_plan=trade_plan,
                    request_id=request_id,
                    alert_id=alert_id,
                    alert_row=alert_row,
                    result=recovered_remote_order,
                )
                self._mark_submitted_from_local_order(conn, request_id, local_order)
                print(
                    f"[execution-worker] recovered remote broker order request_id={request_id} "
                    f"broker_order_id={local_order['broker_order_id']}"
                )
                return

            print(
                f"[execution-worker] submitting order request_id={request_id} "
                f"alert_id={alert_id} notional={config.EXECUTION_ORDER_NOTIONAL}"
            )
            result = submit_validated_order(request_id=request_id, alert_row=alert_row)
            local_order = self._persist_or_reuse_broker_result(
                conn,
                trade_plan=trade_plan,
                request_id=request_id,
                alert_id=alert_id,
                alert_row=alert_row,
                result=result,
            )
            self._mark_submitted_from_local_order(conn, request_id, local_order)
            print(
                f"[execution-worker] submitted request_id={request_id} "
                f"broker_order_id={local_order['broker_order_id']}"
            )
        except Exception as exc:
            try:
                recovered_remote_order = find_existing_broker_order(request_id)
            except Exception as recovery_exc:
                recovered_remote_order = None
                exc = RuntimeError(f"{exc}; recovery_lookup_error={recovery_exc}")

            if recovered_remote_order is not None:
                try:
                    if alert_row is None:
                        alert_row = self._require_alert_row(alert_id)
                    if trade_plan is None:
                        trade_plan = self._get_or_create_trade_plan(
                            conn,
                            request_id=request_id,
                            alert_id=alert_id,
                            alert_row=alert_row,
                        )
                    local_order = self._persist_or_reuse_broker_result(
                        conn,
                        trade_plan=trade_plan,
                        request_id=request_id,
                        alert_id=alert_id,
                        alert_row=alert_row,
                        result=recovered_remote_order,
                    )
                    self._mark_submitted_from_local_order(conn, request_id, local_order)
                    print(
                        f"[execution-worker] recovered broker order after submit exception "
                        f"request_id={request_id} broker_order_id={local_order['broker_order_id']}"
                    )
                    return
                except Exception as persist_exc:
                    exc = RuntimeError(f"{exc}; recovery_persist_error={persist_exc}")

            mark_failed(
                conn,
                request_id,
                code="BROKER_SUBMIT_FAILED",
                message=str(exc),
            )
            print(f"[execution-worker] failed request_id={request_id} error={exc}")

    def _require_alert_row(self, alert_id: str) -> Mapping[str, Any]:
        alert_row = self.alert_lookup(alert_id)
        if alert_row is None:
            raise RuntimeError("alert lookup returned no row during submission")
        return alert_row

    def _persist_or_reuse_broker_result(
        self,
        conn,
        *,
        trade_plan,
        request_id: str,
        alert_id: str,
        result,
        alert_row: Mapping[str, Any] | None = None,
    ):
        existing_order = get_paper_order_by_request_id_and_role(
            conn, request_id, PaperOrderRole.ENTRY.value
        )
        if existing_order is not None:
            return existing_order

        if alert_row is None:
            alert_row = self._require_alert_row(alert_id)

        try:
            return insert_paper_order(
                conn,
                plan_id=str(trade_plan["plan_id"]),
                request_id=request_id,
                alert_id=alert_id,
                broker=result.broker,
                client_order_id=request_id,
                symbol=str(alert_row["symbol"]),
                side=str(alert_row["side"]),
                qty=None,
                notional=config.EXECUTION_ORDER_NOTIONAL,
                order_type="market",
                time_in_force="gtc",
                result=result,
            )
        except sqlite3.IntegrityError:
            existing_order = get_paper_order_by_request_id_and_role(
                conn, request_id, PaperOrderRole.ENTRY.value
            )
            if existing_order is None:
                raise
            print(
                f"[execution-worker] reused local duplicate paper order request_id={request_id} "
                f"broker_order_id={existing_order['broker_order_id']}"
            )
            return existing_order

    def _get_or_create_trade_plan(
        self,
        conn,
        *,
        request_id: str,
        alert_id: str,
        alert_row: Mapping[str, Any],
    ):
        existing_plan = get_paper_trade_plan_by_request_id(conn, request_id)
        if existing_plan is not None:
            return existing_plan

        timeout_seconds = self._coerce_int(alert_row.get("timeout_seconds"))
        signal_ts_utc = self._coerce_optional_str(
            alert_row.get("signal_ts_utc") or alert_row.get("created_ts_utc")
        )
        timeout_ts_utc = None
        if signal_ts_utc and timeout_seconds is not None:
            try:
                timeout_dt = self._parse_iso_datetime(signal_ts_utc)
            except Exception:
                timeout_dt = None
            if timeout_dt is not None:
                timeout_ts_utc = (timeout_dt + timedelta(seconds=timeout_seconds)).isoformat()

        return create_paper_trade_plan(
            conn,
            request_id=request_id,
            alert_id=alert_id,
            symbol=str(alert_row["symbol"]),
            side=str(alert_row["side"]),
            market_type=self._coerce_optional_str(alert_row.get("market_type")),
            strategy_profile=self._coerce_optional_str(alert_row.get("strategy_profile")),
            strategy_version=self._coerce_optional_str(alert_row.get("strategy_version")),
            config_hash=self._coerce_optional_str(alert_row.get("config_hash")),
            strategy_id=self._coerce_optional_str(alert_row.get("strategy_id")),
            setup_family=self._coerce_optional_str(alert_row.get("setup_family")),
            strategy_params_json=self._coerce_optional_str(alert_row.get("strategy_params_json")),
            strategy_params_hash=self._coerce_optional_str(alert_row.get("strategy_params_hash")),
            signal_ts_utc=signal_ts_utc,
            tier=self._coerce_optional_str(alert_row.get("tier")),
            entry_ref_px=self._coerce_float(alert_row.get("signal_px")),
            entry_low=self._coerce_float(alert_row.get("entry_low")),
            entry_high=self._coerce_float(alert_row.get("entry_high")),
            stop_px=self._coerce_float(alert_row.get("stop_px")),
            t1_px=self._coerce_float(alert_row.get("t1_px")),
            t2_px=self._coerce_float(alert_row.get("t2_px")),
            timeout_seconds=timeout_seconds,
            timeout_ts_utc=timeout_ts_utc,
            risk_model=self._coerce_optional_str(alert_row.get("risk_model")),
            risk_usd=self._coerce_float(alert_row.get("risk_usd")),
            qty_model=self._coerce_optional_str(alert_row.get("qty_model")),
            intended_qty=self._coerce_float(alert_row.get("qty")),
            intended_notional=config.EXECUTION_ORDER_NOTIONAL,
        )

    @staticmethod
    def _mark_submitted_from_local_order(conn, request_id: str, local_order) -> None:
        mark_submitted(
            conn,
            request_id,
            broker=str(local_order["broker"]),
            broker_order_id=str(local_order["broker_order_id"]),
            submitted_ts_utc=str(local_order["submitted_ts_utc"]),
        )

    @staticmethod
    def _get_request_id(request_row) -> str:
        return str(request_row["request_id"])

    @staticmethod
    def _get_alert_id(request_row) -> str:
        return str(request_row["alert_id"])

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(float(value))

    @staticmethod
    def _parse_iso_datetime(value: str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)

    @staticmethod
    def _get_plan_row_by_id(conn, plan_id: str):
        return conn.execute(
            """
            SELECT *
            FROM paper_trade_plans
            WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()

    @staticmethod
    def _protective_stop_client_order_id(request_id: str) -> str:
        return f"{request_id}:STOP"


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


def opposite_trade_side(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    raise ValueError(f"Unsupported side for protective stop: {side}")
