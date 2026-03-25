from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from trade_scout import config
from trade_scout.execution.broker import find_existing_broker_order, submit_validated_order
from trade_scout.execution.policy import validate_execution_request
from trade_scout.execution.queue import (
    claim_request,
    fetch_pending,
    fetch_validated,
    get_connection,
    get_paper_order_by_request_id,
    get_request_by_id,
    init_db,
    insert_paper_order,
    mark_expired,
    mark_failed,
    mark_rejected,
    mark_submitted,
    mark_validated,
)


@dataclass(frozen=True)
class ExecutionWorkerConfig:
    db_path: str
    poll_interval_seconds: float = 5.0
    batch_size: int = 10
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

        try:
            existing_order = get_paper_order_by_request_id(conn, request_id)
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
                    request_id=request_id,
                    alert_id=alert_id,
                    alert_row=self._require_alert_row(alert_id),
                    result=recovered_remote_order,
                )
                self._mark_submitted_from_local_order(conn, request_id, local_order)
                print(
                    f"[execution-worker] recovered remote broker order request_id={request_id} "
                    f"broker_order_id={local_order['broker_order_id']}"
                )
                return

            alert_row = self._require_alert_row(alert_id)

            print(
                f"[execution-worker] submitting order request_id={request_id} "
                f"alert_id={alert_id} notional={config.EXECUTION_ORDER_NOTIONAL}"
            )
            result = submit_validated_order(request_id=request_id, alert_row=alert_row)
            local_order = self._persist_or_reuse_broker_result(
                conn,
                request_id=request_id,
                alert_id=alert_id,
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
                    local_order = self._persist_or_reuse_broker_result(
                        conn,
                        request_id=request_id,
                        alert_id=alert_id,
                        alert_row=self._require_alert_row(alert_id),
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
        request_id: str,
        alert_id: str,
        result,
        alert_row: Mapping[str, Any] | None = None,
    ):
        existing_order = get_paper_order_by_request_id(conn, request_id)
        if existing_order is not None:
            return existing_order

        if alert_row is None:
            alert_row = self._require_alert_row(alert_id)

        try:
            insert_paper_order(
                conn,
                request_id=request_id,
                alert_id=alert_id,
                symbol=str(alert_row["symbol"]),
                side=str(alert_row["side"]),
                qty=None,
                notional=config.EXECUTION_ORDER_NOTIONAL,
                order_type="market",
                time_in_force="gtc",
                result=result,
            )
        except sqlite3.IntegrityError:
            existing_order = get_paper_order_by_request_id(conn, request_id)
            if existing_order is None:
                raise
            print(
                f"[execution-worker] reused local duplicate paper order request_id={request_id} "
                f"broker_order_id={existing_order['broker_order_id']}"
            )
            return existing_order

        persisted_order = get_paper_order_by_request_id(conn, request_id)
        if persisted_order is None:
            raise RuntimeError("paper order insert did not persist broker result")
        return persisted_order

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
