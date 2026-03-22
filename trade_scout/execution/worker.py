from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from trade_scout.execution.policy import validate_execution_request
from trade_scout.execution.queue import (
    claim_request,
    fetch_pending,
    get_connection,
    get_request_by_id,
    init_db,
    mark_expired,
    mark_failed,
    mark_rejected,
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
        claimed_count = 0
        try:
            pending_rows = fetch_pending(conn, limit=self.config.batch_size)
            for pending_row in pending_rows:
                request_id = self._get_request_id(pending_row)
                if not claim_request(conn, request_id, self.config.worker_id):
                    continue

                claimed_count += 1
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
        finally:
            conn.close()
        return claimed_count

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
        except Exception as exc:
            mark_failed(conn, request_id, code="PROCESSING_ERROR", message=str(exc))
            print(f"[execution-worker] failed request_id={request_id} error={exc}")

    @staticmethod
    def _get_request_id(request_row) -> str:
        return str(request_row["request_id"])

    @staticmethod
    def _get_alert_id(request_row) -> str:
        return str(request_row["alert_id"])
