from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from trade_scout.execution.reconciler import (
    ExecutionReconciler,
    derive_entry_status_from_order_status,
    normalize_broker_order_status,
)
from trade_scout.execution.repository import ExecutionRepository
from trade_scout.execution.service import ExecutionService


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
        alert_lookup: Callable[[str], Optional[Mapping[str, object]]],
    ):
        self.config = config
        self.alert_lookup = alert_lookup
        self.repository = ExecutionRepository(config.db_path)
        self.reconciler = ExecutionReconciler(self.repository)
        self.service = ExecutionService(
            repository=self.repository,
            reconciler=self.reconciler,
            alert_lookup=alert_lookup,
            max_alert_age_seconds=config.max_alert_age_seconds,
        )

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
        self.repository.init_db()
        conn = self.repository.get_connection()
        attempted_count = 0
        try:
            pending_rows = self.repository.fetch_pending(conn, limit=self.config.batch_size)
            for pending_row in pending_rows:
                request_id = self._get_request_id(pending_row)
                if not self.repository.claim_request(conn, request_id, self.config.worker_id):
                    continue

                attempted_count += 1
                print(
                    f"[execution-worker] claimed request_id={request_id} "
                    f"worker_id={self.config.worker_id}"
                )

                claimed_row = self.repository.get_request_by_id(conn, request_id)
                if claimed_row is None:
                    self.repository.mark_failed(
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
                self.service.process_claimed_request(conn, claimed_row)

            remaining = max(0, self.config.batch_size - attempted_count)
            if remaining > 0:
                validated_rows = self.repository.fetch_validated(conn, limit=remaining)
                for validated_row in validated_rows:
                    attempted_count += 1
                    self.service.process_validated_request(conn, validated_row)

            self.reconciler.sync_open_paper_orders(conn, limit=self.config.sync_batch_size)
            self.service.process_position_actions(conn, limit=self.config.sync_batch_size)
        finally:
            conn.close()
        return attempted_count

    def process_claimed_request(self, conn, request_row) -> None:
        self.service.process_claimed_request(conn, request_row)

    def process_validated_request(self, conn, request_row) -> None:
        self.service.process_validated_request(conn, request_row)

    def sync_open_paper_orders(self, conn, *, limit: int) -> int:
        return self.reconciler.sync_open_paper_orders(conn, limit=limit)

    def reconcile_paper_order(self, conn, paper_order_row, broker_snapshot):
        return self.reconciler.reconcile_paper_order(conn, paper_order_row, broker_snapshot)

    def ingest_entry_fill_if_present(self, conn, paper_order_row, broker_snapshot) -> None:
        self.reconciler.ingest_entry_fill_if_present(conn, paper_order_row, broker_snapshot)

    @staticmethod
    def _get_request_id(request_row) -> str:
        return str(request_row["request_id"])
