from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Optional

from trade_scout.execution.reconciler import ExecutionReconciler
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
    SYNC_FAILURE_LOG_THROTTLE_SECONDS = 60.0

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
        self._sync_failure_log_state: dict[str, tuple[str, float]] = {}

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

            self.sync_open_paper_orders(conn, limit=self.config.sync_batch_size)
            self.service.process_position_actions(conn, limit=self.config.sync_batch_size)
        finally:
            conn.close()
        return attempted_count

    def process_claimed_request(self, conn, request_row) -> None:
        self.service.process_claimed_request(conn, request_row)

    def process_validated_request(self, conn, request_row) -> None:
        self.service.process_validated_request(conn, request_row)

    def sync_open_paper_orders(self, conn, *, limit: int) -> int:
        synced_count = 0
        open_orders = self.repository.get_open_paper_orders_for_sync(conn, limit=limit)
        for paper_order_row in open_orders:
            paper_order_id = str(paper_order_row["paper_order_id"])
            broker_order_id = self.reconciler._coerce_optional_str(paper_order_row["broker_order_id"])
            if not broker_order_id:
                print(
                    f"[execution-worker] skipped sync paper_order_id={paper_order_id} "
                    "reason=missing broker_order_id"
                )
                continue

            try:
                broker_snapshot = self.reconciler._get_broker_order_fetcher()(broker_order_id)
                if broker_snapshot is None:
                    print(
                        f"[execution-worker] skipped sync paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id} reason=broker_order_not_found"
                    )
                    continue

                reconciled = self.reconciler.reconcile_paper_order(conn, paper_order_row, broker_snapshot)
                synced_count += 1
                had_failure_state = paper_order_id in self._sync_failure_log_state
                self._sync_failure_log_state.pop(paper_order_id, None)
                if had_failure_state:
                    print(
                        f"[execution-worker] sync recovered paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id}"
                    )
                if self.reconciler._has_material_order_state_change(paper_order_row, reconciled):
                    print(
                        f"[execution-worker] synced paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id} order_status={reconciled['order_status']}"
                    )
            except Exception as exc:
                if self._should_log_sync_failure(paper_order_id, exc):
                    print(
                        f"[execution-worker] sync failed paper_order_id={paper_order_id} "
                        f"broker_order_id={broker_order_id} error={exc}"
                    )
        return synced_count

    def process_position_triggers(
        self,
        conn=None,
        *,
        price_by_symbol: Mapping[str, float],
        limit: int = 100,
        now_utc: datetime | None = None,
    ) -> int:
        owns_connection = conn is None
        if owns_connection:
            conn = self.repository.get_connection()
        try:
            return self.service.process_position_triggers(
                conn,
                price_by_symbol=price_by_symbol,
                limit=limit,
                now_utc=now_utc,
            )
        finally:
            if owns_connection and conn is not None:
                conn.close()

    def reconcile_paper_order(self, conn, paper_order_row, broker_snapshot):
        return self.reconciler.reconcile_paper_order(conn, paper_order_row, broker_snapshot)

    def ingest_entry_fill_if_present(self, conn, paper_order_row, broker_snapshot) -> None:
        self.reconciler.ingest_entry_fill_if_present(conn, paper_order_row, broker_snapshot)

    @staticmethod
    def _get_request_id(request_row) -> str:
        return str(request_row["request_id"])

    def _should_log_sync_failure(self, paper_order_id: str, error: Exception) -> bool:
        signature = self._sync_failure_signature(error)
        now_ts = self._sync_failure_now()
        prior = self._sync_failure_log_state.get(paper_order_id)
        if prior is not None:
            prior_signature, prior_logged_ts = prior
            if prior_signature == signature and (now_ts - prior_logged_ts) < self.SYNC_FAILURE_LOG_THROTTLE_SECONDS:
                return False
        self._sync_failure_log_state[paper_order_id] = (signature, now_ts)
        return True

    @staticmethod
    def _sync_failure_signature(error: Exception) -> str:
        return f"{error.__class__.__name__}:{error}"

    @staticmethod
    def _sync_failure_now() -> float:
        return time.monotonic()
