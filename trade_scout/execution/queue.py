from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from trade_scout.execution.models import BrokerOrderResult, ExecutionRequest, ExecutionStatus


ACTIVE_REQUEST_STATUSES = (
    ExecutionStatus.PENDING.value,
    ExecutionStatus.CLAIMED.value,
    ExecutionStatus.VALIDATED.value,
)


class ActiveExecutionRequestExistsError(Exception):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.row_factory = sqlite3.Row
    return conn


# NOTE:
# Execution DB uses a fresh-start schema.
# Backward compatibility migrations are intentionally removed.
# If schema changes are needed, recreate the DB.
def init_db(db_path: str) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_requests (
                request_id TEXT PRIMARY KEY,
                alert_id TEXT NOT NULL,
                requested_ts_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                policy_name TEXT,
                policy_version TEXT,
                requested_by TEXT,
                reason TEXT,
                claimed_ts_utc TEXT,
                worker_id TEXT,
                validated_ts_utc TEXT,
                rejection_code TEXT,
                rejection_reason TEXT,
                submitted_ts_utc TEXT,
                completed_ts_utc TEXT,
                broker TEXT,
                broker_order_id TEXT,
                error_code TEXT,
                error_message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                alert_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL,
                notional REAL,
                order_type TEXT NOT NULL,
                time_in_force TEXT NOT NULL,
                broker TEXT NOT NULL,
                broker_order_id TEXT NOT NULL,
                submitted_ts_utc TEXT NOT NULL,
                raw_response_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exec_req_status
            ON execution_requests (status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exec_req_alert_id
            ON execution_requests (alert_id)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_exec_req_alert_active
            ON execution_requests(alert_id)
            WHERE status IN ('PENDING', 'CLAIMED', 'VALIDATED')
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_request(conn: sqlite3.Connection, request: ExecutionRequest) -> None:
    existing = conn.execute(
        """
        SELECT request_id
        FROM execution_requests
        WHERE alert_id = ?
          AND status IN (?, ?, ?)
        """,
        (request.alert_id, *ACTIVE_REQUEST_STATUSES),
    ).fetchone()
    if existing is not None:
        raise ActiveExecutionRequestExistsError(
            "Active request already exists for alert_id"
        )

    requested_ts_utc = utc_now_iso()

    try:
        conn.execute(
            """
            INSERT INTO execution_requests (
                request_id,
                alert_id,
                requested_ts_utc,
                status,
                policy_name,
                policy_version,
                requested_by,
                reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.request_id,
                request.alert_id,
                requested_ts_utc,
                ExecutionStatus.PENDING.value,
                request.policy_name,
                request.policy_version,
                request.requested_by,
                request.reason,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ActiveExecutionRequestExistsError(
            "Active request already exists for alert_id"
        ) from exc


def fetch_pending(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM execution_requests
        WHERE status = ?
        ORDER BY requested_ts_utc ASC
        LIMIT ?
        """,
        (ExecutionStatus.PENDING.value, limit),
    ).fetchall()
    return list(rows)


def fetch_validated(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM execution_requests
        WHERE status = ?
        ORDER BY validated_ts_utc ASC, requested_ts_utc ASC
        LIMIT ?
        """,
        (ExecutionStatus.VALIDATED.value, limit),
    ).fetchall()
    return list(rows)


def get_request_by_id(conn: sqlite3.Connection, request_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM execution_requests
        WHERE request_id = ?
        """,
        (request_id,),
    ).fetchone()


def claim_request(conn: sqlite3.Connection, request_id: str, worker_id: str) -> bool:
    cursor = conn.execute(
        """
        UPDATE execution_requests
        SET status = ?,
            claimed_ts_utc = ?,
            worker_id = ?
        WHERE request_id = ?
          AND status = ?
        """,
        (
            ExecutionStatus.CLAIMED.value,
            utc_now_iso(),
            worker_id,
            request_id,
            ExecutionStatus.PENDING.value,
        ),
    )
    conn.commit()
    return cursor.rowcount == 1


def mark_validated(conn: sqlite3.Connection, request_id: str) -> None:
    conn.execute(
        """
        UPDATE execution_requests
        SET status = ?,
            validated_ts_utc = ?
        WHERE request_id = ?
        """,
        (ExecutionStatus.VALIDATED.value, utc_now_iso(), request_id),
    )
    conn.commit()


def mark_rejected(
    conn: sqlite3.Connection, request_id: str, code: str, reason: str
) -> None:
    conn.execute(
        """
        UPDATE execution_requests
        SET status = ?,
            rejection_code = ?,
            rejection_reason = ?
        WHERE request_id = ?
        """,
        (ExecutionStatus.REJECTED.value, code, reason, request_id),
    )
    conn.commit()


def mark_expired(conn: sqlite3.Connection, request_id: str, reason: str) -> None:
    conn.execute(
        """
        UPDATE execution_requests
        SET status = ?,
            rejection_reason = ?
        WHERE request_id = ?
        """,
        (ExecutionStatus.EXPIRED.value, reason, request_id),
    )
    conn.commit()


def mark_submitted(
    conn: sqlite3.Connection,
    request_id: str,
    broker: str,
    broker_order_id: str,
    submitted_ts_utc: str,
) -> None:
    conn.execute(
        """
        UPDATE execution_requests
        SET status = ?,
            broker = ?,
            broker_order_id = ?,
            submitted_ts_utc = ?
        WHERE request_id = ?
        """,
        (
            ExecutionStatus.SUBMITTED.value,
            broker,
            broker_order_id,
            submitted_ts_utc,
            request_id,
        ),
    )
    conn.commit()


def mark_failed(
    conn: sqlite3.Connection, request_id: str, code: str, message: str
) -> None:
    conn.execute(
        """
        UPDATE execution_requests
        SET status = ?,
            error_code = ?,
            error_message = ?
        WHERE request_id = ?
        """,
        (ExecutionStatus.FAILED.value, code, message, request_id),
    )
    conn.commit()


def get_paper_order_by_request_id(
    conn: sqlite3.Connection, request_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE request_id = ?
        """,
        (request_id,),
    ).fetchone()


def insert_paper_order(
    conn: sqlite3.Connection,
    *,
    request_id: str,
    alert_id: str,
    symbol: str,
    side: str,
    qty: Optional[float],
    notional: Optional[float],
    order_type: str,
    time_in_force: str,
    result: BrokerOrderResult,
) -> None:
    conn.execute(
        """
        INSERT INTO paper_orders (
            request_id,
            alert_id,
            symbol,
            side,
            qty,
            notional,
            order_type,
            time_in_force,
            broker,
            broker_order_id,
            submitted_ts_utc,
            raw_response_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            alert_id,
            symbol,
            side,
            qty,
            notional,
            order_type,
            time_in_force,
            result.broker,
            result.broker_order_id,
            result.submitted_ts_utc,
            result.raw_response_json,
        ),
    )
    conn.commit()
