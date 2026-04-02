from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from trade_scout.execution.models import (
    BrokerOrderResult,
    ExecutionRequest,
    ExecutionStatus,
    PaperFillRole,
    PaperPositionStatus,
    PaperOrderRole,
    PaperOrderStatus,
    PaperTradePlanEntryStatus,
    PaperTradePlanStatus,
)


ACTIVE_REQUEST_STATUSES = (
    ExecutionStatus.PENDING.value,
    ExecutionStatus.CLAIMED.value,
    ExecutionStatus.VALIDATED.value,
)
PAPER_TRADE_PLAN_SCHEMA_VERSION = "1"
PAPER_ORDER_SCHEMA_VERSION = "1"
PAPER_FILL_SCHEMA_VERSION = "1"
PAPER_POSITION_SCHEMA_VERSION = "1"
PAPER_TRADE_PLAN_ENTRY_TYPE_MARKET = "MARKET"
TERMINAL_PAPER_ORDER_STATUSES = (
    PaperOrderStatus.FILLED.value,
    PaperOrderStatus.CANCELED.value,
    PaperOrderStatus.REJECTED.value,
    PaperOrderStatus.EXPIRED.value,
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
            CREATE TABLE IF NOT EXISTS paper_trade_plans (
                plan_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                alert_id TEXT NOT NULL,
                created_ts_utc TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                market_type TEXT,
                strategy_profile TEXT,
                strategy_version TEXT,
                config_hash TEXT,
                strategy_id TEXT,
                setup_family TEXT,
                strategy_params_json TEXT,
                strategy_params_hash TEXT,
                signal_ts_utc TEXT,
                tier TEXT,
                entry_type TEXT NOT NULL,
                entry_ref_px REAL,
                entry_low REAL,
                entry_high REAL,
                stop_px REAL,
                t1_px REAL,
                t2_px REAL,
                timeout_seconds INTEGER,
                timeout_ts_utc TEXT,
                risk_model TEXT,
                risk_usd REAL,
                qty_model TEXT,
                intended_qty REAL,
                intended_notional REAL,
                plan_rules_json TEXT,
                plan_status TEXT NOT NULL,
                entry_status TEXT,
                entry_filled_qty REAL,
                entry_avg_fill_px REAL,
                entry_fill_ts_utc TEXT,
                schema_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_orders (
                paper_order_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                alert_id TEXT NOT NULL,
                broker TEXT NOT NULL,
                broker_order_id TEXT,
                client_order_id TEXT NOT NULL UNIQUE,
                parent_paper_order_id TEXT,
                order_role TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                time_in_force TEXT,
                qty REAL,
                notional REAL,
                limit_price REAL,
                stop_price REAL,
                trail_price REAL,
                trail_percent REAL,
                reduce_only INTEGER NOT NULL DEFAULT 0,
                submitted_ts_utc TEXT NOT NULL,
                accepted_ts_utc TEXT,
                canceled_ts_utc TEXT,
                expired_ts_utc TEXT,
                rejected_ts_utc TEXT,
                filled_ts_utc TEXT,
                broker_status_raw TEXT,
                order_status TEXT NOT NULL,
                status_reason TEXT,
                last_broker_update_ts_utc TEXT,
                raw_response_json TEXT,
                schema_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_fills (
                paper_fill_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                alert_id TEXT NOT NULL,
                paper_order_id TEXT NOT NULL,
                broker TEXT NOT NULL,
                broker_order_id TEXT NOT NULL,
                broker_fill_id TEXT,
                fill_role TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                fill_ts_utc TEXT NOT NULL,
                fill_px REAL NOT NULL,
                fill_qty REAL NOT NULL,
                gross_notional REAL,
                fee REAL,
                liquidity TEXT,
                fill_sequence INTEGER,
                raw_fill_json TEXT,
                schema_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                alert_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                position_status TEXT NOT NULL,
                entry_order_id TEXT,
                entry_client_order_id TEXT,
                entry_filled_qty REAL NOT NULL DEFAULT 0,
                entry_avg_fill_px REAL,
                opened_ts_utc TEXT,
                closed_ts_utc TEXT,
                remaining_qty REAL NOT NULL DEFAULT 0,
                created_ts_utc TEXT NOT NULL,
                updated_ts_utc TEXT NOT NULL,
                schema_version TEXT NOT NULL
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
            CREATE INDEX IF NOT EXISTS idx_paper_trade_plans_alert_id
            ON paper_trade_plans (alert_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_orders_plan_id
            ON paper_orders (plan_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_orders_request_id
            ON paper_orders (request_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_orders_broker_order_id
            ON paper_orders (broker_order_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_fills_broker_order_id
            ON paper_fills (broker_order_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_fills_paper_order_id
            ON paper_fills (paper_order_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_fills_plan_id
            ON paper_fills (plan_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_positions_request_id
            ON paper_positions (request_id)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_fills_order_role
            ON paper_fills (paper_order_id, fill_role)
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
                request.requested_ts_utc,
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


def get_paper_order_by_request_id_and_role(
    conn: sqlite3.Connection, request_id: str, order_role: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE request_id = ?
          AND order_role = ?
        """,
        (request_id, order_role),
    ).fetchone()


def get_paper_order_by_client_order_id(
    conn: sqlite3.Connection, client_order_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE client_order_id = ?
        """,
        (client_order_id,),
    ).fetchone()


def get_open_paper_orders_for_sync(
    conn: sqlite3.Connection, limit: int = 50
) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE order_status NOT IN (?, ?, ?, ?)
        ORDER BY last_broker_update_ts_utc ASC, submitted_ts_utc ASC
        LIMIT ?
        """,
        (*TERMINAL_PAPER_ORDER_STATUSES, limit),
    ).fetchall()
    return list(rows)


def get_paper_trade_plan_by_request_id(
    conn: sqlite3.Connection, request_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_trade_plans
        WHERE request_id = ?
        """,
        (request_id,),
    ).fetchone()


def get_paper_position_by_plan_id(
    conn: sqlite3.Connection, plan_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_positions
        WHERE plan_id = ?
        """,
        (plan_id,),
    ).fetchone()


def get_paper_position_by_request_id(
    conn: sqlite3.Connection, request_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_positions
        WHERE request_id = ?
        """,
        (request_id,),
    ).fetchone()


def create_paper_trade_plan(
    conn: sqlite3.Connection,
    *,
    request_id: str,
    alert_id: str,
    symbol: str,
    side: str,
    market_type: str | None = None,
    strategy_profile: str | None = None,
    strategy_version: str | None = None,
    config_hash: str | None = None,
    strategy_id: str | None = None,
    setup_family: str | None = None,
    strategy_params_json: str | None = None,
    strategy_params_hash: str | None = None,
    signal_ts_utc: str | None = None,
    tier: str | None = None,
    entry_ref_px: float | None = None,
    entry_low: float | None = None,
    entry_high: float | None = None,
    stop_px: float | None = None,
    t1_px: float | None = None,
    t2_px: float | None = None,
    timeout_seconds: int | None = None,
    timeout_ts_utc: str | None = None,
    risk_model: str | None = None,
    risk_usd: float | None = None,
    qty_model: str | None = None,
    intended_qty: float | None = None,
    intended_notional: float | None = None,
    plan_rules_json: str = "{}",
    plan_status: str = PaperTradePlanStatus.CREATED.value,
    schema_version: str = PAPER_TRADE_PLAN_SCHEMA_VERSION,
) -> sqlite3.Row:
    existing = get_paper_trade_plan_by_request_id(conn, request_id)
    if existing is not None:
        return existing

    plan_id = str(uuid4())
    created_ts_utc = utc_now_iso()
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
            PAPER_TRADE_PLAN_ENTRY_TYPE_MARKET,
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
            PaperTradePlanEntryStatus.PENDING.value,
            None,
            None,
            None,
            schema_version,
        ),
    )
    conn.commit()
    created = get_paper_trade_plan_by_request_id(conn, request_id)
    if created is None:
        raise RuntimeError("paper trade plan insert did not persist")
    return created


def ensure_paper_position(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    request_id: str,
    alert_id: str,
    symbol: str,
    side: str,
    entry_order_id: str | None = None,
    entry_client_order_id: str | None = None,
) -> sqlite3.Row:
    existing = get_paper_position_by_plan_id(conn, plan_id)
    now = utc_now_iso()
    if existing is None:
        position_id = str(uuid4())
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
                created_ts_utc,
                updated_ts_utc,
                schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position_id,
                plan_id,
                request_id,
                alert_id,
                symbol,
                side,
                PaperPositionStatus.PENDING_ENTRY.value,
                entry_order_id,
                entry_client_order_id,
                0.0,
                None,
                None,
                None,
                0.0,
                now,
                now,
                PAPER_POSITION_SCHEMA_VERSION,
            ),
        )
        conn.commit()
        created = get_paper_position_by_plan_id(conn, plan_id)
        if created is None:
            raise RuntimeError("paper position insert did not persist")
        return created

    if (
        (entry_order_id and str(existing["entry_order_id"] or "") != entry_order_id)
        or (
            entry_client_order_id
            and str(existing["entry_client_order_id"] or "") != entry_client_order_id
        )
    ):
        conn.execute(
            """
            UPDATE paper_positions
            SET entry_order_id = COALESCE(?, entry_order_id),
                entry_client_order_id = COALESCE(?, entry_client_order_id),
                updated_ts_utc = ?
            WHERE plan_id = ?
            """,
            (entry_order_id, entry_client_order_id, now, plan_id),
        )
        conn.commit()
        existing = get_paper_position_by_plan_id(conn, plan_id)
        if existing is None:
            raise RuntimeError("paper position update did not persist")
    return existing


def insert_paper_order(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    request_id: str,
    alert_id: str,
    broker: str,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: Optional[float],
    notional: Optional[float],
    order_type: str,
    time_in_force: Optional[str],
    result: BrokerOrderResult,
) -> sqlite3.Row:
    existing = get_paper_order_by_client_order_id(conn, client_order_id)
    if existing is not None:
        return existing

    paper_order_id = str(uuid4())
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
            paper_order_id,
            plan_id,
            request_id,
            alert_id,
            broker,
            result.broker_order_id,
            client_order_id,
            None,
            PaperOrderRole.ENTRY.value,
            symbol,
            side,
            order_type,
            time_in_force,
            qty,
            notional,
            result.submitted_ts_utc,
            result.accepted_ts_utc,
            result.broker_status_raw,
            PaperOrderStatus.SUBMITTED.value,
            result.accepted_ts_utc or result.submitted_ts_utc,
            result.raw_response_json,
            PAPER_ORDER_SCHEMA_VERSION,
        ),
    )
    conn.commit()
    inserted = get_paper_order_by_client_order_id(conn, client_order_id)
    if inserted is None:
        raise RuntimeError("paper order insert did not persist")
    return inserted


def reconcile_paper_order_from_broker(
    conn: sqlite3.Connection,
    *,
    paper_order_id: str,
    broker_snapshot: BrokerOrderResult,
    order_status: str,
) -> sqlite3.Row | None:
    existing_row = conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE paper_order_id = ?
        """,
        (paper_order_id,),
    ).fetchone()
    if existing_row is None:
        return None

    conn.execute(
        """
        UPDATE paper_orders
        SET broker_status_raw = ?,
            order_status = ?,
            accepted_ts_utc = ?,
            canceled_ts_utc = ?,
            expired_ts_utc = ?,
            rejected_ts_utc = ?,
            filled_ts_utc = ?,
            status_reason = ?,
            last_broker_update_ts_utc = ?,
            raw_response_json = ?
        WHERE paper_order_id = ?
        """,
        (
            broker_snapshot.broker_status_raw,
            order_status,
            broker_snapshot.accepted_ts_utc or existing_row["accepted_ts_utc"],
            broker_snapshot.canceled_ts_utc or existing_row["canceled_ts_utc"],
            broker_snapshot.expired_ts_utc or existing_row["expired_ts_utc"],
            broker_snapshot.rejected_ts_utc or existing_row["rejected_ts_utc"],
            broker_snapshot.filled_ts_utc or existing_row["filled_ts_utc"],
            broker_snapshot.status_reason or existing_row["status_reason"],
            broker_snapshot.last_broker_update_ts_utc or utc_now_iso(),
            broker_snapshot.raw_response_json,
            paper_order_id,
        ),
    )
    conn.commit()
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE paper_order_id = ?
        """,
        (paper_order_id,),
    ).fetchone()


def get_paper_fill_by_order_id_and_role(
    conn: sqlite3.Connection,
    paper_order_id: str,
    fill_role: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_fills
        WHERE paper_order_id = ?
          AND fill_role = ?
        """,
        (paper_order_id, fill_role),
    ).fetchone()


def _resolve_entry_fill_ts_utc(
    *,
    broker_snapshot: BrokerOrderResult,
    existing_fill_row: sqlite3.Row | None,
    trade_plan_row: sqlite3.Row | None,
) -> str | None:
    if broker_snapshot.filled_ts_utc:
        return broker_snapshot.filled_ts_utc
    if existing_fill_row is not None and existing_fill_row["fill_ts_utc"]:
        return str(existing_fill_row["fill_ts_utc"])
    if trade_plan_row is not None and trade_plan_row["entry_fill_ts_utc"]:
        return str(trade_plan_row["entry_fill_ts_utc"])
    if broker_snapshot.last_broker_update_ts_utc:
        return broker_snapshot.last_broker_update_ts_utc
    return None


def upsert_entry_fill_from_broker_snapshot(
    conn: sqlite3.Connection,
    *,
    paper_order_row: sqlite3.Row,
    broker_snapshot: BrokerOrderResult,
    entry_status: str,
) -> sqlite3.Row | None:
    if broker_snapshot.filled_qty is None or broker_snapshot.filled_avg_price is None:
        return None

    paper_order_id = str(paper_order_row["paper_order_id"])
    fill_role = PaperFillRole.ENTRY.value
    existing_fill = get_paper_fill_by_order_id_and_role(conn, paper_order_id, fill_role)
    trade_plan_row = conn.execute(
        """
        SELECT *
        FROM paper_trade_plans
        WHERE plan_id = ?
        """,
        (str(paper_order_row["plan_id"]),),
    ).fetchone()
    fill_ts_utc = _resolve_entry_fill_ts_utc(
        broker_snapshot=broker_snapshot,
        existing_fill_row=existing_fill,
        trade_plan_row=trade_plan_row,
    )
    if not fill_ts_utc:
        return None

    paper_fill_id = (
        str(existing_fill["paper_fill_id"])
        if existing_fill is not None
        else f"{paper_order_id}:{fill_role}"
    )
    gross_notional = broker_snapshot.filled_qty * broker_snapshot.filled_avg_price

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
        ON CONFLICT(paper_fill_id) DO UPDATE SET
            broker_fill_id = excluded.broker_fill_id,
            fill_ts_utc = excluded.fill_ts_utc,
            fill_px = excluded.fill_px,
            fill_qty = excluded.fill_qty,
            gross_notional = excluded.gross_notional,
            fee = excluded.fee,
            liquidity = excluded.liquidity,
            fill_sequence = excluded.fill_sequence,
            raw_fill_json = excluded.raw_fill_json
        """,
        (
            paper_fill_id,
            str(paper_order_row["plan_id"]),
            str(paper_order_row["request_id"]),
            str(paper_order_row["alert_id"]),
            paper_order_id,
            str(paper_order_row["broker"]),
            str(paper_order_row["broker_order_id"]),
            None,
            fill_role,
            str(paper_order_row["symbol"]),
            str(paper_order_row["side"]),
            fill_ts_utc,
            broker_snapshot.filled_avg_price,
            broker_snapshot.filled_qty,
            gross_notional,
            None,
            None,
            1,
            broker_snapshot.raw_response_json,
            PAPER_FILL_SCHEMA_VERSION,
        ),
    )
    conn.commit()

    update_trade_plan_entry_summary(
        conn,
        plan_id=str(paper_order_row["plan_id"]),
        entry_status=entry_status,
        entry_filled_qty=broker_snapshot.filled_qty,
        entry_avg_fill_px=broker_snapshot.filled_avg_price,
        entry_fill_ts_utc=fill_ts_utc,
    )

    return get_paper_fill_by_order_id_and_role(conn, paper_order_id, fill_role)


def update_paper_position_from_entry_fill(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    request_id: str,
    alert_id: str,
    symbol: str,
    side: str,
    entry_order_id: str | None,
    entry_client_order_id: str | None,
    entry_filled_qty: float,
    entry_avg_fill_px: float,
    opened_ts_utc: str,
) -> sqlite3.Row | None:
    position_row = ensure_paper_position(
        conn,
        plan_id=plan_id,
        request_id=request_id,
        alert_id=alert_id,
        symbol=symbol,
        side=side,
        entry_order_id=entry_order_id,
        entry_client_order_id=entry_client_order_id,
    )
    if position_row is None:
        return None

    current_opened_ts_utc = str(position_row["opened_ts_utc"]) if position_row["opened_ts_utc"] else None
    next_status = (
        PaperPositionStatus.OPEN.value
        if entry_filled_qty > 0
        else str(position_row["position_status"])
    )
    next_opened_ts_utc = current_opened_ts_utc or (opened_ts_utc if entry_filled_qty > 0 else None)
    updated_ts_utc = utc_now_iso()

    conn.execute(
        """
        UPDATE paper_positions
        SET position_status = ?,
            entry_order_id = COALESCE(?, entry_order_id),
            entry_client_order_id = COALESCE(?, entry_client_order_id),
            entry_filled_qty = ?,
            entry_avg_fill_px = ?,
            opened_ts_utc = ?,
            remaining_qty = ?,
            updated_ts_utc = ?
        WHERE plan_id = ?
        """,
        (
            next_status,
            entry_order_id,
            entry_client_order_id,
            entry_filled_qty,
            entry_avg_fill_px,
            next_opened_ts_utc,
            entry_filled_qty,
            updated_ts_utc,
            plan_id,
        ),
    )
    conn.commit()
    return get_paper_position_by_plan_id(conn, plan_id)


def update_trade_plan_entry_summary(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    entry_status: str,
    entry_filled_qty: float,
    entry_avg_fill_px: float,
    entry_fill_ts_utc: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET entry_status = ?,
            entry_filled_qty = ?,
            entry_avg_fill_px = ?,
            entry_fill_ts_utc = ?
        WHERE plan_id = ?
        """,
        (
            entry_status,
            entry_filled_qty,
            entry_avg_fill_px,
            entry_fill_ts_utc,
            plan_id,
        ),
    )
    conn.commit()
    return conn.execute(
        """
        SELECT *
        FROM paper_trade_plans
        WHERE plan_id = ?
        """,
        (plan_id,),
    ).fetchone()
