from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from trade_scout.execution.models import (
    BrokerOrderResult,
    ExecutionRequest,
    ExecutionStatus,
    PaperFillRole,
    PaperOrderRole,
    PaperOrderStatus,
    PaperTradePlanEntryStatus,
    PaperTradePlanPartialExitStatus,
    PaperTradePlanPositionStatus,
    PaperTradePlanStatus,
    PaperTradeOutcomeStatus,
)


ACTIVE_REQUEST_STATUSES = (
    ExecutionStatus.PENDING.value,
    ExecutionStatus.CLAIMED.value,
    ExecutionStatus.VALIDATED.value,
)
PAPER_TRADE_PLAN_SCHEMA_VERSION = "1"
PAPER_ORDER_SCHEMA_VERSION = "1"
PAPER_FILL_SCHEMA_VERSION = "1"
PAPER_TRADE_OUTCOME_SCHEMA_VERSION = "1"
PAPER_TRADE_PLAN_ENTRY_TYPE_MARKET = "MARKET"
TERMINAL_PAPER_ORDER_STATUSES = (
    PaperOrderStatus.FILLED.value,
    PaperOrderStatus.CANCELED.value,
    PaperOrderStatus.REPLACED.value,
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
                position_status TEXT,
                protective_stop_order_id TEXT,
                tp1_order_id TEXT,
                tp1_filled_qty REAL,
                tp1_avg_fill_px REAL,
                tp1_fill_ts_utc TEXT,
                tp2_order_id TEXT,
                tp2_filled_qty REAL,
                tp2_avg_fill_px REAL,
                tp2_fill_ts_utc TEXT,
                remaining_qty REAL,
                partial_exit_status TEXT,
                closed_ts_utc TEXT,
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
                replaced_from_paper_order_id TEXT,
                superseded_by_paper_order_id TEXT,
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
            CREATE TABLE IF NOT EXISTS paper_trade_outcomes (
                outcome_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                alert_id TEXT NOT NULL,
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
                opened_ts_utc TEXT,
                closed_ts_utc TEXT,
                entry_fill_ts_utc TEXT,
                entry_avg_px REAL,
                entry_qty REAL,
                exit_fill_ts_utc TEXT,
                exit_avg_px REAL,
                exit_qty REAL,
                exit_reason TEXT NOT NULL,
                gross_pnl REAL,
                fees REAL,
                net_pnl REAL,
                planned_stop_px REAL,
                planned_t1_px REAL,
                planned_t2_px REAL,
                realized_r REAL,
                entered INTEGER NOT NULL,
                closed INTEGER NOT NULL,
                outcome_status TEXT NOT NULL,
                raw_summary_json TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_paper_trade_outcomes_request_id
            ON paper_trade_outcomes (request_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trade_outcomes_alert_id
            ON paper_trade_outcomes (alert_id)
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
        WHERE order_status NOT IN (?, ?, ?, ?, ?)
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
            position_status,
            protective_stop_order_id,
            tp1_order_id,
            tp1_filled_qty,
            tp1_avg_fill_px,
            tp1_fill_ts_utc,
            tp2_order_id,
            tp2_filled_qty,
            tp2_avg_fill_px,
            tp2_fill_ts_utc,
            remaining_qty,
            partial_exit_status,
            closed_ts_utc,
            schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            PaperTradePlanPositionStatus.PENDING_ENTRY.value,
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
            PaperTradePlanPartialExitStatus.NONE.value,
            None,
            schema_version,
        ),
    )
    conn.commit()
    created = get_paper_trade_plan_by_request_id(conn, request_id)
    if created is None:
        raise RuntimeError("paper trade plan insert did not persist")
    return created


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
    order_role: str = PaperOrderRole.ENTRY.value,
    parent_paper_order_id: str | None = None,
    replaced_from_paper_order_id: str | None = None,
    limit_price: float | None = None,
    stop_price: float | None = None,
    reduce_only: int = 0,
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
            replaced_from_paper_order_id,
            superseded_by_paper_order_id,
            order_role,
            symbol,
            side,
            order_type,
            time_in_force,
            qty,
            notional,
            limit_price,
            stop_price,
            reduce_only,
            submitted_ts_utc,
            accepted_ts_utc,
            broker_status_raw,
            order_status,
            last_broker_update_ts_utc,
            raw_response_json,
            schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_order_id,
            plan_id,
            request_id,
            alert_id,
            broker,
            result.broker_order_id,
            client_order_id,
            parent_paper_order_id,
            replaced_from_paper_order_id,
            None,
            order_role,
            symbol,
            side,
            order_type,
            time_in_force,
            qty,
            notional,
            limit_price,
            stop_price,
            reduce_only,
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


def get_paper_order_by_plan_id_and_role(
    conn: sqlite3.Connection,
    plan_id: str,
    order_role: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE plan_id = ?
          AND order_role = ?
        ORDER BY submitted_ts_utc ASC
        LIMIT 1
        """,
        (plan_id, order_role),
    ).fetchone()


def get_latest_paper_order_by_plan_id_and_role(
    conn: sqlite3.Connection,
    plan_id: str,
    order_role: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE plan_id = ?
          AND order_role = ?
        ORDER BY submitted_ts_utc DESC, paper_order_id DESC
        LIMIT 1
        """,
        (plan_id, order_role),
    ).fetchone()


def get_current_paper_order_by_plan_id_and_role(
    conn: sqlite3.Connection,
    plan_id: str,
    order_role: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_orders
        WHERE plan_id = ?
          AND order_role = ?
          AND order_status NOT IN (?, ?, ?, ?, ?)
          AND superseded_by_paper_order_id IS NULL
        ORDER BY submitted_ts_utc DESC, paper_order_id DESC
        LIMIT 1
        """,
        (
            plan_id,
            order_role,
            PaperOrderStatus.FILLED.value,
            PaperOrderStatus.CANCELED.value,
            PaperOrderStatus.REPLACED.value,
            PaperOrderStatus.REJECTED.value,
            PaperOrderStatus.EXPIRED.value,
        ),
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


def upsert_paper_fill_from_broker_snapshot(
    conn: sqlite3.Connection,
    *,
    paper_order_row: sqlite3.Row,
    broker_snapshot: BrokerOrderResult,
    fill_role: str,
    entry_status: str | None = None,
) -> sqlite3.Row | None:
    if fill_role == PaperFillRole.ENTRY.value:
        return upsert_entry_fill_from_broker_snapshot(
            conn,
            paper_order_row=paper_order_row,
            broker_snapshot=broker_snapshot,
            entry_status=entry_status or PaperTradePlanEntryStatus.FILLED.value,
        )

    if broker_snapshot.filled_qty is None or broker_snapshot.filled_avg_price is None:
        return None
    fill_ts_utc = broker_snapshot.filled_ts_utc
    existing_fill = get_paper_fill_by_order_id_and_role(
        conn, str(paper_order_row["paper_order_id"]), fill_role
    )
    if fill_ts_utc is None and existing_fill is not None and existing_fill["fill_ts_utc"]:
        fill_ts_utc = str(existing_fill["fill_ts_utc"])
    if fill_ts_utc is None:
        fill_ts_utc = broker_snapshot.last_broker_update_ts_utc
    if not fill_ts_utc:
        return None

    paper_order_id = str(paper_order_row["paper_order_id"])
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
    return get_paper_fill_by_order_id_and_role(conn, paper_order_id, fill_role)


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


def update_trade_plan_position_open(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    protective_stop_order_id: str | None = None,
    tp1_order_id: str | None = None,
    remaining_qty: float | None = None,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET position_status = ?,
            protective_stop_order_id = COALESCE(?, protective_stop_order_id),
            tp1_order_id = COALESCE(?, tp1_order_id),
            remaining_qty = COALESCE(?, remaining_qty)
        WHERE plan_id = ?
        """,
        (
            PaperTradePlanPositionStatus.OPEN.value,
            protective_stop_order_id,
            tp1_order_id,
            remaining_qty,
            plan_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def update_trade_plan_position_closed(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    closed_ts_utc: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET position_status = ?,
            closed_ts_utc = ?
        WHERE plan_id = ?
        """,
        (
            PaperTradePlanPositionStatus.CLOSED.value,
            closed_ts_utc,
            plan_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def update_trade_plan_after_tp1_fill(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    tp1_order_id: str,
    tp1_filled_qty: float,
    tp1_avg_fill_px: float,
    tp1_fill_ts_utc: str,
    remaining_qty: float,
    partial_exit_status: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET tp1_order_id = ?,
            tp1_filled_qty = ?,
            tp1_avg_fill_px = ?,
            tp1_fill_ts_utc = ?,
            remaining_qty = ?,
            partial_exit_status = ?,
            position_status = CASE
                WHEN position_status = ? THEN position_status
                ELSE ?
            END
        WHERE plan_id = ?
        """,
        (
            tp1_order_id,
            tp1_filled_qty,
            tp1_avg_fill_px,
            tp1_fill_ts_utc,
            remaining_qty,
            partial_exit_status,
            PaperTradePlanPositionStatus.CLOSED.value,
            PaperTradePlanPositionStatus.OPEN.value,
            plan_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def update_trade_plan_after_tp2_fill(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    tp2_order_id: str,
    tp2_filled_qty: float,
    tp2_avg_fill_px: float,
    tp2_fill_ts_utc: str,
    remaining_qty: float,
    partial_exit_status: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET tp2_order_id = ?,
            tp2_filled_qty = ?,
            tp2_avg_fill_px = ?,
            tp2_fill_ts_utc = ?,
            remaining_qty = ?,
            partial_exit_status = ?,
            position_status = CASE
                WHEN position_status = ? THEN position_status
                ELSE ?
            END
        WHERE plan_id = ?
        """,
        (
            tp2_order_id,
            tp2_filled_qty,
            tp2_avg_fill_px,
            tp2_fill_ts_utc,
            remaining_qty,
            partial_exit_status,
            PaperTradePlanPositionStatus.CLOSED.value,
            PaperTradePlanPositionStatus.OPEN.value,
            plan_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def update_trade_plan_tp1_pointer(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    tp1_order_id: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET tp1_order_id = ?
        WHERE plan_id = ?
        """,
        (tp1_order_id, plan_id),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def update_trade_plan_tp2_pointer(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
    tp2_order_id: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET tp2_order_id = ?
        WHERE plan_id = ?
        """,
        (tp2_order_id, plan_id),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def clear_trade_plan_protective_stop_pointer(
    conn: sqlite3.Connection,
    *,
    plan_id: str,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_trade_plans
        SET protective_stop_order_id = NULL
        WHERE plan_id = ?
        """,
        (plan_id,),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_trade_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()


def update_open_stop_order_qty(
    conn: sqlite3.Connection,
    *,
    paper_order_id: str,
    qty: float,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_orders
        SET qty = ?
        WHERE paper_order_id = ?
          AND order_status NOT IN (?, ?, ?, ?, ?)
        """,
        (qty, paper_order_id, *TERMINAL_PAPER_ORDER_STATUSES),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_orders WHERE paper_order_id = ?",
        (paper_order_id,),
    ).fetchone()


def update_open_paper_order_qty(
    conn: sqlite3.Connection,
    *,
    paper_order_id: str,
    qty: float,
) -> sqlite3.Row | None:
    conn.execute(
        """
        UPDATE paper_orders
        SET qty = ?
        WHERE paper_order_id = ?
          AND order_status NOT IN (?, ?, ?, ?, ?)
        """,
        (qty, paper_order_id, *TERMINAL_PAPER_ORDER_STATUSES),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_orders WHERE paper_order_id = ?",
        (paper_order_id,),
    ).fetchone()


def mark_paper_order_replaced(
    conn: sqlite3.Connection,
    *,
    paper_order_id: str,
    replacement_paper_order_id: str,
    canceled_ts_utc: str | None = None,
    raw_response_json: str | None = None,
) -> sqlite3.Row | None:
    effective_canceled_ts_utc = canceled_ts_utc or utc_now_iso()
    conn.execute(
        """
        UPDATE paper_orders
        SET order_status = ?,
            canceled_ts_utc = COALESCE(?, canceled_ts_utc),
            status_reason = ?,
            superseded_by_paper_order_id = ?,
            last_broker_update_ts_utc = ?,
            raw_response_json = COALESCE(?, raw_response_json)
        WHERE paper_order_id = ?
        """,
        (
            PaperOrderStatus.REPLACED.value,
            effective_canceled_ts_utc,
            "REPLACED",
            replacement_paper_order_id,
            effective_canceled_ts_utc,
            raw_response_json,
            paper_order_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_orders WHERE paper_order_id = ?",
        (paper_order_id,),
    ).fetchone()


def mark_paper_order_canceled_for_replace(
    conn: sqlite3.Connection,
    *,
    paper_order_id: str,
    canceled_ts_utc: str | None = None,
    raw_response_json: str | None = None,
) -> sqlite3.Row | None:
    effective_canceled_ts_utc = canceled_ts_utc or utc_now_iso()
    conn.execute(
        """
        UPDATE paper_orders
        SET order_status = ?,
            canceled_ts_utc = COALESCE(?, canceled_ts_utc),
            status_reason = ?,
            last_broker_update_ts_utc = ?,
            raw_response_json = COALESCE(?, raw_response_json)
        WHERE paper_order_id = ?
        """,
        (
            PaperOrderStatus.CANCELED.value,
            effective_canceled_ts_utc,
            "canceled_for_replace",
            effective_canceled_ts_utc,
            raw_response_json,
            paper_order_id,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM paper_orders WHERE paper_order_id = ?",
        (paper_order_id,),
    ).fetchone()


def get_trade_outcome_by_plan_id(
    conn: sqlite3.Connection,
    plan_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM paper_trade_outcomes
        WHERE plan_id = ?
        """,
        (plan_id,),
    ).fetchone()


def list_exit_fills_by_plan_id(
    conn: sqlite3.Connection,
    plan_id: str,
) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM paper_fills
        WHERE plan_id = ?
          AND fill_role IN (?, ?, ?)
        ORDER BY fill_ts_utc ASC, fill_sequence ASC, paper_fill_id ASC
        """,
        (
            plan_id,
            PaperFillRole.TP1.value,
            PaperFillRole.TP2.value,
            PaperFillRole.STOP.value,
        ),
    ).fetchall()
    return list(rows)


def create_trade_outcome_for_closed_plan(
    conn: sqlite3.Connection,
    *,
    plan_row: sqlite3.Row,
    exit_reason: str,
) -> sqlite3.Row:
    existing = get_trade_outcome_by_plan_id(conn, str(plan_row["plan_id"]))
    if existing is not None:
        return existing

    entry_qty = _optional_float(plan_row["entry_filled_qty"])
    entry_avg_px = _optional_float(plan_row["entry_avg_fill_px"])
    exit_fill_rows = list_exit_fills_by_plan_id(conn, str(plan_row["plan_id"]))
    if not exit_fill_rows:
        raise RuntimeError("closed trade outcome requires at least one exit fill")

    exit_qty = 0.0
    exit_notional = 0.0
    gross_pnl = 0.0
    exit_fill_ids: list[str] = []
    exit_legs: list[dict[str, object | None]] = []
    latest_exit_fill_ts_utc = None
    for exit_fill_row in exit_fill_rows:
        fill_qty = _optional_float(exit_fill_row["fill_qty"])
        fill_px = _optional_float(exit_fill_row["fill_px"])
        fill_ts_utc = _optional_str(exit_fill_row["fill_ts_utc"])
        if fill_qty is None or fill_px is None or fill_ts_utc is None:
            continue

        exit_qty += fill_qty
        exit_notional += fill_px * fill_qty
        gross_pnl += _compute_gross_pnl(
            side=str(plan_row["side"]),
            entry_avg_px=entry_avg_px,
            exit_avg_px=fill_px,
            exit_qty=fill_qty,
        ) or 0.0
        exit_fill_ids.append(str(exit_fill_row["paper_fill_id"]))
        exit_legs.append(
            {
                "fill_id": str(exit_fill_row["paper_fill_id"]),
                "role": str(exit_fill_row["fill_role"]),
                "qty": fill_qty,
                "px": fill_px,
                "ts": fill_ts_utc,
            }
        )
        if latest_exit_fill_ts_utc is None or fill_ts_utc > latest_exit_fill_ts_utc:
            latest_exit_fill_ts_utc = fill_ts_utc

    if exit_qty <= 0 or latest_exit_fill_ts_utc is None:
        raise RuntimeError("closed trade outcome requires usable exit fill truth")

    exit_avg_px = exit_notional / exit_qty
    planned_stop_px = _optional_float(plan_row["stop_px"])
    realized_r = _compute_realized_r(
        side=str(plan_row["side"]),
        entry_avg_px=entry_avg_px,
        entry_qty=entry_qty,
        exit_avg_px=None,
        exit_qty=None,
        planned_stop_px=planned_stop_px,
        gross_pnl=gross_pnl,
    )
    closed_ts_utc = _optional_str(plan_row["closed_ts_utc"]) or latest_exit_fill_ts_utc
    entry_order_row = get_paper_order_by_plan_id_and_role(
        conn,
        str(plan_row["plan_id"]),
        PaperOrderRole.ENTRY.value,
    )
    entry_fill_id = None
    if entry_order_row is not None:
        entry_fill_row = get_paper_fill_by_order_id_and_role(
            conn,
            str(entry_order_row["paper_order_id"]),
            PaperFillRole.ENTRY.value,
        )
        if entry_fill_row is not None:
            entry_fill_id = str(entry_fill_row["paper_fill_id"])
    raw_summary_json = json.dumps(
        {
            "entry_fill_id": entry_fill_id,
            "exit_fill_ids": exit_fill_ids,
            "exit_legs": exit_legs,
            "exit_reason": exit_reason,
            "total_exit_qty": exit_qty,
            "weighted_exit_avg_px": exit_avg_px,
            "realized_gross_pnl": gross_pnl,
        },
        sort_keys=True,
    )
    outcome_id = f"{plan_row['plan_id']}:OUTCOME"
    conn.execute(
        """
        INSERT INTO paper_trade_outcomes (
            outcome_id,
            plan_id,
            request_id,
            alert_id,
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
            opened_ts_utc,
            closed_ts_utc,
            entry_fill_ts_utc,
            entry_avg_px,
            entry_qty,
            exit_fill_ts_utc,
            exit_avg_px,
            exit_qty,
            exit_reason,
            gross_pnl,
            fees,
            net_pnl,
            planned_stop_px,
            planned_t1_px,
            planned_t2_px,
            realized_r,
            entered,
            closed,
            outcome_status,
            raw_summary_json,
            schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            outcome_id,
            str(plan_row["plan_id"]),
            str(plan_row["request_id"]),
            str(plan_row["alert_id"]),
            str(plan_row["symbol"]),
            str(plan_row["side"]),
            _optional_str(plan_row["market_type"]),
            _optional_str(plan_row["strategy_profile"]),
            _optional_str(plan_row["strategy_version"]),
            _optional_str(plan_row["config_hash"]),
            _optional_str(plan_row["strategy_id"]),
            _optional_str(plan_row["setup_family"]),
            _optional_str(plan_row["strategy_params_json"]),
            _optional_str(plan_row["strategy_params_hash"]),
            _optional_str(plan_row["entry_fill_ts_utc"]),
            closed_ts_utc,
            _optional_str(plan_row["entry_fill_ts_utc"]),
            entry_avg_px,
            entry_qty,
            latest_exit_fill_ts_utc,
            exit_avg_px,
            exit_qty,
            exit_reason,
            gross_pnl,
            None,
            gross_pnl,
            planned_stop_px,
            _optional_float(plan_row["t1_px"]),
            _optional_float(plan_row["t2_px"]),
            realized_r,
            1 if entry_qty is not None and entry_avg_px is not None else 0,
            1,
            PaperTradeOutcomeStatus.CLOSED.value,
            raw_summary_json,
            PAPER_TRADE_OUTCOME_SCHEMA_VERSION,
        ),
    )
    conn.commit()
    return get_trade_outcome_by_plan_id(conn, str(plan_row["plan_id"]))


def create_trade_outcome_from_stop_fill(
    conn: sqlite3.Connection,
    *,
    plan_row: sqlite3.Row,
    stop_fill_row: sqlite3.Row,
) -> sqlite3.Row:
    if stop_fill_row is None:
        raise RuntimeError("stop fill row is required for stop-close outcome creation")
    return create_trade_outcome_for_closed_plan(
        conn,
        plan_row=plan_row,
        exit_reason=PaperOrderRole.STOP.value,
    )


def create_trade_outcome_from_tp2_fill(
    conn: sqlite3.Connection,
    *,
    plan_row: sqlite3.Row,
    tp2_fill_row: sqlite3.Row,
) -> sqlite3.Row:
    if tp2_fill_row is None:
        raise RuntimeError("tp2 fill row is required for tp2-close outcome creation")
    return create_trade_outcome_for_closed_plan(
        conn,
        plan_row=plan_row,
        exit_reason=PaperOrderRole.TP2.value,
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _compute_gross_pnl(
    *,
    side: str,
    entry_avg_px: float | None,
    exit_avg_px: float | None,
    exit_qty: float | None,
) -> float | None:
    if entry_avg_px is None or exit_avg_px is None or exit_qty is None:
        return None
    if side == "LONG":
        return (exit_avg_px - entry_avg_px) * exit_qty
    if side == "SHORT":
        return (entry_avg_px - exit_avg_px) * exit_qty
    return None


def _compute_realized_r(
    *,
    side: str,
    entry_avg_px: float | None,
    entry_qty: float | None,
    exit_avg_px: float | None,
    exit_qty: float | None,
    planned_stop_px: float | None,
    gross_pnl: float | None = None,
) -> float | None:
    if gross_pnl is None:
        gross_pnl = _compute_gross_pnl(
            side=side,
            entry_avg_px=entry_avg_px,
            exit_avg_px=exit_avg_px,
            exit_qty=exit_qty,
        )
    if gross_pnl is None or entry_avg_px is None or entry_qty is None or planned_stop_px is None:
        return None
    if side == "LONG":
        risk_per_unit = entry_avg_px - planned_stop_px
    elif side == "SHORT":
        risk_per_unit = planned_stop_px - entry_avg_px
    else:
        return None
    denominator = risk_per_unit * entry_qty
    if denominator <= 0:
        return None
    return gross_pnl / denominator
