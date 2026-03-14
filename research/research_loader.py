from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_DEFAULT_DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", _DEFAULT_LOG_DIR / "research.duckdb"))
_DEFAULT_MARKET_TYPE = os.getenv("MARKET_TYPE", "crypto").strip().lower()


def _resolve_duckdb_path(duckdb_path: str | Path | None = None) -> Path:
    if duckdb_path is None:
        return _DEFAULT_DUCKDB_PATH
    return Path(duckdb_path)


def connect_research_db(
    duckdb_path: str | Path | None = None,
    read_only: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection for research queries."""
    db_path = _resolve_duckdb_path(duckdb_path)
    return duckdb.connect(str(db_path), read_only=read_only)


def _table_columns(
    con: duckdb.DuckDBPyConnection,
    table_or_view: str,
) -> set[str]:
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info('{table_or_view}')").fetchall()}
    except duckdb.Error:
        return set()


def _run_select(
    table_or_view: str,
    *,
    duckdb_path: str | Path | None = None,
    order_by: str | None = None,
    limit: int | None = None,
    market_type: str | None = None,
    alert_status: str | None = None,
    entered_only: bool = False,
    closed_only: bool = False,
) -> pd.DataFrame:
    con = connect_research_db(duckdb_path=duckdb_path, read_only=True)
    try:
        cols = _table_columns(con, table_or_view)
        if not cols:
            return pd.DataFrame()

        clauses: list[str] = []
        params: list[Any] = []

        if market_type and "market_type" in cols:
            clauses.append("market_type = ?")
            params.append(market_type)

        if alert_status and "alert_status" in cols:
            clauses.append("alert_status = ?")
            params.append(alert_status)

        if entered_only and "entered" in cols:
            clauses.append("COALESCE(entered, FALSE) = TRUE")

        if closed_only and "pnl_R" in cols:
            clauses.append("pnl_R IS NOT NULL")

        sql = f"SELECT * FROM {table_or_view}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        return con.execute(sql, params).df()
    finally:
        con.close()


def _safe_select(**kwargs) -> pd.DataFrame:
    try:
        return _run_select(**kwargs)
    except Exception:
        return pd.DataFrame()


def research_object_exists_sql(
    name: str,
    duckdb_path: str | Path | None = None,
) -> bool:
    con = connect_research_db(duckdb_path=duckdb_path, read_only=True)
    try:
        return bool(_table_columns(con, name))
    finally:
        con.close()


def list_research_objects_sql(
    duckdb_path: str | Path | None = None,
) -> pd.DataFrame:
    con = connect_research_db(duckdb_path=duckdb_path, read_only=True)
    try:
        return con.execute("SHOW ALL TABLES").df()
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def load_trades_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    entered_only: bool = False,
    closed_only: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="trades",
        duckdb_path=duckdb_path,
        order_by="signal_ts_utc DESC NULLS LAST",
        limit=limit,
        market_type=market_type,
        entered_only=entered_only,
        closed_only=closed_only,
    )


def load_trades_enriched_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    entered_only: bool = False,
    closed_only: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="trades_enriched",
        duckdb_path=duckdb_path,
        order_by="signal_ts DESC NULLS LAST",
        limit=limit,
        market_type=market_type,
        entered_only=entered_only,
        closed_only=closed_only,
    )


def load_executions_sql(
    duckdb_path: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="executions",
        duckdb_path=duckdb_path,
        order_by="exec_ts_utc DESC NULLS LAST",
        limit=limit,
    )


def load_executions_enriched_sql(
    duckdb_path: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="executions_enriched",
        duckdb_path=duckdb_path,
        order_by="exec_ts DESC NULLS LAST",
        limit=limit,
    )


def load_outcomes_sql(
    duckdb_path: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="outcomes",
        duckdb_path=duckdb_path,
        order_by="exit_ts_utc DESC NULLS LAST",
        limit=limit,
    )


def load_telemetry_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="telemetry",
        duckdb_path=duckdb_path,
        order_by="timestamp_utc DESC NULLS LAST",
        limit=limit,
        market_type=market_type,
    )


def load_alerts_intents_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    alert_status: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="alerts_intents",
        duckdb_path=duckdb_path,
        order_by="signal_ts_utc DESC NULLS LAST",
        limit=limit,
        market_type=market_type,
        alert_status=alert_status,
    )


def load_alert_context_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="alert_context",
        duckdb_path=duckdb_path,
        order_by="signal_ts_utc DESC NULLS LAST",
        limit=limit,
        market_type=market_type,
    )


def load_strategy_summary_sql(
    duckdb_path: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="strategy_summary",
        duckdb_path=duckdb_path,
        order_by="strategy_profile, tier",
        limit=limit,
    )


def load_alert_funnel_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="alert_funnel",
        duckdb_path=duckdb_path,
        order_by="strategy_profile, tier",
        limit=limit,
        market_type=market_type,
    )


def load_symbol_regime_summary_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="symbol_regime_summary",
        duckdb_path=duckdb_path,
        order_by="symbol, anchor_regime, tier",
        limit=limit,
        market_type=market_type,
    )


def load_schema_coverage_summary_sql(
    duckdb_path: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    return _safe_select(
        table_or_view="schema_coverage_summary",
        duckdb_path=duckdb_path,
        order_by="source_name",
        limit=limit,
    )


def load_research_frames_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
) -> dict[str, pd.DataFrame]:
    return {
        "trades": load_trades_enriched_sql(duckdb_path=duckdb_path, market_type=market_type),
        "alerts_intents": load_alerts_intents_sql(duckdb_path=duckdb_path, market_type=market_type),
        "executions": load_executions_sql(duckdb_path=duckdb_path),
        "outcomes": load_outcomes_sql(duckdb_path=duckdb_path),
        "telemetry": load_telemetry_sql(duckdb_path=duckdb_path, market_type=market_type),
        "schema_coverage": load_schema_coverage_summary_sql(duckdb_path=duckdb_path),
    }
