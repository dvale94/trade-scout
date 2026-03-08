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
    clauses: list[str] = []
    params: list[Any] = []

    # Applied only where columns exist in the selected object.
    if market_type:
        clauses.append("market_type = ?")
        params.append(market_type)

    if alert_status:
        clauses.append("alert_status = ?")
        params.append(alert_status)

    if entered_only:
        clauses.append("COALESCE(entered, FALSE) = TRUE")

    if closed_only:
        clauses.append("pnl_R IS NOT NULL")

    sql = f"SELECT * FROM {table_or_view}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    con = connect_research_db(duckdb_path=duckdb_path, read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def _safe_select(**kwargs) -> pd.DataFrame:
    try:
        return _run_select(**kwargs)
    except duckdb.Error:
        return pd.DataFrame()


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
