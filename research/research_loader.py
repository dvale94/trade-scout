from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


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
    """
    Open a DuckDB connection for research queries.

    Parameters:
      - duckdb_path: explicit path to research.duckdb (defaults to DUCKDB_PATH env or logs/research.duckdb)
      - read_only: use read-only mode by default for safer notebook/script querying
    """
    db_path = _resolve_duckdb_path(duckdb_path)
    return duckdb.connect(str(db_path), read_only=read_only)


def _query_df(
    sql: str,
    params: list[Any] | None = None,
    duckdb_path: str | Path | None = None,
) -> pd.DataFrame:
    con = connect_research_db(duckdb_path=duckdb_path, read_only=True)
    try:
        if params is None:
            return con.execute(sql).df()
        return con.execute(sql, params).df()
    finally:
        con.close()


def load_trades_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    where_sql: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM trades"
    conditions: list[str] = []
    params: list[Any] = []

    if market_type:
        conditions.append("market_type = ?")
        params.append(market_type)

    if where_sql:
        conditions.append(f"({where_sql})")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY signal_ts_utc DESC NULLS LAST"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    try:
        return _query_df(query, params=params, duckdb_path=duckdb_path)
    except duckdb.Error:
        return pd.DataFrame()


def load_executions_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    where_sql: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM executions_enriched"
    conditions: list[str] = []
    params: list[Any] = []

    if market_type:
        conditions.append("market_type = ?")
        params.append(market_type)

    if where_sql:
        conditions.append(f"({where_sql})")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY exec_ts_utc DESC NULLS LAST"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    try:
        return _query_df(query, params=params, duckdb_path=duckdb_path)
    except duckdb.Error:
        # Backward compatibility: if the enriched view does not exist yet.
        query = "SELECT * FROM executions"
        conditions = []
        params = []

        if where_sql:
            conditions.append(f"({where_sql})")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY exec_ts_utc DESC NULLS LAST"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        try:
            return _query_df(query, params=params, duckdb_path=duckdb_path)
        except duckdb.Error:
            return pd.DataFrame()


def load_alerts_intents_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
    alert_status: str | None = None,
    where_sql: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM alerts_intents"
    conditions: list[str] = []
    params: list[Any] = []

    if market_type:
        conditions.append("market_type = ?")
        params.append(market_type)

    if alert_status:
        conditions.append("alert_status = ?")
        params.append(alert_status)

    if where_sql:
        conditions.append(f"({where_sql})")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY signal_ts_utc DESC NULLS LAST"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    try:
        return _query_df(query, params=params, duckdb_path=duckdb_path)
    except duckdb.Error:
        return pd.DataFrame()


def load_strategy_summary_sql(
    duckdb_path: str | Path | None = None,
    market_type: str | None = _DEFAULT_MARKET_TYPE,
) -> pd.DataFrame:
    query = "SELECT * FROM strategy_summary"
    params: list[Any] = []

    if market_type:
        query += " WHERE market_type = ?"
        params.append(market_type)

    query += " ORDER BY strategy_profile, tier"

    try:
        return _query_df(query, params=params, duckdb_path=duckdb_path)
    except duckdb.Error:
        return pd.DataFrame()
