from __future__ import annotations

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", LOG_DIR / "research.duckdb"))
MARKET_TYPE = os.getenv("MARKET_TYPE", "crypto").strip().lower()

CSV_MAP = {
    f"{MARKET_TYPE}_alerts.csv": "alerts",
    f"{MARKET_TYPE}_alerts_intents.csv": "alerts_intents",
    f"{MARKET_TYPE}_alert_context.csv": "alert_context_raw",
    f"{MARKET_TYPE}_executions.csv": "executions",
    f"{MARKET_TYPE}_outcomes.csv": "outcomes",
    f"{MARKET_TYPE}_telemetry.csv": "telemetry",
}


def existing_csvs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for filename, table in CSV_MAP.items():
        path = LOG_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            out[table] = path
    return out


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except duckdb.Error:
        return set()


def raw_value_expr(alias: str, cols: set[str], col: str, fallback_expr: str = "NULL") -> str:
    if col in cols:
        return f"{alias}.{col}" if alias else col
    return fallback_expr


def raw_select_expr(alias: str, cols: set[str], col: str, fallback_expr: str = "NULL") -> str:
    return f"{raw_value_expr(alias, cols, col, fallback_expr)} AS {col}"


def raw_numeric_value_expr(alias: str, cols: set[str], col: str) -> str:
    return f"TRY_CAST({raw_value_expr(alias, cols, col)} AS DOUBLE)"


def raw_numeric_select_expr(alias: str, cols: set[str], col: str) -> str:
    return f"{raw_numeric_value_expr(alias, cols, col)} AS {col}"


def raw_boolean_value_expr(alias: str, cols: set[str], col: str) -> str:
    return f"TRY_CAST({raw_value_expr(alias, cols, col)} AS BOOLEAN)"


def raw_boolean_select_expr(alias: str, cols: set[str], col: str) -> str:
    return f"{raw_boolean_value_expr(alias, cols, col)} AS {col}"


def raw_int_select_expr(alias: str, cols: set[str], col: str) -> str:
    return f"TRY_CAST({raw_value_expr(alias, cols, col)} AS INTEGER) AS {col}"


def raw_count_expr(table: str, cols: set[str], col: str) -> str:
    if col in cols:
        return f"COUNT(*) FILTER (WHERE {col} IS NOT NULL) AS {col}_non_null_rows"
    return f"0 AS {col}_non_null_rows"


def safe_row_count(con: duckdb.DuckDBPyConnection, table_or_view: str) -> int | None:
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table_or_view}").fetchone()[0])
    except duckdb.Error:
        return None


def print_build_summary(con: duckdb.DuckDBPyConnection, csvs: dict[str, Path]) -> None:
    raw_tables = list(csvs.keys())
    alert_context_exists = "yes" if safe_row_count(con, "alert_context") is not None else "no"

    print(f"[DUCKDB] raw_tables_loaded={', '.join(raw_tables) if raw_tables else 'none'}")
    for table in raw_tables:
        row_count = safe_row_count(con, table)
        row_count_label = row_count if row_count is not None else "missing"
        print(f"[DUCKDB] raw.{table} rows={row_count_label}")

    print(f"[DUCKDB] view.alert_context exists={alert_context_exists}")

    for view_name in [
        "trades",
        "trades_enriched",
        "executions_enriched",
        "strategy_summary",
        "alert_funnel",
        "schema_coverage_summary",
    ]:
        row_count = safe_row_count(con, view_name)
        if row_count is not None:
            print(f"[DUCKDB] view.{view_name} rows={row_count}")


def create_schema_coverage_summary(con: duckdb.DuckDBPyConnection, csvs: dict[str, Path]) -> None:
    con.execute("DROP VIEW IF EXISTS schema_coverage_summary")

    select_statements: list[str] = []

    source_specs = [
        ("alerts", "alerts"),
        ("alerts_intents", "alerts_intents"),
        ("alert_context", "alert_context_raw"),
        ("executions", "executions"),
        ("outcomes", "outcomes"),
        ("telemetry", "telemetry"),
    ]

    for source_name, raw_table in source_specs:
        if raw_table not in csvs:
            select_statements.append(
                f"""
                SELECT
                  '{source_name}' AS source_name,
                  0 AS row_count,
                  FALSE AS has_schema_version,
                  FALSE AS has_score,
                  0 AS score_non_null_rows,
                  FALSE AS has_signal_hour_utc,
                  0 AS signal_hour_utc_non_null_rows,
                  FALSE AS has_signal_day_of_week,
                  0 AS signal_day_of_week_non_null_rows,
                  FALSE AS has_atr,
                  0 AS atr_non_null_rows,
                  FALSE AS has_vol_z,
                  0 AS vol_z_non_null_rows,
                  FALSE AS has_anchor_regime,
                  0 AS anchor_regime_non_null_rows,
                  FALSE AS has_pnl_R,
                  0 AS pnl_R_non_null_rows
                """
            )
            continue

        cols = table_columns(con, raw_table)
        select_statements.append(
            f"""
            SELECT
              '{source_name}' AS source_name,
              COUNT(*) AS row_count,
              {'TRUE' if 'schema_version' in cols else 'FALSE'} AS has_schema_version,
              {'TRUE' if 'score' in cols else 'FALSE'} AS has_score,
              {raw_count_expr(source_name, cols, 'score')},
              {'TRUE' if 'signal_hour_utc' in cols else 'FALSE'} AS has_signal_hour_utc,
              {raw_count_expr(source_name, cols, 'signal_hour_utc')},
              {'TRUE' if 'signal_day_of_week' in cols else 'FALSE'} AS has_signal_day_of_week,
              {raw_count_expr(source_name, cols, 'signal_day_of_week')},
              {'TRUE' if 'atr' in cols else 'FALSE'} AS has_atr,
              {raw_count_expr(source_name, cols, 'atr')},
              {'TRUE' if 'vol_z' in cols else 'FALSE'} AS has_vol_z,
              {raw_count_expr(source_name, cols, 'vol_z')},
              {'TRUE' if 'anchor_regime' in cols else 'FALSE'} AS has_anchor_regime,
              {raw_count_expr(raw_table, cols, 'anchor_regime')},
              {'TRUE' if 'pnl_R' in cols else 'FALSE'} AS has_pnl_R,
              {raw_count_expr(raw_table, cols, 'pnl_R')}
            FROM {raw_table}
            """
        )

    con.execute("CREATE VIEW schema_coverage_summary AS " + "\nUNION ALL\n".join(select_statements))


def create_views(con: duckdb.DuckDBPyConnection, csvs: dict[str, Path]) -> None:
    con.execute("DROP VIEW IF EXISTS alert_context")
    if "alert_context_raw" in csvs:
        context_cols = table_columns(con, "alert_context_raw")
        con.execute(
            f"""
            CREATE VIEW alert_context AS
            SELECT
              {raw_select_expr('', context_cols, 'schema_version')},
              {raw_select_expr('', context_cols, 'alert_id')},
              {raw_select_expr('', context_cols, 'market_type')},
              {raw_select_expr('', context_cols, 'strategy_profile')},
              {raw_select_expr('', context_cols, 'strategy_version')},
              {raw_select_expr('', context_cols, 'config_hash')},
              {raw_select_expr('', context_cols, 'signal_ts_utc')},
              {raw_int_select_expr('', context_cols, 'signal_hour_utc')},
              {raw_select_expr('', context_cols, 'signal_day_of_week')},
              {raw_select_expr('', context_cols, 'bar_start_ts_utc')},
              {raw_select_expr('', context_cols, 'bar_end_ts_utc')},
              {raw_numeric_select_expr('', context_cols, 'score')},
              {raw_numeric_select_expr('', context_cols, 'atr')},
              {raw_numeric_select_expr('', context_cols, 'ma20')},
              {raw_numeric_select_expr('', context_cols, 'ma50')},
              {raw_numeric_select_expr('', context_cols, 'ma200')},
              {raw_numeric_select_expr('', context_cols, 'breakout_level')},
              {raw_numeric_select_expr('', context_cols, 'breakout_distance_atr')},
              {raw_numeric_select_expr('', context_cols, 'ma20_distance_atr')},
              {raw_numeric_select_expr('', context_cols, 'ma50_distance_atr')},
              {raw_numeric_select_expr('', context_cols, 'ma200_distance_atr')},
              {raw_numeric_select_expr('', context_cols, 'ma20_slope_pct')},
              {raw_numeric_select_expr('', context_cols, 'ma50_slope_pct')},
              {raw_numeric_select_expr('', context_cols, 'range_pct')},
              {raw_numeric_select_expr('', context_cols, 'close_location_in_bar')},
              {raw_numeric_select_expr('', context_cols, 'entry_zone_width_pct')},
              {raw_select_expr('', context_cols, 'anchor_symbol')},
              {raw_select_expr('', context_cols, 'anchor_regime')},
              {raw_numeric_select_expr('', context_cols, 'anchor_close')},
              {raw_numeric_select_expr('', context_cols, 'anchor_ma50')},
              {raw_numeric_select_expr('', context_cols, 'anchor_ma200')},
              {raw_numeric_select_expr('', context_cols, 'vol_z')},
              {raw_numeric_select_expr('', context_cols, 'extended_atr')}
            FROM alert_context_raw
            """
        )

    # 1) trades + 2) trades_enriched
    if "alerts_intents" in csvs and "outcomes" in csvs:
        intent_cols = table_columns(con, "alerts_intents")
        outcome_cols = table_columns(con, "outcomes")

        con.execute("DROP VIEW IF EXISTS trades")
        con.execute(
            f"""
            CREATE VIEW trades AS
            SELECT
              i.alert_id,
              i.symbol,
              i.side,
              i.market_type,
              i.strategy_profile,
              i.strategy_version,
              i.config_hash,
              i.tier,
              i.signal_ts_utc,
              {raw_select_expr('i', intent_cols, 'signal_hour_utc', "EXTRACT(HOUR FROM TRY_CAST(i.signal_ts_utc AS TIMESTAMP))")},
              {raw_select_expr('i', intent_cols, 'signal_day_of_week', "STRFTIME(TRY_CAST(i.signal_ts_utc AS TIMESTAMP), '%a')")},
              {raw_select_expr('i', intent_cols, 'bar_start_ts_utc')},
              {raw_select_expr('i', intent_cols, 'bar_end_ts_utc')},
              {raw_int_select_expr('i', intent_cols, 'timeframe_min')},
              {raw_numeric_select_expr('i', intent_cols, 'score')},
              {raw_numeric_select_expr('i', intent_cols, 'atr')},
              {raw_numeric_select_expr('i', intent_cols, 'ma20')},
              {raw_numeric_select_expr('i', intent_cols, 'ma50')},
              {raw_numeric_select_expr('i', intent_cols, 'ma200')},
              {raw_numeric_select_expr('i', intent_cols, 'breakout_level')},
              {raw_numeric_select_expr('i', intent_cols, 'breakout_distance_atr')},
              {raw_numeric_select_expr('i', intent_cols, 'ma20_distance_atr')},
              {raw_numeric_select_expr('i', intent_cols, 'ma50_distance_atr')},
              {raw_numeric_select_expr('i', intent_cols, 'ma200_distance_atr')},
              {raw_numeric_select_expr('i', intent_cols, 'ma20_slope_pct')},
              {raw_numeric_select_expr('i', intent_cols, 'ma50_slope_pct')},
              {raw_numeric_select_expr('i', intent_cols, 'range_pct')},
              {raw_numeric_select_expr('i', intent_cols, 'close_location_in_bar')},
              {raw_select_expr('i', intent_cols, 'anchor_symbol')},
              {raw_select_expr('i', intent_cols, 'anchor_regime')},
              {raw_numeric_select_expr('i', intent_cols, 'anchor_close')},
              {raw_numeric_select_expr('i', intent_cols, 'anchor_ma50')},
              {raw_numeric_select_expr('i', intent_cols, 'anchor_ma200')},
              {raw_numeric_select_expr('i', intent_cols, 'vol_z')},
              {raw_numeric_select_expr('i', intent_cols, 'extended_atr')},
              {raw_numeric_select_expr('i', intent_cols, 'signal_px')},
              {raw_numeric_select_expr('i', intent_cols, 'entry_low')},
              {raw_numeric_select_expr('i', intent_cols, 'entry_high')},
              {raw_numeric_select_expr('i', intent_cols, 'entry_zone_width_pct')},
              {raw_numeric_select_expr('i', intent_cols, 'stop_px')},
              {raw_numeric_select_expr('i', intent_cols, 't1_px')},
              {raw_numeric_select_expr('i', intent_cols, 't2_px')},
              {raw_select_expr('i', intent_cols, 'alert_status')},
              {raw_boolean_select_expr('o', outcome_cols, 'entered')},
              {raw_select_expr('o', outcome_cols, 'entry_ts_utc')},
              {raw_numeric_select_expr('o', outcome_cols, 'entry_px')},
              {raw_select_expr('o', outcome_cols, 'exit_ts_utc')},
              {raw_numeric_select_expr('o', outcome_cols, 'exit_px')},
              {raw_select_expr('o', outcome_cols, 'exit_reason')},
              {raw_numeric_select_expr('o', outcome_cols, 'pnl_R')},
              {raw_numeric_select_expr('o', outcome_cols, 'mfe_R')},
              {raw_numeric_select_expr('o', outcome_cols, 'mae_R')},
              {raw_numeric_select_expr('o', outcome_cols, 'time_in_trade_s')},
              {raw_numeric_select_expr('o', outcome_cols, 'time_to_t1_s')},
              {raw_numeric_select_expr('o', outcome_cols, 'time_to_t2_s')},
              {raw_numeric_select_expr('o', outcome_cols, 'time_to_stop_s')},
              {raw_boolean_select_expr('o', outcome_cols, 't1_touched')},
              {raw_boolean_select_expr('o', outcome_cols, 't2_touched')},
              {raw_boolean_select_expr('o', outcome_cols, 'stop_touched')}
            FROM alerts_intents i
            LEFT JOIN outcomes o USING (alert_id)
            WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'ALERT_CREATED'
            """
        )

        con.execute("DROP VIEW IF EXISTS trades_enriched")
        if "alert_context_raw" in csvs:
            context_cols = table_columns(con, "alert_context")
            con.execute(
                f"""
                CREATE VIEW trades_enriched AS
                SELECT
                  t.*,
                  {raw_value_expr('c', context_cols, 'schema_version')} AS alert_context_schema_version,
                  {raw_numeric_value_expr('c', context_cols, 'score')} AS context_score,
                  TRY_CAST({raw_value_expr('c', context_cols, 'signal_hour_utc')} AS INTEGER) AS context_signal_hour_utc,
                  {raw_value_expr('c', context_cols, 'signal_day_of_week')} AS context_signal_day_of_week,
                  {raw_numeric_value_expr('c', context_cols, 'atr')} AS context_atr,
                  {raw_numeric_value_expr('c', context_cols, 'ma20')} AS context_ma20,
                  {raw_numeric_value_expr('c', context_cols, 'ma50')} AS context_ma50,
                  {raw_numeric_value_expr('c', context_cols, 'ma200')} AS context_ma200,
                  {raw_numeric_value_expr('c', context_cols, 'breakout_level')} AS context_breakout_level,
                  {raw_numeric_value_expr('c', context_cols, 'breakout_distance_atr')} AS context_breakout_distance_atr,
                  {raw_numeric_value_expr('c', context_cols, 'ma20_distance_atr')} AS context_ma20_distance_atr,
                  {raw_numeric_value_expr('c', context_cols, 'ma50_distance_atr')} AS context_ma50_distance_atr,
                  {raw_numeric_value_expr('c', context_cols, 'ma200_distance_atr')} AS context_ma200_distance_atr,
                  {raw_numeric_value_expr('c', context_cols, 'ma20_slope_pct')} AS context_ma20_slope_pct,
                  {raw_numeric_value_expr('c', context_cols, 'ma50_slope_pct')} AS context_ma50_slope_pct,
                  {raw_numeric_value_expr('c', context_cols, 'range_pct')} AS context_range_pct,
                  {raw_numeric_value_expr('c', context_cols, 'close_location_in_bar')} AS context_close_location_in_bar,
                  {raw_numeric_value_expr('c', context_cols, 'entry_zone_width_pct')} AS context_entry_zone_width_pct,
                  {raw_value_expr('c', context_cols, 'anchor_symbol')} AS context_anchor_symbol,
                  {raw_value_expr('c', context_cols, 'anchor_regime')} AS context_anchor_regime,
                  {raw_numeric_value_expr('c', context_cols, 'anchor_close')} AS context_anchor_close,
                  {raw_numeric_value_expr('c', context_cols, 'anchor_ma50')} AS context_anchor_ma50,
                  {raw_numeric_value_expr('c', context_cols, 'anchor_ma200')} AS context_anchor_ma200,
                  {raw_numeric_value_expr('c', context_cols, 'vol_z')} AS context_vol_z,
                  {raw_numeric_value_expr('c', context_cols, 'extended_atr')} AS context_extended_atr,
                  TRY_CAST(t.signal_ts_utc AS TIMESTAMP) AS signal_ts,
                  CAST(TRY_CAST(t.signal_ts_utc AS TIMESTAMP) AS DATE) AS signal_date,
                  EXTRACT(DOW FROM TRY_CAST(t.signal_ts_utc AS TIMESTAMP)) AS signal_dow_utc,
                  CASE WHEN t.pnl_R IS NOT NULL THEN TRUE ELSE FALSE END AS closed_trade,
                  CASE WHEN t.pnl_R > 0 THEN TRUE ELSE FALSE END AS winner,
                  CASE WHEN t.pnl_R < 0 THEN TRUE ELSE FALSE END AS loser,
                  CASE WHEN t.exit_reason = 'TIMEOUT' THEN TRUE ELSE FALSE END AS timeout_trade,
                  CASE WHEN t.exit_reason = 'NO_ENTRY_FILL' THEN TRUE ELSE FALSE END AS no_entry_fill,
                  CASE WHEN t.exit_reason = 'STOP' THEN TRUE ELSE FALSE END AS stop_exit,
                  CASE WHEN t.exit_reason = 'T2' THEN TRUE ELSE FALSE END AS t2_exit,
                  CASE
                    WHEN COALESCE(t.t1_touched, FALSE) = TRUE AND COALESCE(t.t2_touched, FALSE) = FALSE
                      THEN TRUE
                    ELSE FALSE
                  END AS hit_t1_only,
                  CASE
                    WHEN t.entry_high > 0 AND t.stop_px IS NOT NULL
                      THEN (t.entry_high - t.stop_px) / t.entry_high
                    ELSE NULL
                  END AS risk_pct
                FROM trades t
                LEFT JOIN alert_context c USING (alert_id)
                """
            )
        else:
            con.execute(
                """
                CREATE VIEW trades_enriched AS
                SELECT
                  t.*,
              TRY_CAST(t.signal_ts_utc AS TIMESTAMP) AS signal_ts,
              CAST(TRY_CAST(t.signal_ts_utc AS TIMESTAMP) AS DATE) AS signal_date,
              EXTRACT(DOW FROM TRY_CAST(t.signal_ts_utc AS TIMESTAMP)) AS signal_dow_utc,
              CASE WHEN t.pnl_R IS NOT NULL THEN TRUE ELSE FALSE END AS closed_trade,
              CASE WHEN t.pnl_R > 0 THEN TRUE ELSE FALSE END AS winner,
              CASE WHEN t.pnl_R < 0 THEN TRUE ELSE FALSE END AS loser,
              CASE WHEN t.exit_reason = 'TIMEOUT' THEN TRUE ELSE FALSE END AS timeout_trade,
              CASE WHEN t.exit_reason = 'NO_ENTRY_FILL' THEN TRUE ELSE FALSE END AS no_entry_fill,
              CASE WHEN t.exit_reason = 'STOP' THEN TRUE ELSE FALSE END AS stop_exit,
              CASE WHEN t.exit_reason = 'T2' THEN TRUE ELSE FALSE END AS t2_exit,
              CASE
                WHEN COALESCE(t.t1_touched, FALSE) = TRUE AND COALESCE(t.t2_touched, FALSE) = FALSE
                  THEN TRUE
                ELSE FALSE
              END AS hit_t1_only,
              CASE
                WHEN t.entry_high > 0 AND t.stop_px IS NOT NULL
                  THEN (t.entry_high - t.stop_px) / t.entry_high
                ELSE NULL
                  END AS risk_pct
                FROM trades t
                """
            )

    # 3) executions_enriched
    if "executions" in csvs:
        execution_cols = table_columns(con, "executions")
        con.execute("DROP VIEW IF EXISTS executions_enriched")
        con.execute(
            f"""
            CREATE VIEW executions_enriched AS
            SELECT
              e.*,
              TRY_CAST({raw_value_expr('e', execution_cols, 'exec_ts_utc')} AS TIMESTAMP) AS exec_ts,
              CAST(TRY_CAST({raw_value_expr('e', execution_cols, 'exec_ts_utc')} AS TIMESTAMP) AS DATE) AS exec_date,
              EXTRACT(HOUR FROM TRY_CAST({raw_value_expr('e', execution_cols, 'exec_ts_utc')} AS TIMESTAMP)) AS exec_hour_utc,
              CASE WHEN COALESCE({raw_value_expr('e', execution_cols, 'exec_type')}, '') = 'ENTRY' THEN TRUE ELSE FALSE END AS is_entry,
              CASE WHEN COALESCE({raw_value_expr('e', execution_cols, 'exec_type')}, '') <> 'ENTRY' THEN TRUE ELSE FALSE END AS is_exit
            FROM executions e
            """
        )

    # 4) strategy_summary (real status values only)
    if "alerts_intents" in csvs:
        intent_cols = table_columns(con, "alerts_intents")
        con.execute("DROP VIEW IF EXISTS strategy_summary")
        con.execute(
            f"""
            CREATE VIEW strategy_summary AS
            SELECT
              {raw_select_expr('', intent_cols, 'strategy_profile')},
              {raw_select_expr('', intent_cols, 'tier')},
              COUNT(*) AS intents_total,
              COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('', intent_cols, 'alert_status')}, '') = 'ALERT_CREATED') AS alerts_created,
              COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('', intent_cols, 'alert_status')}, '') = 'SKIPPED_COOLDOWN') AS skipped_cooldown,
              COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('', intent_cols, 'alert_status')}, '') = 'SKIPPED_INVALID_STOP') AS skipped_invalid_stop,
              COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('', intent_cols, 'alert_status')}, '') = 'SKIPPED_TIGHT_STOP') AS skipped_tight_stop,
              COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('', intent_cols, 'alert_status')}, '') = 'ERROR_ALERT_PIPELINE') AS pipeline_errors,
              AVG(TRY_CAST({raw_value_expr('', intent_cols, 'score')} AS DOUBLE)) AS avg_score
            FROM alerts_intents
            GROUP BY 1,2
            ORDER BY 1,2
            """
        )

    # 5) alert_funnel (intent-stage only; no pre-threshold claims)
    if "alerts_intents" in csvs and "outcomes" in csvs:
        intent_cols = table_columns(con, "alerts_intents")
        outcome_cols = table_columns(con, "outcomes")
        con.execute("DROP VIEW IF EXISTS alert_funnel")
        con.execute(
            f"""
            CREATE VIEW alert_funnel AS
            WITH intent_rollup AS (
              SELECT
                {raw_select_expr('i', intent_cols, 'market_type')},
                {raw_select_expr('i', intent_cols, 'strategy_profile')},
                {raw_select_expr('i', intent_cols, 'tier')},
                COUNT(*) AS total_intent_rows,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'ALERT_CREATED') AS alerts_created,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'SKIPPED_COOLDOWN') AS skipped_cooldown,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'SKIPPED_INVALID_STOP') AS skipped_invalid_stop,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'SKIPPED_TIGHT_STOP') AS skipped_tight_stop,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'ERROR_ALERT_PIPELINE') AS pipeline_errors
              FROM alerts_intents i
              GROUP BY 1,2,3
            ),
            created_outcomes AS (
              SELECT
                {raw_select_expr('i', intent_cols, 'market_type')},
                {raw_select_expr('i', intent_cols, 'strategy_profile')},
                {raw_select_expr('i', intent_cols, 'tier')},
                COUNT(*) FILTER (WHERE COALESCE({raw_boolean_value_expr('o', outcome_cols, 'entered')}, FALSE)) AS entered_count,
                COUNT(*) FILTER (WHERE {raw_numeric_value_expr('o', outcome_cols, 'pnl_R')} IS NOT NULL) AS closed_trade_count,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('o', outcome_cols, 'exit_reason')}, '') = 'NO_ENTRY_FILL') AS no_entry_fill_count,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('o', outcome_cols, 'exit_reason')}, '') = 'STOP') AS stop_exit_count,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('o', outcome_cols, 'exit_reason')}, '') = 'TIMEOUT') AS timeout_exit_count,
                COUNT(*) FILTER (WHERE COALESCE({raw_value_expr('o', outcome_cols, 'exit_reason')}, '') = 'T2') AS t2_exit_count
              FROM alerts_intents i
              LEFT JOIN outcomes o USING (alert_id)
              WHERE COALESCE({raw_value_expr('i', intent_cols, 'alert_status')}, '') = 'ALERT_CREATED'
              GROUP BY 1,2,3
            )
            SELECT
              ir.market_type,
              ir.strategy_profile,
              ir.tier,
              ir.total_intent_rows,
              ir.alerts_created,
              ir.skipped_cooldown,
              ir.skipped_invalid_stop,
              ir.skipped_tight_stop,
              ir.pipeline_errors,
              COALESCE(co.entered_count, 0) AS entered_count,
              COALESCE(co.closed_trade_count, 0) AS closed_trade_count,
              COALESCE(co.no_entry_fill_count, 0) AS no_entry_fill_count,
              COALESCE(co.stop_exit_count, 0) AS stop_exit_count,
              COALESCE(co.timeout_exit_count, 0) AS timeout_exit_count,
              COALESCE(co.t2_exit_count, 0) AS t2_exit_count
            FROM intent_rollup ir
            LEFT JOIN created_outcomes co
              ON ir.market_type = co.market_type
             AND ir.strategy_profile = co.strategy_profile
             AND ir.tier = co.tier
            ORDER BY 1,2,3
            """
        )

    # 6) symbol_regime_summary
    if "alerts_intents" in csvs and "outcomes" in csvs:
        con.execute("DROP VIEW IF EXISTS symbol_regime_summary")
        con.execute(
            """
            CREATE VIEW symbol_regime_summary AS
            SELECT
              te.market_type,
              te.symbol,
              te.anchor_regime,
              te.tier,
              COUNT(*) AS trades,
              SUM(CASE WHEN COALESCE(te.entered, FALSE) THEN 1 ELSE 0 END) AS entered_count,
              SUM(CASE WHEN te.closed_trade THEN 1 ELSE 0 END) AS closed_trades,
              SUM(CASE WHEN te.winner THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN te.loser THEN 1 ELSE 0 END) AS losses,
              AVG(CASE WHEN te.closed_trade THEN te.pnl_R END) AS avg_pnl_R,
              MEDIAN(CASE WHEN te.closed_trade THEN te.pnl_R END) AS median_pnl_R,
              AVG(CASE WHEN te.closed_trade THEN te.pnl_R END) AS expectancy_proxy,
              SUM(CASE WHEN te.winner THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(CASE WHEN te.closed_trade THEN 1 ELSE 0 END), 0) AS win_rate
            FROM trades_enriched te
            GROUP BY 1,2,3,4
            ORDER BY 1,2,3,4
            """
        )

    # 7) optional telemetry_summary
    if "telemetry" in csvs:
        telemetry_cols = table_columns(con, "telemetry")
        if "hour_start_utc" in telemetry_cols:
            ts_col = "hour_start_utc"
        elif "timestamp_utc" in telemetry_cols:
            ts_col = "timestamp_utc"
        else:
            ts_col = None

        con.execute("DROP VIEW IF EXISTS telemetry_summary")
        if ts_col:
            con.execute(
                f"""
                CREATE VIEW telemetry_summary AS
                SELECT
                  t.*,
                  TRY_CAST(t.{ts_col} AS TIMESTAMP) AS hour_start_ts,
                  CAST(TRY_CAST(t.{ts_col} AS TIMESTAMP) AS DATE) AS hour_date_utc,
                  EXTRACT(HOUR FROM TRY_CAST(t.{ts_col} AS TIMESTAMP)) AS hour_utc
                FROM telemetry t
                """
            )
        else:
            con.execute(
                """
                CREATE VIEW telemetry_summary AS
                SELECT * FROM telemetry
                """
            )

    create_schema_coverage_summary(con, csvs)


def main() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))

    con.execute("DROP VIEW IF EXISTS alert_context")
    con.execute("DROP TABLE IF EXISTS alert_context")
    con.execute("DROP TABLE IF EXISTS alert_context_raw")

    csvs = existing_csvs()
    for table, path in csvs.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            "CREATE TABLE " + table + " AS SELECT * FROM read_csv_auto(?, header=True, ignore_errors=True)",
            [str(path)],
        )

    create_views(con, csvs)
    print_build_summary(con, csvs)

    con.execute("CHECKPOINT")
    con.close()
    print(f"DuckDB refreshed at {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
