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


def create_views(con: duckdb.DuckDBPyConnection, csvs: dict[str, Path]) -> None:
    # 1) trades + 2) trades_enriched
    if "alerts_intents" in csvs and "outcomes" in csvs:
        intent_cols = {
            row[1] for row in con.execute("PRAGMA table_info('alerts_intents')").fetchall()
        }

        def intent_expr(col: str, fallback_expr: str = "NULL") -> str:
            if col in intent_cols:
                return f"i.{col}"
            return f"{fallback_expr} AS {col}"

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
              {intent_expr('signal_hour_utc', "EXTRACT(HOUR FROM TRY_CAST(i.signal_ts_utc AS TIMESTAMP))")},
              {intent_expr('signal_day_of_week', "STRFTIME(TRY_CAST(i.signal_ts_utc AS TIMESTAMP), '%a')")},
              i.bar_start_ts_utc,
              i.bar_end_ts_utc,
              i.timeframe_min,
              i.score,
              {intent_expr('atr')},
              {intent_expr('ma20')},
              {intent_expr('ma50')},
              {intent_expr('ma200')},
              {intent_expr('breakout_level')},
              {intent_expr('breakout_distance_atr')},
              {intent_expr('ma20_distance_atr')},
              {intent_expr('ma50_distance_atr')},
              {intent_expr('ma200_distance_atr')},
              {intent_expr('ma20_slope_pct')},
              {intent_expr('ma50_slope_pct')},
              {intent_expr('range_pct')},
              {intent_expr('close_location_in_bar')},
              i.anchor_symbol,
              i.anchor_regime,
              i.anchor_close,
              i.anchor_ma50,
              i.anchor_ma200,
              i.vol_z,
              i.extended_atr,
              i.signal_px,
              i.entry_low,
              i.entry_high,
              {intent_expr('entry_zone_width_pct')},
              i.stop_px,
              i.t1_px,
              i.t2_px,
              i.alert_status,
              o.entered,
              o.entry_ts_utc,
              o.entry_px,
              o.exit_ts_utc,
              o.exit_px,
              o.exit_reason,
              o.pnl_R,
              o.mfe_R,
              o.mae_R,
              o.time_in_trade_s,
              o.time_to_t1_s,
              o.time_to_t2_s,
              o.time_to_stop_s,
              o.t1_touched,
              o.t2_touched,
              o.stop_touched
            FROM alerts_intents i
            LEFT JOIN outcomes o USING (alert_id)
            WHERE i.alert_status = 'ALERT_CREATED'
            """
        )

        con.execute("DROP VIEW IF EXISTS trades_enriched")
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
        con.execute("DROP VIEW IF EXISTS executions_enriched")
        con.execute(
            """
            CREATE VIEW executions_enriched AS
            SELECT
              e.*,
              TRY_CAST(e.exec_ts_utc AS TIMESTAMP) AS exec_ts,
              CAST(TRY_CAST(e.exec_ts_utc AS TIMESTAMP) AS DATE) AS exec_date,
              EXTRACT(HOUR FROM TRY_CAST(e.exec_ts_utc AS TIMESTAMP)) AS exec_hour_utc,
              CASE WHEN e.exec_type = 'ENTRY' THEN TRUE ELSE FALSE END AS is_entry,
              CASE WHEN e.exec_type <> 'ENTRY' THEN TRUE ELSE FALSE END AS is_exit
            FROM executions e
            """
        )

    # 4) strategy_summary (real status values only)
    if "alerts_intents" in csvs:
        con.execute("DROP VIEW IF EXISTS strategy_summary")
        con.execute(
            """
            CREATE VIEW strategy_summary AS
            SELECT
              strategy_profile,
              tier,
              COUNT(*) AS intents_total,
              COUNT(*) FILTER (WHERE alert_status = 'ALERT_CREATED') AS alerts_created,
              COUNT(*) FILTER (WHERE alert_status = 'SKIPPED_COOLDOWN') AS skipped_cooldown,
              COUNT(*) FILTER (WHERE alert_status = 'SKIPPED_INVALID_STOP') AS skipped_invalid_stop,
              COUNT(*) FILTER (WHERE alert_status = 'SKIPPED_TIGHT_STOP') AS skipped_tight_stop,
              COUNT(*) FILTER (WHERE alert_status = 'ERROR_ALERT_PIPELINE') AS pipeline_errors,
              AVG(score) AS avg_score
            FROM alerts_intents
            GROUP BY 1,2
            ORDER BY 1,2
            """
        )

    # 5) alert_funnel (intent-stage only; no pre-threshold claims)
    if "alerts_intents" in csvs and "outcomes" in csvs:
        con.execute("DROP VIEW IF EXISTS alert_funnel")
        con.execute(
            """
            CREATE VIEW alert_funnel AS
            WITH intent_rollup AS (
              SELECT
                i.market_type,
                i.strategy_profile,
                i.tier,
                COUNT(*) AS total_intent_rows,
                COUNT(*) FILTER (WHERE i.alert_status = 'ALERT_CREATED') AS alerts_created,
                COUNT(*) FILTER (WHERE i.alert_status = 'SKIPPED_COOLDOWN') AS skipped_cooldown,
                COUNT(*) FILTER (WHERE i.alert_status = 'SKIPPED_INVALID_STOP') AS skipped_invalid_stop,
                COUNT(*) FILTER (WHERE i.alert_status = 'SKIPPED_TIGHT_STOP') AS skipped_tight_stop,
                COUNT(*) FILTER (WHERE i.alert_status = 'ERROR_ALERT_PIPELINE') AS pipeline_errors
              FROM alerts_intents i
              GROUP BY 1,2,3
            ),
            created_outcomes AS (
              SELECT
                i.market_type,
                i.strategy_profile,
                i.tier,
                COUNT(*) FILTER (WHERE COALESCE(o.entered, FALSE)) AS entered_count,
                COUNT(*) FILTER (WHERE o.pnl_R IS NOT NULL) AS closed_trade_count,
                COUNT(*) FILTER (WHERE o.exit_reason = 'NO_ENTRY_FILL') AS no_entry_fill_count,
                COUNT(*) FILTER (WHERE o.exit_reason = 'STOP') AS stop_exit_count,
                COUNT(*) FILTER (WHERE o.exit_reason = 'TIMEOUT') AS timeout_exit_count,
                COUNT(*) FILTER (WHERE o.exit_reason = 'T2') AS t2_exit_count
              FROM alerts_intents i
              LEFT JOIN outcomes o USING (alert_id)
              WHERE i.alert_status = 'ALERT_CREATED'
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
        telemetry_cols = {
            row[1] for row in con.execute("PRAGMA table_info('telemetry')").fetchall()
        }
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


def main() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))

    csvs = existing_csvs()
    for table, path in csvs.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            "CREATE TABLE " + table + " AS SELECT * FROM read_csv_auto(?, header=True, ignore_errors=True)",
            [str(path)],
        )

    create_views(con, csvs)

    con.execute("CHECKPOINT")
    con.close()
    print(f"DuckDB refreshed at {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
