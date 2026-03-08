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
    if "alerts_intents" in csvs and "outcomes" in csvs:
        con.execute("DROP VIEW IF EXISTS trades")
        con.execute(
            """
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
              i.bar_start_ts_utc,
              i.bar_end_ts_utc,
              i.timeframe_min,
              i.score,
              i.anchor_regime,
              i.vol_z,
              i.signal_px,
              i.entry_low,
              i.entry_high,
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
              o.stop_touched,
              (o.time_in_trade_s / 60.0) AS time_in_trade_min
            FROM alerts_intents i
            LEFT JOIN outcomes o USING (alert_id)
            WHERE i.alert_status = 'ALERT_CREATED'
            """
        )

        con.execute("DROP VIEW IF EXISTS edge_report")
        con.execute(
            """
            CREATE VIEW edge_report AS
            SELECT
              market_type,
              strategy_profile,
              tier,
              anchor_regime,
              COUNT(*) AS alerts_created,
              SUM(CASE WHEN entered THEN 1 ELSE 0 END) AS entered_count,
              SUM(CASE WHEN pnl_R IS NOT NULL THEN 1 ELSE 0 END) AS closed_count,
              AVG(CASE WHEN pnl_R IS NOT NULL THEN pnl_R END) AS avg_R,
              AVG(CASE WHEN pnl_R IS NOT NULL AND pnl_R > 0 THEN pnl_R END) AS avg_win_R,
              AVG(CASE WHEN pnl_R IS NOT NULL AND pnl_R < 0 THEN pnl_R END) AS avg_loss_R,
              SUM(CASE WHEN pnl_R IS NOT NULL AND pnl_R > 0 THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(CASE WHEN pnl_R IS NOT NULL THEN 1 ELSE 0 END), 0) AS win_rate,
              AVG(mfe_R) AS avg_mfe_R,
              AVG(mae_R) AS avg_mae_R,
              AVG(time_in_trade_s) AS avg_time_in_trade_s
            FROM trades
            GROUP BY 1,2,3,4
            ORDER BY 1,2,3,4
            """
        )

    if "alerts_intents" in csvs and "executions" in csvs:
        con.execute("DROP VIEW IF EXISTS executions_enriched")
        con.execute(
            """
            CREATE VIEW executions_enriched AS
            SELECT
              e.exec_id,
              e.alert_id,
              i.symbol,
              i.market_type,
              i.strategy_profile,
              i.strategy_version,
              i.tier,
              i.anchor_regime,
              e.side,
              e.exec_type,
              e.exec_ts_utc,
              e.exec_px,
              e.exec_px_source,
              e.qty,
              e.fee_estimate,
              e.slippage_bps_estimate,
              e.latency_ms
            FROM executions e
            LEFT JOIN alerts_intents i USING (alert_id)
            """
        )

    if "alerts_intents" in csvs:
        con.execute("DROP VIEW IF EXISTS strategy_summary")
        con.execute(
            """
            CREATE VIEW strategy_summary AS
            SELECT
              market_type,
              strategy_profile,
              tier,
              COUNT(*) FILTER (WHERE alert_status = 'ALERT_CREATED') AS alerts_created,
              AVG(CASE WHEN alert_status = 'ALERT_CREATED' THEN score END) AS avg_score,
              COUNT(*) FILTER (WHERE alert_status <> 'ALERT_CREATED') AS not_created,
              COUNT(*) FILTER (WHERE alert_status = 'ALERT_SKIPPED') AS skipped,
              COUNT(*) FILTER (WHERE alert_status = 'ALERT_ERROR') AS errors
            FROM alerts_intents
            GROUP BY 1,2,3
            ORDER BY 1,2,3
            """
        )

        con.execute("DROP VIEW IF EXISTS alerts_funnel")
        con.execute(
            """
            CREATE VIEW alerts_funnel AS
            SELECT
              market_type,
              strategy_profile,
              tier,
              alert_status,
              skip_reason,
              COUNT(*) AS rows_count
            FROM alerts_intents
            GROUP BY 1,2,3,4,5
            ORDER BY 1,2,3,4,5
            """
        )


def main() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))

    csvs = existing_csvs()
    for table, path in csvs.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=True, ignore_errors=True)",
            [str(path)],
        )

    create_views(con, csvs)

    con.execute("CHECKPOINT")
    con.close()
    print(f"DuckDB refreshed at {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
