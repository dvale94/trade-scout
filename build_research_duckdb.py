from __future__ import annotations

import os
from pathlib import Path
import duckdb

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", LOG_DIR / "research.duckdb"))
MARKET_TYPE = os.getenv("MARKET_TYPE", "crypto").strip().lower()

CSV_MAP = {
    f"{MARKET_TYPE}_alerts.csv": "alerts",
    f"{MARKET_TYPE}_alerts_intents.csv": "alerts_intents",
    f"{MARKET_TYPE}_executions.csv": "executions",
    f"{MARKET_TYPE}_outcomes_v2.csv": "outcomes_v2",
    f"{MARKET_TYPE}_telemetry.csv": "telemetry",
}


def existing_csvs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for filename, table in CSV_MAP.items():
        path = LOG_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            out[table] = path
    return out


def main() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    csvs = existing_csvs()
    for table, path in csvs.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=True, ignore_errors=True)",
            [str(path)],
        )

    if "alerts_intents" in csvs and "outcomes_v2" in csvs:
        con.execute("DROP VIEW IF EXISTS trades")
        con.execute(
            """
            CREATE VIEW trades AS
            SELECT
              i.alert_id,
              i.symbol,
              i.market_type,
              i.strategy_profile,
              i.strategy_version,
              i.config_hash,
              i.tier,
              i.signal_ts_utc,
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
              o.stop_touched
            FROM alerts_intents i
            LEFT JOIN outcomes_v2 o USING (alert_id)
            WHERE i.alert_status = 'ALERT_CREATED'
            """
        )

    if "alerts_intents" in csvs:
        con.execute("DROP VIEW IF EXISTS strategy_summary")
        con.execute(
            """
            CREATE VIEW strategy_summary AS
            SELECT
              strategy_profile,
              tier,
              COUNT(*) AS alerts_created,
              AVG(score) AS avg_score
            FROM alerts_intents
            WHERE alert_status = 'ALERT_CREATED'
            GROUP BY 1,2
            ORDER BY 1,2
            """
        )

    con.execute("CHECKPOINT")
    con.close()
    print(f"DuckDB refreshed at {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
