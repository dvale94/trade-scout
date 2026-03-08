from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.research_loader import connect_research_db

load_dotenv()

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", LOG_DIR / "research.duckdb"))
EXPORT_DIR = Path("research/exports")


def _where_clause(market_type: str | None) -> tuple[str, list[str]]:
    if market_type:
        return "WHERE market_type = ?", [market_type]
    return "", []


def build_report(duckdb_path: Path, market_type: str | None = None):
    where_sql, params = _where_clause(market_type)

    con = connect_research_db(duckdb_path=duckdb_path, read_only=True)
    try:
        overall_df = con.execute(
            f"""
            SELECT
              COALESCE(market_type, 'ALL') AS market_type,
              COUNT(*) AS alerts_created,
              SUM(CASE WHEN entered THEN 1 ELSE 0 END) AS entered_count,
              SUM(CASE WHEN pnl_R IS NOT NULL THEN 1 ELSE 0 END) AS closed_count,
              SUM(CASE WHEN pnl_R IS NOT NULL AND pnl_R > 0 THEN 1 ELSE 0 END)::DOUBLE
                / NULLIF(SUM(CASE WHEN pnl_R IS NOT NULL THEN 1 ELSE 0 END), 0) AS win_rate,
              AVG(CASE WHEN pnl_R IS NOT NULL THEN pnl_R END) AS avg_R,
              AVG(CASE WHEN pnl_R IS NOT NULL AND pnl_R > 0 THEN pnl_R END) AS avg_win_R,
              AVG(CASE WHEN pnl_R IS NOT NULL AND pnl_R < 0 THEN pnl_R END) AS avg_loss_R,
              AVG(mfe_R) AS avg_mfe_R,
              AVG(mae_R) AS avg_mae_R,
              AVG(time_in_trade_s) AS avg_time_in_trade_s
            FROM trades
            {where_sql}
            GROUP BY 1
            """,
            params,
        ).df()

        by_tier_df = con.execute(
            f"""
            SELECT *
            FROM edge_report
            {where_sql}
            ORDER BY avg_R DESC NULLS LAST, closed_count DESC
            """,
            params,
        ).df()
    finally:
        con.close()

    return overall_df, by_tier_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repeatable research edge report from DuckDB")
    parser.add_argument("--duckdb-path", type=str, default=str(DUCKDB_PATH))
    parser.add_argument("--market-type", type=str, default=os.getenv("MARKET_TYPE", "").strip().lower() or None)
    parser.add_argument("--output-dir", type=str, default=str(EXPORT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_df, by_tier_df = build_report(Path(args.duckdb_path), args.market_type)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = args.market_type if args.market_type else "all"
    overall_path = output_dir / f"edge_report_overall_{suffix}_{ts}.csv"
    by_tier_path = output_dir / f"edge_report_by_tier_{suffix}_{ts}.csv"

    overall_df.to_csv(overall_path, index=False)
    by_tier_df.to_csv(by_tier_path, index=False)

    print(f"Edge report written:\n- {overall_path}\n- {by_tier_path}")


if __name__ == "__main__":
    main()
