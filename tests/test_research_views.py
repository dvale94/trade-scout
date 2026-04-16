from __future__ import annotations

import sys
import types
import unittest

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - local env dependency
    duckdb = None

if duckdb is not None:
    from scripts import build_research_duckdb
else:  # pragma: no cover - local env dependency
    build_research_duckdb = None


@unittest.skipIf(duckdb is None, "duckdb is not installed")
class ResearchViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect(":memory:")

    def tearDown(self) -> None:
        self.con.close()

    def test_strategy_summary_and_alert_funnel_include_strategy_breakdown_columns(self):
        self.con.execute(
            """
            CREATE TABLE alerts_intents (
              alert_id VARCHAR,
              symbol VARCHAR,
              side VARCHAR,
              market_type VARCHAR,
              strategy_profile VARCHAR,
              strategy_version VARCHAR,
              strategy_id VARCHAR,
              setup_family VARCHAR,
              strategy_params_hash VARCHAR,
              config_hash VARCHAR,
              tier VARCHAR,
              signal_ts_utc VARCHAR,
              alert_status VARCHAR
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE alert_context_raw (
              alert_id VARCHAR,
              score DOUBLE
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE outcomes (
              alert_id VARCHAR,
              entered BOOLEAN,
              pnl_R DOUBLE,
              exit_reason VARCHAR
            )
            """
        )

        csvs = {
            "alerts_intents": object(),
            "alert_context_raw": object(),
            "outcomes": object(),
        }
        build_research_duckdb.create_views(self.con, csvs)

        strategy_summary_columns = {
            row[1] for row in self.con.execute("PRAGMA table_info('strategy_summary')").fetchall()
        }
        alert_funnel_columns = {
            row[1] for row in self.con.execute("PRAGMA table_info('alert_funnel')").fetchall()
        }
        alert_context_columns = {
            row[1] for row in self.con.execute("PRAGMA table_info('alert_context')").fetchall()
        }
        trades_enriched_columns = {
            row[1] for row in self.con.execute("PRAGMA table_info('trades_enriched')").fetchall()
        }

        for column in ("strategy_id", "strategy_version", "setup_family", "strategy_params_hash"):
            self.assertIn(column, strategy_summary_columns)
            self.assertIn(column, alert_funnel_columns)
        self.assertIn("experiment_tag", alert_context_columns)
        self.assertIn("experiment_tag", trades_enriched_columns)


if __name__ == "__main__":
    unittest.main()
