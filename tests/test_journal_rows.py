from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass


sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault("numpy", types.SimpleNamespace(isnan=lambda value: False))
sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=object, Series=object))


@dataclass
class _CandidateStub:
    alert_id: str = "a1"
    symbol: str = "BTC/USD"
    timeframe_min: int = 5
    score: float = 7.5
    ts: object = None
    entry_low: float = 100.0
    entry_high: float = 101.0
    stop: float = 95.0
    t1: float = 105.0
    t2: float = 110.0
    r: float = 6.0
    reasons: list[str] = None
    news_sentiment: float = None
    news_summary: str = None


sys.modules.setdefault("trade_scout.strategy", types.SimpleNamespace(Candidate=_CandidateStub))

from trade_scout import journal
from trade_scout.journal_rows import AlertContextRow, AlertIntentRow, ExecutionRow, OutcomeRow


class JournalRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_log_dir = journal.LOG_DIR
        journal.LOG_DIR = self.tmpdir.name

    def tearDown(self) -> None:
        journal.LOG_DIR = self.original_log_dir
        self.tmpdir.cleanup()

    def test_alert_intent_serializes_exact_schema_keys(self):
        row = AlertIntentRow(
            alert_id="a1",
            symbol="BTC/USD",
            side="LONG",
            market_type="crypto",
            strategy_profile="crypto_v1",
            strategy_version="v1",
            config_hash="abc123",
            tier="A",
            signal_ts_utc="2026-01-01T00:00:00+00:00",
            timeframe_min=5,
            signal_px=101.0,
            signal_px_type="bar_close",
            alert_status="ALERT_CREATED",
            skip_reason=None,
            error_message=None,
            entry_low=100.0,
            entry_high=101.0,
            stop_px=95.0,
            t1_px=105.0,
            t2_px=110.0,
            timeout_seconds=14400,
            stop_distance=6.0,
            t1_R=1.5,
            t2_R=2.5,
            risk_model="fixed_usd",
            risk_usd=None,
            qty_model=None,
            qty=None,
            reasons="Close > MA50",
            strategy_id="long_breakout_continuation",
            setup_family="breakout_continuation",
        )
        self.assertEqual(list(row.to_csv_row().keys()), journal.ALERT_INTENT_FIELDS)

    def test_other_row_types_serialize_exact_schema_keys(self):
        context = AlertContextRow(
            alert_id="a1",
            market_type="crypto",
            strategy_profile="crypto_v1",
            strategy_version="v1",
            config_hash="abc123",
            signal_ts_utc="2026-01-01T00:00:00+00:00",
            signal_hour_utc=0,
            signal_day_of_week="Thu",
            bar_start_ts_utc="2025-12-31T23:55:00+00:00",
            bar_end_ts_utc="2026-01-01T00:00:00+00:00",
            score=7.5,
            atr=None,
            ma20=None,
            ma50=None,
            ma200=None,
            breakout_level=None,
            breakout_distance_atr=None,
            ma20_distance_atr=None,
            ma50_distance_atr=None,
            ma200_distance_atr=None,
            ma20_slope_pct=None,
            ma50_slope_pct=None,
            range_pct=None,
            close_location_in_bar=None,
            entry_zone_width_pct=None,
            anchor_symbol=None,
            anchor_regime=None,
            anchor_close=None,
            anchor_ma50=None,
            anchor_ma200=None,
            vol_z=None,
            extended_atr=None,
            market_regime=None,
            volatility_regime=None,
            anchor_trend_1h=None,
            anchor_trend_4h=None,
            anchor_trend_1d=None,
            score_bucket="7_TO_7_9",
            strategy_id="long_breakout_continuation",
            setup_family="breakout_continuation",
        )
        execution = ExecutionRow(
            exec_id="e1",
            alert_id="a1",
            symbol="BTC/USD",
            side="LONG",
            exec_type="ENTRY",
            exec_ts_utc="2026-01-01T00:01:00+00:00",
            exec_px=101.0,
            exec_px_source="zone_touch_entry_high",
            entry_fill_px=101.0,
            entry_fill_ts_utc="2026-01-01T00:01:00+00:00",
            slippage_pct=0.0,
            qty=None,
            fee_estimate=None,
            slippage_bps_estimate=None,
            latency_ms=1000,
            strategy_id="long_breakout_continuation",
            strategy_version="v1",
            setup_family="breakout_continuation",
        )
        outcome = OutcomeRow(
            alert_id="a1",
            symbol="BTC/USD",
            side="LONG",
            strategy_profile="crypto_v1",
            strategy_version="v1",
            config_hash="abc123",
            entry_ts_utc="2026-01-01T00:01:00+00:00",
            entry_px=101.0,
            exit_ts_utc="2026-01-01T00:10:00+00:00",
            exit_px=110.0,
            exit_reason="T2",
            pnl_R=2.5,
            pnl_usd_est=None,
            t1_touched=True,
            t1_ts_utc="2026-01-01T00:05:00+00:00",
            t2_touched=True,
            t2_ts_utc="2026-01-01T00:10:00+00:00",
            stop_touched=False,
            stop_ts_utc=None,
            first_touch="T1",
            first_touch_ts_utc="2026-01-01T00:05:00+00:00",
            mfe_R=2.5,
            mae_R=0.1,
            mfe_r=2.5,
            mae_r=0.1,
            mfe_px=110.0,
            mae_px=100.4,
            time_in_trade_s=540,
            time_to_t1_s=240,
            time_to_t2_s=540,
            time_to_stop_s=None,
            max_spread_bps_seen=None,
            avg_spread_bps_seen=None,
            entered=True,
            strategy_id="long_breakout_continuation",
            setup_family="breakout_continuation",
        )

        self.assertEqual(list(context.to_csv_row().keys()), journal.ALERT_CONTEXT_FIELDS)
        self.assertEqual(list(execution.to_csv_row().keys()), journal.EXECUTION_FIELDS)
        self.assertEqual(list(outcome.to_csv_row().keys()), journal.OUTCOME_FIELDS)

    def test_required_field_validation(self):
        with self.assertRaises(ValueError):
            AlertIntentRow(
                alert_id="",
                symbol="BTC/USD",
                side="LONG",
                market_type="crypto",
                strategy_profile="crypto_v1",
                strategy_version="v1",
                config_hash="abc123",
                tier="A",
                signal_ts_utc="2026-01-01T00:00:00+00:00",
                timeframe_min=5,
                signal_px=101.0,
                signal_px_type="bar_close",
                alert_status=None,
                skip_reason=None,
                error_message=None,
                entry_low=None,
                entry_high=None,
                stop_px=None,
                t1_px=None,
                t2_px=None,
                timeout_seconds=None,
                stop_distance=None,
                t1_R=None,
                t2_R=None,
                risk_model=None,
                risk_usd=None,
                qty_model=None,
                qty=None,
                reasons=None,
                strategy_id="long_breakout_continuation",
                setup_family="breakout_continuation",
            )
        with self.assertRaises(ValueError):
            ExecutionRow(
                exec_id="",
                alert_id="a1",
                symbol="BTC/USD",
                side="LONG",
                exec_type="ENTRY",
                exec_ts_utc="2026-01-01T00:01:00+00:00",
                exec_px=101.0,
                exec_px_source="zone_touch_entry_high",
                entry_fill_px=101.0,
                entry_fill_ts_utc="2026-01-01T00:01:00+00:00",
                slippage_pct=0.0,
                qty=None,
                fee_estimate=None,
                slippage_bps_estimate=None,
                latency_ms=1000,
            )

    def test_journal_logs_typed_rows_with_unchanged_schemas(self):
        intent = AlertIntentRow(
            alert_id="a1",
            symbol="BTC/USD",
            side="LONG",
            market_type="crypto",
            strategy_profile="crypto_v1",
            strategy_version="v1",
            config_hash="abc123",
            tier="A",
            signal_ts_utc="2026-01-01T00:00:00+00:00",
            timeframe_min=5,
            signal_px=101.0,
            signal_px_type="bar_close",
            alert_status="ALERT_CREATED",
            skip_reason=None,
            error_message=None,
            entry_low=100.0,
            entry_high=101.0,
            stop_px=95.0,
            t1_px=105.0,
            t2_px=110.0,
            timeout_seconds=14400,
            stop_distance=6.0,
            t1_R=1.5,
            t2_R=2.5,
            risk_model="fixed_usd",
            risk_usd=None,
            qty_model=None,
            qty=None,
            reasons="Close > MA50",
            strategy_id="long_breakout_continuation",
            setup_family="breakout_continuation",
        )
        journal.log_alert_intent(intent)

        with open(journal.alerts_intents_path(), newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["alert_id"], "a1")
        self.assertEqual(rows[0]["schema_version"], "1")
        self.assertEqual(list(rows[0].keys()), journal.ALERT_INTENT_FIELDS)

    def test_execution_schema_migration_remains_additive(self):
        path = journal.executions_path()
        old_fields = [
            "exec_id",
            "alert_id",
            "symbol",
            "side",
            "exec_type",
            "exec_ts_utc",
            "exec_px",
            "exec_px_source",
            "entry_fill_px",
            "entry_fill_ts_utc",
            "slippage_pct",
            "qty",
            "fee_estimate",
        ]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "exec_id": "old1",
                    "alert_id": "a_old",
                    "symbol": "BTC/USD",
                    "side": "LONG",
                    "exec_type": "ENTRY",
                    "exec_ts_utc": "2026-01-01T00:00:00+00:00",
                    "exec_px": "101.0",
                    "exec_px_source": "legacy",
                    "entry_fill_px": "101.0",
                    "entry_fill_ts_utc": "2026-01-01T00:00:00+00:00",
                    "slippage_pct": "0.0",
                    "qty": "",
                    "fee_estimate": "",
                }
            )

        journal.log_execution(
            ExecutionRow(
                exec_id="e2",
                alert_id="a2",
                symbol="BTC/USD",
                side="LONG",
                exec_type="ENTRY",
                exec_ts_utc="2026-01-01T00:01:00+00:00",
                exec_px=102.0,
                exec_px_source="zone_touch_entry_high",
                entry_fill_px=102.0,
                entry_fill_ts_utc="2026-01-01T00:01:00+00:00",
                slippage_pct=0.1,
                qty=None,
                fee_estimate=None,
                slippage_bps_estimate=None,
                latency_ms=1000,
                strategy_id="long_breakout_continuation",
                strategy_version="v1",
                setup_family="breakout_continuation",
            )
        )

        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(reader.fieldnames, journal.EXECUTION_FIELDS)
        self.assertEqual(list(rows[0].keys()), journal.EXECUTION_FIELDS)
        self.assertEqual(rows[0]["slippage_bps_estimate"], "")
        self.assertEqual(rows[0]["latency_ms"], "")
        self.assertEqual(rows[0]["schema_version"], "")
        self.assertEqual(rows[0]["strategy_id"], "")
        self.assertEqual(rows[0]["strategy_version"], "")
        self.assertEqual(rows[0]["setup_family"], "")
        self.assertEqual(rows[1]["exec_id"], "e2")
        self.assertEqual(rows[1]["schema_version"], "1")
        self.assertEqual(rows[1]["strategy_id"], "long_breakout_continuation")


if __name__ == "__main__":
    unittest.main()
