import csv
import os
import json
from datetime import timezone
from typing import Any

from . import config
from .strategy import Candidate

LOG_DIR = config.LOG_DIR


# `alerts_intents` is the frozen core signal/audit schema.
# `alert_context` is the evolving analytics/context schema.
ALERT_INTENT_FIELDS = [
    "alert_id",
    "symbol",
    "side",
    "market_type",
    "strategy_profile",
    "strategy_version",
    "config_hash",
    "tier",
    "signal_ts_utc",
    "timeframe_min",
    "signal_px",
    "signal_px_type",
    "alert_status",
    "skip_reason",
    "error_message",
    "entry_low",
    "entry_high",
    "stop_px",
    "t1_px",
    "t2_px",
    "timeout_seconds",
    "stop_distance",
    "t1_R",
    "t2_R",
    "risk_model",
    "risk_usd",
    "qty_model",
    "qty",
    "reasons",
    "schema_version",
]

ALERT_CONTEXT_FIELDS = [
    "schema_version",
    "alert_id",
    "market_type",
    "strategy_profile",
    "strategy_version",
    "config_hash",
    "signal_ts_utc",
    "signal_hour_utc",
    "signal_day_of_week",
    "bar_start_ts_utc",
    "bar_end_ts_utc",
    "score",
    "atr",
    "ma20",
    "ma50",
    "ma200",
    "breakout_level",
    "breakout_distance_atr",
    "ma20_distance_atr",
    "ma50_distance_atr",
    "ma200_distance_atr",
    "ma20_slope_pct",
    "ma50_slope_pct",
    "range_pct",
    "close_location_in_bar",
    "entry_zone_width_pct",
    "anchor_symbol",
    "anchor_regime",
    "anchor_close",
    "anchor_ma50",
    "anchor_ma200",
    "vol_z",
    "extended_atr",
]

OUTCOME_FIELDS = [
    "alert_id",
    "symbol",
    "side",
    "strategy_profile",
    "strategy_version",
    "config_hash",
    "entry_ts_utc",
    "entry_px",
    "exit_ts_utc",
    "exit_px",
    "exit_reason",
    "pnl_R",
    "pnl_usd_est",
    "t1_touched",
    "t1_ts_utc",
    "t2_touched",
    "t2_ts_utc",
    "stop_touched",
    "stop_ts_utc",
    "mfe_R",
    "mae_R",
    "time_in_trade_s",
    "time_to_t1_s",
    "time_to_t2_s",
    "time_to_stop_s",
    "max_spread_bps_seen",
    "avg_spread_bps_seen",
    "entered",
    "schema_version",
]

EXECUTION_FIELDS = [
    "exec_id",
    "alert_id",
    "symbol",
    "side",
    "exec_type",
    "exec_ts_utc",
    "exec_px",
    "exec_px_source",
    "qty",
    "fee_estimate",
    "slippage_bps_estimate",
    "latency_ms",
    "schema_version",
]

TELEMETRY_FIELDS = [
    "timestamp_utc",
    "market_type",
    "strategy_profile",
    "timeframe_min",
    "bars_seen",
    "anchor_bars_seen",
    "insufficient_history",
    "indicator_missing",
    "below_score",
    "alerts_sent",
    "skipped_tight_stop",
    "scores_n",
    "score_min",
    "score_avg",
    "score_max",
    "tier_selected",
    "cooldown_blocked",
    "alerts_by_tier",
    "schema_version",
]


def _path(name: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, name)


def alerts_path() -> str:
    return _path(f"{config.MARKET_TYPE}_alerts.csv")


def outcomes_path() -> str:
    return _path(f"{config.MARKET_TYPE}_outcomes.csv")


def telemetry_path() -> str:
    return _path(f"{config.MARKET_TYPE}_telemetry.csv")


def watches_state_path() -> str:
    return _path(f"{config.MARKET_TYPE}_active_watches.json")


def alerts_intents_path() -> str:
    return _path(f"{config.MARKET_TYPE}_alerts_intents.csv")


def alert_context_path(market_type: str) -> str:
    return _path(f"{market_type}_alert_context.csv")


def executions_path() -> str:
    return _path(f"{config.MARKET_TYPE}_executions.csv")


def _write_row(path: str, row: dict[str, Any]):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_row_with_schema(path: str, row: dict[str, Any], fieldnames: list[str]):
    normalized = _normalize_row_for_schema(row, fieldnames, path)
    _migrate_csv_schema_additive(path, fieldnames)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(normalized)


def _normalize_row_for_schema(row: dict[str, Any], fieldnames: list[str], path: str) -> dict[str, Any]:
    row = dict(row)
    if "schema_version" in fieldnames and "schema_version" not in row:
        row["schema_version"] = "1"
    field_set = set(fieldnames)
    extras = sorted([k for k in row.keys() if k not in field_set])
    if extras:
        # Never silently spill schema drift into CSV columns.
        print(f"[JOURNAL] dropping unexpected fields for {path}: {extras}")
    return {k: row.get(k, "") for k in fieldnames}


def _read_csv_header(path: str) -> list[str] | None:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        return next(reader, None)


def _validate_schema_change(path: str, old_fields: list[str], new_fields: list[str]) -> None:
    old_set = set(old_fields)
    new_set = set(new_fields)
    missing = [field for field in old_fields if field not in new_set]
    if missing:
        raise ValueError(
            f"[JOURNAL] destructive schema change for {path}: missing existing columns {missing}"
        )

    old_idx = 0
    for field in new_fields:
        if old_idx < len(old_fields) and field == old_fields[old_idx]:
            old_idx += 1
    if old_idx != len(old_fields):
        raise ValueError(
            f"[JOURNAL] incompatible schema reorder for {path}: "
            f"existing columns must remain in the same relative order"
        )

    if len(new_fields) == len(old_fields) and old_set == new_set:
        raise ValueError(
            f"[JOURNAL] incompatible schema reorder for {path}: "
            f"column order changed without an additive schema update"
        )


def _migrate_csv_schema_additive(path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = _read_csv_header(path)
    if header == fieldnames:
        return
    if header is None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    _validate_schema_change(path, header, fieldnames)

    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def log_alert(candidate: Candidate, computed: dict, tier: str, strategy_profile: str):
    row = {
        "alert_id": candidate.alert_id,
        "market_type": config.MARKET_TYPE,
        "strategy_profile": strategy_profile,
        "timeframe_min": candidate.timeframe_min,
        "timestamp_utc": candidate.ts.astimezone(timezone.utc).isoformat(),
        "symbol": candidate.symbol,
        "tier": tier,
        "score": round(candidate.score, 4),
        "entry_low": round(candidate.entry_low, 6),
        "entry_high": round(candidate.entry_high, 6),
        "stop": round(candidate.stop, 6),
        "t1": round(candidate.t1, 6),
        "t2": round(candidate.t2, 6),
        "R": round(candidate.r, 6),
        "risk_pct": round((candidate.r / candidate.entry_high) * 100.0, 4),
        "atr": computed.get("atr"),
        "ma20": computed.get("ma20"),
        "ma50": computed.get("ma50"),
        "ma200": computed.get("ma200"),
        "breakout_level": computed.get("breakout_level"),
        "anchor_symbol": computed.get("anchor_symbol"),
        "anchor_regime": computed.get("anchor_regime"),
        "anchor_close": computed.get("anchor_close"),
        "anchor_ma50": computed.get("anchor_ma50"),
        "anchor_ma200": computed.get("anchor_ma200"),
        "vol_z": computed.get("vol_z"),
        "extended_atr": computed.get("extended_atr"),
        "news_sentiment": candidate.news_sentiment,
        "news_summary": candidate.news_summary,
        "reasons": " | ".join(candidate.reasons),
    }
    _write_row(alerts_path(), row)


def log_outcome(outcome_row: dict[str, Any]):
    _write_row_with_schema(outcomes_path(), outcome_row, OUTCOME_FIELDS)


def log_telemetry(row: dict[str, Any]):
    _write_row_with_schema(telemetry_path(), row, TELEMETRY_FIELDS)


def log_alert_intent(row: dict[str, Any]):
    _write_row_with_schema(alerts_intents_path(), row, ALERT_INTENT_FIELDS)


def log_alert_context(row: dict[str, Any]):
    _write_row_with_schema(alert_context_path(config.MARKET_TYPE), row, ALERT_CONTEXT_FIELDS)


def log_execution(row: dict[str, Any]):
    _write_row_with_schema(executions_path(), row, EXECUTION_FIELDS)


def save_active_watches(payload: dict):
    with open(watches_state_path(), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_active_watches() -> dict | None:
    path = watches_state_path()
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
