from __future__ import annotations

from datetime import datetime, timezone, timedelta

from . import config
from .journal_rows import AlertContextRow, AlertIntentRow


def _signal_window_fields(bar_ts: datetime, timeframe_min: int) -> tuple[str, str]:
    bar_end = bar_ts.astimezone(timezone.utc).isoformat()
    bar_start = (bar_ts - timedelta(minutes=timeframe_min)).astimezone(timezone.utc).isoformat()
    return bar_start, bar_end


def make_intent_core_row(
    *,
    alert_id: str,
    symbol: str,
    tier: str,
    now: datetime,
    timeframe_min: int,
    computed_for_log: dict,
    config_hash: str,
    strategy_version: str,
) -> AlertIntentRow:
    return AlertIntentRow(
        alert_id=alert_id,
        symbol=symbol,
        side="LONG",
        market_type=config.MARKET_TYPE,
        strategy_profile=config.STRATEGY_PROFILE,
        strategy_version=strategy_version,
        config_hash=config_hash,
        tier=tier,
        signal_ts_utc=now.astimezone(timezone.utc).isoformat(),
        timeframe_min=timeframe_min,
        signal_px=float(computed_for_log.get("close")) if computed_for_log.get("close") is not None else None,
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
    )


def make_alert_context_row(
    *,
    alert_id: str,
    now: datetime,
    bar_ts: datetime,
    timeframe_min: int,
    score: float,
    computed_for_log: dict,
    config_hash: str,
    strategy_version: str,
) -> AlertContextRow:
    bar_start_ts_utc, bar_end_ts_utc = _signal_window_fields(bar_ts, timeframe_min)
    anchor_trend_1h, anchor_trend_4h, anchor_trend_1d = _anchor_trend_snapshot(computed_for_log)
    return AlertContextRow(
        alert_id=alert_id,
        market_type=config.MARKET_TYPE,
        strategy_profile=config.STRATEGY_PROFILE,
        strategy_version=strategy_version,
        config_hash=config_hash,
        signal_ts_utc=now.astimezone(timezone.utc).isoformat(),
        signal_hour_utc=int(now.astimezone(timezone.utc).hour),
        signal_day_of_week=now.astimezone(timezone.utc).strftime("%a"),
        bar_start_ts_utc=bar_start_ts_utc,
        bar_end_ts_utc=bar_end_ts_utc,
        score=round(float(score), 4),
        atr=computed_for_log.get("atr"),
        ma20=computed_for_log.get("ma20"),
        ma50=computed_for_log.get("ma50"),
        ma200=computed_for_log.get("ma200"),
        breakout_level=computed_for_log.get("breakout_level"),
        breakout_distance_atr=computed_for_log.get("breakout_distance_atr"),
        ma20_distance_atr=computed_for_log.get("ma20_distance_atr"),
        ma50_distance_atr=computed_for_log.get("ma50_distance_atr"),
        ma200_distance_atr=computed_for_log.get("ma200_distance_atr"),
        ma20_slope_pct=computed_for_log.get("ma20_slope_pct"),
        ma50_slope_pct=computed_for_log.get("ma50_slope_pct"),
        range_pct=computed_for_log.get("range_pct"),
        close_location_in_bar=computed_for_log.get("close_location_in_bar"),
        entry_zone_width_pct=None,
        anchor_symbol=computed_for_log.get("anchor_symbol"),
        anchor_regime=computed_for_log.get("anchor_regime"),
        anchor_close=computed_for_log.get("anchor_close"),
        anchor_ma50=computed_for_log.get("anchor_ma50"),
        anchor_ma200=computed_for_log.get("anchor_ma200"),
        vol_z=computed_for_log.get("vol_z"),
        extended_atr=computed_for_log.get("extended_atr"),
        market_regime=_market_regime(computed_for_log),
        volatility_regime=_volatility_regime(computed_for_log),
        anchor_trend_1h=anchor_trend_1h,
        anchor_trend_4h=anchor_trend_4h,
        anchor_trend_1d=anchor_trend_1d,
        score_bucket=_score_bucket(score),
    )


def compute_entry_zone_width_pct(entry_low: float | None, entry_high: float | None) -> float | None:
    if entry_low is None or entry_high is None or entry_high <= 0:
        return None
    return (entry_high - entry_low) / entry_high


def _trend_label(lhs: float | None, rhs: float | None) -> str | None:
    if lhs is None or rhs is None:
        return None
    if lhs > rhs:
        return "UP"
    if lhs < rhs:
        return "DOWN"
    return "FLAT"


def _market_regime(computed_for_log: dict) -> str | None:
    regime = computed_for_log.get("anchor_regime")
    return str(regime).upper() if regime is not None else None


def _volatility_regime(computed_for_log: dict) -> str | None:
    vol_z = computed_for_log.get("vol_z")
    try:
        vol_z = float(vol_z) if vol_z is not None else None
    except (TypeError, ValueError):
        vol_z = None

    if vol_z is None:
        return None
    if vol_z >= 1.5:
        return "HIGH"
    if vol_z <= -0.5:
        return "LOW"
    return "NORMAL"


def _anchor_trend_snapshot(computed_for_log: dict) -> tuple[str | None, str | None, str | None]:
    close = computed_for_log.get("anchor_close")
    ma50 = computed_for_log.get("anchor_ma50")
    ma200 = computed_for_log.get("anchor_ma200")
    return (
        _trend_label(close, ma50),
        _trend_label(close, ma200),
        _trend_label(ma50, ma200),
    )


def _score_bucket(score: float) -> str:
    if score < 5.0:
        return "LT_5"
    if score < 6.0:
        return "5_TO_5_9"
    if score < 7.0:
        return "6_TO_6_9"
    if score < 8.0:
        return "7_TO_7_9"
    return "8_PLUS"
