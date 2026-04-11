from __future__ import annotations

from typing import Any


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_shadow_score_features(intent_row: Any, context_row: Any) -> dict[str, object]:
    """Build runtime-safe alert-time features for shadow scoring."""
    fields = {
        "score": getattr(context_row, "score", None),
        "signal_px": getattr(intent_row, "signal_px", None),
        "entry_low": getattr(intent_row, "entry_low", None),
        "entry_high": getattr(intent_row, "entry_high", None),
        "stop_px": getattr(intent_row, "stop_px", None),
        "t1_px": getattr(intent_row, "t1_px", None),
        "t2_px": getattr(intent_row, "t2_px", None),
        "timeframe_min": getattr(intent_row, "timeframe_min", None),
        "timeout_seconds": getattr(intent_row, "timeout_seconds", None),
        "stop_distance": getattr(intent_row, "stop_distance", None),
        "t1_R": getattr(intent_row, "t1_R", None),
        "t2_R": getattr(intent_row, "t2_R", None),
        "risk_usd": getattr(intent_row, "risk_usd", None),
        "qty": getattr(intent_row, "qty", None),
        "signal_hour_utc": getattr(context_row, "signal_hour_utc", None),
        "atr": getattr(context_row, "atr", None),
        "ma20": getattr(context_row, "ma20", None),
        "ma50": getattr(context_row, "ma50", None),
        "ma200": getattr(context_row, "ma200", None),
        "breakout_level": getattr(context_row, "breakout_level", None),
        "breakout_distance_atr": getattr(context_row, "breakout_distance_atr", None),
        "ma20_distance_atr": getattr(context_row, "ma20_distance_atr", None),
        "ma50_distance_atr": getattr(context_row, "ma50_distance_atr", None),
        "ma200_distance_atr": getattr(context_row, "ma200_distance_atr", None),
        "ma20_slope_pct": getattr(context_row, "ma20_slope_pct", None),
        "ma50_slope_pct": getattr(context_row, "ma50_slope_pct", None),
        "range_pct": getattr(context_row, "range_pct", None),
        "close_location_in_bar": getattr(context_row, "close_location_in_bar", None),
        "entry_zone_width_pct": getattr(context_row, "entry_zone_width_pct", None),
        "anchor_close": getattr(context_row, "anchor_close", None),
        "anchor_ma50": getattr(context_row, "anchor_ma50", None),
        "anchor_ma200": getattr(context_row, "anchor_ma200", None),
        "vol_z": getattr(context_row, "vol_z", None),
        "extended_atr": getattr(context_row, "extended_atr", None),
        "symbol": getattr(intent_row, "symbol", None),
        "side": getattr(intent_row, "side", None),
        "market_type": getattr(intent_row, "market_type", None),
        "strategy_profile": getattr(intent_row, "strategy_profile", None),
        "strategy_id": getattr(intent_row, "strategy_id", None),
        "strategy_version": getattr(intent_row, "strategy_version", None),
        "setup_family": getattr(intent_row, "setup_family", None),
        "tier": getattr(intent_row, "tier", None),
        "signal_px_type": getattr(intent_row, "signal_px_type", None),
        "risk_model": getattr(intent_row, "risk_model", None),
        "qty_model": getattr(intent_row, "qty_model", None),
        "anchor_symbol": getattr(context_row, "anchor_symbol", None),
        "anchor_regime": getattr(context_row, "anchor_regime", None),
        "market_regime": getattr(context_row, "market_regime", None),
        "volatility_regime": getattr(context_row, "volatility_regime", None),
        "anchor_trend_1h": getattr(context_row, "anchor_trend_1h", None),
        "anchor_trend_4h": getattr(context_row, "anchor_trend_4h", None),
        "anchor_trend_1d": getattr(context_row, "anchor_trend_1d", None),
    }

    fields["risk_pct"] = _risk_pct(fields.get("entry_high"), fields.get("stop_px"))
    return fields


def _risk_pct(entry_high: object, stop_px: object) -> float | None:
    entry = _float_or_none(entry_high)
    stop = _float_or_none(stop_px)
    if entry is None or stop is None or entry <= 0:
        return None
    return (entry - stop) / entry
