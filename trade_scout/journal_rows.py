from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_nonempty(value: Any, field_name: str) -> None:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} is required")


@dataclass(frozen=True)
class AlertIntentRow:
    alert_id: str
    symbol: str
    side: str
    market_type: str
    strategy_profile: str
    strategy_version: str
    config_hash: str
    tier: str
    signal_ts_utc: str
    timeframe_min: int
    signal_px: float | None
    signal_px_type: str
    alert_status: str | None
    skip_reason: str | None
    error_message: str | None
    entry_low: float | None
    entry_high: float | None
    stop_px: float | None
    t1_px: float | None
    t2_px: float | None
    timeout_seconds: int | None
    stop_distance: float | None
    t1_R: float | None
    t2_R: float | None
    risk_model: str | None
    risk_usd: float | None
    qty_model: str | None
    qty: float | None
    reasons: str | None
    schema_version: str = "1"
    strategy_id: str | None = None
    setup_family: str | None = None
    strategy_params_json: str | None = None
    strategy_params_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "alert_id",
            "symbol",
            "side",
            "market_type",
            "strategy_profile",
            "strategy_version",
            "config_hash",
            "tier",
            "signal_ts_utc",
            "signal_px_type",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "side": self.side,
            "market_type": self.market_type,
            "strategy_profile": self.strategy_profile,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "tier": self.tier,
            "signal_ts_utc": self.signal_ts_utc,
            "timeframe_min": self.timeframe_min,
            "signal_px": self.signal_px,
            "signal_px_type": self.signal_px_type,
            "alert_status": self.alert_status,
            "skip_reason": self.skip_reason,
            "error_message": self.error_message,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop_px": self.stop_px,
            "t1_px": self.t1_px,
            "t2_px": self.t2_px,
            "timeout_seconds": self.timeout_seconds,
            "stop_distance": self.stop_distance,
            "t1_R": self.t1_R,
            "t2_R": self.t2_R,
            "risk_model": self.risk_model,
            "risk_usd": self.risk_usd,
            "qty_model": self.qty_model,
            "qty": self.qty,
            "reasons": self.reasons,
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "setup_family": self.setup_family,
            "strategy_params_json": self.strategy_params_json,
            "strategy_params_hash": self.strategy_params_hash,
        }


@dataclass(frozen=True)
class AlertContextRow:
    alert_id: str
    market_type: str
    strategy_profile: str
    strategy_version: str
    config_hash: str
    signal_ts_utc: str
    signal_hour_utc: int | None
    signal_day_of_week: str | None
    bar_start_ts_utc: str | None
    bar_end_ts_utc: str | None
    score: float | None
    atr: float | None
    ma20: float | None
    ma50: float | None
    ma200: float | None
    breakout_level: float | None
    breakout_distance_atr: float | None
    ma20_distance_atr: float | None
    ma50_distance_atr: float | None
    ma200_distance_atr: float | None
    ma20_slope_pct: float | None
    ma50_slope_pct: float | None
    range_pct: float | None
    close_location_in_bar: float | None
    entry_zone_width_pct: float | None
    anchor_symbol: str | None
    anchor_regime: str | None
    anchor_close: float | None
    anchor_ma50: float | None
    anchor_ma200: float | None
    vol_z: float | None
    extended_atr: float | None
    market_regime: str | None
    volatility_regime: str | None
    anchor_trend_1h: str | None
    anchor_trend_4h: str | None
    anchor_trend_1d: str | None
    score_bucket: str | None
    schema_version: str = "1"
    strategy_id: str | None = None
    setup_family: str | None = None
    strategy_params_json: str | None = None
    strategy_params_hash: str | None = None
    experiment_tag: str | None = None
    score_model_version: str | None = None
    feature_set_version: str | None = None
    score_recipe_version: str | None = None
    score_recipe_mode: str | None = None
    score_bucket_version: str | None = None
    shadow_score_status: str | None = None
    pred_expected_r: float | None = None
    pred_win_prob: float | None = None
    pred_expected_mfe: float | None = None
    score_shadow: float | None = None
    score_shadow_bucket: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "alert_id",
            "market_type",
            "strategy_profile",
            "strategy_version",
            "config_hash",
            "signal_ts_utc",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alert_id": self.alert_id,
            "market_type": self.market_type,
            "strategy_profile": self.strategy_profile,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "signal_ts_utc": self.signal_ts_utc,
            "signal_hour_utc": self.signal_hour_utc,
            "signal_day_of_week": self.signal_day_of_week,
            "bar_start_ts_utc": self.bar_start_ts_utc,
            "bar_end_ts_utc": self.bar_end_ts_utc,
            "score": self.score,
            "atr": self.atr,
            "ma20": self.ma20,
            "ma50": self.ma50,
            "ma200": self.ma200,
            "breakout_level": self.breakout_level,
            "breakout_distance_atr": self.breakout_distance_atr,
            "ma20_distance_atr": self.ma20_distance_atr,
            "ma50_distance_atr": self.ma50_distance_atr,
            "ma200_distance_atr": self.ma200_distance_atr,
            "ma20_slope_pct": self.ma20_slope_pct,
            "ma50_slope_pct": self.ma50_slope_pct,
            "range_pct": self.range_pct,
            "close_location_in_bar": self.close_location_in_bar,
            "entry_zone_width_pct": self.entry_zone_width_pct,
            "anchor_symbol": self.anchor_symbol,
            "anchor_regime": self.anchor_regime,
            "anchor_close": self.anchor_close,
            "anchor_ma50": self.anchor_ma50,
            "anchor_ma200": self.anchor_ma200,
            "vol_z": self.vol_z,
            "extended_atr": self.extended_atr,
            "market_regime": self.market_regime,
            "volatility_regime": self.volatility_regime,
            "anchor_trend_1h": self.anchor_trend_1h,
            "anchor_trend_4h": self.anchor_trend_4h,
            "anchor_trend_1d": self.anchor_trend_1d,
            "score_bucket": self.score_bucket,
            "strategy_id": self.strategy_id,
            "setup_family": self.setup_family,
            "strategy_params_json": self.strategy_params_json,
            "strategy_params_hash": self.strategy_params_hash,
            "experiment_tag": self.experiment_tag,
            "score_model_version": self.score_model_version,
            "feature_set_version": self.feature_set_version,
            "score_recipe_version": self.score_recipe_version,
            "score_recipe_mode": self.score_recipe_mode,
            "score_bucket_version": self.score_bucket_version,
            "shadow_score_status": self.shadow_score_status,
            "pred_expected_r": self.pred_expected_r,
            "pred_win_prob": self.pred_win_prob,
            "pred_expected_mfe": self.pred_expected_mfe,
            "score_shadow": self.score_shadow,
            "score_shadow_bucket": self.score_shadow_bucket,
        }


@dataclass(frozen=True)
class ExecutionRow:
    exec_id: str
    alert_id: str
    symbol: str
    side: str
    exec_type: str
    exec_ts_utc: str
    exec_px: float
    exec_px_source: str
    entry_fill_px: float | None
    entry_fill_ts_utc: str | None
    slippage_pct: float | None
    qty: float | None
    fee_estimate: float | None
    slippage_bps_estimate: float | None
    latency_ms: int | None
    schema_version: str = "1"
    strategy_id: str | None = None
    strategy_version: str | None = None
    setup_family: str | None = None
    strategy_params_json: str | None = None
    strategy_params_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "exec_id",
            "alert_id",
            "symbol",
            "side",
            "exec_type",
            "exec_ts_utc",
            "exec_px_source",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "side": self.side,
            "exec_type": self.exec_type,
            "exec_ts_utc": self.exec_ts_utc,
            "exec_px": self.exec_px,
            "exec_px_source": self.exec_px_source,
            "entry_fill_px": self.entry_fill_px,
            "entry_fill_ts_utc": self.entry_fill_ts_utc,
            "slippage_pct": self.slippage_pct,
            "qty": self.qty,
            "fee_estimate": self.fee_estimate,
            "slippage_bps_estimate": self.slippage_bps_estimate,
            "latency_ms": self.latency_ms,
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "setup_family": self.setup_family,
            "strategy_params_json": self.strategy_params_json,
            "strategy_params_hash": self.strategy_params_hash,
        }


@dataclass(frozen=True)
class OutcomeRow:
    alert_id: str
    symbol: str
    side: str
    strategy_profile: str | None
    strategy_version: str | None
    config_hash: str | None
    entry_ts_utc: str | None
    entry_px: float | None
    exit_ts_utc: str | None
    exit_px: float | None
    exit_reason: str | None
    pnl_R: float | None
    pnl_usd_est: float | None
    t1_touched: bool
    t1_ts_utc: str | None
    t2_touched: bool
    t2_ts_utc: str | None
    stop_touched: bool
    stop_ts_utc: str | None
    first_touch: str | None
    first_touch_ts_utc: str | None
    mfe_R: float | None
    mae_R: float | None
    mfe_r: float | None
    mae_r: float | None
    mfe_px: float | None
    mae_px: float | None
    time_in_trade_s: int | None
    time_to_t1_s: int | None
    time_to_t2_s: int | None
    time_to_stop_s: int | None
    max_spread_bps_seen: float | None
    avg_spread_bps_seen: float | None
    entered: bool
    schema_version: str = "1"
    strategy_id: str | None = None
    setup_family: str | None = None
    strategy_params_json: str | None = None
    strategy_params_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("alert_id", "symbol", "side"):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "side": self.side,
            "strategy_profile": self.strategy_profile,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "entry_ts_utc": self.entry_ts_utc,
            "entry_px": self.entry_px,
            "exit_ts_utc": self.exit_ts_utc,
            "exit_px": self.exit_px,
            "exit_reason": self.exit_reason,
            "pnl_R": self.pnl_R,
            "pnl_usd_est": self.pnl_usd_est,
            "t1_touched": self.t1_touched,
            "t1_ts_utc": self.t1_ts_utc,
            "t2_touched": self.t2_touched,
            "t2_ts_utc": self.t2_ts_utc,
            "stop_touched": self.stop_touched,
            "stop_ts_utc": self.stop_ts_utc,
            "first_touch": self.first_touch,
            "first_touch_ts_utc": self.first_touch_ts_utc,
            "mfe_R": self.mfe_R,
            "mae_R": self.mae_R,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "mfe_px": self.mfe_px,
            "mae_px": self.mae_px,
            "time_in_trade_s": self.time_in_trade_s,
            "time_to_t1_s": self.time_to_t1_s,
            "time_to_t2_s": self.time_to_t2_s,
            "time_to_stop_s": self.time_to_stop_s,
            "max_spread_bps_seen": self.max_spread_bps_seen,
            "avg_spread_bps_seen": self.avg_spread_bps_seen,
            "entered": self.entered,
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "setup_family": self.setup_family,
            "strategy_params_json": self.strategy_params_json,
            "strategy_params_hash": self.strategy_params_hash,
        }
