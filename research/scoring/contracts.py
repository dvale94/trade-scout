from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class ScoringContract:
    """Explicit feature, metadata, and target contract for scoring research.

    Feature columns are limited to alert-time information. Target columns are
    populated from later outcome truth and must never be used as model inputs.
    """

    feature_set_version: str
    metadata_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    leakage_columns: tuple[str, ...]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_feature_columns + self.categorical_feature_columns


DEFAULT_SCORING_CONTRACT = ScoringContract(
    feature_set_version="score_features_v1",
    metadata_columns=(
        "alert_id",
        "signal_ts_utc",
        "symbol",
        "side",
        "market_type",
        "strategy_profile",
        "strategy_id",
        "strategy_version",
        "setup_family",
        "strategy_params_hash",
        "config_hash",
        "tier",
        "score",
        "score_bucket",
        "anchor_regime",
    ),
    numeric_feature_columns=(
        "score",
        "signal_px",
        "entry_low",
        "entry_high",
        "stop_px",
        "t1_px",
        "t2_px",
        "timeframe_min",
        "timeout_seconds",
        "stop_distance",
        "t1_R",
        "t2_R",
        "risk_usd",
        "qty",
        "signal_hour_utc",
        "signal_dow_utc",
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
        "anchor_close",
        "anchor_ma50",
        "anchor_ma200",
        "vol_z",
        "extended_atr",
        "risk_pct",
    ),
    categorical_feature_columns=(
        "symbol",
        "side",
        "market_type",
        "strategy_profile",
        "strategy_id",
        "strategy_version",
        "setup_family",
        "tier",
        "signal_px_type",
        "risk_model",
        "qty_model",
        "anchor_symbol",
        "anchor_regime",
        "market_regime",
        "volatility_regime",
        "anchor_trend_1h",
        "anchor_trend_4h",
        "anchor_trend_1d",
    ),
    target_columns=(
        "pnl_R",
        "profitable",
        "hit_t1",
        "hit_t2",
        "mfe_R",
        "mae_R",
        "time_in_trade_s",
    ),
    leakage_columns=(
        "entered",
        "entry_ts_utc",
        "entry_px",
        "exit_ts_utc",
        "exit_px",
        "exit_reason",
        "first_touch",
        "first_touch_ts_utc",
        "mfe_r",
        "mae_r",
        "mfe_px",
        "mae_px",
        "time_to_t1_s",
        "time_to_t2_s",
        "time_to_stop_s",
        "t1_touched",
        "t2_touched",
        "stop_touched",
        "closed_trade",
        "winner",
        "loser",
        "timeout_trade",
        "no_entry_fill",
        "stop_exit",
        "t2_exit",
        "hit_t1_only",
    ),
)


def available_feature_columns(
    frame: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> list[str]:
    """Return allowed model-input columns that are present in a frame."""
    present = set(frame.columns)
    return [col for col in contract.feature_columns if col in present]


def missing_contract_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    present = set(frame.columns)
    return [col for col in columns if col not in present]
