from dataclasses import dataclass

@dataclass(frozen=True)
class StrategyProfile:
    name: str
    timeframe_min: int
    lookback_bars: int
    seed_minutes: int

    ma_fast: int
    ma_mid: int
    ma_slow: int
    atr_period: int

    breakout_lookback: int
    persist_bars: int

    vol_lookback: int
    volume_z: float

    atr_k: float
    swing_low_lookback: int
    entry_buffer_pct: float
    extended_atr_max: float

    a_threshold: float
    a_plus_threshold: float

    retest_atr_band: float
    retest_hold_bars: int

    cooldown_min: int
    watch_timeout_min: int


PROFILES = {
    "crypto_v1": StrategyProfile(
        name="crypto_v1",
        timeframe_min=5,
        lookback_bars=800,
        seed_minutes=3 * 24 * 60,

        ma_fast=20, ma_mid=50, ma_slow=200,
        atr_period=14,

        breakout_lookback=55,
        persist_bars=2,

        vol_lookback=50,
        volume_z=1.0,

        atr_k=2.2,
        swing_low_lookback=25,
        entry_buffer_pct=0.0012,
        extended_atr_max=1.35,

        a_threshold=7.6,
        a_plus_threshold=8.6,

        retest_atr_band=0.9,
        retest_hold_bars=2,

        cooldown_min=60,
        watch_timeout_min=240,
    ),

    "stocks_v1": StrategyProfile(
        name="stocks_v1",
        timeframe_min=5,
        lookback_bars=1200,
        seed_minutes=10 * 24 * 60,

        ma_fast=20, ma_mid=50, ma_slow=200,
        atr_period=14,

        breakout_lookback=30,
        persist_bars=2,

        vol_lookback=40,
        volume_z=1.2,

        atr_k=2.0,
        swing_low_lookback=20,
        entry_buffer_pct=0.0010,
        extended_atr_max=1.75,

        a_threshold=7.8,
        a_plus_threshold=8.8,

        retest_atr_band=0.8,
        retest_hold_bars=2,

        cooldown_min=90,
        watch_timeout_min=240,
    ),
}