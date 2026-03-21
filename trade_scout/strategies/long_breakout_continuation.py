from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from trade_scout.strategy_profiles import StrategyProfile

from .base import Strategy
from .types import SetupCandidate, StrategyMetadata, TradePlan


@dataclass(frozen=True)
class LongBreakoutContinuationStrategy(Strategy):
    strategy_id: str = "long_breakout_continuation"
    strategy_version: str = "v1"
    setup_family: str = "breakout_continuation"

    def metadata_for_journaling(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            setup_family=self.setup_family,
        )

    def evaluate_candidate(self, df: pd.DataFrame, profile: StrategyProfile) -> SetupCandidate:
        score, reasons, computed = self._score_candidate(df, profile)
        return SetupCandidate(
            score=score,
            reasons=tuple(reasons),
            computed=computed,
            tier=self._pick_tier(score, df, profile, computed),
        )

    def build_trade_plan(
        self,
        symbol: str,
        profile: StrategyProfile,
        candidate: SetupCandidate,
    ) -> TradePlan | None:
        return self._build_trade_plan(
            symbol=symbol,
            profile=profile,
            computed=candidate.computed,
            reasons=list(candidate.reasons),
        )

    def _score_candidate(self, df: pd.DataFrame, profile: StrategyProfile) -> tuple[float, list[str], dict]:
        reasons: list[str] = []
        score = 0.0

        close = float(df["close"].iloc[-1])
        vol = float(df["volume"].iloc[-1])
        last_high = float(df["high"].iloc[-1])
        last_low = float(df["low"].iloc[-1])

        ma20 = df["close"].rolling(profile.ma_fast).mean()
        ma50 = df["close"].rolling(profile.ma_mid).mean()
        ma200 = df["close"].rolling(profile.ma_slow).mean()

        ma20_last = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else np.nan
        ma50_last = float(ma50.iloc[-1]) if not np.isnan(ma50.iloc[-1]) else np.nan
        ma200_last = float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan

        atr = self.compute_atr(df, profile.atr_period)

        # Trend (0-4)
        if not np.isnan(ma50_last) and close > ma50_last:
            score += 1.5
            reasons.append("Close > MA50")
        if not np.isnan(ma200_last) and close > ma200_last:
            score += 1.5
            reasons.append("Close > MA200")
        if not np.isnan(ma20_last) and not np.isnan(ma50_last) and ma20_last > ma50_last:
            score += 1.0
            reasons.append("MA20 > MA50 (uptrend acceleration)")

        # Breakout + persistence (0-3.5)
        lb = profile.breakout_lookback
        persist = profile.persist_bars
        breakout_level = None
        breakout_ok = False
        if len(df) >= lb + persist + 2:
            breakout_level = float(df["high"].iloc[-(lb + persist + 1) : -(persist + 1)].max())
            last_closes = df["close"].iloc[-persist:]
            if (last_closes > breakout_level).all():
                breakout_ok = True
                score += 3.5
                reasons.append(f"Breakout holds {persist} bars above {breakout_level:.2f}")
            else:
                reasons.append(f"No persistent breakout (level {breakout_level:.2f})")

        # Volume z-score (0-1.5)
        vol_z = None
        vz_lb = profile.vol_lookback
        if len(df) >= vz_lb + 2:
            vols = df["volume"].iloc[-(vz_lb + 1) : -1]
            mu = float(vols.mean())
            sigma = float(vols.std(ddof=0))
            if sigma > 0:
                vol_z = (vol - mu) / sigma
                if vol_z >= profile.volume_z:
                    score += 1.5
                    reasons.append(f"Volume z-score {vol_z:.2f} ≥ {profile.volume_z:.2f}")
                else:
                    reasons.append(f"Volume z-score {vol_z:.2f} (below threshold)")
            else:
                reasons.append("Volume std ~0 (z-score unavailable)")

        # Don't chase (penalty up to -2)
        extended_atr = None
        if atr is not None and not np.isnan(ma20_last) and atr > 0:
            extended_atr = (close - ma20_last) / atr
            if extended_atr > profile.extended_atr_max:
                penalty = min(2.0, (extended_atr - profile.extended_atr_max) * 1.0)
                score -= penalty
                reasons.append(f"Too extended: {extended_atr:.2f} ATR above MA20 (penalty -{penalty:.1f})")

        # ATR bonus
        if atr is not None:
            score += 0.5
            reasons.append(f"ATR({profile.atr_period})={atr:.2f}")

        # Stops for risk engine
        swing_lb = profile.swing_low_lookback
        structure_stop = float(df["low"].iloc[-swing_lb:].min()) if len(df) >= swing_lb else float(df["low"].min())
        atr_stop = (close - profile.atr_k * atr) if (atr is not None) else None

        breakout_distance_atr = self._distance_atr(breakout_level, close, atr)
        ma20_distance_atr = self._distance_atr(None if np.isnan(ma20_last) else ma20_last, close, atr)
        ma50_distance_atr = self._distance_atr(None if np.isnan(ma50_last) else ma50_last, close, atr)
        ma200_distance_atr = self._distance_atr(None if np.isnan(ma200_last) else ma200_last, close, atr)
        ma20_slope_pct = self._ma_slope_pct(ma20, lookback=5)
        ma50_slope_pct = self._ma_slope_pct(ma50, lookback=5)
        range_pct = float((last_high - last_low) / close) if close > 0 else None
        close_location_in_bar = None
        bar_range = last_high - last_low
        if bar_range > 0:
            close_location_in_bar = float((close - last_low) / bar_range)

        computed = {
            "close": close,
            "atr": atr,
            "ma20": ma20_last,
            "ma50": ma50_last,
            "ma200": ma200_last,
            "breakout_level": breakout_level,
            "breakout_ok": breakout_ok,
            "vol_z": vol_z,
            "extended_atr": extended_atr,
            "structure_stop": structure_stop,
            "atr_stop": atr_stop,
            "breakout_distance_atr": breakout_distance_atr,
            "ma20_distance_atr": ma20_distance_atr,
            "ma50_distance_atr": ma50_distance_atr,
            "ma200_distance_atr": ma200_distance_atr,
            "ma20_slope_pct": ma20_slope_pct,
            "ma50_slope_pct": ma50_slope_pct,
            "range_pct": range_pct,
            "close_location_in_bar": close_location_in_bar,
        }
        return score, reasons, computed

    def _pick_tier(
        self,
        score: float,
        df: pd.DataFrame,
        profile: StrategyProfile,
        computed: dict,
    ) -> str | None:
        if score >= profile.a_plus_threshold and self.is_retest_setup(df, profile, computed):
            return "A_PLUS_RETEST"
        if score >= profile.a_threshold:
            return "A_CONTINUATION"
        return None

    def is_retest_setup(self, df: pd.DataFrame, profile: StrategyProfile, computed: dict) -> bool:
        breakout = computed.get("breakout_level")
        atr = computed.get("atr")
        close = computed.get("close")
        if breakout is None or atr is None or atr <= 0 or close is None:
            return False

        band = profile.retest_atr_band * atr
        if close > breakout + band:
            return False

        hold = profile.retest_hold_bars
        if len(df) < hold:
            return False

        last_closes = df["close"].iloc[-hold:]
        return bool((last_closes > breakout).all())

    @staticmethod
    def true_range(high, low, prev_close):
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    @classmethod
    def compute_atr(cls, df: pd.DataFrame, period: int) -> float | None:
        if len(df) < period + 1:
            return None
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        trs = []
        for i in range(1, len(df)):
            trs.append(cls.true_range(highs[i], lows[i], closes[i - 1]))
        atr = pd.Series(trs).rolling(period).mean().iloc[-1]
        return float(atr) if not math.isnan(atr) else None

    @staticmethod
    def _distance_atr(ref: float | None, px: float, atr_val: float | None) -> float | None:
        if ref is None or atr_val is None or atr_val <= 0:
            return None
        return float((px - ref) / atr_val)

    @staticmethod
    def _ma_slope_pct(ma_series: pd.Series, lookback: int = 5) -> float | None:
        if len(ma_series) < lookback + 1:
            return None
        now_v = ma_series.iloc[-1]
        prev_v = ma_series.iloc[-(lookback + 1)]
        if np.isnan(now_v) or np.isnan(prev_v) or prev_v == 0:
            return None
        return float((now_v - prev_v) / prev_v)

    @staticmethod
    def _build_trade_plan(
        *,
        symbol: str,
        profile: StrategyProfile,
        computed: dict,
        reasons: list[str],
    ) -> TradePlan | None:
        close = computed["close"]
        structure_stop = computed["structure_stop"]
        atr_stop = computed["atr_stop"]

        buf = close * profile.entry_buffer_pct
        entry_low = close - buf
        entry_high = close + buf

        stop_candidates = [structure_stop]
        if atr_stop is not None:
            stop_candidates.append(atr_stop)
        stop = float(min(stop_candidates))

        if stop >= entry_low:
            reasons.append("Stop not valid (too tight / above entry zone)")
            return None

        stop_distance = entry_high - stop
        t1 = entry_high + 1.5 * stop_distance
        t2 = entry_high + 2.5 * stop_distance

        return TradePlan(
            symbol=symbol,
            side="LONG",
            timeframe_min=profile.timeframe_min,
            entry_low=float(entry_low),
            entry_high=float(entry_high),
            stop_px=float(stop),
            t1_px=float(t1),
            t2_px=float(t2),
            stop_distance=float(stop_distance),
        )
