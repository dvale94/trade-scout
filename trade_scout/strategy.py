from dataclasses import dataclass
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd

from trade_scout.strategy_profiles import StrategyProfile

@dataclass
class Candidate:
    symbol: str
    score: float
    ts: datetime
    entry_low: float
    entry_high: float
    stop: float
    t1: float
    t2: float
    r: float
    reasons: list[str]
    news_sentiment: float | None = None
    news_summary: str | None = None


def true_range(high, low, prev_close):
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr(df: pd.DataFrame, period: int) -> float | None:
    if len(df) < period + 1:
        return None
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    trs = []
    for i in range(1, len(df)):
        trs.append(true_range(highs[i], lows[i], closes[i - 1]))
    atr = pd.Series(trs).rolling(period).mean().iloc[-1]
    return float(atr) if not math.isnan(atr) else None


def score_candidate(df: pd.DataFrame, profile: StrategyProfile) -> tuple[float, list[str], dict]:
    reasons: list[str] = []
    score = 0.0

    close = float(df["close"].iloc[-1])
    vol = float(df["volume"].iloc[-1])

    ma20 = df["close"].rolling(profile.ma_fast).mean()
    ma50 = df["close"].rolling(profile.ma_mid).mean()
    ma200 = df["close"].rolling(profile.ma_slow).mean()

    ma20_last = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else np.nan
    ma50_last = float(ma50.iloc[-1]) if not np.isnan(ma50.iloc[-1]) else np.nan
    ma200_last = float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan

    atr = compute_atr(df, profile.atr_period)

    # Trend (0–4)
    if not np.isnan(ma50_last) and close > ma50_last:
        score += 1.5
        reasons.append("Close > MA50")
    if not np.isnan(ma200_last) and close > ma200_last:
        score += 1.5
        reasons.append("Close > MA200")
    if not np.isnan(ma20_last) and not np.isnan(ma50_last) and ma20_last > ma50_last:
        score += 1.0
        reasons.append("MA20 > MA50 (uptrend acceleration)")

    # Breakout + persistence (0–3.5)
    lb = profile.breakout_lookback
    persist = profile.persist_bars
    breakout_level = None
    breakout_ok = False
    if len(df) >= lb + persist + 2:
        breakout_level = float(df["high"].iloc[-(lb + persist + 1):-(persist + 1)].max())
        last_closes = df["close"].iloc[-persist:]
        if (last_closes > breakout_level).all():
            breakout_ok = True
            score += 3.5
            reasons.append(f"Breakout holds {persist} bars above {breakout_level:.2f}")
        else:
            reasons.append(f"No persistent breakout (level {breakout_level:.2f})")

    # Volume z-score (0–1.5)
    vol_z = None
    vz_lb = profile.vol_lookback
    if len(df) >= vz_lb + 2:
        vols = df["volume"].iloc[-(vz_lb + 1):-1]
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
    }
    return score, reasons, computed


def is_retest_setup(df: pd.DataFrame, profile: StrategyProfile, computed: dict) -> bool:
    breakout = computed.get("breakout_level")
    atr = computed.get("atr")
    close = computed.get("close")
    if breakout is None or atr is None or atr <= 0 or close is None:
        return False

    band = profile.retest_atr_band * atr
    # near breakout (not far above it)
    if close > breakout + band:
        return False

    hold = profile.retest_hold_bars
    if len(df) < hold:
        return False

    last_closes = df["close"].iloc[-hold:]
    return bool((last_closes > breakout).all())


def pick_tier(score: float, df: pd.DataFrame, profile: StrategyProfile, computed: dict) -> str | None:
    if score >= profile.a_plus_threshold and is_retest_setup(df, profile, computed):
        return "A_PLUS_RETEST"
    if score >= profile.a_threshold:
        return "A_CONTINUATION"
    return None


def get_news_sentiment_stub(symbol: str):
    return None, None


def compute_entry_stop_targets(symbol: str, profile: StrategyProfile, computed: dict, reasons: list[str], score: float) -> Candidate | None:
    now = datetime.now(timezone.utc)

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

    r = entry_high - stop
    t1 = entry_high + 1.5 * r
    t2 = entry_high + 2.5 * r

    news_sent, news_summary = get_news_sentiment_stub(symbol)

    return Candidate(
        symbol=symbol,
        score=score,
        ts=now,
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        stop=float(stop),
        t1=float(t1),
        t2=float(t2),
        r=float(r),
        reasons=reasons,
        news_sentiment=news_sent,
        news_summary=news_summary,
    )