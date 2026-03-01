from datetime import datetime, timezone
import pandas as pd
from uuid import uuid4

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from . import config
from .strategy_profiles import PROFILES
from .market import make_stream
from .seed import seed_history
from .strategy import Candidate, score_candidate, pick_tier, compute_entry_stop_targets
from .notifier import discord_notify
from . import journal
from .watch import Watch, WatchManager


class Cooldown:
    def __init__(self, minutes: int):
        from datetime import timedelta
        self.cooldown = timedelta(minutes=minutes)
        self.last_alert: dict[str, datetime] = {}

    def allowed(self, symbol: str, now: datetime) -> bool:
        last = self.last_alert.get(symbol)
        if not last:
            return True
        return (now - last) >= self.cooldown

    def mark(self, symbol: str, now: datetime):
        self.last_alert[symbol] = now


def format_alert(candidate: Candidate, tier: str) -> str:
    reasons = "\n".join([f"• {r}" for r in candidate.reasons][:6])
    news_line = ""
    if candidate.news_sentiment is not None:
        news_line = f"\n**News sentiment:** {candidate.news_sentiment:.2f} — {candidate.news_summary or ''}"

    return (
        f"🚨 **TRADE SCOUT ALERT** — **{candidate.symbol}**\n"
        f"**Tier:** {tier}\n"
        f"**Score:** {candidate.score:.1f}/10\n"
        f"**Time:** {candidate.ts.astimezone().strftime('%Y-%m-%d %H:%M %Z')}\n\n"
        f"**Entry zone:** {candidate.entry_low:.2f} – {candidate.entry_high:.2f}\n"
        f"**Stop:** {candidate.stop:.2f}\n"
        f"**Targets:** {candidate.t1:.2f} (1.5R), {candidate.t2:.2f} (2.5R)\n"
        f"**R (risk/unit):** {candidate.r:.2f} ({(candidate.r/candidate.entry_high)*100:.2f}%)\n\n"
        f"**Why:**\n{reasons}"
        f"{news_line}\n\n"
        f"⚠️ Not financial advice. Validate spread/liquidity + scheduled events."
    )


def run():
    profile = PROFILES.get(config.STRATEGY_PROFILE)
    if profile is None:
        raise ValueError(f"Unknown STRATEGY_PROFILE: {config.STRATEGY_PROFILE}")

    print(f"MARKET_TYPE={config.MARKET_TYPE} PROFILE={profile.name} timeframe={profile.timeframe_min}m symbols={config.SYMBOLS}")

    print("Seeding history...")
    data = seed_history(config.SYMBOLS, timeframe_min=profile.timeframe_min, seed_minutes=profile.seed_minutes, lookback_bars=profile.lookback_bars)
    print("Seed complete:", {k: len(v) for k, v in data.items()})

    cooldown = Cooldown(profile.cooldown_min)
    watch_mgr = WatchManager(timeout_minutes=profile.watch_timeout_min, max_active_per_symbol=1)

    stream = make_stream()

    async def on_bar(bar):
        symbol = bar.symbol
        ts = pd.to_datetime(bar.timestamp, utc=True).to_pydatetime()

        # Update existing watches every bar
        watch_mgr.update(symbol, float(bar.close), ts)

        df = data.get(symbol)
        if df is None:
            df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        new_row = {
            "ts": pd.to_datetime(bar.timestamp, utc=True),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        if len(df) > profile.lookback_bars:
            df = df.iloc[-profile.lookback_bars:].reset_index(drop=True)
        data[symbol] = df

        # Need minimum bars for indicators
        min_needed = max(
            profile.ma_mid + 5,
            profile.atr_period + 5,
            profile.breakout_lookback + profile.persist_bars + 5,
        )
        if len(df) < min_needed:
            return

        score, reasons, computed = score_candidate(df, profile)

        tier = pick_tier(score, df, profile, computed)
        if tier is None:
            return

        now = datetime.now(timezone.utc)
        if not cooldown.allowed(symbol, now):
            return

        try:
            candidate = compute_entry_stop_targets(symbol, profile, computed, reasons, score)
            if candidate is None:
                return
 
            # sanity: avoid ultra-tight stops
            if candidate.r / candidate.entry_high < 0.004:
                return

            msg = format_alert(candidate, tier)
            discord_notify(msg)
            journal.log_alert(candidate, computed, tier, config.STRATEGY_PROFILE)

            watch_mgr.add(Watch(
                alert_id=candidate.alert_id,
                symbol=symbol,
                timeframe_min=profile.timeframe_min,
                tier=tier,
                created_ts=now,
                entry_low=candidate.entry_low,
                entry_high=candidate.entry_high,
                stop=candidate.stop,
                t1=candidate.t1,
                t2=candidate.t2,
                R=candidate.r,
                score=candidate.score,
                max_favorable=float(candidate.entry_high),
                max_adverse=float(candidate.entry_high),
            ))

            cooldown.mark(symbol, now)
            print(f"ALERT {symbol} tier={tier} score={score:.2f}")

        except Exception as e:
            print(f"ALERT ERROR for {symbol}: {e}")

    # Subscribe bars
    tf = TimeFrame(profile.timeframe_min, TimeFrameUnit.Minute)
    for sym in config.SYMBOLS:
        stream.subscribe_bars(on_bar, sym)

    print("Trade Scout running...")
    stream.run()


if __name__ == "__main__":
    run()