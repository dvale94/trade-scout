from datetime import datetime, timezone
import pandas as pd

from .telemetry import TelemetryTracker

from . import config
from . import strategy_profiles
from . import market
from . import seed
from . import strategy
from . import notifier
from . import journal
from . import watch
from . import anchors


class Cooldown:
    def __init__(self, minutes: int):
        from datetime import timedelta
        self.cooldown = timedelta(minutes=minutes)
        self.last_alert: dict[tuple[str, str], datetime] = {}

    def allowed(self, symbol: str, tier: str, now: datetime) -> bool:
        key = (symbol, tier)
        last = self.last_alert.get(key)
        if not last:
            return True
        return (now - last) >= self.cooldown

    def mark(self, symbol: str, tier: str, now: datetime):
        self.last_alert[(symbol, tier)] = now


def format_alert(candidate: strategy.Candidate, tier: str) -> str:
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


def append_bar_to_df(df: pd.DataFrame, bar, max_keep: int) -> pd.DataFrame:
    new_row = {
        "ts": pd.to_datetime(bar.timestamp, utc=True),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    if len(df) > max_keep:
        df = df.iloc[-max_keep:].reset_index(drop=True)
    return df


def run():
    profile = strategy_profiles.PROFILES.get(config.STRATEGY_PROFILE)
    if profile is None:
        raise ValueError(f"Unknown STRATEGY_PROFILE: {config.STRATEGY_PROFILE}")
    
    min_needed = max(
        profile.ma_mid + 5,
        profile.atr_period + 5,
        profile.breakout_lookback + profile.persist_bars + 5,
    )
    max_keep = max(profile.lookback_bars, min_needed)
    
    anchor = anchors.AnchorTracker()
    symbols_to_run = list(dict.fromkeys(config.SYMBOLS + [anchor.anchor_symbol]))  # de-dupe, preserve order

    print(f"MARKET_TYPE={config.MARKET_TYPE} PROFILE={profile.name} timeframe={profile.timeframe_min}m symbols={symbols_to_run}")

    print("Seeding history...")
    data = seed.seed_history(
        symbols_to_run,
        timeframe_min=profile.timeframe_min,
        seed_minutes=profile.seed_minutes,
        lookback_bars=profile.lookback_bars,
    )
    print("Seed complete:", {k: len(v) for k, v in data.items()})

    cooldown = Cooldown(profile.cooldown_min)
    watch_mgr = watch.WatchManager(timeout_minutes=profile.watch_timeout_min, max_active_per_symbol=1)
    watch_mgr.load_state()
    print(f"Restored active watches: {sum(len(v) for v in watch_mgr.active.values())}")

    # --- Telemetry (hourly) ---
    telemetry = TelemetryTracker(config.STRATEGY_PROFILE, profile.timeframe_min)

    stream = market.make_stream()

    async def on_bar(bar):
        symbol = bar.symbol
        ts = pd.to_datetime(bar.timestamp, utc=True).to_pydatetime()
        now = datetime.now(timezone.utc)

        telemetry.record_bar(is_anchor=anchor.is_anchor(symbol))
        # update outcomes for active watches
        watch_mgr.update(symbol, float(bar.high), float(bar.low), ts)

        df = data.get(symbol)
        if df is None:
            return
        df = append_bar_to_df(df, bar, max_keep)
        data[symbol] = df

        if len(df) < min_needed:
            telemetry.record_insufficient_history()
            return

        score, reasons, computed = strategy.score_candidate(df, profile)

        telemetry.record_score(score)

        # If you rely on some indicators and they're not ready yet
        if computed.get("ma200") is None or computed.get("atr") is None:
            telemetry.record_indicator_missing()
            return

        # If this symbol is an anchor, update anchor_state
        if anchor.is_anchor(symbol):
            anchor.update_from_computed(computed, ts)
            # skip alerting if anchor is not in configured symbols
            if symbol not in config.SYMBOLS:
                return

        tier = strategy.pick_tier(score, df, profile, computed)
        if tier is None:
            telemetry.record_below_score()
            return
        telemetry.record_tier_selected(tier)

        if not cooldown.allowed(symbol, tier, now):
            telemetry.record_cooldown_blocked(tier)
            return

        try:
            candidate = strategy.compute_entry_stop_targets(symbol, profile, computed, reasons, score)
            if candidate is None:
                return
 
            # sanity: avoid ultra-tight stops
            if candidate.r / candidate.entry_high < 0.004:
                telemetry.record_skipped_tight_stop()
                return

            computed_for_log = anchor.inject_into_computed(computed)

            journal.log_alert(candidate, computed_for_log, tier, config.STRATEGY_PROFILE)
            notifier.discord_notify(format_alert(candidate, tier))

            telemetry.record_alert_sent(tier)

            watch_mgr.add(watch.Watch(
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

            cooldown.mark(symbol, tier, now)
            print(f"ALERT {symbol} tier={tier} score={score:.2f}")

        except Exception as e:
            print(f"ALERT ERROR for {symbol}: {e}")

        telemetry.maybe_flush(now)

    # Subscribe bars
    for sym in symbols_to_run:
        stream.subscribe_bars(on_bar, sym)

    print("Trade Scout running...")
    stream.run()


if __name__ == "__main__":
    run()