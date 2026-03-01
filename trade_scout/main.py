from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd
import json

from . import config
from . import strategy_profiles
from . import market
from . import seed
from . import strategy
from . import notifier
from . import journal
from . import watch


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


def run():
    profile = strategy_profiles.PROFILES.get(config.STRATEGY_PROFILE)
    if profile is None:
        raise ValueError(f"Unknown STRATEGY_PROFILE: {config.STRATEGY_PROFILE}")
    
    anchor = market.anchor_symbol()
    symbols_to_run = list(dict.fromkeys(config.SYMBOLS + [anchor]))  # de-dupe, preserve order

    print(f"MARKET_TYPE={config.MARKET_TYPE} PROFILE={profile.name} timeframe={profile.timeframe_min}m symbols={symbols_to_run}")

    print("Seeding history...")
    data = seed.seed_history(
        symbols_to_run,
        timeframe_min=profile.timeframe_min,
        seed_minutes=profile.seed_minutes,
        lookback_bars=profile.lookback_bars,
    )
    print("Seed complete:", {k: len(v) for k, v in data.items()})

    anchor_state = {"symbol": anchor, "regime": None, "close": None, "ma50": None, "ma200": None}

    cooldown = Cooldown(profile.cooldown_min)
    watch_mgr = watch.WatchManager(timeout_minutes=profile.watch_timeout_min, max_active_per_symbol=1)

    # --- Telemetry (hourly) ---
    telemetry = {
        "bars_seen": 0,
        "anchor_bars_seen": 0,
        "insufficient_history": 0,
        "indicator_missing": 0,
        "below_score": 0,
        "alerts_sent": 0,
        "scores_n": 0,
        "scores_sum": 0.0,
        "score_min": None,
        "score_max": None,
    }

    tier_selected = defaultdict(int)       # tier -> count
    cooldown_blocked = defaultdict(int)    # tier -> count
    alerts_by_tier = defaultdict(int)      # tier -> count

    telemetry_window_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    next_flush = telemetry_window_start + timedelta(hours=1)

    stream = market.make_stream()

    async def on_bar(bar):
        telemetry["bars_seen"] += 1
        symbol = bar.symbol
        ts = pd.to_datetime(bar.timestamp, utc=True).to_pydatetime()

        # Update existing watches every bar
        watch_mgr.update(symbol, float(bar.high), float(bar.low), ts)

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
            telemetry["insufficient_history"] += 1
            return

        score, reasons, computed = strategy.score_candidate(df, profile)

        telemetry["scores_n"] += 1
        telemetry["scores_sum"] += float(score)
        telemetry["score_min"] = float(score) if telemetry["score_min"] is None else min(telemetry["score_min"], float(score))
        telemetry["score_max"] = float(score) if telemetry["score_max"] is None else max(telemetry["score_max"], float(score))

        # If you rely on some indicators and they're not ready yet
        if computed.get("ma200") is None or computed.get("atr") is None:
            telemetry["indicator_missing"] += 1
            return

        # If this symbol is an anchor, update anchor_state
        if symbol == anchor:
            telemetry["anchor_bars_seen"] += 1
            anchor_state["close"] = computed.get("close")
            anchor_state["ma50"] = computed.get("ma50")
            anchor_state["ma200"] = computed.get("ma200")
            anchor_state["regime"] = market.classify_anchor_regime(computed)
            # skip alerting if anchor is not in configured symbols
            if symbol not in config.SYMBOLS:
                return

        tier = strategy.pick_tier(score, df, profile, computed)
        if tier is None:
            telemetry["below_score"] += 1
            return
        tier_selected[tier] += 1

        now = datetime.now(timezone.utc)
        if not cooldown.allowed(symbol, tier, now):
            cooldown_blocked[tier] += 1
            return

        try:
            candidate = strategy.compute_entry_stop_targets(symbol, profile, computed, reasons, score)
            if candidate is None:
                return
 
            # sanity: avoid ultra-tight stops
            if candidate.r / candidate.entry_high < 0.004:
                return

            msg = format_alert(candidate, tier)
            notifier.discord_notify(msg)
            journal.log_alert(candidate, computed, anchor_state, tier, config.STRATEGY_PROFILE)

            telemetry["alerts_sent"] += 1
            alerts_by_tier[tier] += 1

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

        now_utc = datetime.now(timezone.utc)
        if now_utc >= next_flush:
            avg_score = (telemetry["scores_sum"] / telemetry["scores_n"]) if telemetry["scores_n"] > 0 else None

            row = {
                "timestamp_utc": next_flush.astimezone(timezone.utc).isoformat(),
                "market_type": config.MARKET_TYPE,
                "strategy_profile": config.STRATEGY_PROFILE,
                "timeframe_min": profile.timeframe_min,

                "bars_seen": telemetry["bars_seen"],
                "anchor_bars_seen": telemetry["anchor_bars_seen"],
                "insufficient_history": telemetry["insufficient_history"],
                "indicator_missing": telemetry["indicator_missing"],
                "below_score": telemetry["below_score"],
                "alerts_sent": telemetry["alerts_sent"],

                "scores_n": telemetry["scores_n"],
                "score_min": telemetry["score_min"],
                "score_avg": avg_score,
                "score_max": telemetry["score_max"],

                # store dicts as JSON strings to keep CSV clean + schema-stable
                "tier_selected": json.dumps(dict(tier_selected), separators=(",", ":")),
                "cooldown_blocked": json.dumps(dict(cooldown_blocked), separators=(",", ":")),
                "alerts_by_tier": json.dumps(dict(alerts_by_tier), separators=(",", ":")),
            }

            journal.log_telemetry(row)

            # Optional: one-line console print (once per hour, not spammy)
            print(
                f"[TELEMETRY] bars={row['bars_seen']} alerts={row['alerts_sent']} "
                f"below_score={row['below_score']} ind_missing={row['indicator_missing']} "
                f"tiers={row['tier_selected']} cooldown={row['cooldown_blocked']}"
            )

            # reset hourly window
            telemetry.clear()
            telemetry.update({
                "bars_seen": 0,
                "anchor_bars_seen": 0,
                "insufficient_history": 0,
                "indicator_missing": 0,
                "below_score": 0,
                "alerts_sent": 0,
                "scores_n": 0,
                "scores_sum": 0.0,
                "score_min": None,
                "score_max": None,
            })
            tier_selected.clear()
            cooldown_blocked.clear()
            alerts_by_tier.clear()

            # advance window
            telemetry_window_start = next_flush
            next_flush = telemetry_window_start + timedelta(hours=1)

    # Subscribe bars
    for sym in symbols_to_run:
        stream.subscribe_bars(on_bar, sym)

    print("Trade Scout running...")
    stream.run()


if __name__ == "__main__":
    run()