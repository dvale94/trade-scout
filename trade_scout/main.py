import json
import hashlib
import time
import random
from uuid import uuid4
from datetime import datetime, timezone, timedelta
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
from . import scheduler_tasks


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


def compute_config_hash(profile) -> str:
    """
    Stable hash of strategy + config + profile knobs.
    Keep this deterministic; only include things you consider part of the 'experiment configuration'.
    """
    payload = {
        "market_type": config.MARKET_TYPE,
        "strategy_profile": config.STRATEGY_PROFILE,
        "alpaca_stock_feed": config.ALPACA_STOCK_FEED,
        "symbols": list(config.SYMBOLS),

        # profile parameters (dataclass -> dict)
        "profile": profile.__dict__,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def make_intent_base(
    *,
    alert_id: str,
    symbol: str,
    tier: str,
    now: datetime,
    bar_ts: datetime,
    timeframe_min: int,
    score: float,
    computed_for_log: dict,
    config_hash: str,
    strategy_version: str,
) -> dict:
    bar_end = bar_ts
    bar_start = bar_ts - timedelta(minutes=timeframe_min)

    row = {
        # identifiers
        "alert_id": alert_id,
        "symbol": symbol,
        "side": "LONG",
        "market_type": config.MARKET_TYPE,
        "strategy_profile": config.STRATEGY_PROFILE,
        "strategy_version": strategy_version,
        "config_hash": config_hash,
        "tier": tier,

        # timestamps
        "signal_ts_utc": now.astimezone(timezone.utc).isoformat(),
        "signal_hour_utc": int(now.astimezone(timezone.utc).hour),
        "signal_day_of_week": now.astimezone(timezone.utc).strftime("%a"),
        "bar_start_ts_utc": bar_start.astimezone(timezone.utc).isoformat(),
        "bar_end_ts_utc": bar_end.astimezone(timezone.utc).isoformat(),
        "timeframe_min": timeframe_min,

        # signal price (what we scored on)
        "signal_px": float(computed_for_log.get("close")) if computed_for_log.get("close") is not None else None,
        "signal_px_type": "bar_close",

        # status (filled later)
        "alert_status": None,
        "skip_reason": None,
        "error_message": None,

        # quick context
        "score": round(float(score), 4),

        # carry forward your computed snapshot fields (anchor injected)
        "atr": computed_for_log.get("atr"),
        "ma20": computed_for_log.get("ma20"),
        "ma50": computed_for_log.get("ma50"),
        "ma200": computed_for_log.get("ma200"),
        "breakout_level": computed_for_log.get("breakout_level"),
        "breakout_distance_atr": computed_for_log.get("breakout_distance_atr"),
        "ma20_distance_atr": computed_for_log.get("ma20_distance_atr"),
        "ma50_distance_atr": computed_for_log.get("ma50_distance_atr"),
        "ma200_distance_atr": computed_for_log.get("ma200_distance_atr"),
        "ma20_slope_pct": computed_for_log.get("ma20_slope_pct"),
        "ma50_slope_pct": computed_for_log.get("ma50_slope_pct"),
        "range_pct": computed_for_log.get("range_pct"),
        "close_location_in_bar": computed_for_log.get("close_location_in_bar"),
        "anchor_symbol": computed_for_log.get("anchor_symbol"),
        "anchor_regime": computed_for_log.get("anchor_regime"),
        "anchor_close": computed_for_log.get("anchor_close"),
        "anchor_ma50": computed_for_log.get("anchor_ma50"),
        "anchor_ma200": computed_for_log.get("anchor_ma200"),
        "vol_z": computed_for_log.get("vol_z"),
        "extended_atr": computed_for_log.get("extended_atr"),

        # planned levels / sizing placeholders (keep alerts_intents schema stable)
        "entry_low": None,
        "entry_high": None,
        "entry_zone_width_pct": None,
        "stop_px": None,
        "t1_px": None,
        "t2_px": None,
        "timeout_seconds": None,
        "stop_distance": None,
        "t1_R": None,
        "t2_R": None,
        "risk_model": None,
        "risk_usd": None,
        "qty_model": None,
        "qty": None,
        "reasons": None,
    }
    return row


def _compute_entry_zone_width_pct(entry_low: float | None, entry_high: float | None) -> float | None:
    if entry_low is None or entry_high is None or entry_high <= 0:
        return None
    return (entry_high - entry_low) / entry_high


def websocket_backoff(attempt: int, base: float = 5.0, cap: float = 120.0) -> float:
    """
    Exponential backoff with jitter.
    attempt=1 -> ~5s
    attempt=2 -> ~10s
    attempt=3 -> ~20s
    capped at 120s
    """
    delay = min(cap, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, delay * 0.25)
    sleep_for = delay + jitter
    time.sleep(sleep_for)
    return sleep_for


def run():
    profile = strategy_profiles.PROFILES.get(config.STRATEGY_PROFILE)
    if profile is None:
        raise ValueError(f"Unknown STRATEGY_PROFILE: {config.STRATEGY_PROFILE}")
    
    strategy_version = "v3_research_step2"
    config_hash = compute_config_hash(profile)
    
    min_needed = max(
        profile.ma_mid + 5,
        profile.atr_period + 5,
        profile.breakout_lookback + profile.persist_bars + 5,
    )
    max_keep = max(profile.lookback_bars, min_needed)
    
    anchor = anchors.AnchorTracker()
    symbols_to_run = list(dict.fromkeys(config.SYMBOLS + [anchor.anchor_symbol]))  # de-dupe, preserve order

    print(
        f"MARKET_TYPE={config.MARKET_TYPE} PROFILE={profile.name} timeframe={profile.timeframe_min}m "
        f"symbols={symbols_to_run} config_hash={config_hash} strat_ver={strategy_version}"
    )

    print("Seeding history...")
    data = seed.seed_history(
        symbols_to_run,
        timeframe_min=profile.timeframe_min,
        seed_minutes=profile.seed_minutes,
        lookback_bars=profile.lookback_bars,
    )
    print("Seed complete:", {k: len(v) for k, v in data.items()})

    cooldown = Cooldown(profile.cooldown_min)
    watch_mgr = watch.WatchManager(timeout_minutes=profile.watch_timeout_min, max_active_per_symbol=50)
    watch_mgr.load_state()
    print(f"Restored active watches: {sum(len(v) for v in watch_mgr.active.values())}")

    # --- Telemetry (hourly) ---
    telemetry = TelemetryTracker(config.STRATEGY_PROFILE, profile.timeframe_min)

    scheduler_tasks.start_scheduler()
    print("Maintenance scheduler started (UTC): 6h backups, daily snapshot, nightly DuckDB, monthly archive")

    async def on_bar(bar):
        nonlocal reconnect_attempt

        now = datetime.now(timezone.utc)

        try:
            if reconnect_attempt > 0:
                print(f"Websocket stabilized after reconnect attempt {reconnect_attempt}; resetting backoff.")
                reconnect_attempt = 0

            symbol = bar.symbol
            ts = pd.to_datetime(bar.timestamp, utc=True).to_pydatetime()

            telemetry.record_bar(is_anchor=anchor.is_anchor(symbol))
            watch_mgr.update(symbol, float(bar.high), float(bar.low), float(bar.close), ts)

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

            if computed.get("ma200") is None or computed.get("atr") is None:
                telemetry.record_indicator_missing()
                return

            if anchor.is_anchor(symbol):
                anchor.update_from_computed(computed, ts)
                if symbol not in config.SYMBOLS:
                    return

            tier = strategy.pick_tier(score, df, profile, computed)
            if tier is None:
                telemetry.record_below_score()
                return
            telemetry.record_tier_selected(tier)

            computed_for_log = anchor.inject_into_computed(computed)
            provisional_alert_id = str(uuid4())

            intent_row = make_intent_base(
                alert_id=provisional_alert_id,
                symbol=symbol,
                tier=tier,
                now=now,
                bar_ts=ts,
                timeframe_min=profile.timeframe_min,
                score=score,
                computed_for_log=computed_for_log,
                config_hash=config_hash,
                strategy_version=strategy_version,
            )

            if not cooldown.allowed(symbol, tier, now):
                telemetry.record_cooldown_blocked(tier)
                intent_row["alert_status"] = "SKIPPED_COOLDOWN"
                intent_row["skip_reason"] = "COOLDOWN"
                journal.log_alert_intent(intent_row)
                return

            try:
                candidate = strategy.compute_entry_stop_targets(symbol, profile, computed, reasons, score)
                if candidate is None:
                    intent_row["alert_status"] = "SKIPPED_INVALID_STOP"
                    intent_row["skip_reason"] = "CANDIDATE_NONE"
                    intent_row["reasons"] = " | ".join(reasons) if reasons else None
                    journal.log_alert_intent(intent_row)
                    return

                intent_row["alert_id"] = candidate.alert_id
                intent_row["entry_low"] = round(candidate.entry_low, 6)
                intent_row["entry_high"] = round(candidate.entry_high, 6)
                entry_zone_width_pct = _compute_entry_zone_width_pct(candidate.entry_low, candidate.entry_high)
                if entry_zone_width_pct is not None:
                    intent_row["entry_zone_width_pct"] = round(entry_zone_width_pct, 6)
                intent_row["stop_px"] = round(candidate.stop, 6)
                intent_row["t1_px"] = round(candidate.t1, 6)
                intent_row["t2_px"] = round(candidate.t2, 6)
                intent_row["timeout_seconds"] = int(profile.watch_timeout_min * 60)
                intent_row["stop_distance"] = round(candidate.r, 6)
                intent_row["t1_R"] = 1.5
                intent_row["t2_R"] = 2.5
                intent_row["risk_model"] = "fixed_usd"
                intent_row["risk_usd"] = None
                intent_row["qty_model"] = None
                intent_row["qty"] = None

                if candidate.r / candidate.entry_high < 0.004:
                    telemetry.record_skipped_tight_stop()
                    intent_row["alert_status"] = "SKIPPED_TIGHT_STOP"
                    intent_row["skip_reason"] = "TIGHT_STOP"
                    intent_row["reasons"] = " | ".join(candidate.reasons) if candidate.reasons else None
                    journal.log_alert_intent(intent_row)
                    return

                journal.log_alert(candidate, computed_for_log, tier, config.STRATEGY_PROFILE)
                notifier.discord_notify(format_alert(candidate, tier))
                telemetry.record_alert_sent(tier)

                intent_row["alert_status"] = "ALERT_CREATED"
                intent_row["reasons"] = " | ".join(candidate.reasons) if candidate.reasons else None
                journal.log_alert_intent(intent_row)

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
                    strategy_profile=config.STRATEGY_PROFILE,
                    strategy_version=strategy_version,
                    config_hash=config_hash,
                    side="LONG",
                ))

                cooldown.mark(symbol, tier, now)
                print(f"ALERT {symbol} tier={tier} score={score:.2f}")

            except Exception as e:
                print(f"ALERT ERROR for {symbol}: {e}")
                intent_row["alert_status"] = "ERROR_ALERT_PIPELINE"
                intent_row["error_message"] = str(e)
                journal.log_alert_intent(intent_row)

        finally:
            telemetry.maybe_flush(datetime.now(timezone.utc))

    reconnect_attempt = 0

    while True:
        stream = None
        try:
            stream = market.make_stream()

            for sym in symbols_to_run:
                stream.subscribe_bars(on_bar, sym)

            if reconnect_attempt == 0:
                print("Trade Scout running...")
            else:
                print(f"Trade Scout reconnecting (attempt {reconnect_attempt})...")

            stream.run()

            # If run() exits without exception treat as disconnect
            reconnect_attempt += 1
            delay = websocket_backoff(reconnect_attempt)
            print(f"Websocket stopped. Reconnecting in {delay:.1f}s...")

        except KeyboardInterrupt:
            print("Shutting down Trade Scout.")
            raise

        except Exception as e:
            reconnect_attempt += 1
            msg = str(e).lower()

            if "429" in msg or "connection limit exceeded" in msg:
                print(f"Websocket rate limit / connection cap hit: {e}")
            else:
                print(f"Websocket error: {e}")

            delay = websocket_backoff(reconnect_attempt)
            print(f"Retrying websocket in {delay:.1f}s...")

        finally:
            if stream is not None:
                try:
                    close_fn = getattr(stream, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as close_err:
                    print(f"Error while closing websocket stream: {close_err}")


if __name__ == "__main__":
    run()
