import time
import random

from . import config
from . import strategy_profiles
from . import market
from . import seed
from . import anchors
from . import scheduler_tasks
from .telemetry import TelemetryTracker
from . import watch
from .bar_processor import BarProcessor
from .cooldown import Cooldown
from .runtime_context import RuntimeContext, compute_config_hash
from .strategy_engine import CurrentStrategyAdapter


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

    telemetry = TelemetryTracker(config.STRATEGY_PROFILE, profile.timeframe_min)
    strategy_engine = CurrentStrategyAdapter()
    runtime = RuntimeContext(
        profile=profile,
        strategy_engine=strategy_engine,
        strategy_version=strategy_version,
        config_hash=config_hash,
        min_needed=min_needed,
        max_keep=max_keep,
        data=data,
        anchor=anchor,
        cooldown=cooldown,
        watch_mgr=watch_mgr,
        telemetry=telemetry,
    )
    processor = BarProcessor(runtime)

    scheduler_tasks.start_scheduler()
    print("Maintenance scheduler started (UTC): 6h backups, daily snapshot, nightly DuckDB, monthly archive")

    while True:
        stream = None
        try:
            stream = market.make_stream()

            for sym in symbols_to_run:
                stream.subscribe_bars(processor.on_bar, sym)

            if runtime.reconnect_attempt == 0:
                print("Trade Scout running...")
            else:
                print(f"Trade Scout reconnecting (attempt {runtime.reconnect_attempt})...")

            stream.run()

            # If run() exits without exception treat as disconnect
            runtime.reconnect_attempt += 1
            delay = websocket_backoff(runtime.reconnect_attempt)
            print(f"Websocket stopped. Reconnecting in {delay:.1f}s...")

        except KeyboardInterrupt:
            print("Shutting down Trade Scout.")
            raise

        except Exception as e:
            runtime.reconnect_attempt += 1
            msg = str(e).lower()

            if "429" in msg or "connection limit exceeded" in msg:
                print(f"Websocket rate limit / connection cap hit: {e}")
            else:
                print(f"Websocket error: {e}")

            delay = websocket_backoff(runtime.reconnect_attempt)
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
