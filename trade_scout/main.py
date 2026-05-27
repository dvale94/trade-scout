import time
import threading

from . import config
from . import strategy_profiles
from . import market
from . import seed
from . import anchors
from . import scheduler_tasks
from . import journal
from .execution.worker import ExecutionWorker, ExecutionWorkerConfig
from .telemetry import TelemetryTracker
from . import watch
from .bar_processor import BarProcessor
from .cooldown import Cooldown
from .runtime_context import EnabledStrategy, RuntimeContext, compute_config_hash
from .strategies.registry import build_strategy_metadata, load_enabled_strategies
from .websocket_control import (
    WebsocketLifecycleGuard,
    classify_websocket_failure,
    format_websocket_reason,
    release_run_guard,
    stop_stream_client,
    try_acquire_run_guard,
)


IMMEDIATE_DISCONNECT_SECONDS = 10.0


def run():
    if not try_acquire_run_guard():
        print("[websocket] duplicate startup skipped reason=already_running")
        return

    profile = strategy_profiles.PROFILES.get(config.STRATEGY_PROFILE)
    try:
        if profile is None:
            raise ValueError(f"Unknown STRATEGY_PROFILE: {config.STRATEGY_PROFILE}")

        config_hash = compute_config_hash(profile)

        min_needed = max(
            profile.ma_mid + 5,
            profile.atr_period + 5,
            profile.breakout_lookback + profile.persist_bars + 5,
        )
        max_keep = max(profile.lookback_bars, min_needed)

        anchor = anchors.AnchorTracker()
        symbols_to_run = list(dict.fromkeys(config.SYMBOLS + [anchor.anchor_symbol]))  # de-dupe, preserve order

        strategies = load_enabled_strategies(config.ENABLED_STRATEGY_IDS, config.STRATEGY_PARAMS)
        if not strategies:
            raise ValueError("At least one enabled strategy is required")
        enabled_strategies = [
            EnabledStrategy(
                strategy=strategy,
                metadata=build_strategy_metadata(
                    strategy,
                    config.STRATEGY_PARAMS.get(strategy.strategy_id),
                ),
            )
            for strategy in strategies
        ]
        strategy_version = ",".join(strategy.strategy_version for strategy in enabled_strategies)

        print(
            f"MARKET_TYPE={config.MARKET_TYPE} PROFILE={profile.name} timeframe={profile.timeframe_min}m "
            f"symbols={symbols_to_run} config_hash={config_hash} strat_ver={strategy_version} "
            f"strategies={[strategy.metadata.strategy_id for strategy in enabled_strategies]}"
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
        runtime = RuntimeContext(
            profile=profile,
            strategies=enabled_strategies,
            config_hash=config_hash,
            min_needed=min_needed,
            max_keep=max_keep,
            data=data,
            anchor=anchor,
            cooldown=cooldown,
            watch_mgr=watch_mgr,
            telemetry=telemetry,
        )
        runtime.websocket_lifecycle = None
        execution_worker = ExecutionWorker(
            ExecutionWorkerConfig(
                db_path=config.EXECUTION_DB_PATH,
                poll_interval_seconds=config.EXECUTION_POLL_INTERVAL_SECONDS,
                batch_size=config.EXECUTION_BATCH_SIZE,
                max_alert_age_seconds=config.EXECUTION_MAX_ALERT_AGE_SECONDS,
                worker_id=config.EXECUTION_WORKER_ID,
                enabled=config.EXECUTION_ENABLED,
            ),
            alert_lookup=journal.load_alert_intent_by_id,
        )
        processor = BarProcessor(
            runtime,
            execution_worker=execution_worker if config.EXECUTION_ENABLED else None,
        )

        scheduler_tasks.start_scheduler()
        print("Maintenance scheduler started (UTC): 6h backups, daily snapshot, nightly DuckDB, monthly archive")

        if config.EXECUTION_ENABLED:
            execution_thread = threading.Thread(
                target=execution_worker.run_forever,
                name=config.EXECUTION_WORKER_ID,
                daemon=True,
            )
            execution_thread.start()
            print(
                "Execution worker started "
                f"(db_path={config.EXECUTION_DB_PATH}, worker_id={config.EXECUTION_WORKER_ID})"
            )

        lifecycle = WebsocketLifecycleGuard()
        runtime.websocket_lifecycle = lifecycle

        while True:
            now_monotonic = time.monotonic()
            if lifecycle.is_cooling_down(now_monotonic):
                remaining = lifecycle.cooldown_remaining_seconds(now_monotonic)
                print(
                    "[websocket] cooldown active "
                    f"live_bar_streaming_paused=true remaining_seconds={remaining:.1f}"
                )
                time.sleep(remaining)
                continue

            if runtime.reconnect_attempt > 0:
                print(
                    "[websocket] reconnect attempt allowed "
                    f"retry={runtime.reconnect_attempt} state={lifecycle.current_state()}"
                )

            if not lifecycle.try_mark_starting():
                print("[websocket] startup skipped reason=already_running")
                time.sleep(1.0)
                continue

            stream = None
            stream_started_monotonic = time.monotonic()
            try:
                stream = market.make_stream()
                lifecycle.bind_client(stream)

                for sym in symbols_to_run:
                    stream.subscribe_bars(processor.on_bar, sym)

                if runtime.reconnect_attempt == 0:
                    print("Trade Scout running...")
                else:
                    print(f"Trade Scout reconnecting (attempt {runtime.reconnect_attempt})...")

                lifecycle.mark_running()
                stream.run()

                reason = classify_websocket_failure(
                    None,
                    immediate_disconnect=(time.monotonic() - stream_started_monotonic) <= IMMEDIATE_DISCONNECT_SECONDS,
                )
                if lifecycle.stop_existing_client():
                    print("[websocket] stopped existing client before reconnect")
                delay, cooldown_triggered = lifecycle.record_failure(
                    reason=reason,
                    now_monotonic=time.monotonic(),
                )
                runtime.reconnect_attempt = lifecycle.circuit_breaker.reconnect_attempt
                log_reason = format_websocket_reason(reason)
                if cooldown_triggered:
                    print(
                        "[websocket] entering cooldown "
                        f"retry={runtime.reconnect_attempt} reason={log_reason} "
                        f"cooldown_seconds={delay:.1f} live_bar_streaming_paused=true"
                    )
                else:
                    print(
                        "[websocket] reconnect scheduled "
                        f"retry={runtime.reconnect_attempt} reason={log_reason} delay_seconds={delay:.1f}"
                    )
                time.sleep(delay)

            except KeyboardInterrupt:
                print("Shutting down Trade Scout.")
                raise

            except Exception as exc:
                reason = classify_websocket_failure(exc)
                if lifecycle.stop_existing_client():
                    print("[websocket] stopped existing client before reconnect")
                delay, cooldown_triggered = lifecycle.record_failure(
                    reason=reason,
                    now_monotonic=time.monotonic(),
                )
                runtime.reconnect_attempt = lifecycle.circuit_breaker.reconnect_attempt
                log_reason = format_websocket_reason(reason)
                if cooldown_triggered:
                    print(
                        "[websocket] entering cooldown "
                        f"retry={runtime.reconnect_attempt} reason={log_reason} "
                        f"cooldown_seconds={delay:.1f} live_bar_streaming_paused=true"
                    )
                else:
                    print(
                        "[websocket] reconnect scheduled "
                        f"retry={runtime.reconnect_attempt} reason={log_reason} "
                        f"delay_seconds={delay:.1f} error={exc}"
                    )
                time.sleep(delay)

            finally:
                if stream is not None:
                    try:
                        stop_stream_client(stream)
                    except Exception as close_err:
                        print(f"Error while closing websocket stream: {close_err}")
                if not lifecycle.is_cooling_down(time.monotonic()):
                    lifecycle.mark_stopped()
    finally:
        release_run_guard()


if __name__ == "__main__":
    run()
