from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from . import config
from .alert_pipeline import AlertPipeline
from .runtime_context import RuntimeContext

if TYPE_CHECKING:
    from .execution.worker import ExecutionWorker


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


class BarProcessor:
    def __init__(
        self,
        context: RuntimeContext,
        *,
        execution_worker: ExecutionWorker | None = None,
    ):
        self.context = context
        self.alerts = AlertPipeline(context)
        self.execution_worker = execution_worker

    def _process_execution_triggers(self, *, symbol: str, close_price: float) -> None:
        if self.execution_worker is None:
            return
        try:
            processed = self.execution_worker.process_position_triggers(
                price_by_symbol={symbol: close_price}
            )
            if processed > 0:
                print(
                    f"[execution-worker] processed tp triggers symbol={symbol} "
                    f"count={processed}"
                )
        except Exception as exc:
            print(f"[execution-worker] trigger processing failed symbol={symbol} error={exc}")

    async def on_bar(self, bar):
        now = datetime.now(timezone.utc)
        symbol = bar.symbol
        close_price = float(bar.close)

        try:
            if self.context.reconnect_attempt > 0:
                print(
                    f"Websocket stabilized after reconnect attempt {self.context.reconnect_attempt}; "
                    "resetting backoff."
                )
                self.context.reconnect_attempt = 0
                websocket_lifecycle = getattr(self.context, "websocket_lifecycle", None)
                if websocket_lifecycle is not None:
                    websocket_lifecycle.record_success()

            ts = pd.to_datetime(bar.timestamp, utc=True).to_pydatetime()

            self.context.telemetry.record_bar(is_anchor=self.context.anchor.is_anchor(symbol))
            self.context.watch_mgr.update(symbol, float(bar.high), float(bar.low), close_price, ts)

            df = self.context.data.get(symbol)
            if df is None:
                return

            df = append_bar_to_df(df, bar, self.context.max_keep)
            self.context.data[symbol] = df

            if len(df) < self.context.min_needed:
                self.context.telemetry.record_insufficient_history()
                return

            anchor_updated = False
            for enabled_strategy in self.context.strategies:
                setup_candidate = enabled_strategy.strategy.evaluate_candidate(df, self.context.profile)
                score = float(setup_candidate.score)
                computed = dict(setup_candidate.computed)
                self.context.telemetry.record_score(score)

                if computed.get("ma200") is None or computed.get("atr") is None:
                    self.context.telemetry.record_indicator_missing()
                    continue

                if self.context.anchor.is_anchor(symbol) and not anchor_updated:
                    self.context.anchor.update_from_computed(computed, ts)
                    anchor_updated = True
                    if symbol not in config.SYMBOLS:
                        return

                tier = setup_candidate.tier
                if tier is None:
                    self.context.telemetry.record_below_score()
                    continue
                self.context.telemetry.record_tier_selected(tier)

                computed_for_log = self.context.anchor.inject_into_computed(computed)
                self.alerts.process(
                    enabled_strategy=enabled_strategy,
                    symbol=symbol,
                    now=now,
                    ts=ts,
                    setup_candidate=setup_candidate,
                    computed_for_log=computed_for_log,
                )

        finally:
            self._process_execution_triggers(symbol=symbol, close_price=close_price)
            self.context.telemetry.maybe_flush(datetime.now(timezone.utc))
