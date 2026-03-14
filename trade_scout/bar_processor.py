from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config
from .alert_pipeline import AlertPipeline
from .runtime_context import RuntimeContext


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
    def __init__(self, context: RuntimeContext):
        self.context = context
        self.alerts = AlertPipeline(context)

    async def on_bar(self, bar):
        now = datetime.now(timezone.utc)

        try:
            if self.context.reconnect_attempt > 0:
                print(
                    f"Websocket stabilized after reconnect attempt {self.context.reconnect_attempt}; "
                    "resetting backoff."
                )
                self.context.reconnect_attempt = 0

            symbol = bar.symbol
            ts = pd.to_datetime(bar.timestamp, utc=True).to_pydatetime()

            self.context.telemetry.record_bar(is_anchor=self.context.anchor.is_anchor(symbol))
            self.context.watch_mgr.update(symbol, float(bar.high), float(bar.low), float(bar.close), ts)

            df = self.context.data.get(symbol)
            if df is None:
                return

            df = append_bar_to_df(df, bar, self.context.max_keep)
            self.context.data[symbol] = df

            if len(df) < self.context.min_needed:
                self.context.telemetry.record_insufficient_history()
                return

            score, reasons, computed = self.context.strategy_engine.score_candidate(df, self.context.profile)
            self.context.telemetry.record_score(score)

            if computed.get("ma200") is None or computed.get("atr") is None:
                self.context.telemetry.record_indicator_missing()
                return

            if self.context.anchor.is_anchor(symbol):
                self.context.anchor.update_from_computed(computed, ts)
                if symbol not in config.SYMBOLS:
                    return

            tier = self.context.strategy_engine.pick_tier(score, df, self.context.profile, computed)
            if tier is None:
                self.context.telemetry.record_below_score()
                return
            self.context.telemetry.record_tier_selected(tier)

            computed_for_log = self.context.anchor.inject_into_computed(computed)
            self.alerts.process(
                symbol=symbol,
                now=now,
                ts=ts,
                score=score,
                reasons=reasons,
                computed=computed,
                computed_for_log=computed_for_log,
                tier=tier,
            )

        finally:
            self.context.telemetry.maybe_flush(datetime.now(timezone.utc))
