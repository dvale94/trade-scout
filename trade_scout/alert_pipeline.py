from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from . import config
from . import journal
from . import notifier
from . import watch
from .journal_builders import (
    compute_entry_zone_width_pct,
    make_alert_context_row,
    make_intent_core_row,
)
from .runtime_context import EnabledStrategy, RuntimeContext
from .strategies.types import SetupCandidate
from .strategy import Candidate


def format_alert(candidate, tier: str) -> str:
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


def _candidate_from_trade_plan(
    *,
    alert_id: str,
    symbol: str,
    now: datetime,
    setup_candidate: SetupCandidate,
    trade_plan,
) -> Candidate:
    return Candidate(
        alert_id=alert_id,
        symbol=symbol,
        timeframe_min=trade_plan.timeframe_min,
        score=float(setup_candidate.score),
        ts=now,
        entry_low=float(trade_plan.entry_low),
        entry_high=float(trade_plan.entry_high),
        stop=float(trade_plan.stop_px),
        t1=float(trade_plan.t1_px),
        t2=float(trade_plan.t2_px),
        r=float(trade_plan.stop_distance),
        reasons=list(setup_candidate.reasons),
        news_sentiment=None,
        news_summary=None,
    )


class AlertPipeline:
    def __init__(self, context: RuntimeContext):
        self.context = context

    def _log_intent_and_context(self, symbol: str, core_row, context_row) -> None:
        journal.log_alert_intent(core_row)
        try:
            journal.log_alert_context(context_row)
        except Exception as context_err:
            print(f"ALERT CONTEXT LOG ERROR for {symbol}: {context_err}")

    def process(
        self,
        *,
        enabled_strategy: EnabledStrategy,
        symbol: str,
        now: datetime,
        ts: datetime,
        setup_candidate: SetupCandidate,
        computed_for_log: dict,
    ) -> None:
        profile = self.context.profile
        strategy = enabled_strategy.strategy
        strategy_metadata = enabled_strategy.metadata
        score = float(setup_candidate.score)
        reasons = list(setup_candidate.reasons)
        tier = setup_candidate.tier

        if tier is None:
            raise ValueError("AlertPipeline.process requires a tiered setup candidate")

        provisional_alert_id = str(uuid4())
        intent_row = make_intent_core_row(
            alert_id=provisional_alert_id,
            symbol=symbol,
            tier=tier,
            now=now,
            timeframe_min=profile.timeframe_min,
            computed_for_log=computed_for_log,
            config_hash=self.context.config_hash,
            strategy_metadata=strategy_metadata,
        )
        alert_context_row = make_alert_context_row(
            alert_id=provisional_alert_id,
            now=now,
            bar_ts=ts,
            timeframe_min=profile.timeframe_min,
            score=score,
            computed_for_log=computed_for_log,
            config_hash=self.context.config_hash,
            strategy_metadata=strategy_metadata,
        )

        if not self.context.cooldown.allowed(symbol, tier, now, strategy_metadata.strategy_id):
            self.context.telemetry.record_cooldown_blocked(tier)
            intent_row = replace(intent_row, alert_status="SKIPPED_COOLDOWN", skip_reason="COOLDOWN")
            self._log_intent_and_context(symbol, intent_row, alert_context_row)
            return

        try:
            trade_plan = strategy.build_trade_plan(
                symbol,
                profile,
                setup_candidate,
            )
            if trade_plan is None:
                intent_row = replace(
                    intent_row,
                    alert_status="SKIPPED_INVALID_STOP",
                    skip_reason="CANDIDATE_NONE",
                    reasons=" | ".join(reasons) if reasons else None,
                )
                self._log_intent_and_context(symbol, intent_row, alert_context_row)
                return

            candidate = _candidate_from_trade_plan(
                alert_id=str(uuid4()),
                symbol=symbol,
                now=now,
                setup_candidate=setup_candidate,
                trade_plan=trade_plan,
            )

            intent_row = replace(
                intent_row,
                alert_id=candidate.alert_id,
                entry_low=round(candidate.entry_low, 6),
                entry_high=round(candidate.entry_high, 6),
                stop_px=round(candidate.stop, 6),
                t1_px=round(candidate.t1, 6),
                t2_px=round(candidate.t2, 6),
                timeout_seconds=int(profile.watch_timeout_min * 60),
                stop_distance=round(candidate.r, 6),
                t1_R=1.5,
                t2_R=2.5,
                risk_model="fixed_usd",
                risk_usd=None,
                qty_model=None,
                qty=None,
            )
            alert_context_row = replace(alert_context_row, alert_id=candidate.alert_id)
            entry_zone_width_pct = compute_entry_zone_width_pct(candidate.entry_low, candidate.entry_high)
            if entry_zone_width_pct is not None:
                alert_context_row = replace(alert_context_row, entry_zone_width_pct=round(entry_zone_width_pct, 6))

            if candidate.r / candidate.entry_high < 0.004:
                self.context.telemetry.record_skipped_tight_stop()
                intent_row = replace(
                    intent_row,
                    alert_status="SKIPPED_TIGHT_STOP",
                    skip_reason="TIGHT_STOP",
                    reasons=" | ".join(candidate.reasons) if candidate.reasons else None,
                )
                self._log_intent_and_context(symbol, intent_row, alert_context_row)
                return

            journal.log_alert(candidate, computed_for_log, tier, config.STRATEGY_PROFILE)
            notifier.discord_notify(format_alert(candidate, tier))
            self.context.telemetry.record_alert_sent(tier)

            intent_row = replace(
                intent_row,
                alert_status="ALERT_CREATED",
                reasons=" | ".join(candidate.reasons) if candidate.reasons else None,
            )
            self._log_intent_and_context(symbol, intent_row, alert_context_row)

            self.context.watch_mgr.add(
                watch.Watch(
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
                    strategy_id=strategy_metadata.strategy_id,
                    strategy_version=strategy_metadata.strategy_version,
                    setup_family=strategy_metadata.setup_family,
                    strategy_params_json=strategy_metadata.strategy_params_json,
                    strategy_params_hash=strategy_metadata.strategy_params_hash,
                    config_hash=self.context.config_hash,
                    side="LONG",
                )
            )

            self.context.cooldown.mark(symbol, tier, now, strategy_metadata.strategy_id)
            print(f"ALERT {symbol} tier={tier} score={score:.2f}")

        except Exception as e:
            print(f"ALERT ERROR for {symbol}: {e}")
            intent_row = replace(intent_row, alert_status="ERROR_ALERT_PIPELINE", error_message=str(e))
            self._log_intent_and_context(symbol, intent_row, alert_context_row)
