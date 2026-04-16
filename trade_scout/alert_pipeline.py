from __future__ import annotations

import json
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
from .policy.evaluator import evaluate_policy
from .policy.models import PolicyConfig
from .policy.repository import PolicyRepository
from .policy.shadow_variants import seed_shadow_policy_variants
from .runtime_context import EnabledStrategy, RuntimeContext
from .scoring import RuntimeShadowScoreConfig, RuntimeShadowScorer, active_shadow_score_config
from .scoring.scorer import SHADOW_SCORE_STATUS_ERROR
from .strategies.types import SetupCandidate
from .strategy import Candidate
from .execution.models import ExecutionRequest, ExecutionStatus
from .execution.queue import ActiveExecutionRequestExistsError, get_connection, insert_request, utc_now_iso

POLICY_NAME = "execution_policy"


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
    def __init__(
        self,
        context: RuntimeContext,
        *,
        policy_repository: PolicyRepository | None = None,
        policy_name: str = POLICY_NAME,
        shadow_scorer: RuntimeShadowScorer | None = None,
    ):
        self.context = context
        self.policy_repository = policy_repository or PolicyRepository(config.EXECUTION_DB_PATH)
        self.execution_db_path = getattr(self.policy_repository, "db_path", config.EXECUTION_DB_PATH)
        self.policy_name = policy_name
        self.shadow_scorer = shadow_scorer or self._build_shadow_scorer()
        self.shadow_policy_variant_names = list(config.SHADOW_POLICY_VARIANT_NAMES)
        self._policy_runtime_ready = False
        try:
            self.policy_repository.init_db()
            if self.shadow_policy_variant_names:
                seed_shadow_policy_variants(self.policy_repository, self.shadow_policy_variant_names)
            self._policy_runtime_ready = True
        except Exception as exc:
            print(
                f"[policy] error alert_id=UNKNOWN stage=init_db "
                f"policy_name={self.policy_name} mode=SHADOW error={exc}"
            )
        self._log_collection_config()

    def _build_shadow_scorer(self) -> RuntimeShadowScorer:
        try:
            return RuntimeShadowScorer(active_shadow_score_config())
        except Exception as exc:
            print(f"[shadow-score] error alert_id=UNKNOWN mode=SHADOW stage=config error={exc}")
            return RuntimeShadowScorer(
                RuntimeShadowScoreConfig(
                    score_model_version="shadow_score_disabled",
                    feature_set_version="score_features_v1",
                    score_recipe_version="score_recipe_disabled",
                    score_recipe_mode="disabled",
                    score_bucket_version="shadow_score_buckets_v1",
                    enabled=False,
                )
            )

    def _log_collection_config(self) -> None:
        variant_names = ",".join(self.shadow_policy_variant_names) or "<none>"
        experiment_tag = config.EXPERIMENT_TAG or "<unset>"
        print(
            f"[startup] collection_config experiment_tag={experiment_tag} "
            f"shadow_policy_variant_names={variant_names} "
            f"policy_runtime_ready={self._policy_runtime_ready}"
        )

    def _log_intent_and_context(self, symbol: str, core_row, context_row) -> None:
        journal.log_alert_intent(core_row)
        try:
            journal.log_alert_context(context_row)
        except Exception as context_err:
            print(f"ALERT CONTEXT LOG ERROR for {symbol}: {context_err}")

    def _process_policy_shadow(self, intent_row, context_row) -> None:
        alert_id = str(intent_row.alert_id)
        if not self._policy_runtime_ready:
            return

        try:
            conn = self.policy_repository.get_connection()
        except Exception as exc:
            self._log_policy_error(alert_id=alert_id, stage="get_connection", error=exc)
            return

        try:
            policy_names = [self.policy_name, *self.shadow_policy_variant_names]
            try:
                active_policies = self.policy_repository.get_active_policies(
                    conn,
                    policy_names,
                    str(intent_row.signal_ts_utc),
                    mode="SHADOW",
                )
            except Exception as exc:
                self._log_policy_error(alert_id=alert_id, stage="get_active_policy", error=exc)
                return

            if not active_policies:
                print(
                    f"[policy] skipped alert_id={alert_id} reason=no_active_policy "
                    f"policy_name={self.policy_name} mode=SHADOW"
                )
                return

            for active_policy in active_policies:
                try:
                    policy_config = _policy_config_from_definition_row(active_policy)
                except Exception as exc:
                    self._log_policy_error(alert_id=alert_id, stage="parse_config", error=exc, policy_name=active_policy.policy_name)
                    continue

                try:
                    decision = evaluate_policy(
                        intent_row,
                        context_row,
                        policy_config,
                        mode="SHADOW",
                    )
                except Exception as exc:
                    self._log_policy_error(alert_id=alert_id, stage="evaluate", error=exc, policy_name=active_policy.policy_name)
                    continue

                try:
                    decision_row = decision.to_row(active_policy.policy_id)
                    persisted = self.policy_repository.insert_policy_decision(conn, decision_row)
                except Exception as exc:
                    self._log_policy_error(alert_id=alert_id, stage="persist", error=exc, policy_name=active_policy.policy_name)
                    continue

                print(
                    f"[policy] evaluated alert_id={alert_id} decision={persisted.decision} "
                    f"policy_name={persisted.policy_name} policy_version={persisted.policy_version} "
                    f"mode=SHADOW"
                )
        finally:
            conn.close()

    def _log_policy_error(
        self,
        *,
        alert_id: str,
        stage: str,
        error: Exception,
        policy_name: str | None = None,
    ) -> None:
        print(
            f"[policy] error alert_id={alert_id} stage={stage} "
            f"policy_name={policy_name or self.policy_name} mode=SHADOW error={error}"
        )

    def _apply_shadow_score(self, intent_row, context_row):
        alert_id = str(intent_row.alert_id)
        try:
            output = self.shadow_scorer.score(intent_row, context_row)
        except Exception as exc:
            print(f"[shadow-score] error alert_id={alert_id} mode=SHADOW stage=score error={exc}")
            return replace(context_row, shadow_score_status=SHADOW_SCORE_STATUS_ERROR)

        print(
            f"[shadow-score] evaluated alert_id={alert_id} "
            f"score_model_version={output.score_model_version} "
            f"score_recipe_mode={output.score_recipe_mode} "
            f"status={output.shadow_score_status} has_score={output.score_shadow is not None} mode=SHADOW"
        )
        return replace(context_row, **output.to_context_updates())

    def _should_auto_create_execution_request(self, *, symbol: str) -> bool:
        mode = _normalized_auto_execution_mode(config.AUTO_EXECUTION_MODE)
        normalized_symbol = _normalized_symbol(symbol)
        if mode == "OFF":
            return False
        if mode == "ALL":
            return True
        return normalized_symbol in _normalized_symbol_set(config.AUTO_EXECUTION_SYMBOLS)

    def _process_auto_execution(self, *, alert_id: str, symbol: str) -> None:
        mode = _normalized_auto_execution_mode(config.AUTO_EXECUTION_MODE)
        if not self._should_auto_create_execution_request(symbol=symbol):
            return

        try:
            conn = get_connection(self.execution_db_path)
        except Exception as exc:
            print(
                f"[execution-auto] error alert_id={alert_id} symbol={symbol} "
                f"mode={mode} stage=get_connection error={exc}"
            )
            return

        try:
            try:
                insert_request(
                    conn,
                    ExecutionRequest(
                        request_id=str(uuid4()),
                        alert_id=alert_id,
                        requested_ts_utc=utc_now_iso(),
                        status=ExecutionStatus.PENDING,
                        requested_by="auto_execution",
                        reason=f"AUTO_EXECUTION_{mode}",
                    ),
                )
            except ActiveExecutionRequestExistsError:
                return
        except Exception as exc:
            print(
                f"[execution-auto] error alert_id={alert_id} symbol={symbol} "
                f"mode={mode} stage=insert_request error={exc}"
            )
            return
        finally:
            conn.close()

        print(
            f"[execution-auto] inserted alert_id={alert_id} symbol={symbol} mode={mode}"
        )

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

            alert_context_row = self._apply_shadow_score(intent_row, alert_context_row)

            self._process_policy_shadow(intent_row, alert_context_row)

            journal.log_alert(candidate, computed_for_log, tier, config.STRATEGY_PROFILE)
            notifier.discord_notify(format_alert(candidate, tier))
            self.context.telemetry.record_alert_sent(tier)

            intent_row = replace(
                intent_row,
                alert_status="ALERT_CREATED",
                reasons=" | ".join(candidate.reasons) if candidate.reasons else None,
            )
            self._log_intent_and_context(symbol, intent_row, alert_context_row)

            self._process_auto_execution(
                alert_id=str(candidate.alert_id),
                symbol=symbol,
            )

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


def _policy_config_from_definition_row(policy_definition_row) -> PolicyConfig:
    raw_config = json.loads(str(policy_definition_row.config_json))
    if not isinstance(raw_config, dict):
        raise ValueError("policy config_json must decode to an object")

    raw_filters = raw_config.get("filters", raw_config)
    if not isinstance(raw_filters, dict):
        raise ValueError("policy config filters must decode to an object")

    description = raw_config.get("description")
    if description is None:
        description = policy_definition_row.description or ""

    return PolicyConfig(
        policy_name=str(policy_definition_row.policy_name),
        policy_version=str(policy_definition_row.policy_version),
        policy_hash=str(policy_definition_row.policy_hash),
        description=str(description),
        filters=dict(raw_filters),
    )


def _normalized_auto_execution_mode(value: object) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"OFF", "FILTERED", "ALL"}:
        return normalized
    return "OFF"


def _normalized_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _normalized_symbol_set(values: object) -> set[str]:
    if isinstance(values, set):
        return {_normalized_symbol(value) for value in values if _normalized_symbol(value)}
    if isinstance(values, (list, tuple)):
        return {_normalized_symbol(value) for value in values if _normalized_symbol(value)}
    return set()
