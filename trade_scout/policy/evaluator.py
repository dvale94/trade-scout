from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from trade_scout.journal_rows import AlertContextRow, AlertIntentRow
from trade_scout.policy.models import (
    PolicyConfig,
    PolicyCriterionResult,
    PolicyDecision,
    PolicyFeatureSnapshot,
)


SCORE_BUCKET_ORDER = {
    "LT_5": 0,
    "5_TO_5_9": 1,
    "6_TO_6_9": 2,
    "7_TO_7_9": 3,
    "8_PLUS": 4,
}


@dataclass(frozen=True)
class PolicyEvaluationResult:
    decision: PolicyDecision


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_policy(
    alert_intent: AlertIntentRow,
    alert_context: AlertContextRow,
    policy_config: PolicyConfig,
    *,
    mode: str = "SHADOW",
    policy_decision_id: str | None = None,
    decision_ts_utc: str | None = None,
) -> PolicyDecision:
    feature_snapshot = build_feature_snapshot(alert_intent, alert_context)
    criteria_results: list[PolicyCriterionResult] = []
    reasons: list[str] = []

    allowed_setup_families = _normalized_filter_values(
        policy_config.filters.get("allowed_setup_families")
    )
    if allowed_setup_families:
        passed = feature_snapshot.setup_family in allowed_setup_families
        criteria_results.append(
            PolicyCriterionResult(
                criterion_name="allowed_setup_families",
                passed=passed,
                actual_value=feature_snapshot.setup_family,
                expected_value=allowed_setup_families,
            )
        )
        if not passed:
            reasons.append(f"setup_family '{feature_snapshot.setup_family}' not allowed")

    allowed_anchor_regimes = _normalized_filter_values(
        policy_config.filters.get("allowed_anchor_regimes")
    )
    if allowed_anchor_regimes:
        passed = feature_snapshot.anchor_regime in allowed_anchor_regimes
        criteria_results.append(
            PolicyCriterionResult(
                criterion_name="allowed_anchor_regimes",
                passed=passed,
                actual_value=feature_snapshot.anchor_regime,
                expected_value=allowed_anchor_regimes,
            )
        )
        if not passed:
            reasons.append(f"anchor_regime '{feature_snapshot.anchor_regime}' not allowed")

    min_score_bucket = _normalized_optional_str(policy_config.filters.get("min_score_bucket"))
    if min_score_bucket:
        actual_rank = SCORE_BUCKET_ORDER.get(feature_snapshot.score_bucket)
        expected_rank = SCORE_BUCKET_ORDER.get(min_score_bucket)
        passed = actual_rank is not None and expected_rank is not None and actual_rank >= expected_rank
        criteria_results.append(
            PolicyCriterionResult(
                criterion_name="min_score_bucket",
                passed=passed,
                actual_value=feature_snapshot.score_bucket,
                expected_value=min_score_bucket,
            )
        )
        if expected_rank is None:
            reasons.append(f"configured min_score_bucket '{min_score_bucket}' is unknown")
        elif actual_rank is None:
            reasons.append(f"score_bucket '{feature_snapshot.score_bucket}' is unknown")
        elif not passed:
            reasons.append(
                f"score_bucket '{feature_snapshot.score_bucket}' below required minimum '{min_score_bucket}'"
            )

    allowed_symbols = _normalized_filter_values(policy_config.filters.get("allowed_symbols"))
    if allowed_symbols:
        passed = feature_snapshot.symbol in allowed_symbols
        criteria_results.append(
            PolicyCriterionResult(
                criterion_name="allowed_symbols",
                passed=passed,
                actual_value=feature_snapshot.symbol,
                expected_value=allowed_symbols,
            )
        )
        if not passed:
            reasons.append(f"symbol '{feature_snapshot.symbol}' not allowed")

    all_passed = all(result.passed for result in criteria_results)
    decision = "ALLOW" if all_passed else "BLOCK"
    if not reasons and decision == "ALLOW":
        reasons.append("all configured policy criteria passed")

    return PolicyDecision(
        policy_decision_id=policy_decision_id or str(uuid4()),
        alert_id=feature_snapshot.alert_id,
        policy_name=policy_config.policy_name,
        policy_version=policy_config.policy_version,
        policy_hash=policy_config.policy_hash,
        mode=mode,
        decision=decision,
        decision_ts_utc=decision_ts_utc or utc_now_iso(),
        reasons=reasons,
        criteria_results=criteria_results,
        feature_snapshot=feature_snapshot,
        execution_candidate=decision == "ALLOW",
        request_id=None,
    )


def build_feature_snapshot(
    alert_intent: AlertIntentRow,
    alert_context: AlertContextRow,
) -> PolicyFeatureSnapshot:
    if alert_intent.alert_id != alert_context.alert_id:
        raise ValueError(
            "alert_intent.alert_id and alert_context.alert_id must match for policy evaluation"
        )
    if alert_intent.setup_family is None or not str(alert_intent.setup_family).strip():
        raise ValueError("policy evaluation requires alert_intent.setup_family")
    if alert_context.anchor_regime is None or not str(alert_context.anchor_regime).strip():
        raise ValueError("policy evaluation requires alert_context.anchor_regime")
    if alert_context.score is None:
        raise ValueError("policy evaluation requires alert_context.score")
    if alert_context.score_bucket is None or not str(alert_context.score_bucket).strip():
        raise ValueError("policy evaluation requires alert_context.score_bucket")

    return PolicyFeatureSnapshot(
        alert_id=alert_intent.alert_id,
        symbol=alert_intent.symbol,
        setup_family=str(alert_intent.setup_family).strip(),
        anchor_regime=str(alert_context.anchor_regime).strip(),
        score=float(alert_context.score),
        score_bucket=str(alert_context.score_bucket).strip(),
    )


def _normalized_filter_values(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("policy filter list values must be lists")
    normalized = []
    for item in value:
        text = _normalized_optional_str(item)
        if text is not None:
            normalized.append(text)
    return normalized


def _normalized_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
