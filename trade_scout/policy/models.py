from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _require_nonempty(value: Any, field_name: str) -> None:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} is required")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True)
class PolicyConfig:
    policy_name: str
    policy_version: str
    policy_hash: str
    description: str
    filters: dict[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "policy_name",
            "policy_version",
            "policy_hash",
            "description",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "description": self.description,
            "filters": self.filters,
        }


@dataclass(frozen=True)
class PolicyFeatureSnapshot:
    alert_id: str
    symbol: str
    setup_family: str
    anchor_regime: str
    score: float
    score_bucket: str

    def __post_init__(self) -> None:
        for field_name in (
            "alert_id",
            "symbol",
            "setup_family",
            "anchor_regime",
            "score_bucket",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "setup_family": self.setup_family,
            "anchor_regime": self.anchor_regime,
            "score": self.score,
            "score_bucket": self.score_bucket,
        }


@dataclass(frozen=True)
class PolicyCriterionResult:
    criterion_name: str
    passed: bool
    actual_value: Any
    expected_value: Any

    def __post_init__(self) -> None:
        _require_nonempty(self.criterion_name, "criterion_name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_name": self.criterion_name,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
        }


@dataclass(frozen=True)
class PolicyDecision:
    policy_decision_id: str
    alert_id: str
    policy_name: str
    policy_version: str
    policy_hash: str
    mode: str
    decision: str
    decision_ts_utc: str
    reasons: list[str]
    criteria_results: list[PolicyCriterionResult]
    feature_snapshot: PolicyFeatureSnapshot
    execution_candidate: bool
    request_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "policy_decision_id",
            "alert_id",
            "policy_name",
            "policy_version",
            "policy_hash",
            "mode",
            "decision",
            "decision_ts_utc",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_row(self, policy_id: str) -> PolicyDecisionRow:
        _require_nonempty(policy_id, "policy_id")
        return PolicyDecisionRow(
            policy_decision_id=self.policy_decision_id,
            alert_id=self.alert_id,
            policy_id=policy_id,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            policy_hash=self.policy_hash,
            mode=self.mode,
            decision=self.decision,
            decision_ts_utc=self.decision_ts_utc,
            decision_reasons_json=_json_dumps(self.reasons),
            criteria_results_json=_json_dumps(
                [criterion_result.to_dict() for criterion_result in self.criteria_results]
            ),
            feature_snapshot_json=_json_dumps(self.feature_snapshot.to_dict()),
            execution_candidate=int(self.execution_candidate),
            request_id=self.request_id,
        )


@dataclass(frozen=True)
class PolicyDefinitionRow:
    policy_id: str
    policy_name: str
    policy_version: str
    policy_hash: str
    status: str
    config_json: str
    description: str | None
    created_ts_utc: str

    def __post_init__(self) -> None:
        for field_name in (
            "policy_id",
            "policy_name",
            "policy_version",
            "policy_hash",
            "status",
            "config_json",
            "created_ts_utc",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "status": self.status,
            "config_json": self.config_json,
            "description": self.description,
            "created_ts_utc": self.created_ts_utc,
        }


@dataclass(frozen=True)
class PolicyDecisionRow:
    policy_decision_id: str
    alert_id: str
    policy_id: str
    policy_name: str
    policy_version: str
    policy_hash: str
    mode: str
    decision: str
    decision_ts_utc: str
    decision_reasons_json: str
    criteria_results_json: str
    feature_snapshot_json: str
    execution_candidate: int
    request_id: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "policy_decision_id",
            "alert_id",
            "policy_id",
            "policy_name",
            "policy_version",
            "policy_hash",
            "mode",
            "decision",
            "decision_ts_utc",
            "decision_reasons_json",
            "criteria_results_json",
            "feature_snapshot_json",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_decision_id": self.policy_decision_id,
            "alert_id": self.alert_id,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "mode": self.mode,
            "decision": self.decision,
            "decision_ts_utc": self.decision_ts_utc,
            "decision_reasons_json": self.decision_reasons_json,
            "criteria_results_json": self.criteria_results_json,
            "feature_snapshot_json": self.feature_snapshot_json,
            "execution_candidate": self.execution_candidate,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class PolicyActivationRow:
    activation_id: str
    policy_id: str
    policy_name: str
    policy_version: str
    mode: str
    effective_from_ts_utc: str
    effective_to_ts_utc: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "activation_id",
            "policy_id",
            "policy_name",
            "policy_version",
            "mode",
            "effective_from_ts_utc",
        ):
            _require_nonempty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_id": self.activation_id,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "mode": self.mode,
            "effective_from_ts_utc": self.effective_from_ts_utc,
            "effective_to_ts_utc": self.effective_to_ts_utc,
        }
