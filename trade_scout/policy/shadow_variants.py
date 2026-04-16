from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from trade_scout.execution.queue import utc_now_iso
from trade_scout.policy.models import PolicyActivationRow, PolicyDefinitionRow
from trade_scout.policy.repository import PolicyRepository


@dataclass(frozen=True)
class ShadowPolicyVariant:
    policy_name: str
    policy_version: str
    policy_hash: str
    description: str
    filters: dict[str, object]


VACATION_SHADOW_POLICY_VARIANTS: dict[str, ShadowPolicyVariant] = {
    "shadow_score_gte_8_0": ShadowPolicyVariant(
        policy_name="shadow_score_gte_8_0",
        policy_version="vacation_baseline_v1",
        policy_hash="shadow_score_gte_8_0:v1",
        description="Shadow comparison: allow alerts with score >= 8.0",
        filters={"min_score": 8.0},
    ),
    "shadow_score_gte_7_5": ShadowPolicyVariant(
        policy_name="shadow_score_gte_7_5",
        policy_version="vacation_baseline_v1",
        policy_hash="shadow_score_gte_7_5:v1",
        description="Shadow comparison: allow alerts with score >= 7.5",
        filters={"min_score": 7.5},
    ),
    "shadow_block_score_lt_6_5": ShadowPolicyVariant(
        policy_name="shadow_block_score_lt_6_5",
        policy_version="vacation_baseline_v1",
        policy_hash="shadow_block_score_lt_6_5:v1",
        description="Shadow comparison: block alerts with score < 6.5",
        filters={"min_score": 6.5},
    ),
    "shadow_score_gte_8_0_bear": ShadowPolicyVariant(
        policy_name="shadow_score_gte_8_0_bear",
        policy_version="vacation_baseline_v1",
        policy_hash="shadow_score_gte_8_0_bear:v1",
        description="Shadow comparison: allow alerts with score >= 8.0 in bear anchor regime",
        filters={"min_score": 8.0, "allowed_anchor_regimes": ["bear"]},
    ),
}


def seed_shadow_policy_variants(
    repository: PolicyRepository,
    policy_names: Iterable[str],
    *,
    effective_from_ts_utc: str = "1970-01-01T00:00:00+00:00",
) -> None:
    selected = [VACATION_SHADOW_POLICY_VARIANTS[name] for name in policy_names if name in VACATION_SHADOW_POLICY_VARIANTS]
    if not selected:
        return

    conn = repository.get_connection()
    try:
        for variant in selected:
            definition = repository.insert_policy_definition(
                conn,
                PolicyDefinitionRow(
                    policy_id=f"{variant.policy_name}:{variant.policy_version}",
                    policy_name=variant.policy_name,
                    policy_version=variant.policy_version,
                    policy_hash=variant.policy_hash,
                    status="SHADOW_ONLY",
                    config_json=json.dumps({"filters": variant.filters}, sort_keys=True),
                    description=variant.description,
                    created_ts_utc=utc_now_iso(),
                ),
            )
            repository.activate_policy(
                conn,
                PolicyActivationRow(
                    activation_id=f"{variant.policy_name}:{variant.policy_version}:shadow",
                    policy_id=definition.policy_id,
                    policy_name=definition.policy_name,
                    policy_version=definition.policy_version,
                    mode="SHADOW",
                    effective_from_ts_utc=effective_from_ts_utc,
                    effective_to_ts_utc=None,
                ),
            )
    finally:
        conn.close()
