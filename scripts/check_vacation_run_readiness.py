from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trade_scout import config  # noqa: E402
from trade_scout.policy.shadow_variants import VACATION_SHADOW_POLICY_VARIANTS  # noqa: E402


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    severity: str
    count: int
    detail: str = ""

    @property
    def status(self) -> str:
        if self.count <= 0:
            return "PASS"
        return self.severity


def connect_readonly(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def configured_variant_names() -> list[str]:
    return list(config.SHADOW_POLICY_VARIANT_NAMES)


def build_readiness_checks(
    conn: sqlite3.Connection | None,
    *,
    variant_names: list[str],
    now_ts_utc: str | None = None,
) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    known_names = set(VACATION_SHADOW_POLICY_VARIANTS)
    unknown_names = [name for name in variant_names if name not in known_names]

    checks.append(
        ReadinessCheck(
            name="experiment_tag_unset",
            severity="WARN",
            count=0 if config.EXPERIMENT_TAG else 1,
            detail="set EXPERIMENT_TAG=vacation_baseline_v1 for this collection run",
        )
    )
    checks.append(
        ReadinessCheck(
            name="shadow_policy_variants_unset",
            severity="WARN",
            count=0 if variant_names else 1,
            detail="set SHADOW_POLICY_VARIANT_NAMES to the vacation variant list",
        )
    )
    checks.append(
        ReadinessCheck(
            name="unknown_shadow_policy_variant_names",
            severity="WARN",
            count=len(unknown_names),
            detail=",".join(unknown_names),
        )
    )

    if conn is None:
        checks.append(
            ReadinessCheck(
                name="execution_db_unavailable",
                severity="WARN",
                count=1,
                detail="run after the execution DB exists; this script does not initialize schemas",
            )
        )
        return checks

    required_tables = {"policy_definitions", "policy_activation_history"}
    missing_tables = [table for table in sorted(required_tables) if not table_exists(conn, table)]
    checks.append(
        ReadinessCheck(
            name="policy_tables_missing",
            severity="WARN",
            count=len(missing_tables),
            detail=",".join(missing_tables),
        )
    )
    if missing_tables:
        return checks

    selected_names = [name for name in variant_names if name in known_names]
    missing_definitions = _missing_policy_definitions(conn, selected_names)
    missing_activations = _missing_active_shadow_activations(
        conn,
        selected_names,
        now_ts_utc=now_ts_utc or datetime.now(timezone.utc).isoformat(),
    )
    checks.append(
        ReadinessCheck(
            name="configured_policy_definitions_missing",
            severity="WARN",
            count=len(missing_definitions),
            detail=",".join(missing_definitions),
        )
    )
    checks.append(
        ReadinessCheck(
            name="configured_shadow_activations_missing",
            severity="WARN",
            count=len(missing_activations),
            detail=",".join(missing_activations),
        )
    )
    return checks


def _missing_policy_definitions(conn: sqlite3.Connection, policy_names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in policy_names:
        variant = VACATION_SHADOW_POLICY_VARIANTS[name]
        row = conn.execute(
            """
            SELECT 1
            FROM policy_definitions
            WHERE policy_name = ? AND policy_version = ?
            LIMIT 1
            """,
            (variant.policy_name, variant.policy_version),
        ).fetchone()
        if row is None:
            missing.append(name)
    return missing


def _missing_active_shadow_activations(
    conn: sqlite3.Connection,
    policy_names: list[str],
    *,
    now_ts_utc: str,
) -> list[str]:
    missing: list[str] = []
    for name in policy_names:
        variant = VACATION_SHADOW_POLICY_VARIANTS[name]
        row = conn.execute(
            """
            SELECT 1
            FROM policy_activation_history
            WHERE policy_name = ?
              AND policy_version = ?
              AND mode = 'SHADOW'
              AND effective_from_ts_utc <= ?
              AND (effective_to_ts_utc IS NULL OR effective_to_ts_utc > ?)
            LIMIT 1
            """,
            (variant.policy_name, variant.policy_version, now_ts_utc, now_ts_utc),
        ).fetchone()
        if row is None:
            missing.append(name)
    return missing


def print_readiness_report(*, db_path: str, checks: list[ReadinessCheck]) -> None:
    print(f"execution_db_path={Path(db_path).expanduser().resolve()}")
    print(f"EXPERIMENT_TAG={config.EXPERIMENT_TAG or '<unset>'}")
    print(f"SHADOW_POLICY_VARIANT_NAMES={','.join(configured_variant_names()) or '<none>'}")
    for check in checks:
        suffix = f" detail={check.detail}" if check.detail and check.count > 0 else ""
        print(f"[{check.status}] {check.name} = {check.count}{suffix}")
    print(f"OVERALL: {_overall_status(checks)}")


def _overall_status(checks: list[ReadinessCheck]) -> str:
    if any(check.status == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check vacation baseline collection readiness.")
    parser.add_argument("--db-path", default=config.EXECUTION_DB_PATH)
    args = parser.parse_args(argv)

    conn: sqlite3.Connection | None = None
    try:
        conn = connect_readonly(args.db_path)
    except sqlite3.Error:
        conn = None

    try:
        checks = build_readiness_checks(conn, variant_names=configured_variant_names())
    finally:
        if conn is not None:
            conn.close()

    print_readiness_report(db_path=args.db_path, checks=checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
