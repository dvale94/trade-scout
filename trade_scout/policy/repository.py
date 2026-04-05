from __future__ import annotations

import sqlite3

from trade_scout.policy.models import (
    PolicyActivationRow,
    PolicyDecisionRow,
    PolicyDefinitionRow,
)


class PolicyRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def init_db(self) -> None:
        from trade_scout.execution import queue

        queue.init_db(self.db_path)

    def get_connection(self) -> sqlite3.Connection:
        from trade_scout.execution import queue

        return queue.get_connection(self.db_path)

    def insert_policy_definition(
        self,
        conn: sqlite3.Connection,
        row: PolicyDefinitionRow,
    ) -> PolicyDefinitionRow:
        return insert_policy_definition(conn, row)

    def get_policy_definition(
        self,
        conn: sqlite3.Connection,
        policy_name: str,
        policy_version: str,
    ) -> PolicyDefinitionRow | None:
        return get_policy_definition(conn, policy_name, policy_version)

    def insert_policy_decision(
        self,
        conn: sqlite3.Connection,
        row: PolicyDecisionRow,
    ) -> PolicyDecisionRow:
        return insert_policy_decision(conn, row)

    def get_policy_decision(
        self,
        conn: sqlite3.Connection,
        alert_id: str,
        policy_name: str,
        policy_version: str,
    ) -> PolicyDecisionRow | None:
        return get_policy_decision(conn, alert_id, policy_name, policy_version)

    def get_active_policy(
        self,
        conn: sqlite3.Connection,
        policy_name: str,
        at_ts_utc: str,
        *,
        mode: str = "SHADOW",
    ) -> PolicyDefinitionRow | None:
        return get_active_policy(conn, policy_name, at_ts_utc, mode=mode)

    def activate_policy(
        self,
        conn: sqlite3.Connection,
        row: PolicyActivationRow,
    ) -> PolicyActivationRow:
        return activate_policy(conn, row)


def insert_policy_definition(
    conn: sqlite3.Connection,
    row: PolicyDefinitionRow,
) -> PolicyDefinitionRow:
    existing = get_policy_definition(conn, row.policy_name, row.policy_version)
    if existing is not None:
        return existing

    try:
        conn.execute(
            """
            INSERT INTO policy_definitions (
                policy_id,
                policy_name,
                policy_version,
                policy_hash,
                status,
                config_json,
                description,
                created_ts_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.policy_id,
                row.policy_name,
                row.policy_version,
                row.policy_hash,
                row.status,
                row.config_json,
                row.description,
                row.created_ts_utc,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        existing = get_policy_definition(conn, row.policy_name, row.policy_version)
        if existing is not None:
            return existing
        raise

    inserted = get_policy_definition(conn, row.policy_name, row.policy_version)
    if inserted is None:
        raise RuntimeError("policy definition insert did not persist")
    return inserted


def get_policy_definition(
    conn: sqlite3.Connection,
    policy_name: str,
    policy_version: str,
) -> PolicyDefinitionRow | None:
    row = conn.execute(
        """
        SELECT *
        FROM policy_definitions
        WHERE policy_name = ? AND policy_version = ?
        """,
        (policy_name, policy_version),
    ).fetchone()
    return _policy_definition_row_from_db(row)


def insert_policy_decision(
    conn: sqlite3.Connection,
    row: PolicyDecisionRow,
) -> PolicyDecisionRow:
    existing = get_policy_decision(conn, row.alert_id, row.policy_name, row.policy_version)
    if existing is not None:
        return existing

    try:
        conn.execute(
            """
            INSERT INTO policy_decisions (
                policy_decision_id,
                alert_id,
                policy_id,
                policy_name,
                policy_version,
                policy_hash,
                mode,
                decision,
                decision_ts_utc,
                decision_reasons_json,
                criteria_results_json,
                feature_snapshot_json,
                execution_candidate,
                request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.policy_decision_id,
                row.alert_id,
                row.policy_id,
                row.policy_name,
                row.policy_version,
                row.policy_hash,
                row.mode,
                row.decision,
                row.decision_ts_utc,
                row.decision_reasons_json,
                row.criteria_results_json,
                row.feature_snapshot_json,
                row.execution_candidate,
                row.request_id,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        existing = get_policy_decision(conn, row.alert_id, row.policy_name, row.policy_version)
        if existing is not None:
            return existing
        raise

    inserted = get_policy_decision(conn, row.alert_id, row.policy_name, row.policy_version)
    if inserted is None:
        raise RuntimeError("policy decision insert did not persist")
    return inserted


def get_policy_decision(
    conn: sqlite3.Connection,
    alert_id: str,
    policy_name: str,
    policy_version: str,
) -> PolicyDecisionRow | None:
    row = conn.execute(
        """
        SELECT *
        FROM policy_decisions
        WHERE alert_id = ? AND policy_name = ? AND policy_version = ?
        """,
        (alert_id, policy_name, policy_version),
    ).fetchone()
    return _policy_decision_row_from_db(row)


def get_active_policy(
    conn: sqlite3.Connection,
    policy_name: str,
    at_ts_utc: str,
    *,
    mode: str = "SHADOW",
) -> PolicyDefinitionRow | None:
    row = conn.execute(
        """
        SELECT pd.*
        FROM policy_activation_history pah
        JOIN policy_definitions pd
            ON pd.policy_id = pah.policy_id
        WHERE pah.policy_name = ?
          AND pah.mode = ?
          AND pah.effective_from_ts_utc <= ?
          AND (pah.effective_to_ts_utc IS NULL OR pah.effective_to_ts_utc > ?)
        ORDER BY pah.effective_from_ts_utc DESC
        LIMIT 1
        """,
        (policy_name, mode, at_ts_utc, at_ts_utc),
    ).fetchone()
    return _policy_definition_row_from_db(row)


def activate_policy(
    conn: sqlite3.Connection,
    row: PolicyActivationRow,
) -> PolicyActivationRow:
    existing = _get_policy_activation_row(
        conn,
        policy_id=row.policy_id,
        policy_name=row.policy_name,
        policy_version=row.policy_version,
        mode=row.mode,
        effective_from_ts_utc=row.effective_from_ts_utc,
    )
    if existing is not None:
        return existing

    conn.execute(
        """
        UPDATE policy_activation_history
        SET effective_to_ts_utc = ?
        WHERE policy_name = ?
          AND mode = ?
          AND effective_from_ts_utc < ?
          AND (effective_to_ts_utc IS NULL OR effective_to_ts_utc > ?)
        """,
        (
            row.effective_from_ts_utc,
            row.policy_name,
            row.mode,
            row.effective_from_ts_utc,
            row.effective_from_ts_utc,
        ),
    )
    try:
        conn.execute(
            """
            INSERT INTO policy_activation_history (
                activation_id,
                policy_id,
                policy_name,
                policy_version,
                mode,
                effective_from_ts_utc,
                effective_to_ts_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.activation_id,
                row.policy_id,
                row.policy_name,
                row.policy_version,
                row.mode,
                row.effective_from_ts_utc,
                row.effective_to_ts_utc,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        existing = _get_policy_activation_row(
            conn,
            policy_id=row.policy_id,
            policy_name=row.policy_name,
            policy_version=row.policy_version,
            mode=row.mode,
            effective_from_ts_utc=row.effective_from_ts_utc,
        )
        if existing is not None:
            return existing
        raise

    inserted = _get_policy_activation_row(
        conn,
        policy_id=row.policy_id,
        policy_name=row.policy_name,
        policy_version=row.policy_version,
        mode=row.mode,
        effective_from_ts_utc=row.effective_from_ts_utc,
    )
    if inserted is None:
        raise RuntimeError("policy activation insert did not persist")
    return inserted


def _get_policy_activation_row(
    conn: sqlite3.Connection,
    *,
    policy_id: str,
    policy_name: str,
    policy_version: str,
    mode: str,
    effective_from_ts_utc: str,
) -> PolicyActivationRow | None:
    row = conn.execute(
        """
        SELECT *
        FROM policy_activation_history
        WHERE policy_id = ?
          AND policy_name = ?
          AND policy_version = ?
          AND mode = ?
          AND effective_from_ts_utc = ?
        LIMIT 1
        """,
        (
            policy_id,
            policy_name,
            policy_version,
            mode,
            effective_from_ts_utc,
        ),
    ).fetchone()
    return _policy_activation_row_from_db(row)


def _policy_definition_row_from_db(row: sqlite3.Row | None) -> PolicyDefinitionRow | None:
    if row is None:
        return None
    return PolicyDefinitionRow(
        policy_id=str(row["policy_id"]),
        policy_name=str(row["policy_name"]),
        policy_version=str(row["policy_version"]),
        policy_hash=str(row["policy_hash"]),
        status=str(row["status"]),
        config_json=str(row["config_json"]),
        description=_optional_str(row["description"]),
        created_ts_utc=str(row["created_ts_utc"]),
    )


def _policy_decision_row_from_db(row: sqlite3.Row | None) -> PolicyDecisionRow | None:
    if row is None:
        return None
    return PolicyDecisionRow(
        policy_decision_id=str(row["policy_decision_id"]),
        alert_id=str(row["alert_id"]),
        policy_id=str(row["policy_id"]),
        policy_name=str(row["policy_name"]),
        policy_version=str(row["policy_version"]),
        policy_hash=str(row["policy_hash"]),
        mode=str(row["mode"]),
        decision=str(row["decision"]),
        decision_ts_utc=str(row["decision_ts_utc"]),
        decision_reasons_json=str(row["decision_reasons_json"]),
        criteria_results_json=str(row["criteria_results_json"]),
        feature_snapshot_json=str(row["feature_snapshot_json"]),
        execution_candidate=int(row["execution_candidate"]),
        request_id=_optional_str(row["request_id"]),
    )


def _policy_activation_row_from_db(row: sqlite3.Row | None) -> PolicyActivationRow | None:
    if row is None:
        return None
    return PolicyActivationRow(
        activation_id=str(row["activation_id"]),
        policy_id=str(row["policy_id"]),
        policy_name=str(row["policy_name"]),
        policy_version=str(row["policy_version"]),
        mode=str(row["mode"]),
        effective_from_ts_utc=str(row["effective_from_ts_utc"]),
        effective_to_ts_utc=_optional_str(row["effective_to_ts_utc"]),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
