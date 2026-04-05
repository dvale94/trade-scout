from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_EXECUTION_DB_PATH = Path(os.getenv("EXECUTION_DB_PATH", "./execution.db"))


def _resolve_execution_db_path(execution_db_path: str | Path | None = None) -> Path:
    if execution_db_path is None:
        return _DEFAULT_EXECUTION_DB_PATH
    return Path(execution_db_path)


def connect_policy_db(
    execution_db_path: str | Path | None = None,
    *,
    read_only: bool = True,
) -> sqlite3.Connection:
    db_path = _resolve_execution_db_path(execution_db_path)
    uri = f"file:{db_path}?mode=ro" if read_only else str(db_path)
    conn = sqlite3.connect(uri, uri=read_only)
    conn.row_factory = sqlite3.Row
    return conn


def load_policy_decisions(
    execution_db_path: str | Path | None = None,
    *,
    policy_name: str | None = None,
    policy_version: str | None = None,
    mode: str | None = None,
    decision: str | None = None,
) -> pd.DataFrame:
    conn = connect_policy_db(execution_db_path, read_only=True)
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if policy_name:
            clauses.append("policy_name = ?")
            params.append(policy_name)
        if policy_version:
            clauses.append("policy_version = ?")
            params.append(policy_version)
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if decision:
            clauses.append("decision = ?")
            params.append(decision)

        sql = """
            SELECT
                alert_id,
                policy_name,
                policy_version,
                policy_hash,
                mode,
                decision,
                decision_ts_utc,
                execution_candidate,
                request_id,
                decision_reasons_json,
                criteria_results_json,
                feature_snapshot_json
            FROM policy_decisions
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY decision_ts_utc DESC, alert_id"
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def flatten_policy_feature_snapshot(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()

    out = decisions.copy()
    features = out["feature_snapshot_json"].map(_parse_json_object)
    feature_df = pd.DataFrame(
        [
            {
                "symbol": feature.get("symbol"),
                "setup_family": feature.get("setup_family"),
                "anchor_regime": feature.get("anchor_regime"),
                "score": feature.get("score"),
                "score_bucket": feature.get("score_bucket"),
            }
            for feature in features
        ]
    )
    return pd.concat([out.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def flatten_policy_criteria_results(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "policy_name",
                "policy_version",
                "criterion_name",
                "passed",
                "actual_value",
                "expected_value",
            ]
        )

    rows: list[dict[str, Any]] = []
    for decision_row in decisions.to_dict(orient="records"):
        criteria_results = _parse_json_list(decision_row.get("criteria_results_json"))
        for criterion in criteria_results:
            if not isinstance(criterion, dict):
                continue
            rows.append(
                {
                    "alert_id": decision_row.get("alert_id"),
                    "policy_name": decision_row.get("policy_name"),
                    "policy_version": decision_row.get("policy_version"),
                    "criterion_name": criterion.get("criterion_name"),
                    "passed": criterion.get("passed"),
                    "actual_value": criterion.get("actual_value"),
                    "expected_value": criterion.get("expected_value"),
                }
            )
    return pd.DataFrame(rows)


def flatten_policy_decision_reasons(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=["alert_id", "policy_name", "policy_version", "reason"])

    rows: list[dict[str, Any]] = []
    for decision_row in decisions.to_dict(orient="records"):
        reasons = _parse_json_list(decision_row.get("decision_reasons_json"))
        for reason in reasons:
            rows.append(
                {
                    "alert_id": decision_row.get("alert_id"),
                    "policy_name": decision_row.get("policy_name"),
                    "policy_version": decision_row.get("policy_version"),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def counts_by_decision(decisions: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_frame(decisions, "decision")


def counts_by_policy_version(decisions: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_frame(decisions, "policy_version")


def counts_by_mode(decisions: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_frame(decisions, "mode")


def counts_by_feature(decisions: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    feature_df = flatten_policy_feature_snapshot(decisions)
    if feature_df.empty or feature_column not in feature_df.columns:
        return pd.DataFrame(columns=[feature_column, "count"])
    return (
        feature_df[feature_column]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis(feature_column)
        .reset_index(name="count")
    )


def top_block_reasons(decisions: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    blocked = decisions[decisions["decision"] == "BLOCK"].copy() if "decision" in decisions.columns else pd.DataFrame()
    reasons = flatten_policy_decision_reasons(blocked)
    if reasons.empty:
        return pd.DataFrame(columns=["reason", "count"])
    return (
        reasons["reason"]
        .value_counts()
        .head(limit)
        .rename_axis("reason")
        .reset_index(name="count")
    )


def top_failed_criteria(decisions: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    criteria = flatten_policy_criteria_results(decisions)
    if criteria.empty or "passed" not in criteria.columns:
        return pd.DataFrame(columns=["criterion_name", "count"])
    failed = criteria[criteria["passed"] == False].copy()
    if failed.empty:
        return pd.DataFrame(columns=["criterion_name", "count"])
    return (
        failed["criterion_name"]
        .value_counts()
        .head(limit)
        .rename_axis("criterion_name")
        .reset_index(name="count")
    )


def join_ready_policy_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "policy_name",
                "policy_version",
                "policy_hash",
                "mode",
                "decision",
                "decision_ts_utc",
                "execution_candidate",
                "request_id",
                "symbol",
                "setup_family",
                "anchor_regime",
                "score",
                "score_bucket",
            ]
        )

    feature_df = flatten_policy_feature_snapshot(decisions)
    cols = [
        "alert_id",
        "policy_name",
        "policy_version",
        "policy_hash",
        "mode",
        "decision",
        "decision_ts_utc",
        "execution_candidate",
        "request_id",
        "symbol",
        "setup_family",
        "anchor_regime",
        "score",
        "score_bucket",
    ]
    available = [col for col in cols if col in feature_df.columns]
    return (
        feature_df[available]
        .sort_values(["alert_id", "decision_ts_utc"], ascending=[True, False], na_position="last")
        .drop_duplicates(subset=["alert_id"], keep="first")
        .reset_index(drop=True)
    )


def summarize_policy_shadow(decisions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feature_df = flatten_policy_feature_snapshot(decisions)
    return {
        "by_decision": counts_by_decision(decisions),
        "by_policy_version": counts_by_policy_version(decisions),
        "by_mode": counts_by_mode(decisions),
        "by_symbol": counts_by_feature(feature_df, "symbol"),
        "by_setup_family": counts_by_feature(feature_df, "setup_family"),
        "by_anchor_regime": counts_by_feature(feature_df, "anchor_regime"),
        "top_block_reasons": top_block_reasons(decisions),
        "top_failed_criteria": top_failed_criteria(decisions),
    }


def _value_counts_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    return (
        df[column]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )


def _parse_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []
