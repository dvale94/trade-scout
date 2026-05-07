from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from research.metrics import summarize_trade_performance


_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_EXECUTION_DB_PATH = Path(os.getenv("EXECUTION_DB_PATH", "./execution.db"))


def _resolve_execution_db_path(execution_db_path: str | Path | None = None) -> Path:
    raw_path = _DEFAULT_EXECUTION_DB_PATH if execution_db_path is None else Path(execution_db_path)
    expanded_path = raw_path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return (_REPO_ROOT / expanded_path).resolve()


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
    feature_columns = {"symbol", "setup_family", "anchor_regime", "score", "score_bucket"}
    existing_columns = set(decisions.columns)
    if feature_column in existing_columns and feature_columns.intersection(existing_columns):
        feature_df = decisions.copy()
    else:
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


def summarize_policy_variant_performance(
    trades_enriched: pd.DataFrame,
    policy_decisions: pd.DataFrame,
    *,
    experiment_tag: str | None = None,
    policy_version: str | None = None,
    min_trades: int = 1,
) -> pd.DataFrame:
    """
    Summarize shadow-policy trade performance by policy variant and decision.

    Example notebook usage:
    summary = summarize_policy_variant_performance(
        trades_enriched,
        policy_decisions,
        experiment_tag="vacation_baseline_v1",
    )
    """
    joined = _join_policy_trades(
        trades_enriched,
        policy_decisions,
        experiment_tag=experiment_tag,
        policy_version=policy_version,
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "policy_version",
                "mode",
                "decision",
                "trade_count",
                "share_of_policy_trades",
                "win_rate",
                "expectancy_R",
                "median_R",
                "total_R",
                "avg_mfe_R",
                "avg_mae_R",
                "profit_factor",
                "closed_trade_count",
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_key, group_df in joined.groupby(
        ["policy_name", "policy_version", "mode", "decision"],
        observed=False,
        dropna=False,
    ):
        policy_name, grouped_policy_version, mode, decision = group_key
        trade_count = int(len(group_df))
        if trade_count < min_trades:
            continue

        performance = summarize_trade_performance(group_df)
        closed_trade_mask = _closed_trade_mask(group_df)
        closed_trade_count = int(closed_trade_mask.sum())
        pnl = _numeric_series(group_df, "pnl_R").dropna()
        mfe = _numeric_series(group_df, "mfe_R").dropna()
        mae = _numeric_series(group_df, "mae_R").dropna()

        policy_group_df = joined.loc[
            joined["policy_name"].fillna("UNKNOWN").astype(str).eq(str(policy_name))
            & joined["policy_version"].fillna("UNKNOWN").astype(str).eq(str(grouped_policy_version))
            & joined["mode"].fillna("UNKNOWN").astype(str).eq(str(mode))
        ]
        policy_total_trade_count = int(len(policy_group_df))

        rows.append(
            {
                "policy_name": policy_name,
                "policy_version": grouped_policy_version,
                "mode": mode,
                "decision": decision,
                "trade_count": trade_count,
                "share_of_policy_trades": (
                    float(trade_count / policy_total_trade_count) if policy_total_trade_count else None
                ),
                "win_rate": performance["win_rate"],
                "expectancy_R": performance["expectancy_R"],
                "median_R": float(pnl.median()) if not pnl.empty else None,
                "total_R": float(pnl.sum()) if not pnl.empty else None,
                "avg_mfe_R": float(mfe.mean()) if not mfe.empty else None,
                "avg_mae_R": float(mae.mean()) if not mae.empty else None,
                "profit_factor": performance["profit_factor"],
                "closed_trade_count": closed_trade_count,
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["policy_name", "policy_version", "mode", "decision"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def compare_policy_variant_allow_block(
    trades_enriched: pd.DataFrame,
    policy_decisions: pd.DataFrame,
    *,
    experiment_tag: str | None = None,
    policy_version: str | None = None,
    min_trades: int = 1,
) -> pd.DataFrame:
    """
    Compare ALLOW vs BLOCK behavior for each policy variant.
    """
    joined = _join_policy_trades(
        trades_enriched,
        policy_decisions,
        experiment_tag=experiment_tag,
        policy_version=policy_version,
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "policy_version",
                "mode",
                "allow_count",
                "block_count",
                "block_rate",
                "allow_expectancy_R",
                "block_expectancy_R",
                "expectancy_delta_R",
                "blocked_vs_allowed_delta_R",
                "winners_removed_pct",
                "losers_removed_pct",
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_key, group_df in joined.groupby(
        ["policy_name", "policy_version", "mode"],
        observed=False,
        dropna=False,
    ):
        policy_name, grouped_policy_version, mode = group_key
        total_count = int(len(group_df))
        if total_count < min_trades:
            continue

        allow_df = group_df.loc[group_df["decision"].fillna("").astype(str).eq("ALLOW")].copy()
        block_df = group_df.loc[group_df["decision"].fillna("").astype(str).eq("BLOCK")].copy()
        overall_pnl = _numeric_series(group_df, "pnl_R").dropna()
        allow_pnl = _numeric_series(allow_df, "pnl_R").dropna()
        block_pnl = _numeric_series(block_df, "pnl_R").dropna()

        overall_expectancy = float(overall_pnl.mean()) if not overall_pnl.empty else None
        allow_expectancy = float(allow_pnl.mean()) if not allow_pnl.empty else None
        block_expectancy = float(block_pnl.mean()) if not block_pnl.empty else None

        overall_winners = int((overall_pnl > 0).sum()) if not overall_pnl.empty else 0
        overall_losers = int((overall_pnl < 0).sum()) if not overall_pnl.empty else 0
        blocked_winners = int((block_pnl > 0).sum()) if not block_pnl.empty else 0
        blocked_losers = int((block_pnl < 0).sum()) if not block_pnl.empty else 0

        rows.append(
            {
                "policy_name": policy_name,
                "policy_version": grouped_policy_version,
                "mode": mode,
                "allow_count": int(len(allow_df)),
                "block_count": int(len(block_df)),
                "block_rate": (float(len(block_df) / total_count) if total_count else None),
                "allow_expectancy_R": allow_expectancy,
                "block_expectancy_R": block_expectancy,
                "expectancy_delta_R": (
                    float(allow_expectancy - overall_expectancy)
                    if allow_expectancy is not None and overall_expectancy is not None
                    else None
                ),
                "blocked_vs_allowed_delta_R": (
                    float(allow_expectancy - block_expectancy)
                    if allow_expectancy is not None and block_expectancy is not None
                    else None
                ),
                "winners_removed_pct": (
                    float(blocked_winners / overall_winners) if overall_winners > 0 else None
                ),
                "losers_removed_pct": (
                    float(blocked_losers / overall_losers) if overall_losers > 0 else None
                ),
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["policy_name", "policy_version", "mode"], ascending=[True, True, True])
        .reset_index(drop=True)
    )


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


def _join_policy_trades(
    trades_enriched: pd.DataFrame,
    policy_decisions: pd.DataFrame,
    *,
    experiment_tag: str | None,
    policy_version: str | None,
) -> pd.DataFrame:
    if trades_enriched.empty or policy_decisions.empty:
        return pd.DataFrame()

    trades = trades_enriched.copy()
    decisions = policy_decisions.copy()
    if experiment_tag is not None and "experiment_tag" in trades.columns:
        trades = trades.loc[trades["experiment_tag"].astype(str).eq(experiment_tag)].copy()
    if policy_version is not None and "policy_version" in decisions.columns:
        decisions = decisions.loc[decisions["policy_version"].astype(str).eq(policy_version)].copy()
    if trades.empty or decisions.empty or "alert_id" not in trades.columns or "alert_id" not in decisions.columns:
        return pd.DataFrame()

    joined = decisions.merge(
        trades,
        on="alert_id",
        how="inner",
        suffixes=("_policy", ""),
    )
    return joined.reset_index(drop=True)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _closed_trade_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="bool")
    if "closed_trade" in df.columns:
        raw = df["closed_trade"]
        if raw.dtype == bool:
            return raw.fillna(False)
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0).astype(int).ne(0)
        text = raw.fillna("").astype(str).str.strip().str.upper()
        return text.isin({"TRUE", "T", "YES", "Y", "1"})
    return _numeric_series(df, "pnl_R").notna()


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
