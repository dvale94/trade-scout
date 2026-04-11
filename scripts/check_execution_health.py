from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.execution_analysis import (  # noqa: E402
    _resolve_execution_db_path,
    build_execution_fill_summary,
    load_execution_frames,
)


QTY_TOLERANCE = 1e-8
DEFAULT_RESIDUAL_DUST_NOTIONAL = 1.0
OLD_REQUEST_MINUTES = 30
STALE_PENDING_ACTION_MINUTES = 15
NONTERMINAL_REQUEST_STATUSES = {"PENDING", "CLAIMED", "VALIDATED"}
TERMINAL_PRE_PLAN_REQUEST_STATUSES = {"REJECTED", "EXPIRED", "FAILED"}
OPEN_LIKE_POSITION_STATUSES = {"OPEN", "PARTIAL", "CLOSING"}
KNOWN_TRANSIENT_NO_STOP_ACTIONS = {
    "PLACE_INITIAL_STOP",
    "CANCEL_STOP_FOR_TP1",
    "SUBMIT_TP1",
    "WAIT_TP1_FILL",
    "REARM_STOP_AFTER_TP1",
    "CANCEL_STOP_FOR_TP2",
    "SUBMIT_TP2",
}


@dataclass(frozen=True)
class HealthCheck:
    name: str
    severity: str
    count: int
    detail: str = ""

    @property
    def status(self) -> str:
        if self.severity == "INFO":
            return "INFO" if self.count > 0 else "PASS"
        if self.count <= 0:
            return "PASS"
        return self.severity


def _to_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _request_age_minutes(execution_requests: pd.DataFrame) -> pd.Series:
    requested_ts = pd.to_datetime(execution_requests.get("requested_ts_utc"), errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return (now - requested_ts).dt.total_seconds() / 60.0


def residual_dust_notional_threshold() -> float:
    """Read the execution residual dust notional threshold without mutating runtime state."""
    raw = os.getenv("EXECUTION_RESIDUAL_DUST_NOTIONAL", str(DEFAULT_RESIDUAL_DUST_NOTIONAL)).strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return DEFAULT_RESIDUAL_DUST_NOTIONAL


def _position_action_age_minutes(paper_positions: pd.DataFrame) -> pd.Series:
    # There is no dedicated action-updated timestamp today, so updated_ts_utc is
    # the best read-only proxy for how long the pending action has been present.
    updated_ts = pd.to_datetime(paper_positions.get("updated_ts_utc"), errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return (now - updated_ts).dt.total_seconds() / 60.0


def _format_counts(df: pd.DataFrame, column: str, *, label_map: dict[str, str] | None = None) -> str:
    if df.empty or column not in df.columns:
        return "none"
    counts = df[column].fillna("<NULL>").astype(str).value_counts(dropna=False)
    if label_map:
        return ", ".join(f"{label_map.get(key, key)}={int(value)}" for key, value in counts.items())
    return ", ".join(f"{key}={int(value)}" for key, value in counts.items())


def _reference_price_estimate(df: pd.DataFrame) -> pd.Series:
    # The runtime dust policy estimates notional from the best local execution
    # summary price. This mirrors that analysis-only heuristic:
    # max(entry_avg_fill_px, stop_price, stop_limit_price).
    candidates = [
        _to_numeric(df, "entry_avg_fill_px"),
        _to_numeric(df, "stop_price"),
        _to_numeric(df, "stop_limit_price"),
    ]
    if not candidates:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.concat(candidates, axis=1).max(axis=1).fillna(0.0)


def build_health_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], bool]:
    execution_requests = frames["execution_requests"].copy()
    paper_trade_plans = frames["paper_trade_plans"].copy()
    paper_positions = frames["paper_positions"].copy()
    paper_orders = frames["paper_orders"].copy()
    paper_fills = frames["paper_fills"].copy()

    stop_filled_qty_available = "stop_filled_qty" in paper_positions.columns
    fill_summary = build_execution_fill_summary(paper_fills)
    stop_from_fills = (
        fill_summary[["plan_id", "stop_fill_qty"]].copy()
        if "stop_fill_qty" in fill_summary.columns
        else pd.DataFrame(columns=["plan_id", "stop_fill_qty"])
    )

    positions_with_qty = paper_positions.merge(
        stop_from_fills,
        on="plan_id",
        how="left",
        validate="1:1",
    )
    positions_with_qty["stop_qty_for_invariant"] = (
        _to_numeric(positions_with_qty, "stop_filled_qty")
        if stop_filled_qty_available
        else _to_numeric(positions_with_qty, "stop_fill_qty")
    )
    positions_with_qty["expected_remaining_qty"] = (
        _to_numeric(positions_with_qty, "entry_filled_qty")
        - _to_numeric(positions_with_qty, "tp1_filled_qty")
        - _to_numeric(positions_with_qty, "tp2_filled_qty")
        - _to_numeric(positions_with_qty, "stop_qty_for_invariant")
    ).clip(lower=0.0)
    positions_with_qty["remaining_qty_numeric"] = _to_numeric(positions_with_qty, "remaining_qty")
    positions_with_qty["remaining_qty_delta"] = (
        positions_with_qty["remaining_qty_numeric"] - positions_with_qty["expected_remaining_qty"]
    ).abs()
    positions_with_qty["reference_price_estimate"] = _reference_price_estimate(positions_with_qty)
    positions_with_qty["remaining_notional_estimate"] = (
        positions_with_qty["remaining_qty_numeric"].abs() * positions_with_qty["reference_price_estimate"]
    )
    positions_with_qty["expected_remaining_notional_estimate"] = (
        positions_with_qty["expected_remaining_qty"].abs() * positions_with_qty["reference_price_estimate"]
    )
    dust_threshold = residual_dust_notional_threshold()

    if execution_requests.empty:
        requests_without_plans = execution_requests.copy()
    else:
        requests_without_plans = execution_requests.loc[
            ~execution_requests["request_id"].astype(str).isin(paper_trade_plans["request_id"].astype(str))
        ].copy()
    if not requests_without_plans.empty:
        requests_without_plans["age_minutes"] = _request_age_minutes(requests_without_plans)
    expected_terminal_requests_without_plans = requests_without_plans.loc[
        requests_without_plans["status"].astype(str).isin(TERMINAL_PRE_PLAN_REQUEST_STATUSES)
    ].copy()
    unexpected_requests_without_plans = requests_without_plans.loc[
        (
            requests_without_plans["status"].astype(str).eq("SUBMITTED")
            | (
                requests_without_plans["status"].astype(str).isin(NONTERMINAL_REQUEST_STATUSES)
                & (requests_without_plans.get("age_minutes", pd.Series(dtype="float64")) > OLD_REQUEST_MINUTES)
            )
        )
    ].copy()

    if paper_trade_plans.empty or paper_positions.empty:
        plans_without_positions = paper_trade_plans.copy()
    else:
        plans_without_positions = paper_trade_plans.loc[
            ~paper_trade_plans["plan_id"].astype(str).isin(paper_positions["plan_id"].astype(str))
        ].copy()

    active_stop_missing_order = (
        paper_positions.loc[
            paper_positions["active_stop_paper_order_id"].notna()
            & ~paper_positions["active_stop_paper_order_id"].astype(str).isin(paper_orders["paper_order_id"].astype(str))
        ].copy()
        if not paper_positions.empty and "active_stop_paper_order_id" in paper_positions.columns
        else _empty_like(paper_positions)
    )

    pending_action_positions = paper_positions.loc[
        ~paper_positions["pending_action_status"].fillna("NONE").astype(str).eq("NONE")
    ].copy()
    if not pending_action_positions.empty:
        pending_action_positions["action_age_minutes"] = _position_action_age_minutes(pending_action_positions)
    stale_pending_action_positions = pending_action_positions.loc[
        pending_action_positions.get("action_age_minutes", pd.Series(dtype="float64")) > STALE_PENDING_ACTION_MINUTES
    ].copy()

    # Missing active stop can be normal during cancel/submit/rearm windows. Warn
    # only when the position has remaining live qty, no active stop/TP2, and the
    # pending action does not clearly explain the temporary gap.
    suspicious_open_missing_active_stop = (
        paper_positions.loc[
            paper_positions["position_status"].astype(str).isin(OPEN_LIKE_POSITION_STATUSES)
            & (_to_numeric(paper_positions, "remaining_qty") > QTY_TOLERANCE)
            & paper_positions["active_stop_paper_order_id"].isna()
            & paper_positions["active_tp2_paper_order_id"].isna()
            & (
                paper_positions["pending_action_status"].fillna("NONE").astype(str).isin({"NONE", "WAIT_TP2_FILL"})
                | ~paper_positions["pending_action_status"].fillna("NONE").astype(str).isin(KNOWN_TRANSIENT_NO_STOP_ACTIONS)
            )
        ].copy()
        if not paper_positions.empty
        and {
            "position_status",
            "pending_action_status",
            "active_stop_paper_order_id",
            "active_tp2_paper_order_id",
        }.issubset(paper_positions.columns)
        else _empty_like(paper_positions)
    )

    nonterminal_requests = execution_requests.loc[
        execution_requests["status"].astype(str).isin(NONTERMINAL_REQUEST_STATUSES)
    ].copy()
    if not nonterminal_requests.empty:
        nonterminal_requests["age_minutes"] = _request_age_minutes(nonterminal_requests)
    old_nonterminal_requests = nonterminal_requests.loc[
        nonterminal_requests.get("age_minutes", pd.Series(dtype="float64")) > OLD_REQUEST_MINUTES
    ].copy()

    closed_leftover = positions_with_qty.loc[
        positions_with_qty["position_status"].astype(str).eq("CLOSED")
        & (positions_with_qty["remaining_qty_numeric"].abs() > QTY_TOLERANCE)
    ].copy()
    acceptable_closed_dust_residuals = closed_leftover.loc[
        closed_leftover["remaining_notional_estimate"] <= dust_threshold
    ].copy()
    unexpected_closed_leftover = closed_leftover.loc[
        closed_leftover["remaining_notional_estimate"] > dust_threshold
    ].copy()

    invariant_mismatches = positions_with_qty.loc[
        positions_with_qty["remaining_qty_delta"] > QTY_TOLERANCE
    ].copy()
    acceptable_dust_invariant_mismatches = invariant_mismatches.loc[
        invariant_mismatches["expected_remaining_notional_estimate"] <= dust_threshold
    ].copy()
    unexpected_invariant_failures = invariant_mismatches.loc[
        invariant_mismatches["expected_remaining_notional_estimate"] > dust_threshold
    ].copy()

    health_frames = {
        "error_positions": paper_positions.loc[paper_positions["position_status"].astype(str).eq("ERROR")].copy(),
        "unexpected_closed_positions_with_leftover_qty": unexpected_closed_leftover,
        "acceptable_closed_dust_residuals": acceptable_closed_dust_residuals,
        "unexpected_remaining_qty_invariant_failures": unexpected_invariant_failures,
        "acceptable_dust_invariant_mismatches": acceptable_dust_invariant_mismatches,
        "plans_without_positions": plans_without_positions,
        "active_stop_pointer_missing_order": active_stop_missing_order,
        "expected_terminal_requests_without_plans": expected_terminal_requests_without_plans,
        "unexpected_requests_without_plans": unexpected_requests_without_plans,
        "pending_action_positions": pending_action_positions,
        "stale_pending_action_positions": stale_pending_action_positions,
        "suspicious_open_positions_missing_active_stop": suspicious_open_missing_active_stop,
        "old_nonterminal_requests": old_nonterminal_requests,
    }
    return health_frames, stop_filled_qty_available


def build_health_checks(health_frames: dict[str, pd.DataFrame]) -> list[HealthCheck]:
    fail_checks = [
        "error_positions",
        "unexpected_closed_positions_with_leftover_qty",
        "unexpected_remaining_qty_invariant_failures",
        "plans_without_positions",
        "active_stop_pointer_missing_order",
    ]
    warn_checks = [
        "unexpected_requests_without_plans",
        "stale_pending_action_positions",
        "suspicious_open_positions_missing_active_stop",
        "old_nonterminal_requests",
    ]
    info_checks = [
        "acceptable_closed_dust_residuals",
        "acceptable_dust_invariant_mismatches",
    ]

    checks: list[HealthCheck] = []
    for name in fail_checks:
        checks.append(HealthCheck(name=name, severity="FAIL", count=int(len(health_frames[name]))))
    for name in warn_checks:
        checks.append(HealthCheck(name=name, severity="WARN", count=int(len(health_frames[name]))))
    for name in info_checks:
        checks.append(HealthCheck(name=name, severity="INFO", count=int(len(health_frames[name]))))
    return checks


def print_summary(
    *,
    db_path: Path,
    stop_filled_qty_available: bool,
    frames: dict[str, pd.DataFrame],
    checks: list[HealthCheck],
) -> int:
    print(f"Execution DB: {db_path}")
    print(f"paper_positions.stop_filled_qty present: {stop_filled_qty_available}")
    print(f"residual_dust_notional_threshold: {residual_dust_notional_threshold():.2f}")
    if not stop_filled_qty_available:
        print("Quantity invariant fallback: using STOP fill quantity aggregated from paper_fills")
    print()

    print(
        "request_status_counts: "
        f"{_format_counts(frames['execution_requests'], 'status')}"
    )
    print(
        "position_status_counts: "
        f"{_format_counts(frames['paper_positions'], 'position_status')}"
    )
    print(
        "row_counts: "
        f"requests={len(frames['execution_requests'])}, "
        f"plans={len(frames['paper_trade_plans'])}, "
        f"positions={len(frames['paper_positions'])}, "
        f"orders={len(frames['paper_orders'])}, "
        f"fills={len(frames['paper_fills'])}"
    )
    print()

    overall = "PASS"
    for check in checks:
        print(f"[{check.status}] {check.name} = {check.count}")
        if check.status == "FAIL":
            overall = "FAIL"
        elif check.status == "WARN" and overall != "FAIL":
            overall = "WARN"

    print()
    print(f"OVERALL: {overall}")
    return 1 if overall == "FAIL" else 0


def main() -> int:
    db_path = _resolve_execution_db_path()
    frames = load_execution_frames(db_path)
    health_frames, stop_filled_qty_available = build_health_frames(frames)
    checks = build_health_checks(health_frames)
    return print_summary(
        db_path=db_path,
        stop_filled_qty_available=stop_filled_qty_available,
        frames=frames,
        checks=checks,
    )


if __name__ == "__main__":
    raise SystemExit(main())
