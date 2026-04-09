from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_EXECUTION_DB_PATH = Path(os.getenv("EXECUTION_DB_PATH", "./execution.db"))


def _resolve_execution_db_path(execution_db_path: str | Path | None = None) -> Path:
    """Resolve the execution SQLite path relative to the repo root."""
    raw_path = _DEFAULT_EXECUTION_DB_PATH if execution_db_path is None else Path(execution_db_path)
    expanded_path = raw_path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return (_REPO_ROOT / expanded_path).resolve()


def connect_execution_db(
    execution_db_path: str | Path | None = None,
    *,
    read_only: bool = True,
) -> sqlite3.Connection:
    """Open a SQLite connection for read-only execution analysis by default."""
    db_path = _resolve_execution_db_path(execution_db_path)
    uri = f"file:{db_path}?mode=ro" if read_only else str(db_path)
    conn = sqlite3.connect(uri, uri=read_only)
    conn.row_factory = sqlite3.Row
    return conn


def load_execution_table(
    table_name: str,
    execution_db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a single execution table into a pandas DataFrame."""
    conn = connect_execution_db(execution_db_path, read_only=True)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def load_execution_frames(
    execution_db_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the core execution-analysis tables from the SQLite DB."""
    return {
        "execution_requests": load_execution_table("execution_requests", execution_db_path),
        "paper_trade_plans": load_execution_table("paper_trade_plans", execution_db_path),
        "paper_positions": load_execution_table("paper_positions", execution_db_path),
        "paper_orders": load_execution_table("paper_orders", execution_db_path),
        "paper_fills": load_execution_table("paper_fills", execution_db_path),
    }


def build_execution_requests_latest(execution_requests: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest request row per alert_id using a deterministic sort rule."""
    if execution_requests.empty:
        return execution_requests.copy()

    required = {"alert_id", "request_id", "requested_ts_utc"}
    if not required.issubset(execution_requests.columns):
        return execution_requests.copy()

    out = execution_requests.copy()
    out["requested_ts_utc"] = pd.to_datetime(out["requested_ts_utc"], errors="coerce", utc=True)
    out["_request_id_sort"] = out["request_id"].astype(str)
    out = (
        out.sort_values(
            ["alert_id", "requested_ts_utc", "_request_id_sort"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["alert_id"], keep="first")
        .drop(columns=["_request_id_sort"])
        .reset_index(drop=True)
    )
    return out


def build_execution_fill_summary(paper_fills: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized fill truth to one row per plan_id and fill role."""
    if paper_fills.empty:
        return pd.DataFrame(columns=["plan_id"])

    required = {"plan_id", "fill_role", "fill_qty", "fill_px"}
    if not required.issubset(paper_fills.columns):
        return pd.DataFrame(columns=["plan_id"])

    fills = paper_fills.copy()
    fills["fill_qty"] = pd.to_numeric(fills["fill_qty"], errors="coerce")
    fills["fill_px"] = pd.to_numeric(fills["fill_px"], errors="coerce")

    rows: list[dict[str, object]] = []
    for plan_id, sub in fills.groupby("plan_id", dropna=False):
        row: dict[str, object] = {"plan_id": plan_id}
        realized_exit_qty_total = 0.0
        for role in ["ENTRY", "TP1", "TP2", "STOP"]:
            role_df = sub[sub["fill_role"].astype(str).eq(role)].copy()
            qty = pd.to_numeric(role_df["fill_qty"], errors="coerce").sum(min_count=1)
            notional = (
                pd.to_numeric(role_df["fill_qty"], errors="coerce")
                * pd.to_numeric(role_df["fill_px"], errors="coerce")
            ).sum(min_count=1)
            avg_px = (notional / qty) if pd.notna(qty) and qty not in (0, 0.0) else pd.NA
            fill_ts = role_df["fill_ts_utc"].dropna().min() if "fill_ts_utc" in role_df.columns and not role_df.empty else None
            row[f"{role.lower()}_fill_qty"] = float(qty) if pd.notna(qty) else pd.NA
            row[f"{role.lower()}_fill_avg_px"] = float(avg_px) if pd.notna(avg_px) else pd.NA
            row[f"{role.lower()}_fill_ts_utc"] = fill_ts
            if role in {"TP1", "TP2", "STOP"} and pd.notna(qty):
                realized_exit_qty_total += float(qty)
        row["realized_exit_qty_total"] = realized_exit_qty_total
        rows.append(row)

    return pd.DataFrame(rows)


def build_execution_plan_summary(
    paper_trade_plans: pd.DataFrame,
    *,
    execution_requests: pd.DataFrame | None = None,
    paper_positions: pd.DataFrame | None = None,
    execution_fill_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a plan-grain execution summary without mutating runtime truth."""
    if paper_trade_plans.empty:
        return paper_trade_plans.copy()

    out = paper_trade_plans.copy()

    if execution_requests is not None and not execution_requests.empty and "request_id" in out.columns:
        request_columns = [
            col
            for col in [
                "request_id",
                "alert_id",
                "status",
                "requested_ts_utc",
                "submitted_ts_utc",
                "completed_ts_utc",
                "requested_by",
                "reason",
                "error_code",
                "error_message",
                "broker_reject_category",
                "broker_reject_code",
                "rejection_code",
                "rejection_reason",
                "broker",
                "broker_order_id",
            ]
            if col in execution_requests.columns
        ]
        out = out.merge(
            execution_requests[request_columns],
            on="request_id",
            how="left",
            suffixes=("_plan", "_request"),
            validate="1:1",
        )

    if paper_positions is not None and not paper_positions.empty and "plan_id" in out.columns:
        out = out.merge(
            paper_positions,
            on="plan_id",
            how="left",
            suffixes=("", "_position"),
            validate="1:1",
        )

    if execution_fill_summary is not None and not execution_fill_summary.empty and "plan_id" in out.columns:
        out = out.merge(
            execution_fill_summary,
            on="plan_id",
            how="left",
            validate="1:1",
        )

    if "stop_filled_qty" not in out.columns:
        out["stop_filled_qty"] = out.get("stop_fill_qty")
    else:
        out["stop_filled_qty"] = pd.to_numeric(out["stop_filled_qty"], errors="coerce")
        out["stop_filled_qty"] = out["stop_filled_qty"].fillna(
            pd.to_numeric(out.get("stop_fill_qty"), errors="coerce")
        )

    return out
