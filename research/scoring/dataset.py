from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.research_loader import load_trades_enriched_sql, load_trades_sql
from research.research_metrics_v3 import to_boolish, to_dt, to_num

from .contracts import DEFAULT_SCORING_CONTRACT, ScoringContract, present_export_columns


def _load_authoritative_trade_surface(
    duckdb_path: str | Path | None = None,
    market_type: str | None = None,
) -> pd.DataFrame:
    enriched = load_trades_enriched_sql(duckdb_path=duckdb_path, market_type=market_type)
    if not enriched.empty:
        return enriched
    return load_trades_sql(duckdb_path=duckdb_path, market_type=market_type)


def _dedupe_one_row_per_alert(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "alert_id" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    sort_cols = [col for col in ("signal_ts_utc", "alert_id") if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last")
    return out.drop_duplicates(subset=["alert_id"], keep="last").reset_index(drop=True)


def _derive_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pnl_R" in out.columns:
        pnl = pd.to_numeric(out["pnl_R"], errors="coerce")
        out["profitable"] = pnl.gt(0).where(pnl.notna())

    if "t1_touched" in out.columns:
        out["hit_t1"] = out["t1_touched"]
    elif {"mfe_R", "t1_R"}.issubset(out.columns):
        mfe = pd.to_numeric(out["mfe_R"], errors="coerce")
        t1_r = pd.to_numeric(out["t1_R"], errors="coerce")
        out["hit_t1"] = mfe.ge(t1_r).where(mfe.notna() & t1_r.notna())

    if "t2_touched" in out.columns:
        out["hit_t2"] = out["t2_touched"]
    elif {"mfe_R", "t2_R"}.issubset(out.columns):
        mfe = pd.to_numeric(out["mfe_R"], errors="coerce")
        t2_r = pd.to_numeric(out["t2_R"], errors="coerce")
        out["hit_t2"] = mfe.ge(t2_r).where(mfe.notna() & t2_r.notna())

    return out


def build_scoring_modeling_dataset(
    duckdb_path: str | Path | None = None,
    *,
    market_type: str | None = None,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    """Build one row per alert for scoring research.

    Feature columns are limited by the explicit contract and come from alert-time
    trade/context fields. Outcome-derived fields remain in target columns only.
    """
    frame = _load_authoritative_trade_surface(duckdb_path=duckdb_path, market_type=market_type)
    return _build_scoring_modeling_dataset_from_source(frame, contract=contract)


def _build_scoring_modeling_dataset_from_source(
    frame: pd.DataFrame,
    *,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(contract.metadata_columns + contract.target_columns))

    out = _dedupe_one_row_per_alert(frame)
    out = _derive_targets(out)

    dt_cols = [col for col in ("signal_ts_utc", "signal_ts", "signal_date") if col in out.columns]
    out = to_dt(out, dt_cols)

    numeric_cols = [
        col
        for col in set(contract.numeric_feature_columns + contract.derived_feature_columns + contract.target_columns)
        if col in out.columns and col not in {"profitable", "hit_t1", "hit_t2"}
    ]
    out = to_num(out, numeric_cols)
    out = to_boolish(out, [col for col in ("profitable", "hit_t1", "hit_t2") if col in out.columns])

    # Add missing target columns as nullable values so downstream model code can
    # fail clearly on target availability rather than schema presence.
    for col in contract.target_columns:
        if col not in out.columns:
            out[col] = pd.NA

    return out[present_export_columns(out, contract)].copy()


def audit_scoring_dataset_export(
    source_frame: pd.DataFrame,
    exported_frame: pd.DataFrame,
    *,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> dict[str, object]:
    """Audit strict export decisions for the scoring dataset."""
    source_columns = list(source_frame.columns)
    exported_columns = list(exported_frame.columns)
    exported = set(exported_columns)
    approved = set(contract.export_columns)
    leakage = set(contract.leakage_columns)
    dropped = [col for col in source_columns if col not in exported]
    return {
        "feature_set_version": contract.feature_set_version,
        "source_columns_seen": source_columns,
        "exported_columns": exported_columns,
        "dropped_columns": dropped,
        "columns_dropped_for_leakage": [col for col in dropped if col in leakage],
        "columns_dropped_unknown_unapproved": [
            col for col in dropped if col not in approved and col not in leakage
        ],
        "missing_expected_export_columns": [
            col for col in contract.export_columns if col not in exported
        ],
    }


def build_scoring_modeling_dataset_with_audit(
    duckdb_path: str | Path | None = None,
    *,
    market_type: str | None = None,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = _load_authoritative_trade_surface(duckdb_path=duckdb_path, market_type=market_type)
    dataset = _build_scoring_modeling_dataset_from_source(source, contract=contract)
    return dataset, audit_scoring_dataset_export(source, dataset, contract=contract)
