from __future__ import annotations

import pandas as pd

from .contracts import DEFAULT_SCORING_CONTRACT, ScoringContract, contract_coverage_summary


def column_availability_summary(
    frame: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    return contract_coverage_summary(frame, contract)


def feature_missingness_summary(
    frame: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    rows: list[dict] = []
    for col in contract.feature_columns:
        if col not in frame.columns:
            rows.append({"feature": col, "present": False, "missing_pct": 1.0, "non_null_rows": 0})
            continue
        rows.append(
            {
                "feature": col,
                "present": True,
                "missing_pct": float(frame[col].isna().mean()) if len(frame) else 0.0,
                "non_null_rows": int(frame[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["present", "missing_pct", "feature"], ascending=[True, False, True])


def target_availability_summary(
    frame: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    rows: list[dict] = []
    for col in contract.target_columns:
        if col not in frame.columns:
            rows.append({"target": col, "present": False, "non_null_rows": 0, "coverage_pct": 0.0})
            continue
        rows.append(
            {
                "target": col,
                "present": True,
                "non_null_rows": int(frame[col].notna().sum()),
                "coverage_pct": float(frame[col].notna().mean()) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def numeric_constant_feature_summary(
    frame: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    rows: list[dict] = []
    for col in contract.numeric_feature_columns + contract.derived_feature_columns:
        if col not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce").dropna()
        rows.append(
            {
                "feature": col,
                "non_null_rows": int(len(numeric)),
                "unique_values": int(numeric.nunique()),
                "is_constant": bool(len(numeric) > 0 and numeric.nunique() <= 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_constant", "feature"], ascending=[False, True])


def categorical_cardinality_summary(
    frame: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    rows: list[dict] = []
    for col in contract.categorical_feature_columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        rows.append(
            {
                "feature": col,
                "non_null_rows": int(series.notna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "top_value": series.dropna().mode().iloc[0] if not series.dropna().mode().empty else pd.NA,
            }
        )
    return pd.DataFrame(rows).sort_values(["unique_values", "feature"], ascending=[False, True])


def train_test_schema_check(
    train: pd.DataFrame,
    test: pd.DataFrame,
    contract: ScoringContract = DEFAULT_SCORING_CONTRACT,
) -> pd.DataFrame:
    rows: list[dict] = []
    for col in contract.export_columns:
        rows.append(
            {
                "column": col,
                "in_train": col in train.columns,
                "in_test": col in test.columns,
                "train_non_null_rows": int(train[col].notna().sum()) if col in train.columns else 0,
                "test_non_null_rows": int(test[col].notna().sum()) if col in test.columns else 0,
            }
        )
    out = pd.DataFrame(rows)
    out["schema_match"] = out["in_train"].eq(out["in_test"])
    return out
