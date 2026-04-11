from __future__ import annotations

import numpy as np
import pandas as pd


def _valid_outcome_rows(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if frame.empty or value_col not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out[out[value_col].notna()].copy()


def _bucket_series(values: pd.Series, buckets: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unique_values = numeric.nunique(dropna=True)
    if unique_values <= 1:
        return pd.Series(["all"] * len(values), index=values.index, dtype="object")
    q = max(1, min(int(buckets), int(unique_values)))
    return pd.qcut(numeric, q=q, duplicates="drop")


def bucket_outcome_summary(
    frame: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str = "pnl_R",
    bucket_col: str = "score_bucket_eval",
    buckets: int = 5,
    small_sample_threshold: int = 10,
) -> pd.DataFrame:
    if frame.empty or score_col not in frame.columns:
        return pd.DataFrame()
    df = _valid_outcome_rows(frame, outcome_col)
    if df.empty:
        return pd.DataFrame()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df[score_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df[bucket_col] = _bucket_series(df[score_col], buckets)
    rows: list[dict] = []
    for bucket, group in df.groupby(bucket_col, observed=False, dropna=False):
        outcome = pd.to_numeric(group[outcome_col], errors="coerce").dropna()
        rows.append(
            {
                bucket_col: str(bucket),
                "count": int(len(outcome)),
                "expectancy_R": float(outcome.mean()) if len(outcome) else np.nan,
                "median_R": float(outcome.median()) if len(outcome) else np.nan,
                "win_rate": float((outcome > 0).mean()) if len(outcome) else np.nan,
                "min_score": float(group[score_col].min()),
                "max_score": float(group[score_col].max()),
                "small_sample_flag": bool(len(outcome) < small_sample_threshold),
            }
        )
    return pd.DataFrame(rows)


def top_bottom_comparison(summary: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    if summary.empty or bucket_col not in summary.columns:
        return pd.DataFrame()
    first = summary.iloc[[0]].copy()
    last = summary.iloc[[-1]].copy()
    full = pd.DataFrame(
        [
            {
                bucket_col: "full_universe",
                "count": int(summary["count"].sum()),
                "expectancy_R": np.average(summary["expectancy_R"], weights=summary["count"])
                if summary["count"].sum()
                else np.nan,
                "win_rate": np.average(summary["win_rate"], weights=summary["count"])
                if summary["count"].sum()
                else np.nan,
            }
        ]
    )
    first[bucket_col] = "bottom_bucket"
    last[bucket_col] = "top_bucket"
    return pd.concat([full, first, last], ignore_index=True, sort=False)


def monotonicity_check(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty or "expectancy_R" not in summary.columns:
        return {"bucket_count": 0, "spearman_like_corr": np.nan, "monotonic_non_decreasing": None}
    values = pd.to_numeric(summary["expectancy_R"], errors="coerce").dropna()
    if len(values) < 2:
        return {"bucket_count": int(len(values)), "spearman_like_corr": np.nan, "monotonic_non_decreasing": None}
    ranks = pd.Series(range(len(values)), index=values.index, dtype="float64")
    value_ranks = values.rank(method="average")
    return {
        "bucket_count": int(len(values)),
        "spearman_like_corr": float(ranks.corr(value_ranks, method="pearson")),
        "monotonic_non_decreasing": bool(values.is_monotonic_increasing),
    }


def evaluate_current_score(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    outcome_col: str = "pnl_R",
    buckets: int = 5,
) -> dict[str, object]:
    summary = bucket_outcome_summary(
        frame,
        score_col=score_col,
        outcome_col=outcome_col,
        bucket_col="current_score_bucket_eval",
        buckets=buckets,
    )
    return {
        "bucket_summary": summary,
        "top_bottom": top_bottom_comparison(summary, "current_score_bucket_eval"),
        "monotonicity": monotonicity_check(summary),
    }


def evaluate_prediction_buckets(
    frame: pd.DataFrame,
    *,
    prediction_col: str = "score_shadow",
    outcome_col: str = "pnl_R",
    buckets: int = 5,
) -> dict[str, object]:
    summary = bucket_outcome_summary(
        frame,
        score_col=prediction_col,
        outcome_col=outcome_col,
        bucket_col=f"{prediction_col}_bucket_eval",
        buckets=buckets,
    )
    return {
        "bucket_summary": summary,
        "top_bottom": top_bottom_comparison(summary, f"{prediction_col}_bucket_eval"),
        "monotonicity": monotonicity_check(summary),
    }


def compare_shadow_to_baseline(
    baseline_eval: dict[str, object],
    shadow_eval: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for label, evaluation in (("current_score", baseline_eval), ("shadow_score", shadow_eval)):
        top_bottom = evaluation.get("top_bottom")
        if not isinstance(top_bottom, pd.DataFrame) or top_bottom.empty:
            continue
        top = top_bottom[top_bottom.iloc[:, 0].astype(str).eq("top_bucket")]
        full = top_bottom[top_bottom.iloc[:, 0].astype(str).eq("full_universe")]
        rows.append(
            {
                "score_source": label,
                "full_expectancy_R": float(full["expectancy_R"].iloc[0]) if not full.empty else np.nan,
                "top_bucket_expectancy_R": float(top["expectancy_R"].iloc[0]) if not top.empty else np.nan,
                "top_vs_full_delta_R": (
                    float(top["expectancy_R"].iloc[0] - full["expectancy_R"].iloc[0])
                    if not top.empty and not full.empty
                    else np.nan
                ),
                "top_bucket_win_rate": float(top["win_rate"].iloc[0]) if not top.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_train_test_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str = "pnl_R",
    buckets: int = 5,
) -> dict[str, object]:
    train_eval = evaluate_prediction_buckets(
        train,
        prediction_col=score_col,
        outcome_col=outcome_col,
        buckets=buckets,
    )
    test_eval = evaluate_prediction_buckets(
        test,
        prediction_col=score_col,
        outcome_col=outcome_col,
        buckets=buckets,
    )
    return {
        "train": train_eval,
        "test": test_eval,
        "train_vs_test": pd.DataFrame(
            [
                {
                    "split": "train",
                    **train_eval.get("monotonicity", {}),
                },
                {
                    "split": "test",
                    **test_eval.get("monotonicity", {}),
                },
            ]
        ),
    }


def segmented_score_evaluation(
    frame: pd.DataFrame,
    *,
    segment_col: str,
    score_col: str = "score_shadow",
    outcome_col: str = "pnl_R",
    min_count: int = 10,
) -> pd.DataFrame:
    if frame.empty or segment_col not in frame.columns or score_col not in frame.columns:
        return pd.DataFrame()
    df = _valid_outcome_rows(frame, outcome_col)
    if df.empty:
        return pd.DataFrame()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df[score_col].notna()].copy()
    rows: list[dict] = []
    for value, group in df.groupby(segment_col, dropna=False):
        outcome = pd.to_numeric(group[outcome_col], errors="coerce").dropna()
        if outcome.empty:
            continue
        rows.append(
            {
                segment_col: value,
                "count": int(len(outcome)),
                "expectancy_R": float(outcome.mean()),
                "win_rate": float((outcome > 0).mean()),
                "avg_score": float(pd.to_numeric(group[score_col], errors="coerce").mean()),
                "small_sample_flag": bool(len(outcome) < min_count),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "expectancy_R"], ascending=[False, False])


def promotion_readiness_summary(
    *,
    baseline_eval: dict[str, object],
    candidate_eval: dict[str, object],
    min_test_rows: int = 30,
) -> pd.DataFrame:
    comparison = compare_shadow_to_baseline(baseline_eval, candidate_eval)
    if comparison.empty:
        return pd.DataFrame()
    wide = comparison.set_index("score_source")
    candidate = wide.loc["shadow_score"] if "shadow_score" in wide.index else pd.Series(dtype="float64")
    baseline = wide.loc["current_score"] if "current_score" in wide.index else pd.Series(dtype="float64")
    candidate_top = candidate.get("top_bucket_expectancy_R", np.nan)
    baseline_top = baseline.get("top_bucket_expectancy_R", np.nan)
    top_delta = candidate_top - baseline_top if pd.notna(candidate_top) and pd.notna(baseline_top) else np.nan

    top_bottom = candidate_eval.get("top_bottom")
    test_rows = 0
    if isinstance(top_bottom, pd.DataFrame) and not top_bottom.empty:
        full = top_bottom[top_bottom.iloc[:, 0].astype(str).eq("full_universe")]
        if not full.empty:
            test_rows = int(full["count"].iloc[0])

    return pd.DataFrame(
        [
            {
                "candidate_top_vs_current_top_delta_R": top_delta,
                "test_rows": test_rows,
                "small_sample_flag": bool(test_rows < min_test_rows),
                "candidate_beats_current_top_bucket": bool(pd.notna(top_delta) and top_delta > 0),
            }
        ]
    )
