from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    split_ts_utc: pd.Timestamp | None
    train_rows: int
    test_rows: int


def time_train_test_split(
    frame: pd.DataFrame,
    *,
    ts_col: str = "signal_ts_utc",
    test_size: float = 0.25,
    min_train_rows: int = 20,
    min_test_rows: int = 5,
) -> TimeSplitResult:
    """Create a deterministic time-ordered train/test split.

    This intentionally avoids random shuffling because scoring research should
    approximate future out-of-sample behavior.
    """
    if frame.empty or ts_col not in frame.columns:
        empty = frame.iloc[0:0].copy()
        return TimeSplitResult(empty, empty, None, 0, 0)

    out = frame.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    out = out[out[ts_col].notna()].sort_values(ts_col).reset_index(drop=True)
    if out.empty:
        return TimeSplitResult(out.copy(), out.copy(), None, 0, 0)

    n = len(out)
    test_rows = max(min_test_rows, int(round(n * test_size)))
    split_idx = max(min_train_rows, n - test_rows)
    split_idx = min(max(split_idx, 1), n)

    train = out.iloc[:split_idx].copy()
    test = out.iloc[split_idx:].copy()
    split_ts = test[ts_col].min() if not test.empty else None
    return TimeSplitResult(train, test, split_ts, int(len(train)), int(len(test)))
