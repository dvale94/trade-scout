from __future__ import annotations

from pathlib import Path
import pandas as pd

from .research_loader import load_trades_sql


def load_trades(log_dir: str | Path, market_type: str = "crypto") -> pd.DataFrame:
    """
    Backward-compatible wrapper for old CSV loader API.

    V3 uses DuckDB-first loading. Given a log directory, this resolves
    the co-located research.duckdb file and loads from the `trades` view.
    """
    duckdb_path = Path(log_dir) / "research.duckdb"
    return load_trades_sql(duckdb_path=duckdb_path, market_type=market_type)
