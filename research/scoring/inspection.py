from __future__ import annotations

from typing import Any

import pandas as pd


def coefficient_summary(model: Any, *, top_n: int = 20) -> pd.DataFrame:
    """Return best-effort coefficient summary for sklearn Pipeline models."""
    if model is None or not hasattr(model, "named_steps"):
        return pd.DataFrame()
    if "preprocess" not in model.named_steps or "model" not in model.named_steps:
        return pd.DataFrame()

    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame()

    preprocessor = model.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(estimator.coef_.ravel()))]

    coef = estimator.coef_.ravel()
    rows = pd.DataFrame({"feature": names, "coefficient": coef})
    rows["abs_coefficient"] = rows["coefficient"].abs()
    return rows.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)
