from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd


PolicySimulator = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ExitPolicySpec:
    name: str
    simulator: PolicySimulator
    params: dict[str, Any] = field(default_factory=dict)

    def simulate(self, trades: pd.DataFrame) -> pd.Series:
        return self.simulator(trades)


def _empty_series_like(trades: pd.DataFrame) -> pd.Series:
    return pd.Series(index=trades.index, dtype="float64")


def _coerce_trade_metric(trades: pd.DataFrame, column: str) -> pd.Series:
    if column not in trades.columns:
        return _empty_series_like(trades)
    return pd.to_numeric(trades[column], errors="coerce")


def _coerce_trade_metrics(trades: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        _coerce_trade_metric(trades, "mfe_R"),
        _coerce_trade_metric(trades, "mae_R"),
        _coerce_trade_metric(trades, "pnl_R"),
    )


def simulate_policy_hold_t2_or_stop(trades: pd.DataFrame) -> pd.Series:
    """
    Current strategy approximation:
      - If MAE >= 1R => STOP (-1R)
      - Else if MFE >= 2.5R => T2 (+2.5R)
      - Else => TIMEOUT exit at whatever pnl_R already computed (fallback to pnl_R)
    """
    mfe, mae, pnl = _coerce_trade_metrics(trades)
    sim = pnl.copy()
    sim = sim.where(~(mae >= 1.0), other=-1.0)
    sim = sim.where(~((mae < 1.0) & (mfe >= 2.5)), other=2.5)
    return sim


def simulate_policy_full_exit_t1(trades: pd.DataFrame) -> pd.Series:
    """
    Full exit at T1 when possible:
      - If MAE >= 1R before exit => STOP (-1R)
      - Else if MFE >= 1.5R => +1.5R
      - Else fallback to pnl_R
    """
    mfe, mae, pnl = _coerce_trade_metrics(trades)
    sim = pnl.copy()
    sim = sim.where(~(mae >= 1.0), other=-1.0)
    sim = sim.where(~((mae < 1.0) & (mfe >= 1.5)), other=1.5)
    return sim


def simulate_policy_half_t1_rest_t2_be(trades: pd.DataFrame) -> pd.Series:
    """
    Strategy A:
      - Take 50% at T1 (+0.75R)
      - Move stop to breakeven on remainder once T1 hit
      - Remainder to T2 (+1.25R) if MFE >= 2.5R
      - If after T1 price later would have hit original stop, we assume BE on remainder (0R), but overall keeps the +0.75R.
      - If stop before T1: -1R
    """
    mfe, mae, pnl = _coerce_trade_metrics(trades)
    sim = pnl.copy()

    sim = sim.where(~(mae >= 1.0), other=-1.0)

    reached_t1 = (mae < 1.0) & (mfe >= 1.5)
    sim = sim.where(~reached_t1, other=0.75)

    reached_t2 = (mae < 1.0) & (mfe >= 2.5)
    sim = sim.where(~reached_t2, other=2.0)
    return sim


def simulate_policy_break_even_after_trigger(trades: pd.DataFrame, trigger_r: float = 0.5) -> pd.Series:
    """
    Approximation:
      - If MAE >= 1.0R and the trigger was never reached, assume full stop at -1R.
      - If MFE >= trigger_r and MAE < 1.0R and trade does not ultimately reach T2, assume stop is moved to breakeven and final result becomes 0R.
      - If MFE >= 2.5R, assume full target at +2.5R.
      - If both trigger and stop are hit but T2 is not reached, ordering is ambiguous with aggregate MFE/MAE only, so fall back to recorded pnl_R.
      - Otherwise fall back to recorded pnl_R.

    This uses only MFE/MAE/pnl_R aggregates, so it does not know the true intratrade order of excursions.
    """
    mfe, mae, pnl = _coerce_trade_metrics(trades)
    sim = pnl.copy()

    reached_t2 = mfe >= 2.5
    reached_trigger = mfe >= trigger_r
    clear_stop = (mae >= 1.0) & ~reached_trigger
    clean_be_runner = reached_trigger & (mae < 1.0) & ~reached_t2

    sim = sim.where(~clear_stop, other=-1.0)
    sim = sim.where(~clean_be_runner, other=0.0)
    sim = sim.where(~reached_t2, other=2.5)
    return sim


def simulate_policy_partial_and_runner(
    trades: pd.DataFrame,
    partial_trigger_r: float = 1.0,
    partial_fraction: float = 0.5,
    final_target_r: float = 2.5,
    move_stop_to_be: bool = True,
) -> pd.Series:
    """
    Approximation:
      - If stopped before trigger, assume -1R.
      - If trigger is reached, bank partial_fraction * partial_trigger_r.
      - If the final target is also reached, add the remaining fraction * final_target_r.
      - If trigger is reached and BE is enabled but final target is not reached, assume the runner exits at 0R.
      - If trigger is reached and BE is disabled, approximate the runner using the recorded pnl_R on the remaining fraction.
      - If both trigger and stop are hit but final target is not reached, ordering is ambiguous with aggregate MFE/MAE only, so fall back to recorded pnl_R.
      - Otherwise fall back to recorded pnl_R.

    This uses only aggregated trade metrics, so trigger/stop ordering is inferred conservatively from MFE/MAE thresholds.
    """
    mfe, mae, pnl = _coerce_trade_metrics(trades)
    sim = pnl.copy()

    partial_bank = partial_fraction * partial_trigger_r
    runner_fraction = 1.0 - partial_fraction

    reached_trigger = mfe >= partial_trigger_r
    reached_final_target = mfe >= final_target_r
    clear_stop = (mae >= 1.0) & ~reached_trigger
    clean_trigger_no_target = reached_trigger & (mae < 1.0) & ~reached_final_target

    sim = sim.where(~clear_stop, other=-1.0)

    if move_stop_to_be:
        sim = sim.where(~clean_trigger_no_target, other=partial_bank)
    else:
        runner_pnl = partial_bank + (runner_fraction * pnl)
        sim = sim.where(~clean_trigger_no_target, other=runner_pnl)

    sim = sim.where(
        ~reached_final_target,
        other=partial_bank + (runner_fraction * final_target_r),
    )
    return sim


def expectancy(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    wins = s[s > 0]
    losses = s[s < 0]
    win_rate = len(wins) / len(s)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))


def summarize_simulated_series(policy: str, series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "policy": policy,
            "trades": 0,
            "avg_R": None,
            "median_R": None,
            "expectancy_R": None,
            "win_rate": None,
        }

    return {
        "policy": policy,
        "trades": int(s.shape[0]),
        "avg_R": float(s.mean()),
        "median_R": float(s.median()),
        "expectancy_R": expectancy(s),
        "win_rate": float((s > 0).mean()),
    }


def make_policy(name: str, simulator: PolicySimulator, **params: Any) -> ExitPolicySpec:
    if not params:
        return ExitPolicySpec(name=name, simulator=simulator)

    def _bound(trades: pd.DataFrame) -> pd.Series:
        return simulator(trades, **params)

    return ExitPolicySpec(name=name, simulator=_bound, params=params)


def default_exit_policies() -> list[ExitPolicySpec]:
    return [
        make_policy("hold_t2_or_stop", simulate_policy_hold_t2_or_stop),
        make_policy("full_exit_t1", simulate_policy_full_exit_t1),
        make_policy("half_t1_rest_t2_be", simulate_policy_half_t1_rest_t2_be),
        make_policy("break_even_after_0_5R", simulate_policy_break_even_after_trigger, trigger_r=0.5),
        make_policy(
            "partial_1_0R_half_runner_2_5R_be",
            simulate_policy_partial_and_runner,
            partial_trigger_r=1.0,
            partial_fraction=0.5,
            final_target_r=2.5,
            move_stop_to_be=True,
        ),
    ]


def compare_exit_policies(
    trades: pd.DataFrame,
    policies: Iterable[ExitPolicySpec] | None = None,
) -> pd.DataFrame:
    """
    Return a tidy summary table of simulated expectancy by policy.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=["policy", "trades", "avg_R", "median_R", "expectancy_R", "win_rate"]
        )

    policy_specs = list(default_exit_policies() if policies is None else policies)
    if not policy_specs:
        return pd.DataFrame(
            columns=["policy", "trades", "avg_R", "median_R", "expectancy_R", "win_rate"]
        )

    rows = [
        summarize_simulated_series(policy.name, policy.simulate(trades))
        for policy in policy_specs
    ]
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["expectancy_R", "trades"], ascending=[False, False], na_position="last")
