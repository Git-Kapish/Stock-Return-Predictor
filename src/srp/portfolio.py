"""Decile long/short portfolio with proportional costs.

Conventions
- ``weights.loc[t]`` is the target book set at the CLOSE of day t from predictions made with data
  <= t. Long leg sums to +1, short leg to -1 (dollar-neutral, gross exposure 2). Returns are
  therefore per $1 of long-leg notional.
- A return series is indexed by the day the position starts earning (close t -> close t+1).
- ``lag`` = days between signal and execution: lag 0 trades at the signal close, lag 1 enters one
  close later (the book decided at t starts earning at t+1).
- Turnover/cost use target-weight changes and ignore intraday drift (second-order for 1 day).
"""

from __future__ import annotations

import pandas as pd


def decile_weights(pred: pd.Series, n_deciles: int) -> pd.DataFrame:
    """Wide ``(date x ticker)`` weights: top decile long, bottom decile short, equal weight.

    Uses only the same-day cross-section of ``pred``. Legs are normalised to +1 / -1.
    """
    pct = pred.groupby(level="date").rank(pct=True)
    long_ = (pct > 1 - 1 / n_deciles).astype(float)
    short = (pct <= 1 / n_deciles).astype(float)
    w = long_ / long_.groupby(level="date").transform("sum") - short / short.groupby(
        level="date"
    ).transform("sum")
    return w.unstack("ticker").fillna(0.0)


def backtest(
    weights: pd.DataFrame, ret_fwd: pd.DataFrame, cost_bps: float, lag: int = 0
) -> pd.DataFrame:
    """Daily gross/net returns, turnover and cost.

    ``ret_fwd`` is the wide raw next-day return (close t -> t+1) stamped at t. Weights set at t
    are applied at t + lag, so lag=1 earns the return starting one day after the signal.
    """
    w = weights.shift(lag).dropna(how="all").fillna(0.0) if lag else weights
    r = ret_fwd.reindex(index=w.index, columns=w.columns)
    gross = (w * r).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()  # building the first book is a full trade
    cost = turnover * cost_bps / 1e4
    out = pd.DataFrame({"gross": gross, "turnover": turnover, "cost": cost})
    out["net"] = out["gross"] - out["cost"]
    return out.dropna(subset=["gross"])


def equal_weight_benchmark(ret_fwd: pd.DataFrame) -> pd.Series:
    """Daily-rebalanced equal-weight long-only universe return (no costs)."""
    return ret_fwd.mean(axis=1).dropna()
