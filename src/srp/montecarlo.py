"""Monte Carlo evaluation: stationary block bootstrap and a within-day permutation null."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano indices, shape ``(n_boot, n)``: continue the block w.p. 1-1/mean_block."""
    start = rng.integers(0, n, (n_boot, n))
    new = rng.random((n_boot, n)) < 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = start[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_stats(
    r: np.ndarray, n_boot: int, mean_block: float, rng: np.random.Generator, periods: int = 252
) -> dict[str, np.ndarray]:
    """Bootstrap distributions of Sharpe, max drawdown and final wealth of daily returns ``r``."""
    idx = stationary_bootstrap_indices(len(r), n_boot, mean_block, rng)
    s = r[idx]
    sh = s.mean(1) / s.std(1, ddof=1) * np.sqrt(periods)
    w = np.cumprod(1 + s, axis=1)
    dd = (w / np.maximum.accumulate(w, axis=1) - 1).min(1)
    return {"sharpe": sh, "max_drawdown": dd, "final_wealth": w[:, -1]}


def summarize(dist: dict[str, np.ndarray], level: float) -> dict[str, float]:
    """Percentile CIs for each bootstrap statistic and P(Sharpe <= 0)."""
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out: dict[str, float] = {}
    for k, v in dist.items():
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(x) for x in np.percentile(v, [lo, hi]))
    out["p_sharpe_le0"] = float((dist["sharpe"] <= 0).mean())
    return out


def permutation_null_sharpes(
    pred: pd.DataFrame,
    ret: pd.DataFrame,
    n_deciles: int,
    n_perm: int,
    rng: np.random.Generator,
    periods: int = 252,
) -> np.ndarray:
    """Gross Sharpe of the decile book when predictions are shuffled across tickers each day.

    ``pred`` and ``ret`` are wide ``(date x ticker)``; names without a prediction stay out. Each
    permutation re-assigns the day's prediction values to random tickers, then applies the same
    top/bottom-decile rule, so any skill is destroyed while the marginal prediction distribution
    and the tradable set are kept.
    """
    ret = ret.reindex_like(pred)
    valid = (pred.notna() & ret.notna()).to_numpy()
    r = np.nan_to_num(ret.to_numpy())
    n_valid = valid.sum(1, keepdims=True)
    out = np.empty(n_perm)
    for b in range(n_perm):
        keys = np.where(valid, rng.random(valid.shape), 2.0)
        order = keys.argsort(1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(valid.shape[1])[None, :].repeat(len(valid), 0), 1)
        pct = (rank + 1) / n_valid
        lg, sh = valid & (pct > 1 - 1 / n_deciles), valid & (pct <= 1 / n_deciles)
        pnl = (r * lg).sum(1) / lg.sum(1) - (r * sh).sum(1) / sh.sum(1)
        out[b] = pnl.mean() / pnl.std(ddof=1) * np.sqrt(periods)
    return out


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided p = (1 + #{null >= observed}) / (1 + B)."""
    return float((1 + (null >= observed).sum()) / (1 + len(null)))
