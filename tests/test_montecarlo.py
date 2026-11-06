import numpy as np
import pandas as pd

from srp.montecarlo import (
    bootstrap_stats,
    permutation_null_sharpes,
    permutation_pvalue,
    stationary_bootstrap_indices,
)


def test_bootstrap_indices_valid_and_block_length():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 200, 10, rng)
    assert idx.min() >= 0 and idx.max() < 500
    cont = (np.diff(idx, axis=1) % 500 == 1).mean()  # fraction continuing a block
    assert abs(cont - (1 - 1 / 10)) < 0.02  # mean block ~10


def test_bootstrap_sharpe_ci_covers_truth_and_zero_mean_straddles_zero():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, 1500)
    d = bootstrap_stats(r, 1000, 10, rng)
    lo, hi = np.percentile(d["sharpe"], [2.5, 97.5])
    assert lo < 0 < hi
    assert (d["max_drawdown"] <= 0).all()


def test_permutation_null_centered_and_detects_skill():
    rng = np.random.default_rng(2)
    D, N = 400, 40
    idx, cols = pd.bdate_range("2020-01-01", periods=D), [f"T{i}" for i in range(N)]
    ret = pd.DataFrame(rng.normal(0, 0.01, (D, N)), index=idx, columns=cols)
    skill = ret * 3 + pd.DataFrame(
        rng.normal(0, 0.01, (D, N)), index=idx, columns=cols
    )  # informative
    null = permutation_null_sharpes(skill, ret, 10, 300, rng)
    assert abs(null.mean()) < 0.2  # no skill after shuffling
    long = skill.rank(axis=1, pct=True)
    top, bot = (long > 0.9), (long <= 0.1)
    pnl = (ret * top).sum(1) / top.sum(1) - (ret * bot).sum(1) / bot.sum(1)
    obs = pnl.mean() / pnl.std() * np.sqrt(252)
    assert permutation_pvalue(obs, null) < 0.01
