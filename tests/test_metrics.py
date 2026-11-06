import numpy as np
import pandas as pd

from srp.metrics import daily_ic, ic_summary, max_drawdown, oos_r2, sharpe, sortino


def test_sharpe_hand_computed():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert np.isclose(sharpe(r), r.mean() / r.std() * np.sqrt(252))
    assert np.isclose(r.mean(), 0.005) and np.isclose(r.std(), 0.012909944487358056)
    assert np.isclose(sharpe(r), 0.005 / 0.012909944487358056 * np.sqrt(252))


def test_max_drawdown_hand_computed():
    r = pd.Series([0.10, -0.20, 0.05])
    # wealth 1.10, 0.88, 0.924 -> trough 0.88 vs peak 1.10 = -20%
    assert np.isclose(max_drawdown(r), -0.2)


def test_sortino_hand_computed():
    r = pd.Series([0.02, -0.01, 0.01, -0.03])
    dd = np.sqrt((0.01**2 + 0.03**2) / 4)
    assert np.isclose(sortino(r), r.mean() / dd * np.sqrt(252))


def test_daily_ic_perfect_and_inverse():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=3), list("ABCDE")], names=["date", "ticker"]
    )
    ret = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 3), index=idx)
    assert np.allclose(daily_ic(ret * 10, ret), 1.0)
    assert np.allclose(daily_ic(-ret, ret), -1.0)
    s = ic_summary(daily_ic(ret, ret))
    assert s["ic_mean"] == 1.0 and s["n_days"] == 3


def test_oos_r2_vs_zero():
    y = pd.Series([1.0, -1.0, 2.0])
    assert np.isclose(oos_r2(y, y), 1.0)
    assert np.isclose(oos_r2(y * 0, y), 0.0)
    assert oos_r2(-y, y) < 0


def test_psr_half_at_own_sharpe_and_monotone():
    from srp.metrics import psr

    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    sr = r.mean() / r.std()
    assert np.isclose(psr(r, sr), 0.5)
    assert psr(r, 0.0) > psr(r, sr) > psr(r, 2 * sr)


def test_expected_max_sharpe_hand_computed():
    from scipy import stats

    from srp.metrics import dsr, expected_max_sharpe, psr

    s = np.array([0.0, 0.1, 0.2, 0.3])
    g = 0.5772156649015329
    z = (1 - g) * stats.norm.ppf(1 - 1 / 4) + g * stats.norm.ppf(1 - 1 / (4 * np.e))
    assert np.isclose(expected_max_sharpe(s), np.std(s, ddof=1) * z)
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 500))
    assert np.isclose(dsr(r, np.array([0.05])), psr(r, 0.0))  # one trial -> no deflation
    assert dsr(r, s) < psr(r, 0.0)  # more trials -> harder to be significant
