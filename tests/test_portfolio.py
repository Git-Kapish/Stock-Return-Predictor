import numpy as np
import pandas as pd

from srp.portfolio import backtest, decile_weights


def _pred(n_days=6, n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=n_days), [f"T{i}" for i in range(n)]],
        names=["date", "ticker"],
    )
    return pd.Series(rng.normal(size=len(idx)), index=idx)


def test_dollar_neutral_and_leg_sizes():
    w = decile_weights(_pred(), 10)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.clip(lower=0).sum(axis=1), 1)
    assert np.allclose(w.clip(upper=0).sum(axis=1), -1)
    assert ((w > 0).sum(axis=1) == 2).all()  # 20 names / 10 deciles


def test_long_top_short_bottom():
    p = _pred()
    w = decile_weights(p, 10)
    d = p.index.get_level_values("date")[0]
    day = p.xs(d, level="date")
    assert w.loc[d, day.idxmax()] > 0 and w.loc[d, day.idxmin()] < 0


def test_zero_turnover_zero_cost_after_first_day():
    w = pd.DataFrame(
        [[0.5, -0.5]] * 4, index=pd.bdate_range("2020-01-01", periods=4), columns=["A", "B"]
    )
    r = pd.DataFrame(0.01, index=w.index, columns=w.columns)
    out = backtest(w, r, cost_bps=5)
    assert (out["turnover"].iloc[1:] == 0).all()
    assert (out["cost"].iloc[1:] == 0).all()
    assert np.isclose(out["cost"].iloc[0], 1.0 * 5 / 1e4)  # first book is a full trade


def test_fully_reversed_book_costs_two_times_gross():
    idx = pd.bdate_range("2020-01-01", periods=3)
    w = pd.DataFrame({"A": [1.0, -1.0, 1.0], "B": [-1.0, 1.0, -1.0]}, index=idx)
    r = pd.DataFrame(0.0, index=idx, columns=w.columns)
    out = backtest(w, r, cost_bps=10)
    # each flip trades |2| + |2| = 4 of notional -> 4 * 10bps
    assert np.allclose(out["turnover"].iloc[1:], 4.0)
    assert np.allclose(out["cost"].iloc[1:], 4.0 * 10 / 1e4)
    assert np.allclose(out["net"].iloc[1:], -out["cost"].iloc[1:])


def test_lag_delays_the_book_by_one_day():
    idx = pd.bdate_range("2020-01-01", periods=4)
    w = pd.DataFrame({"A": [1.0, 0, 0, 0], "B": [-1.0, 0, 0, 0]}, index=idx)
    r = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4], "B": [0.0] * 4}, index=idx)
    assert np.isclose(backtest(w, r, 0, lag=0)["gross"].iloc[0], 0.1)
    out1 = backtest(w, r, 0, lag=1)
    assert np.isclose(out1.loc[idx[1], "gross"], 0.2)  # book from day 0 earns day 1's return
