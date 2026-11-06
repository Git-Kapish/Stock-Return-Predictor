import numpy as np
import pandas as pd
import pytest

from srp.config import load_config
from srp.features import build_features
from srp.labels import build_labels


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _panel(n=400, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = [f"T{i}" for i in range(k)]
    close = pd.DataFrame(
        100 * np.exp(rng.normal(0, 0.01, (n, k)).cumsum(0)), index=idx, columns=cols
    )
    return {
        "close": close,
        "open": close * 1.001,
        "high": close * (1 + rng.uniform(0, 0.01, (n, k))),
        "low": close * (1 - rng.uniform(0, 0.01, (n, k))),
        "volume": pd.DataFrame(rng.uniform(1e5, 1e6, (n, k)), index=idx, columns=cols),
    }


def test_future_mutation_does_not_change_features(cfg):
    p = _panel()
    t = p["close"].index[300]
    base = build_features(p, cfg)
    q = {f: df.copy() for f, df in p.items()}
    rng = np.random.default_rng(1)
    for df in q.values():
        after = df.index > t
        df.loc[after] = df.loc[after] * rng.uniform(0.5, 2.0, df.loc[after].shape)
    mut = build_features(q, cfg)
    a, b = (
        base[base.index.get_level_values("date") <= t],
        mut[mut.index.get_level_values("date") <= t],
    )
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


def test_features_in_range_and_no_nan(cfg):
    f = build_features(_panel(), cfg)
    stock = f.drop(columns=["mkt_ret", "mkt_vol"])
    assert stock.min().min() >= -1 and stock.max().max() <= 1
    assert not f.isna().any().any()


def test_label_is_next_day_return_and_demeaned(cfg):
    p = _panel()
    lab = build_labels(p, cfg)
    t = p["close"].index[50]
    expect = p["close"].iloc[51] / p["close"].iloc[50] - 1
    got = lab.loc[t, "ret_fwd"]
    np.testing.assert_allclose(got.values, expect[got.index].values)
    assert abs(lab["y"].groupby(level="date").mean()).max() < 1e-12
    assert p["close"].index[-1] not in lab.index.get_level_values("date")
