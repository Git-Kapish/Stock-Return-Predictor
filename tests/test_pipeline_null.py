import numpy as np
import pandas as pd

from srp.config import load_config
from srp.features import build_features
from srp.labels import build_labels
from srp.metrics import daily_ic
from srp.models import get_model
from srp.walkforward import make_folds, walk_forward


def _panel_with_reversal(n=900, k=30, seed=0, rho=-0.3):
    """Returns follow r_t = rho * r_{t-1} + noise, so ret_1 genuinely predicts next day."""
    rng = np.random.default_rng(seed)
    r = np.zeros((n, k))
    for t in range(1, n):
        r[t] = rho * r[t - 1] + rng.normal(0, 0.01, k)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = [f"T{i}" for i in range(k)]
    close = pd.DataFrame(100 * np.exp(r.cumsum(0)), index=idx, columns=cols)
    vol = pd.DataFrame(rng.uniform(1e5, 1e6, (n, k)), index=idx, columns=cols)
    return {
        "close": close,
        "open": close,
        "high": close * 1.005,
        "low": close * 0.995,
        "volume": vol,
    }


def _oos_ic(y_train_labels, X, lab, cfg):
    dates = (
        X.index.get_level_values("date")
        .unique()
        .intersection(lab.index.get_level_values("date").unique())
    )
    folds = make_folds(dates.sort_values(), 300, 63, cfg["walkforward"]["purge_days"])
    pred = walk_forward(get_model("ridge", cfg), X, y_train_labels, folds)
    return daily_ic(pred, lab["ret_fwd"]).mean()


def _xy(rho):
    p = _panel_with_reversal(rho=rho)
    X, lab = build_features(p, load_config()), build_labels(p, load_config())
    common = X.index.intersection(lab.index)
    return X.loc[common], lab.loc[common]


def test_planted_signal_is_detected():
    cfg = load_config()
    X, lab = _xy(rho=-0.3)
    assert _oos_ic(lab["y"], X, lab, cfg) > 0.05


def test_no_signal_gives_zero_ic_with_real_and_permuted_labels():
    # Null must use a panel with NO planted signal: with a signal present, a ridge fit to shuffled
    # labels gets a random coefficient direction that projects onto it, so IC is +-0.08 by chance.
    # Without a signal, any non-zero IC would mean information leaks into the pipeline.
    cfg = load_config()
    X, lab = _xy(rho=0.0)
    rng = np.random.default_rng(1)
    perm = lab["y"].groupby(level="date").transform(lambda s: rng.permutation(s.to_numpy()))
    assert abs(_oos_ic(lab["y"], X, lab, cfg)) < 0.03
    assert abs(_oos_ic(perm, X, lab, cfg)) < 0.02
