import numpy as np
import pandas as pd

from srp.walkforward import make_folds


def test_folds_purge_no_overlap_and_tile():
    dates = pd.bdate_range("2015-01-01", periods=1000)
    initial, block, purge = 504, 63, 5
    folds = make_folds(dates, initial, block, purge)
    pos = {d: i for i, d in enumerate(dates)}
    assert len(folds[0].train_dates) == initial
    for f in folds:
        gap = pos[f.test_dates[0]] - pos[f.train_dates[-1]] - 1
        assert gap >= purge
        assert f.train_dates.max() < f.test_dates.min()
        assert f.train_dates[0] == dates[0]  # expanding
    tests = np.concatenate([f.test_dates.values for f in folds])
    assert len(set(tests)) == len(tests)  # no overlap
    expected = dates[initial + purge :].values
    assert (tests == expected).all()  # blocks tile the sample with no gaps


def test_train_grows_each_fold():
    dates = pd.bdate_range("2015-01-01", periods=800)
    sizes = [len(f.train_dates) for f in make_folds(dates, 300, 50, 5)]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
