"""Expanding-window walk-forward splitter with a purge gap, and the OOS prediction loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """Train on ``train_dates`` (all <= test start - purge - 1), predict ``test_dates``."""

    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def make_folds(
    dates: pd.DatetimeIndex, initial: int, block: int, purge: int, expanding: bool = True
) -> list[Fold]:
    """Split sorted unique ``dates`` into walk-forward folds.

    The first training set has ``initial`` dates. Test blocks of ``block`` dates tile the rest of
    the sample with no gaps or overlap. The last ``purge`` dates before each test block are
    removed from training so labels (which look ``horizon`` days ahead) cannot overlap the test.
    """
    dates = pd.DatetimeIndex(dates)
    assert dates.is_monotonic_increasing and dates.is_unique
    folds = []
    for start in range(initial + purge, len(dates), block):
        end = min(start + block, len(dates))
        train_end = start - purge
        train_start = 0 if expanding else max(0, train_end - initial)
        folds.append(Fold(dates[train_start:train_end], dates[start:end]))
    return folds


def walk_forward(
    model_factory: Callable[[], Any], X: pd.DataFrame, y: pd.Series, folds: list[Fold]
) -> pd.Series:
    """Fit a fresh model per fold and return concatenated OOS predictions.

    ``X`` and ``y`` are long ``MultiIndex(date, ticker)``. A model's ``predict`` for date t may
    read ``X`` rows at dates <= t (features only), never ``y`` beyond the training dates.
    """
    out = []
    for i, f in enumerate(folds):
        model = model_factory()
        model.fit(X, y, f.train_dates)
        p = model.predict(X, f.test_dates)
        log.info(
            "fold %d/%d train %s..%s test %s..%s",
            i + 1,
            len(folds),
            f.train_dates[0].date(),
            f.train_dates[-1].date(),
            f.test_dates[0].date(),
            f.test_dates[-1].date(),
        )
        out.append(p)
    return pd.concat(out).rename("pred")
