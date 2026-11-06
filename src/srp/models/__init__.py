"""Models share one interface: ``fit(X, y, train_dates)`` and ``predict(X, test_dates)``.

``X``/``y`` are full long ``MultiIndex(date, ticker)`` frames; each model selects its own rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rows(df: pd.DataFrame | pd.Series, dates: pd.DatetimeIndex):
    """Rows of a long frame whose date level is in ``dates``."""
    return df[df.index.get_level_values("date").isin(dates)]


class SklearnModel:
    """Base for sklearn regressors: subclasses set ``self.est``."""

    est: Any

    def fit(self, X: pd.DataFrame, y: pd.Series, train_dates: pd.DatetimeIndex) -> None:
        Xt = rows(X, train_dates)
        self.cols = list(Xt.columns)
        self.est.fit(Xt.to_numpy(), y.loc[Xt.index].to_numpy())

    def predict(self, X: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        Xs = rows(X, test_dates)
        return pd.Series(self.est.predict(Xs[self.cols].to_numpy()), index=Xs.index)

    def importance(self) -> pd.Series:
        raise NotImplementedError


def get_model(name: str, cfg: dict[str, Any]):
    """Factory returning a zero-arg constructor for model ``name`` (params from config)."""
    from srp.models.ridge import RidgeModel

    reg = {"ridge": RidgeModel}
    try:
        from srp.models.rf import RFModel

        reg["rf"] = RFModel
    except ImportError:  # pragma: no cover
        pass
    try:
        from srp.models.lstm import LSTMModel

        reg["lstm"] = LSTMModel
    except ImportError:  # pragma: no cover
        pass
    cls = reg[name]
    return lambda: cls(cfg["models"][name], cfg["seed"], cfg)


__all__ = ["SklearnModel", "get_model", "rows", "np"]
