"""Ridge baseline with a fixed alpha from config."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

from srp.models import SklearnModel


class RidgeModel(SklearnModel):
    def __init__(self, params: dict, seed: int, cfg: dict | None = None) -> None:
        self.est = Ridge(alpha=params["alpha"])

    def importance(self) -> pd.Series:
        """Absolute standardised coefficient per feature (features are already on one scale)."""
        return pd.Series(abs(self.est.coef_), index=self.cols)
