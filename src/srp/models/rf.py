"""Random Forest: shallow trees, huge leaves, subsampled features (hyper-parameters in config)."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from srp.models import SklearnModel


class RFModel(SklearnModel):
    def __init__(self, params: dict, seed: int, cfg: dict | None = None) -> None:
        self.est = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=params["n_jobs"],
            random_state=seed,
        )

    def importance(self) -> pd.Series:
        """Impurity-based importance per feature (sums to 1)."""
        return pd.Series(self.est.feature_importances_, index=self.cols)
