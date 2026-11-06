import numpy as np
import pandas as pd

from srp.data import select_universe


def _panel(n_days=300):
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    close = pd.DataFrame(10.0, index=idx, columns=["A", "B", "C", "D"])
    vol = pd.DataFrame(
        {"A": 100.0, "B": 300.0, "C": 200.0, "D": 1e9}, index=idx
    )  # D has huge volume but is listed late
    close.loc[idx[:100], "D"] = np.nan
    return {"close": close, "volume": vol}


def test_select_universe_ranks_by_early_dollar_volume():
    p = _panel()
    assert select_universe(p, n=2, window=252, min_frac=0.98) == ["B", "C"]


def test_select_universe_ignores_future_data():
    p = _panel()
    p["volume"].iloc[252:, 0] = 1e12  # mutate after the selection window
    assert select_universe(p, n=2, window=252, min_frac=0.98) == ["B", "C"]
