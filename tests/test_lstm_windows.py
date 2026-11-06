import numpy as np
import pandas as pd

from srp.models.lstm import make_windows


def test_windows_end_at_t_and_use_only_past_rows():
    dates = pd.bdate_range("2020-01-01", periods=30)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    t = dates[25]
    W, widx = make_windows(X, ["f"], 5, pd.DatetimeIndex([t]))
    assert set(widx.get_level_values("date")) == {t}
    a = widx.get_level_values("ticker") == "A"
    expect = X.xs("A", level="ticker")["f"].loc[dates[21] : t].to_numpy()
    np.testing.assert_array_equal(W[a][0, :, 0], expect)
    # first dates lack a full history -> no windows
    W0, _ = make_windows(X, ["f"], 5, pd.DatetimeIndex([dates[2]]))
    assert len(W0) == 0
