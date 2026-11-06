"""Labels. The ONLY place a forward shift (``shift(-k)``) is allowed."""

from __future__ import annotations

from typing import Any

import pandas as pd

from srp.data import Panel


def build_labels(panel: Panel, cfg: dict[str, Any]) -> pd.DataFrame:
    """Long ``MultiIndex(date, ticker)`` frame with:

    - ``ret_fwd``: raw return from close t to close t+h (used by the portfolio, known only at t+h).
    - ``y``: ``ret_fwd`` winsorised and demeaned across tickers within day t (training target).
    The last h dates have no label and are dropped.
    """
    h = cfg["label"]["horizon"]
    lo, hi = cfg["label"]["winsor_quantiles"]
    c = panel["close"]
    fwd = (c.shift(-h) / c - 1).rename_axis(index="date", columns="ticker")
    q_lo, q_hi = fwd.quantile(lo, axis=1), fwd.quantile(hi, axis=1)
    y = fwd.clip(lower=q_lo, upper=q_hi, axis=0)
    if cfg["label"]["demean_cross_section"]:
        y = y.sub(y.mean(axis=1), axis=0)
    out = pd.DataFrame({"ret_fwd": fwd.stack(future_stack=True), "y": y.stack(future_stack=True)})
    return out.dropna()
