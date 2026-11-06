"""Leak-free feature engineering.

Every feature at date t uses data from dates <= t only (trailing windows, no centered windows,
no negative shifts). Stock features are cross-sectionally rank-normalised each day to [-1, 1]
(Gu, Kelly & Xiu 2020); this only uses the same-day cross-section, so it is safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from srp.data import Panel

MARKET_COLS = ["mkt_ret", "mkt_vol"]


def _rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder RSI at t from closes <= t (causal EWM), scaled to 0..100."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def raw_features(panel: Panel, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Unnormalised stock features as wide ``(date x ticker)`` frames; value at t uses <= t."""
    f = cfg["features"]
    c, h, lo, v = panel["close"], panel["high"], panel["low"], panel["volume"]
    r1 = c.pct_change()
    out: dict[str, pd.DataFrame] = {}
    for k in f["return_lags"]:
        out[f"ret_{k}"] = c.pct_change(k)
    for w in f["ma_windows"]:
        out[f"ma_ratio_{w}"] = c / c.rolling(w).mean() - 1
    out[f"rsi_{f['rsi_window']}"] = _rsi(c, f["rsi_window"])
    for w in f["vol_windows"]:
        out[f"vol_{w}"] = r1.rolling(w).std()
    short, long = f["vol_windows"][0], f["vol_windows"][-1]
    out["vol_ratio"] = out[f"vol_{short}"] / out[f"vol_{long}"]
    w = f["volume_z_window"]
    out["volume_z"] = (v - v.rolling(w).mean()) / v.rolling(w).std()
    out["hl_range"] = (h - lo) / c
    return out


def market_features(panel: Panel, cfg: dict[str, Any]) -> pd.DataFrame:
    """Equal-weight market return and 21d market vol, z-scored with an expanding (causal) window."""
    r1 = panel["close"].pct_change()
    mkt = r1.mean(axis=1)
    vol = mkt.rolling(cfg["features"]["vol_windows"][-1]).std()
    df = pd.DataFrame({"mkt_ret": mkt, "mkt_vol": vol})
    mu = df.expanding(min_periods=63).mean()
    sd = df.expanding(min_periods=63).std()
    return (df - mu) / sd


def _long(wide: pd.DataFrame) -> pd.Series:
    wide = wide.rename_axis(index="date", columns="ticker")
    return wide.stack(future_stack=True)


def build_features(panel: Panel, cfg: dict[str, Any]) -> pd.DataFrame:
    """Long ``MultiIndex(date, ticker)`` features in [-1, 1]; row at t uses data <= t only.

    Rows with any missing raw feature (warm-up, halts) are dropped, not filled.
    """
    raw = pd.DataFrame({k: _long(v) for k, v in raw_features(panel, cfg).items()})
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    ranked = raw.groupby(level="date").rank(pct=True) * 2 - 1
    if cfg["features"]["market_features"]:
        mkt = market_features(panel, cfg).reindex(ranked.index.get_level_values("date"))
        mkt.index = ranked.index
        ranked = pd.concat([ranked, mkt], axis=1).dropna()
    return ranked
