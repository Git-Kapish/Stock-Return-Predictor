"""Data loading, caching and universe selection.

Panels are dicts of ``(date x ticker)`` frames keyed by field. Dates are tz-naive business days.
A value at date t is that day's close-of-day observation (adjusted for splits/dividends).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)
FIELDS = ["open", "high", "low", "close", "volume"]
Panel = dict[str, pd.DataFrame]


def _load_yfinance(tickers: list[str], start: str, end: str | None) -> Panel:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    return {f: raw[f.capitalize()].copy() for f in FIELDS}


def _load_csv(tickers: list[str], start: str, end: str | None, path: str = "") -> Panel:
    """Long CSV with columns date,ticker,open,high,low,close,volume (swap-in source)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["ticker"].isin(tickers)]
    return {f: df.pivot(index="date", columns="ticker", values=f) for f in FIELDS}


LOADERS: dict[str, Callable[..., Panel]] = {"yfinance": _load_yfinance, "csv": _load_csv}


def _clean(panel: Panel, start: str, end: str | None) -> Panel:
    """Tz-naive sorted DatetimeIndex, trimmed to [start, end], all-NaN columns dropped."""
    out = {}
    for f, df in panel.items():
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df.sort_index().loc[start:end].dropna(axis=1, how="all")
        out[f] = df
    return out


def load_prices(cfg: dict[str, Any], tickers: list[str] | None = None) -> Panel:
    """Load OHLCV for ``tickers`` (default: config candidates), using the raw cache if present."""
    d = cfg["data"]
    tickers = sorted(set(tickers or d["candidates"]))
    cache = Path(cfg["paths"]["raw"]) / f"{d['source']}_{d['start']}_{d['end']}.pkl"
    if cache.exists():
        log.info("using cached raw data %s", cache)
        return pd.read_pickle(cache)
    panel = _clean(LOADERS[d["source"]](tickers, d["start"], d["end"]), d["start"], d["end"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panel, cache)
    missing = sorted(set(tickers) - set(panel["close"].columns))
    if missing:
        log.warning("no data for %d tickers: %s", len(missing), missing)
    return panel


def select_universe(panel: Panel, n: int, window: int, min_frac: float) -> list[str]:
    """Top-``n`` tickers by mean dollar volume over the FIRST ``window`` rows only.

    Uses early-sample data exclusively (no look-ahead in selection). Tickers must have at least
    ``min_frac`` of the window observed.
    """
    close, vol = panel["close"].iloc[:window], panel["volume"].iloc[:window]
    ok = close.notna().mean() >= min_frac
    dv = (close * vol).mean()[ok].dropna().sort_values(ascending=False)
    return sorted(dv.index[:n])


def subset(panel: Panel, tickers: list[str]) -> Panel:
    """Restrict every field to ``tickers``."""
    return {f: df[tickers] for f, df in panel.items()}


def build_panel(cfg: dict[str, Any]) -> Panel:
    """Download/cache candidates, pick the universe on early data, save the processed panel."""
    d = cfg["data"]
    full = load_prices(cfg)
    tickers = select_universe(
        full, d["universe_size"], d["selection_window_days"], d["min_history_frac"]
    )
    panel = subset(full, tickers)
    assert panel["close"].shape[1] == len(tickers) <= d["universe_size"]
    assert panel["close"].shape[0] > 252 * 5, "need several years of history"
    out = Path(cfg["paths"]["processed"]) / "panel.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panel, out)
    log.info("panel %s -> %s (universe %d)", panel["close"].shape, out, len(tickers))
    return panel


def load_panel(cfg: dict[str, Any]) -> Panel:
    """Read the processed universe panel written by ``build_panel``."""
    return pd.read_pickle(Path(cfg["paths"]["processed"]) / "panel.pkl")
