"""Evaluation metrics. All series are indexed by date (daily) unless stated."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ic(pred: pd.Series, ret: pd.Series) -> pd.Series:
    """Per-day Spearman rank correlation between ``pred`` at t and ``ret`` (next-day return).

    Inputs are long ``MultiIndex(date, ticker)``; output is indexed by date t.
    """
    df = pd.concat({"p": pred, "r": ret}, axis=1).dropna()
    rk = df.groupby(level="date").rank()
    rk = rk - rk.groupby(level="date").transform("mean")
    g = lambda s: s.groupby(level="date").sum()  # noqa: E731
    return g(rk["p"] * rk["r"]) / np.sqrt(g(rk["p"] ** 2) * g(rk["r"] ** 2))


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean IC, its std, naive t-stat (daily ICs treated as independent) and hit rate."""
    n = ic.notna().sum()
    sd = ic.std()
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(sd),
        "ic_t": float(ic.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan"),
        "ic_pos_frac": float((ic > 0).mean()),
        "n_days": int(n),
    }


def oos_r2(pred: pd.Series, y: pd.Series) -> float:
    """OOS R^2 versus a zero forecast: 1 - sum((y-p)^2) / sum(y^2). Positive = beats zero."""
    df = pd.concat({"p": pred, "y": y}, axis=1).dropna()
    return float(1 - ((df["y"] - df["p"]) ** 2).sum() / (df["y"] ** 2).sum())


def sharpe(r: pd.Series, periods: int = 252) -> float:
    """Annualised Sharpe of daily returns (zero risk-free; long/short book is self-financing)."""
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(r: pd.Series, periods: int = 252) -> float:
    """Annualised mean over downside deviation (target 0, divided by all observations)."""
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / dd * np.sqrt(periods)) if dd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall of compounded wealth (a negative number)."""
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def perf_summary(df: pd.DataFrame, periods: int = 252, col: str = "net") -> dict[str, float]:
    """Annualised stats of ``df[col]`` plus hit rate and mean turnover from ``df['turnover']``."""
    r = df[col]
    return {
        "ann_return": float(r.mean() * periods),
        "ann_vol": float(r.std() * np.sqrt(periods)),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "daily_turnover": float(df["turnover"].mean()),
    }
