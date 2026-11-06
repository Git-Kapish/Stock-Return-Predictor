"""Figures and markdown tables from the saved pipeline outputs. No computation of results here."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from srp.metrics import daily_ic  # noqa: E402

log = logging.getLogger(__name__)
MODELS = ["ridge", "rf", "lstm"]


def _save(fig: plt.Figure, cfg: dict[str, Any], name: str) -> None:
    out = Path(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)


def fig_equity(series: dict, cfg: dict[str, Any]) -> None:
    """Log-scale wealth of each model (gross and net of base cost, lag 0) vs equal-weight B&H."""
    base = cfg["portfolio"]["base_cost_bps"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, c in zip(cfg["models"]["enabled"], ["C0", "C1", "C2"], strict=False):
        bt = series[(m, 0, base)]
        ax.plot((1 + bt["gross"]).cumprod(), c, ls="--", lw=1, label=f"{m} gross")
        ax.plot((1 + bt["net"]).cumprod(), c, lw=1.3, label=f"{m} net {base}bps")
    ax.plot(
        (1 + series[("benchmark", 0, 0)]["net"]).cumprod(), "k", lw=1.5, label="equal-weight B&H"
    )
    ax.set_yscale("log")
    ax.set_xlabel("date (position start)")
    ax.set_ylabel("wealth (log scale, start = 1)")
    ax.set_title("Decile long/short vs buy-and-hold (lag 0). One path, not evidence.")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, cfg, "equity_curves.png")


def fig_bootstrap_vs_null(dists: dict, cfg: dict[str, Any]) -> None:
    """Bootstrap distribution of gross lag-0 Sharpe against the permutation-null distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, m in zip(axes, cfg["models"]["enabled"], strict=False):
        ax.hist(dists[(m, "null")], bins=40, alpha=0.5, density=True, label="permutation null")
        ax.hist(dists[(m, 0, 0)], bins=40, alpha=0.5, density=True, label="bootstrap (observed)")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(m)
        ax.set_xlabel("annualised gross Sharpe")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    _save(fig, cfg, "bootstrap_vs_null.png")


def fig_importance(cfg: dict[str, Any]) -> None:
    """Mean per-fold feature importance for Ridge (|coef|) and RF (impurity)."""
    tdir = Path(cfg["paths"]["tables"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, m, lab in zip(
        axes, ["ridge", "rf"], ["mean |coef|", "mean impurity importance"], strict=True
    ):
        s = pd.read_csv(tdir / f"importance_{m}.csv", index_col=0)["importance"].sort_values()
        ax.barh(s.index, s.values)
        ax.set_xlabel(lab)
        ax.set_title(f"{m} feature importance")
    _save(fig, cfg, "feature_importance.png")


def fig_ic_by_month(cfg: dict[str, Any], ret_fwd: pd.Series) -> None:
    """Mean daily rank IC per calendar month for each model."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in cfg["models"]["enabled"]:
        pred = pd.read_pickle(Path(cfg["paths"]["processed"]) / f"preds_{m}.pkl")
        ic = daily_ic(pred, ret_fwd)
        ax.plot(
            ic.groupby(ic.index.to_period("M")).mean().rolling(6).mean().to_timestamp(), label=m
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("month")
    ax.set_ylabel("monthly mean IC (6-month rolling avg)")
    ax.set_title("Rank IC over time")
    ax.legend()
    _save(fig, cfg, "ic_by_month.png")


def fig_sensitivity(sens: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Net Sharpe by cost and execution lag for each model."""
    m = sens[sens["model"] != "equal_weight_bh"]
    costs = sorted(m["cost_bps"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lag in zip(axes, sorted(m["lag"].unique()), strict=True):
        for i, name in enumerate(cfg["models"]["enabled"]):
            v = (
                m[(m["model"] == name) & (m["lag"] == lag)]
                .set_index("cost_bps")
                .loc[costs, "sharpe"]
            )
            ax.bar(np.arange(len(costs)) + i * 0.27, v.values, 0.27, label=name)
        ax.set_xticks(np.arange(len(costs)) + 0.27, [f"{c} bps" for c in costs])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"execution lag {lag}")
        ax.set_xlabel("cost per side")
    axes[0].set_ylabel("net Sharpe")
    axes[0].legend()
    _save(fig, cfg, "cost_lag_sensitivity.png")


def _md(df: pd.DataFrame) -> str:
    """Minimal GitHub markdown table (avoids the optional tabulate dependency)."""
    nl = chr(10)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return nl.join(lines)


def write_tables(cfg: dict[str, Any]) -> None:
    """Markdown copies of the result tables for the README."""
    tdir = Path(cfg["paths"]["tables"])
    for name in ["oos_summary", "sensitivity", "montecarlo"]:
        df = pd.read_csv(tdir / f"{name}.csv")
        if df.columns[0].startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "model"})
        (tdir / f"{name}.md").write_text(_md(df.round(3)), encoding="utf-8")


def make_report(cfg: dict[str, Any]) -> None:
    """Build all figures and markdown tables from saved results."""
    proc = Path(cfg["paths"]["processed"])
    series = pd.read_pickle(proc / "backtest.pkl")
    dists = pd.read_pickle(proc / "mc.pkl")
    lab = pd.read_pickle(proc / "labels.pkl")
    sens = pd.read_csv(Path(cfg["paths"]["tables"]) / "sensitivity.csv")
    fig_equity(series, cfg)
    fig_bootstrap_vs_null(dists, cfg)
    fig_importance(cfg)
    fig_ic_by_month(cfg, lab["ret_fwd"])
    fig_sensitivity(sens, cfg)
    write_tables(cfg)
    log.info("report written to %s", cfg["paths"]["figures"])
