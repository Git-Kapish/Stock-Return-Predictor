"""Stage logic: walk-forward training of all enabled models. Scripts only call these."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from srp.metrics import daily_ic, ic_summary, oos_r2
from srp.models import get_model
from srp.trials import log_trial
from srp.walkforward import make_folds, walk_forward

log = logging.getLogger(__name__)
MAX_PLAUSIBLE_IC = 0.1  # CLAUDE.md rule 8: above this, assume a bug


def load_xy(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and labels aligned on their common (date, ticker) rows."""
    p = Path(cfg["paths"]["processed"])
    X, lab = pd.read_pickle(p / "features.pkl"), pd.read_pickle(p / "labels.pkl")
    idx = X.index.intersection(lab.index)
    return X.loc[idx], lab.loc[idx]


def train_models(cfg: dict[str, Any]) -> pd.DataFrame:
    """Walk-forward fit every enabled model, save OOS predictions and an IC summary table."""
    X, lab = load_xy(cfg)
    w = cfg["walkforward"]
    dates = X.index.get_level_values("date").unique().sort_values()
    folds = make_folds(
        dates, w["initial_train_days"], w["test_block_days"], w["purge_days"], w["expanding"]
    )
    log.info(
        "%d folds, OOS %s..%s",
        len(folds),
        folds[0].test_dates[0].date(),
        folds[-1].test_dates[-1].date(),
    )
    rows = {}
    for name in cfg["models"]["enabled"]:
        imps: list = []
        pred = walk_forward(get_model(name, cfg), X, lab["y"], folds, imps)
        if imps:
            tdir = Path(cfg["paths"]["tables"])
            tdir.mkdir(parents=True, exist_ok=True)
            pd.concat(imps, axis=1).mean(axis=1).rename("importance").to_csv(
                tdir / f"importance_{name}.csv"
            )
        pred.to_pickle(Path(cfg["paths"]["processed"]) / f"preds_{name}.pkl")
        ic = daily_ic(pred, lab["ret_fwd"])
        rows[name] = {**ic_summary(ic), "oos_r2": oos_r2(pred, lab["y"])}
        log_trial(cfg, name, {"ic_mean": rows[name]["ic_mean"], "ic_t": rows[name]["ic_t"]})
        log.info("%s: %s", name, rows[name])
        if rows[name]["ic_mean"] > MAX_PLAUSIBLE_IC:
            raise RuntimeError(
                f"{name} IC {rows[name]['ic_mean']:.3f} > {MAX_PLAUSIBLE_IC}: check for leakage"
            )
    out = pd.DataFrame(rows).T
    tdir = Path(cfg["paths"]["tables"])
    tdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tdir / "oos_summary.csv")
    return out


def run_backtest(cfg: dict[str, Any]) -> pd.DataFrame:
    """Decile long/short backtest for every model over costs x execution lags, plus benchmark.

    Writes ``results/tables/sensitivity.csv`` and ``data/processed/backtest.pkl`` (daily frames
    keyed by (model, lag, cost_bps)); records base-cost Sharpe per model in the trials log.
    """
    from srp.metrics import perf_summary
    from srp.portfolio import backtest, decile_weights, equal_weight_benchmark

    pc = cfg["portfolio"]
    _, lab = load_xy(cfg)
    ret = lab["ret_fwd"].unstack("ticker")
    rows, series = [], {}
    for name in cfg["models"]["enabled"]:
        pred = pd.read_pickle(Path(cfg["paths"]["processed"]) / f"preds_{name}.pkl")
        w = decile_weights(pred, pc["n_deciles"])
        for lag in pc["execution_lags"]:
            for bps in pc["cost_bps_per_side"]:
                bt = backtest(w, ret, bps, lag)
                series[(name, lag, bps)] = bt
                rows.append(
                    {
                        "model": name,
                        "lag": lag,
                        "cost_bps": bps,
                        "gross_sharpe": perf_summary(bt, pc["periods_per_year"], "gross")["sharpe"],
                        **perf_summary(bt, pc["periods_per_year"]),
                    }
                )
            base = next(
                r
                for r in rows
                if r["model"] == name and r["lag"] == lag and r["cost_bps"] == pc["base_cost_bps"]
            )
            log_trial(cfg, name, {f"sharpe_net_lag{lag}": base["sharpe"]})
            if base["sharpe"] > 2:
                raise RuntimeError(
                    f"{name} lag{lag} Sharpe {base['sharpe']:.2f} > 2: check for leakage"
                )
    first = next(iter(series.values()))
    bench = equal_weight_benchmark(ret).reindex(first.index).dropna()
    bdf = pd.DataFrame({"net": bench, "gross": bench, "turnover": 0.0})
    series[("benchmark", 0, 0)] = bdf
    rows.append(
        {
            "model": "equal_weight_bh",
            "lag": 0,
            "cost_bps": 0,
            **perf_summary(bdf, pc["periods_per_year"]),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(Path(cfg["paths"]["tables"]) / "sensitivity.csv", index=False)
    pd.to_pickle(series, Path(cfg["paths"]["processed"]) / "backtest.pkl")
    log.info("\n%s", out.round(3).to_string())
    return out
