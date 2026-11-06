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
        pred = walk_forward(get_model(name, cfg), X, lab["y"], folds)
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
