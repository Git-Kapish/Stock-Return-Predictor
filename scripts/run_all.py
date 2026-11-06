"""Thin CLI wiring the pipeline stages together. Stages are added per milestone."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from srp.config import load_config, set_seed, setup_logging
from srp.data import build_panel, load_panel
from srp.features import build_features
from srp.labels import build_labels
from srp.pipeline import train_models

log = logging.getLogger("run_all")


def build_features_stage(cfg: dict) -> None:
    panel = load_panel(cfg)
    feats, labels = build_features(panel, cfg), build_labels(panel, cfg)
    out = Path(cfg["paths"]["processed"])
    feats.to_pickle(out / "features.pkl")
    labels.to_pickle(out / "labels.pkl")
    log.info("features %s labels %s", feats.shape, labels.shape)


STAGES: dict[str, callable] = {
    "data": build_panel,
    "features": build_features_stage,
    "train": train_models,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["data", "features", "train", "backtest", "report", "all"])
    p.add_argument("--config", default="configs/default.yaml")
    a = p.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    todo = list(STAGES) if a.stage == "all" else [a.stage]
    for s in todo:
        if s not in STAGES:
            raise SystemExit(f"stage '{s}' not implemented yet")
        log.info("stage: %s", s)
        STAGES[s](cfg)


if __name__ == "__main__":
    main()
