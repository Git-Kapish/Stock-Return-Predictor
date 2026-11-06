"""Append-only-ish log of every model/config evaluated (feeds the Deflated Sharpe Ratio)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def log_trial(cfg: dict[str, Any], model: str, fields: dict[str, Any]) -> None:
    """Upsert one row keyed by (model, params-json); new keys add columns, nothing is deleted."""
    path = Path(cfg["paths"]["trials"])
    path.parent.mkdir(parents=True, exist_ok=True)
    params = json.dumps(cfg["models"][model], sort_keys=True)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "params"])
    hit = (df["model"] == model) & (df["params"] == params)
    if hit.any():
        for k, v in fields.items():
            if k not in df.columns:
                df[k] = pd.NA
            df.loc[hit, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([{"model": model, "params": params, **fields}])])
    df.to_csv(path, index=False)
