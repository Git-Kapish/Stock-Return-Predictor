"""Small LSTM on the last ``seq_len`` days of a few features.

Timing: the sequence ending at date t holds feature rows t-seq_len+1..t (features only look back),
so a test-date sequence may reach into the training period without using any label.
Early stopping uses the LAST ``val_frac`` of the training dates (time-ordered), separated from the
fitting dates by the purge gap. Minibatch order is shuffled with a seeded generator; this only
reorders independent 20-day windows inside the training slice and never mixes train and test.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

log = logging.getLogger(__name__)


def make_windows(
    X: pd.DataFrame, cols: list[str], seq_len: int, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.MultiIndex]:
    """Windows ``(M, seq_len, F)`` ending at each (date in ``dates``, ticker) with a full history.

    Windows with any missing value are dropped. ``X`` is the long feature frame.
    """
    all_dates = X.index.get_level_values("date").unique().sort_values()
    wide = [X[c].unstack("ticker").reindex(all_dates) for c in cols]
    tickers = wide[0].columns
    arr = np.stack([w.to_numpy(dtype=np.float32) for w in wide], axis=-1)  # (T, N, F)
    win = sliding_window_view(arr, seq_len, axis=0)  # (T-seq+1, N, F, seq)
    end_dates = all_dates[seq_len - 1 :]
    take = np.flatnonzero(end_dates.isin(dates))
    win = win[take].transpose(0, 1, 3, 2)  # (D, N, seq, F)
    ok = ~np.isnan(win).any(axis=(2, 3))
    di, ti = np.nonzero(ok)
    idx = pd.MultiIndex.from_arrays([end_dates[take][di], tickers[ti]], names=["date", "ticker"])
    return np.ascontiguousarray(win[di, ti]), idx


class _Net(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMModel:
    def __init__(self, params: dict, seed: int, cfg: dict) -> None:
        self.p, self.seed = params, seed
        self.cols = list(cfg["features"]["lstm_features"])
        self.purge = cfg["walkforward"]["purge_days"]

    def _tensors(self, X, y, dates):
        W, idx = make_windows(X, self.cols, self.p["seq_len"], dates)
        tgt = y.reindex(idx)
        ok = tgt.notna().to_numpy()
        return torch.from_numpy(W[ok]), torch.from_numpy(tgt.to_numpy(np.float32)[ok])

    def fit(self, X: pd.DataFrame, y: pd.Series, train_dates: pd.DatetimeIndex) -> None:
        p = self.p
        torch.manual_seed(self.seed)
        n_val = max(1, int(len(train_dates) * p["val_frac"]))
        fit_d, val_d = train_dates[: -n_val - self.purge], train_dates[-n_val:]
        xt, yt = self._tensors(X, y, fit_d)
        xv, yv = self._tensors(X, y, val_d)
        self.scale = float(yt.std())  # target scale from the fitting slice only
        yt, yv = yt / self.scale, yv / self.scale
        self.net = _Net(len(self.cols), p["hidden_size"], p["num_layers"], p["dropout"])
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        epochs = 0
        for _ in range(p["max_epochs"]):
            epochs += 1
            self.net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), p["batch_size"]):
                b = perm[i : i + p["batch_size"]]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(nn.functional.mse_loss(self.net(xv), yv))
            if val < best - 1e-6:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        log.info("lstm fit: %d epochs, best val mse %.4f", epochs, best)

    def predict(self, X: pd.DataFrame, test_dates: pd.DatetimeIndex) -> pd.Series:
        W, idx = make_windows(X, self.cols, self.p["seq_len"], test_dates)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(W)).numpy() * self.scale
        return pd.Series(out, index=idx)
