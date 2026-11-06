# Progress log

## M0: scaffold
- Repo layout, Makefile, config, ruff + pytest, CI workflow.
- Python 3.12 instead of 3.11 (only 3.12+ installed locally).

## M1: data
- `data.py`: yfinance loader (swappable via `LOADERS`), raw cache in `data/raw/`, universe = top 75 by 2012 dollar volume.
- Result: panel 3733 days x 75 tickers. 6 candidates had no data (ANTM, BK, GPS, K, MON, WBA: renamed/delisted/ticker changes), which is itself survivorship bias to state in the README.

## M2: features and labels
- 18 columns: 6 return lags, 4 MA ratios, RSI14, vol 5/21 + ratio, volume z, hl range (rank-normalised to [-1,1] per day) plus 2 market-state columns (expanding z-score, causal).
- Label: next-day return, per-day winsorised (1/99%) and demeaned; raw `ret_fwd` kept for the portfolio.
- Shapes: features 273,750 rows, labels 279,900 rows (75 tickers). Mutating prices after t leaves features at <= t identical.

## M3: walk-forward + Ridge
- Expanding window: 504-day initial train, 63-day test blocks, 5-day purge -> 50 folds, OOS 2014-05-13..2026-11-04 (3,140 days).
- Ridge (alpha=1000, fixed in config before the run): mean IC 0.0071, t=1.98 (naive, daily ICs treated as independent), OOS R2 vs zero -0.0002.
- Lesson: the permuted-label null test must use a panel with no planted signal. With a signal, a ridge fit to shuffled labels gets a random coefficient direction and IC swings +-0.08 by chance (checked over 8 seeds). That is not a leak.
