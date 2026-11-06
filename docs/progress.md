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

## M4: RF and LSTM (same protocol as Ridge)
| model | mean IC | IC t (naive) | OOS R2 vs 0 |
|---|---|---|---|
| ridge | 0.0071 | 1.98 | -0.00021 |
| rf | 0.0053 | 1.56 | -0.00014 |
| lstm | 0.0034 | 0.91 | -0.00021 |
- Configs were fixed before running (no tuning); each is one row in `results/trials.csv`. Full train takes ~22 min on CPU (LSTM dominates).
- None beats zero in R2; the linear baseline is the best IC, so trees/nets add nothing here.
- LSTM minibatch order is shuffled (seeded) inside the training slice only; train/val/test stay time-ordered.

## M5: portfolio + backtest + sensitivity
Decile long/short (long leg +1, short leg -1, equal weight, rebalanced daily). Sharpe is annualised daily, no risk-free. OOS 2014-05..2026-11. Lag 0 = trade at signal close, lag 1 = enter next close. Net = gross minus bps per side on traded notional.

| model | lag | gross Sharpe | net Sharpe @5bps | net @10bps | daily turnover (sum abs dw) |
|---|---|---|---|---|---|
| ridge | 0 | -0.21 | -1.52 | -2.83 | 2.21 |
| ridge | 1 | 0.14 | -1.18 | -2.49 | 2.21 |
| rf | 0 | 0.18 | -1.33 | -2.85 | 2.56 |
| rf | 1 | -0.07 | -1.63 | -3.18 | 2.56 |
| lstm | 0 | 0.01 | -0.92 | -1.84 | 1.58 |
| lstm | 1 | 0.06 | -0.87 | -1.80 | 1.58 |
| equal-weight B&H (long only) | - | 0.85 (no costs) | | | 0 |
- No model has a gross edge distinguishable from zero, and turnover (~1.1x the gross book per day) makes every net Sharpe strongly negative. Gross Sharpe flips sign between lag 0 and lag 1 for ridge and rf, which is what noise looks like.
- Turnover ignores intraday weight drift. The benchmark is long-only so its Sharpe is not comparable to a market-neutral book.
