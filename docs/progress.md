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
