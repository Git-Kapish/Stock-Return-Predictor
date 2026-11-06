# Progress log

## M0: scaffold
- Repo layout, Makefile, config, ruff + pytest, CI workflow.
- Python 3.12 instead of 3.11 (only 3.12+ installed locally).

## M1: data
- `data.py`: yfinance loader (swappable via `LOADERS`), raw cache in `data/raw/`, universe = top 75 by 2012 dollar volume.
- Result: panel 3733 days x 75 tickers. 6 candidates had no data (ANTM, BK, GPS, K, MON, WBA: renamed/delisted/ticker changes), which is itself survivorship bias to state in the README.
