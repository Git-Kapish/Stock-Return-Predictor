# CLAUDE.md — Stock Return Predictor

## What this project is
ML pipeline that predicts next-day US equity returns (Ridge / Random Forest / LSTM), backtests a decile long/short portfolio, and evaluates it with Monte Carlo resampling. **Rigor over returns.** Read `PRD.md` before starting any task.

## Stack
Python 3.12, numpy, pandas, scikit-learn, PyTorch (CPU), yfinance, scipy, matplotlib, pyyaml, pytest, ruff. No exotic dependencies without asking.

## Commands
```
make setup      # venv + install
make data       # download + cache OHLCV
make features
make train      # walk-forward fit, writes OOS predictions
make backtest
make report     # figures + tables into results/
make test       # pytest
make lint       # ruff check + format
```
Keep these targets working at every milestone.

## Layout
`src/srp/` holds all logic (data, features, labels, models, walkforward, portfolio, metrics, montecarlo, report). `configs/default.yaml` holds every parameter. `notebooks/` is exploration only and is never imported by `src/`.

## Non-negotiable rules (look-ahead / leakage)
1. **Never shuffle** time-series data. No `train_test_split(shuffle=True)`, no `KFold(shuffle=True)`.
2. A feature at date t may use data from dates ≤ t only. No centered rolling windows, no `shift(-k)` in features (it is allowed only when building the label).
3. Fit any scaler/transform on train data only. Per-day cross-sectional ranks are fine.
4. Purge the last N training dates before each test block (N ≥ label horizon).
5. Early stopping/validation uses a *later* time slice of the training window, never random rows.
6. Pick the universe using early-sample data only.
7. Hyperparameters are fixed in the config *before* looking at test results. Every config evaluated is appended to `results/trials.csv` (feeds the Deflated Sharpe Ratio).
8. If a result looks too good (Sharpe > 2, IC > 0.1), assume a bug and investigate before reporting.

## Conventions
- Type hints on public functions; docstrings state the **timing** (what date a value refers to).
- Seeds from config; every run is deterministic (`numpy`, `torch`, `sklearn`).
- No magic numbers: costs, deciles, windows live in the config.
- Pure functions in `src/`; scripts only wire things together.
- Dates: `DatetimeIndex`, tz-naive, business days. Panels are `(date × ticker)`; long form is `MultiIndex(date, ticker)`.
- Plots: matplotlib, saved to `results/figures/` as PNG, labelled axes, no plt.show() in library code.
- Log with `logging`, not `print`.

## Testing (write tests alongside code, not after)
- `test_no_lookahead.py`: mutate prices after t, features at t unchanged.
- `test_walkforward.py`: train < test with purge gap, no overlap, blocks tile the sample.
- `test_portfolio.py`: costs, dollar-neutrality, turnover.
- `test_metrics.py`: Sharpe/drawdown/IC/DSR vs hand-computed values.
- `test_pipeline_null.py`: with permuted labels the OOS IC ≈ 0.
Run `make test` and `make lint` before declaring any milestone done.

## Workflow
1. Work milestone by milestone (M0 → M7 in the PRD). One milestone per branch/commit series.
2. Before coding a milestone, restate its done-criteria; after, show the evidence (test output, a table).
3. Ask before: adding dependencies, changing the evaluation protocol, or changing the universe/period.
4. Do not hardcode paths; use the config. Do not commit `data/` or large artifacts.
5. Commit messages: imperative, small commits.

## Definition of done (per milestone)
Code + tests pass + lint clean + a short note in `docs/progress.md` (what changed, what was learned, any numbers).

## Reporting rules
Report every performance number with its uncertainty (bootstrap CI, permutation p-value, DSR). State costs and the execution-lag assumption next to every Sharpe. Write limitations (survivorship bias, regime, costs) in the README. Never present a single equity curve as evidence.

## Out of scope
Live trading, intraday data, leverage optimisation, large hyperparameter searches.
