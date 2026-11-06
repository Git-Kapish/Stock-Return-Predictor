# Stock Return Predictor

Predict next-day US equity returns with Ridge, Random Forest and an LSTM, then test the signal with a
leak-free walk-forward backtest, a decile long/short portfolio with costs, and Monte Carlo
evaluation (block bootstrap, permutation null, Deflated Sharpe Ratio).

**Headline result: no tradable edge.** Gross Sharpe is statistically indistinguishable from zero for
all three models, with and without a 1-day execution lag, and every net-of-cost result is strongly
negative. The point of the project is the evidence for that statement, not a pretty equity curve.

## What was built
`data → features → labels → walk-forward models → decile long/short → metrics → Monte Carlo → report`,
all in `src/srp/`, all parameters in `configs/default.yaml`.

- **Data:** yfinance daily adjusted OHLCV, 2012-01-01 to 2026-11-06. Universe: top 75 of ~150 large-cap
  candidates by mean dollar volume in the **first 252 trading days only** (no look-ahead in selection).
  Panel: 3,733 days x 75 tickers.
- **Features (all use data <= t):** returns over 1/2/3/5/10/21d, MA ratios (5/10/20/50), RSI(14),
  vol (5d, 21d, ratio), volume z-score, high-low range, each rank-normalised to [-1, 1] across stocks
  each day; plus equal-weight market return and market vol (expanding z-score).
- **Target:** next-day return (close t to close t+1), winsorised 1/99% and demeaned across stocks per
  day. The portfolio is evaluated on the raw return.
- **Validation:** expanding window, 504-day initial train, refit every 63 days, **5-day purge** before
  each test block, 50 folds, out-of-sample 2014-05-13 to 2026-11-04 (3,140 days). No shuffling of
  train/test; LSTM early stopping uses the last 15% of training dates.
- **Models (hyper-parameters fixed in the config before any test result; no tuning):** Ridge
  (alpha 1000), Random Forest (200 trees, depth 6, `min_samples_leaf` 500, `max_features` 0.33),
  LSTM (20-day sequence of 7 features, hidden size 16).
- **Portfolio:** each day long the top decile, short the bottom decile of predictions, equal weight,
  dollar-neutral (long leg +1, short leg -1). Returns are per $1 of long-leg notional.

## How to run
```
make setup      # venv + deps (Python 3.12, CPU PyTorch)
make data       # download + cache OHLCV, pick universe
make features
make train      # ~22 min on CPU, dominated by the LSTM
make backtest
make montecarlo
make report     # figures + tables into results/
make test       # 26 tests
make lint
```
`python scripts/run_all.py all` runs every stage.

## Results
All numbers are out-of-sample, 2014-05 to 2026-11. Sharpe is annualised, daily returns, zero risk-free.
**Execution:** lag 0 trades at the close the signal was computed on; lag 1 enters one close later.
**Costs:** bps per side on traded notional.

### Predictive power
| model | mean IC | IC t-stat (naive) | OOS R² vs zero |
|---|---|---|---|
| Ridge | 0.0071 | 1.98 | -0.0002 |
| Random Forest | 0.0053 | 1.56 | -0.0001 |
| LSTM | 0.0034 | 0.91 | -0.0002 |

ICs are tiny and no model beats a zero forecast in R². The naive t-stats treat daily ICs as
independent and are the optimistic reading; none is convincing after considering six trials.

### Portfolio: cost and lag sensitivity (net Sharpe)
| model | lag | 0 bps (gross) | 5 bps | 10 bps |
|---|---|---|---|---|
| Ridge | 0 | -0.21 | -1.52 | -2.83 |
| Ridge | 1 | 0.14 | -1.18 | -2.49 |
| RF | 0 | 0.18 | -1.33 | -2.85 |
| RF | 1 | -0.07 | -1.63 | -3.18 |
| LSTM | 0 | 0.01 | -0.92 | -1.84 |
| LSTM | 1 | 0.06 | -0.87 | -1.80 |
| Equal-weight buy-and-hold (long-only, no costs) | | 0.85 | | |

Daily turnover is 1.6 to 2.6 (sum of absolute weight changes on a gross book of 2), which is why 5 bps
per side costs roughly 20 to 32% a year (LSTM lowest, RF highest). The benchmark is long-only; its Sharpe is not comparable to a
market-neutral book.

### Uncertainty (5,000 bootstrap draws, 1,000 permutations)
| model | lag | gross Sharpe | 95% bootstrap CI | P(Sharpe <= 0) | permutation p (lag 0) | DSR (6 trials) |
|---|---|---|---|---|---|---|
| Ridge | 0 | -0.21 | [-0.73, 0.37] | 0.76 | 0.79 | 0.02 |
| Ridge | 1 | 0.14 | [-0.42, 0.76] | 0.32 | | 0.18 |
| RF | 0 | 0.18 | [-0.32, 0.69] | 0.25 | 0.26 | 0.21 |
| RF | 1 | -0.07 | [-0.58, 0.47] | 0.61 | | 0.05 |
| LSTM | 0 | 0.01 | [-0.54, 0.55] | 0.49 | 0.47 | 0.08 |
| LSTM | 1 | 0.06 | [-0.54, 0.66] | 0.43 | | 0.11 |

Stationary block bootstrap (mean block 10 days). Permutation null shuffles predictions across tickers
within each day. DSR uses six trials (3 models x 2 lags at 5 bps, per-day Sharpes) and PSR/DSR account
for skew and kurtosis. With 5 bps per side every net CI lies entirely below zero. Full tables are in
`results/tables/`.

Figures in `results/figures/`: `equity_curves.png` (log scale, vs benchmark), `bootstrap_vs_null.png`,
`feature_importance.png`, `ic_by_month.png`, `cost_lag_sensitivity.png`.

![equity](results/figures/equity_curves.png)
![bootstrap vs null](results/figures/bootstrap_vs_null.png)

### Reading it
- **No edge survives costs or a 1-day lag**; before costs there is nothing to survive. Signs of gross
  Sharpe flip between lag 0 and lag 1 for Ridge and RF, which is what noise does.
- **Ridge was the best model by IC.** Trees and the LSTM add nothing on this feature set.
- **Not just 1-day reversal:** Ridge weights are spread over RSI, MA ratios and 21d return; `ret_1` is
  not dominant. (RF puts most importance on the two market-state features, which are identical for
  every stock on a given day and cannot help rank stocks; that is the forest fitting day-level noise.)
- Overall results were sanity-checked against the "too good" rule (Sharpe > 2, IC > 0.1): nothing near.

## Tests
`make test` runs 26 tests, including: mutating prices after t leaves features at t unchanged; train
indices precede test indices by at least the purge gap and blocks tile the sample; costs, dollar
neutrality and turnover; Sharpe/Sortino/drawdown/IC/PSR/DSR against hand-computed values; a planted
signal is detected and permuted labels on a no-signal panel give IC close to 0.

## Limitations (read these)
- **Survivorship bias.** Candidates are today's large caps; 6 candidates (ANTM, BK, GPS, K, MON, WBA)
  returned no yfinance data (renamed or delisted) and dropped out. Past winners are over-represented,
  which flatters long-biased results more than a market-neutral one, but it is not removed.
- **Universe/period** are one draw: 75 stocks, one regime. IC and Sharpe have wide intervals.
- **Costs:** a flat bps per side; no borrow fees, market impact, or short-sale constraints. Turnover
  ignores intraday weight drift. Real costs for a daily-rebalanced book of this turnover would be
  higher for the short leg.
- **Adjusted prices** from yfinance; close-to-close execution at the exact close is optimistic.
- **Daily naive IC t-stats** ignore autocorrelation; use the bootstrap/permutation numbers.
- **Trial count:** six configs were evaluated for the DSR, but design choices made while building
  (feature set, universe size) were not counted.
- Hyper-parameters were set once and not tuned, on purpose; a better-tuned model is not ruled out, but
  tuning would add trials and weaken any resulting Sharpe via the DSR.
- yfinance data can change; results will shift slightly if re-downloaded later.

## What I learned / why most ML funds fail
A tiny positive IC (0.003 to 0.007) is common in daily data and gives a gross Sharpe near zero; once
you charge 5 bps per side on a book that turns over about once a day, the signal is far too small to
pay for the trading. López de Prado's *The 10 Reasons Most Machine Learning Funds Fail* describes the
mechanisms this project tried to guard against: backtests that are run many times until one looks good
(the DSR and a trials log address this), leakage through splits and labels (purge gap, time-ordered
validation, look-ahead tests), and treating one equity curve as evidence (the bootstrap and permutation
tests show how wide the plausible range is). Flexible models (RF, LSTM) did not beat a regularised linear baseline here. Gu, Kelly & Xiu (2020) report ML gains at a monthly horizon with a much larger cross-section and richer features; this project is daily, 75 stocks and 18 simple features, so it is not a replication of that result and says little about it.

## References
- Gu, Kelly, Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
- López de Prado (2018), *The 10 Reasons Most Machine Learning Funds Fail*; *Advances in Financial Machine Learning*.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Politis & Romano (1994), *The Stationary Bootstrap*.
