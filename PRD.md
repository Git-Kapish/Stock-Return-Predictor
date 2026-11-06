# PRD — Stock Return Predictor (ML + Quant Finance, Project 1)

**Owner:** Kapish · **Status:** Draft v1 · **Build tool:** Claude Code

## 1. One-line summary
Train Random Forest and LSTM models to predict next-day stock returns from engineered technical features, then test them with a leak-free walk-forward backtest, a long/short portfolio with transaction costs, and Monte Carlo resampling, instead of trusting a single equity curve.

## 2. Why this project
- Shows the *whole* quant-ML loop: data → features → model → portfolio → honest evaluation.
- The point is rigor. Most people show a pretty equity curve; this project shows how likely that curve is to be luck. That is the differentiator for software/ML/quant-adjacent roles.
- Expected honest outcome: a **small or zero edge** after costs. A clear, well-evidenced "the edge is weak, and here is the statistics" is a successful result. A Sharpe of 3 is a bug, not a win.

## 3. Goals / Non-goals
**Goals**
1. Reproducible pipeline: one command downloads data, builds features, trains, backtests, writes a report.
2. Zero look-ahead: time-ordered splits only, purge gap, features use data ≤ t only.
3. Three models compared on equal footing: Ridge (linear baseline), Random Forest, LSTM.
4. Long/short decile portfolio vs buy-and-hold, with costs and a cost/lag sensitivity table.
5. Monte Carlo evaluation: block bootstrap CI on Sharpe/drawdown, permutation null test, Deflated Sharpe Ratio.
6. Clean GitHub README with results and honest caveats.

**Non-goals**
- Live trading, intraday data, order-book modelling.
- Hyperparameter search at scale (keeps overfitting risk low; if any tuning is done, count it as trials for the DSR).
- Claiming a tradable strategy.

## 4. Data
- **Source:** `yfinance` daily OHLCV (adjusted close) as default; keep a loader interface so a CSV/other source can be swapped in.
- **Universe:** 50–100 liquid US large caps. Choose the universe using only data available at the *start* of the sample (e.g. dollar volume over the first year) to avoid look-ahead in selection.
- **Period:** at least 10 years if available (e.g. 2012 → latest).
- **Known bias to document in README:** survivorship bias (today's index members are winners). State it; do not hide it.
- Cache raw downloads to `data/raw/` (git-ignored). Never re-download in tests.

## 5. Methodology

### 5.1 Features (all computed from data ≤ t)
Lagged returns (1, 2, 3, 5, 10, 21d), moving-average ratios (5/10/20/50), RSI(14), rolling volatility (5d, 21d, ratio), volume z-score, high-low range. Cross-sectionally rank-normalise each feature to [-1, 1] each day (as in Gu, Kelly & Xiu 2020). Add market-state features (equal-weight market return, market volatility).

### 5.2 Target
Next-day return (close t → close t+1), cross-sectionally demeaned for training; winsorise extreme values. The portfolio is evaluated on *raw* next-day returns.

### 5.3 Models
| Model | Notes |
|---|---|
| Ridge | Linear baseline; fixed alpha. If trees/nets can't beat this, say so. |
| Random Forest | `min_samples_leaf` large (hundreds), shallow depth, subsampled features. |
| LSTM (PyTorch) | Sequence of last ~20 days of a small feature set; small hidden size; early stopping on a *later* time slice, never random. |

### 5.4 Validation: walk-forward, expanding window
- Initial train ≈ 2 years, then refit every ~63 trading days; predict the next block out-of-sample.
- Purge the last N training dates (≥ label horizon) before each test block.
- **No shuffling anywhere.** Scalers/rank transforms are fit on train only (per-day cross-sectional ranks are safe).
- Fix all hyperparameters before looking at test performance. Log every config tried (feeds the DSR).

### 5.5 Portfolio and backtest
- Each day: long top decile, short bottom decile of predictions, equal weight, dollar-neutral.
- Costs: default 5 bps per side on traded notional; report 0 / 5 / 10 bps.
- **Execution assumption:** baseline trades at the close the signal was computed on; also report a **1-day execution lag** variant (enter at next close). If the edge disappears with a 1-day lag, say so.
- Benchmarks: equal-weight buy-and-hold of the universe; optionally SPY.
- Metrics: annualised return/vol, Sharpe, Sortino, max drawdown, hit rate, turnover, mean daily IC (rank correlation), IC t-stat, OOS R² vs zero.

### 5.6 Monte Carlo evaluation
1. **Stationary block bootstrap** (mean block ≈ 10d, ≥ 5,000 draws) of daily net returns → CI for Sharpe, max drawdown, final wealth; report P(Sharpe ≤ 0).
2. **Permutation null:** shuffle predictions across tickers within each day (≥ 1,000 runs), compute gross Sharpe under no skill → p-value for the observed Sharpe.
3. **Deflated Sharpe Ratio** (Bailey & López de Prado) with `n_trials` = every model/config evaluated; also PSR.
4. (Stretch) Combinatorial Purged CV for multiple backtest paths.

## 6. Repo structure
```
stock-return-predictor/
├── CLAUDE.md
├── README.md
├── pyproject.toml / requirements.txt
├── Makefile                    # make data | features | train | backtest | report | test
├── configs/default.yaml        # all parameters live here
├── data/{raw,processed}/       # git-ignored
├── src/srp/
│   ├── data.py                 # download + cache + universe selection
│   ├── features.py             # feature engineering (leak-free)
│   ├── labels.py
│   ├── models/{ridge.py,rf.py,lstm.py}
│   ├── walkforward.py          # splitter with purge
│   ├── portfolio.py            # decile long/short, costs
│   ├── metrics.py              # Sharpe, DD, IC, PSR, DSR
│   ├── montecarlo.py           # bootstrap + permutation
│   └── report.py               # figures + markdown tables
├── scripts/run_all.py
├── notebooks/                  # exploration only, never imported
├── results/{figures,tables}/
└── tests/
```

## 7. Milestones
| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold, config, CI (ruff + pytest) | `make test` green |
| M1 | Data loader + universe + cache | panel of OHLCV saved, shape asserted |
| M2 | Features + labels + **no-look-ahead tests** | tests pass (see §8) |
| M3 | Walk-forward splitter + Ridge baseline | OOS predictions + IC reported |
| M4 | RF and LSTM | same OOS protocol, comparable table |
| M5 | Portfolio + backtest + sensitivity | equity curves, cost/lag table |
| M6 | Monte Carlo + DSR | bootstrap CI, permutation p-value, DSR |
| M7 | README, figures, results writeup | reproducible from clean clone |

## 8. Testing requirements
- Altering prices *after* date t must not change features at t.
- Train indices all precede test indices by ≥ purge gap.
- Permuting labels in train produces OOS IC ≈ 0 (sanity: pipeline has no leak).
- Cost model: zero turnover → zero cost; fully reversed book → expected cost.
- Metrics checked against hand-computed small examples.

## 9. Deliverables
1. GitHub repo with a clean README: what was built, data used, how to run, key results, limitations.
2. Figures: equity curves (log scale) vs benchmark, bootstrap Sharpe distribution vs permutation null, feature importance, IC by month, cost/lag sensitivity.
3. A short "What I learned / why most ML funds fail" section citing López de Prado.

## 10. Risks and pitfalls
| Risk | Mitigation |
|---|---|
| Look-ahead via features, scalers, or universe choice | Tests in §8; universe chosen on early data only |
| Overfitting through repeated tuning | Fixed configs; count trials in DSR |
| Unrealistically good result | Treat Sharpe > 2 as a bug until proven otherwise; check lag variant |
| Survivorship bias | Document; optionally test on a delisting-aware dataset |
| Short-term reversal dominating signal | Report feature importance; check if edge is just 1-day reversal |

## 11. Success criteria
- Pipeline reproduces from scratch with one command.
- All leak tests pass.
- Report states results *with uncertainty* (CI, p-value, DSR) and says plainly whether an edge survives costs and a 1-day lag.

## 12. References (from the source post)
- Gu, Kelly, Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
- López de Prado (2018), *The 10 Reasons Most Machine Learning Funds Fail*.
- Stefan Jansen, *Machine Learning for Trading* (GitHub).
- QuantStart, *Forecasting Financial Time Series, Part 1*.
- Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014); López de Prado, *Advances in Financial Machine Learning* (purging/embargo, CPCV).
