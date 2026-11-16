# Systematic multi-asset trend and carry

**Question.** Does time-series momentum on liquid futures still work after it was published,
once realistic costs are charged, and does it still protect a 60/40 portfolio when equities fall?

This project replicates the trend strategy in Hurst, Ooi and Pedersen (2017), "A Century of
Evidence on Trend-Following Investing", on 54 futures markets from 1990, then tests it on
2017 to 2026, a period the paper never saw. It adds three portfolio construction methods,
carry from the futures curve, tick-based trading and roll costs, bootstrap and deflated Sharpe
ratios, attribution, stress tests and a 60/40 overlay.

The design was fixed in [HYPOTHESES.md](HYPOTHESES.md) before any real-data backtest, and the
2017 to 2026 period is locked in code so it can only be run once with those settings.

## Replication check against AQR

_To be filled from `results/real/full/tables/replication_vs_aqr_factor.md` and
`replication_by_period.md` once the locked run exists._

## Findings

_To be written after the locked run. Every figure will trace to a file in `results/`._

## How to run

```bash
pip install -e ".[dev]"
pytest                                         # 51 tests, including the look-ahead tests

# synthetic data (no Bloomberg needed)
trendcarry synthetic                           # writes workbooks to data/synthetic
trendcarry run --dataset synthetic             # every table and chart, 1990-2016

# real data (Bloomberg exports placed in data/raw, see below)
trendcarry checks --dataset real               # data checks only
trendcarry run --dataset real                  # in-sample, 1990-2016
trendcarry run --dataset real --period full    # locked run including 2017-2026
```

Outputs go to `results/<dataset>/<period>/` as `tables/` (CSV and Markdown), `figures/`,
`data_checks/`, `summary.json` and `report.md`.

## Data

Bloomberg data is licensed and is not in this repository. The loader expects these files in
`data/raw/` (values copies of the pull workbooks, names ending in `final`):

| File pattern | Contents |
|---|---|
| `TrendPull_1*final*.xlsx` | Daily ratio-adjusted generic first contracts, 31 financial futures |
| `TrendPull_2*final*.xlsx` | Daily ratio-adjusted generic first contracts, 23 commodity futures |
| `TrendPull_3*final*.xlsx` | Month-end generic 1 and 2, benchmarks, contract specs |
| `carry_revised*.xlsx` | Re-pulled carry blocks (optional; the block with more data wins) |
| `Time_Series_Momentum_Factors_Monthly*.xlsx` | AQR's free TSMOM factor data |

Synthetic workbooks in the same layout, with known faults injected, let the whole pipeline
and every test run without Bloomberg.

## Method in brief

* **Signal.** Sign of the past 1, 3 and 12-month excess return, averaged.
* **Timing.** Signals use closes up to the day before the month-end rebalance. A test scrambles
  all later data and checks that no weight changes, and a planted leak proves the test works.
* **Sizing.** Each market is scaled by its ex-ante volatility, then the book is scaled to 10%
  ex-ante volatility using a 36-month covariance.
* **Weighting.** Equal risk per market, equal risk per asset class, or equal risk contribution
  with a Ledoit-Wolf shrunk correlation.
* **Carry.** Annualised log spread between the first two contracts, with contract changes and
  untraded back months removed.
* **Costs.** Half a tick per unit traded and at every roll, with quarter-tick and one-tick
  sensitivities. HOP's 2 and 20 fees are applied for the replication comparison.
* **Statistics.** Stationary block bootstrap intervals, Newey-West t-statistics and a deflated
  Sharpe ratio across every variant tried.

## Layout

```
src/trendcarry/
  config.py      universe, parameters, periods
  data.py        Bloomberg workbook loader and cache
  checks.py      data checks and cleaning rules
  synthetic.py   synthetic futures in the Bloomberg layout
  rolls.py       contract cycles, carry, roll frequency
  signals.py     calendar, trend and carry signals
  volatility.py  ex-ante volatility, covariance, shrunk correlation
  portfolio.py   weighting schemes and volatility targeting
  costs.py       trading, roll and fee costs
  backtest.py    monthly engine and attribution of P&L
  stats.py       Sharpe, bootstrap, Newey-West, deflated Sharpe
  analysis.py    attribution, stress tests, 60/40 overlay, replication
  reporting.py   tables and charts
  pipeline.py    one command for every output, with the locked-period gate
tests/           pytest suite
```
