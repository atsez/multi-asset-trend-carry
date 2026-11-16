# Hypotheses and pre-registered design

**Status:** Fixed. Commit this file before the first run on real data.
Nothing in this file may change after that commit without a dated entry in the change log at
the bottom, and any change to a parameter counts as an extra trial in the deflated Sharpe ratio.

**Dated:** 16 November 2026

**In-sample period:** January 1990 to December 2016 (signals warm up from 1990, so the first
live month depends on each market's history).

**Locked test period:** January 2017 to October 2026. Run once, with the settings below, after
the in-sample work is finished. The code refuses a second run with different settings unless a
reason is given, and that reason is logged.

## 1. The question

Does a simple, published time-series momentum strategy still earn a positive risk-adjusted
return on liquid futures after it was published, once realistic costs are charged, and does it
still protect a 60/40 portfolio in equity sell-offs?

## 2. Primary specification (the one result the project stands on)

All values are set in `src/trendcarry/config.py` (`BASELINE`).

| Choice | Setting |
|---|---|
| Universe | 54 futures (11 equity index, 12 government bond, 8 currency, 23 commodity) |
| Prices | Bloomberg ratio-adjusted generic first contracts, 5 days before expiry or first notice |
| Signal | Sign of the past 1, 3 and 12-month excess return, equally weighted |
| Information lag | Signals use closes up to the trading day before the rebalance close |
| Rebalance | Monthly, at the last trading day's close |
| Position size | Signal divided by ex-ante volatility (EWMA of daily returns, centre of mass 60 days) |
| Weighting | Equal risk per market |
| Volatility target | 10% a year, ex ante, from an equally weighted 36-month covariance of monthly returns (at least 24 months) |
| Leverage cap | Gross notional at most 10 times capital |
| Eligibility | 24 monthly returns of history, a 12-month lookback and a price no older than 10 trading days |
| Costs | Half a tick per unit of notional traded, plus two half-tick trades at every roll |
| Returns | Excess returns (futures are funded); cash added only for fees and the 60/40 overlay |

**Primary test:** the net-of-cost Sharpe ratio of this specification over the locked period,
with a block bootstrap 95% interval and a Newey-West t-statistic.

## 3. Secondary specifications (each counts as a trial)

| Name | Change from primary |
|---|---|
| hop_replication | Same-close information (no lag), as in Hurst, Ooi and Pedersen |
| equal_class | Equal risk per asset class, split equally within each class |
| erc_lw | Equal risk contribution using a Ledoit-Wolf shrunk correlation of 156 weekly returns |
| trend_carry | Half the risk budget to trend, half to carry |
| carry_only | Carry signal alone |
| trend_1m, trend_3m, trend_12m | Each lookback on its own |

Sensitivities that are not counted as competing strategies: costs of a quarter tick and one
tick per trade, and trend on gold, silver, copper and platinum only (the metals check).

## 4. Carry definition

Carry = log(generic 1 / generic 2) x 12 / months between the two contracts, using month-end
unadjusted prices. Contract spacing comes from the listed cycle in FUT_GEN_MONTH. The signal is
the sign of carry from the previous month end (the curve at month end t is only known at that
close, so it is used from month t+1).

Curve cleaning, applied before carry is computed:
1. Month ends where generic 1 equals generic 2 are removed (the back month did not trade).
2. Month ends where |log(G2/G1)| exceeds a cap are removed. The cap is 5% for bond futures,
   which removes notional coupon and deliverable changes (US Treasuries in late 1999 and
   2014 to 2015, Long Gilt in 1997 to 1998 and 2011, Canada 10y in 2000) and illiquid
   back-month prints. It is 10% for equity indices and 15% for currencies, which removes bad
   prints (TSX 60 in late 2000) but keeps genuine crisis-era carry (KOSPI in 1997, the peso
   from 1995 to 1999). Commodity spreads are not capped because large seasonal spreads are real.

## 5. Data rules (decided before any backtest)

1. Text such as "#N/A N/A" is missing and is never filled.
2. Non-positive prices are removed.
3. From the third identical close in a row onwards, prints are removed. The rule only looks
   backwards, so it cannot leak future information. It removes 415 daily prints, mostly in
   slow-moving short-dated bonds, early NZD and the JGB in late 1999. Because a removed print
   equals the one kept before it, month-end prices and signals do not change; the effect is on
   daily volatility, and a market whose price has not moved for more than ten trading days is
   treated as not trading until it moves again.
4. Manual exceptions are listed with a reason in `data/reference/data_exceptions.yaml`
   (currently one, CAC 40 on 4 January 1999).
5. Outliers that match a known market event are kept. The data check finds 120 daily moves
   larger than seven times trailing volatility. Four partly reverse the next day (S&P 500 on
   27 October 1997, TSX 60 on 12 March 2020, JGB on 15 June 2022, natural gas on 14 November
   2018), and all four match known market events, so no move is treated as a bad print.
6. AQMIX and DBMF are price series. Distributions are added back only where the published
   amount matches the price drop on the ex-date. Both are after fees, so they are a live
   comparison, not a like-for-like index.

## 6. Hypotheses and what would count against them

Expectations are written down now so the results can be judged against them, not the reverse.

**H1. Replication.** Over 1990 to 2016 the HOP-style specification has a positive gross Sharpe
ratio and a correlation of at least 0.7 with AQR's published TSMOM factor. A correlation below
0.5 would mean the pipeline does not reproduce the published strategy and must be fixed before
anything else is reported.

**H2. Decay after publication.** The net Sharpe ratio over 2017 to 2026 is lower than over 1990
to 2016. AQR's own factor file already shows this (computed from the AQR data: Sharpe 1.21 for
1985 to 2016 and 0.24 for 2017 to May 2026), so a lower out-of-sample result is the base case,
not a failure. The primary test is whether the locked-period Sharpe is above zero.

**H3. Portfolio construction.** The correlation-aware scheme has a higher in-sample Sharpe ratio
than equal risk per market, by less than 0.2. A larger gap would be treated as likely overfit.

**H4. Carry.** Trend and carry returns have a correlation below 0.3, and the combined portfolio
has a Sharpe ratio at least as high as trend alone in sample.

**H5. Costs.** Half-tick costs reduce the Sharpe ratio by less than 0.15. A larger fall would
mean the 1-month signal is too expensive to trade.

**H6. Crisis behaviour.** Trend is positive in most of the ten worst 60/40 drawdowns covered by
the data and positive over calendar 2022.

**H7. 60/40 overlay.** Moving 20% of a 60/40 portfolio into trend reduces its maximum drawdown.

**H8. Weak spots.** Trend loses money or is flat in the quietest fifth of months (lowest average
absolute 12-month move per unit of volatility).

**H9. Metals.** Trend on gold, silver, copper and platinum alone has a positive net Sharpe
ratio over the full sample. This is a standalone check on mining-related metals; it is not
compared with any other project inside this repository.

## 7. Sanity bounds

A trend portfolio usually shows a Sharpe ratio of about 0.4 to 0.8 before costs. If the primary
specification shows more than 1.0 in sample, it is treated as a bug until shown otherwise. The
look-ahead tests in `tests/test_lookahead.py` must pass before any number is reported.

## 8. Change log

| Date | Change | Reason |
|---|---|---|
| | | |
