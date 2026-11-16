"""Trend and carry signals on a monthly rebalance calendar.

Every signal for rebalance month t is computed from prices at the information date i_t,
which is `signal_lag_days` trading days before the rebalance close. With the default lag of
one day, nothing known only at the rebalance close itself is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Calendar:
    months: pd.PeriodIndex          # rebalance months
    rebal: pd.DatetimeIndex         # rebalance close (last trading day of each month)
    info: pd.DatetimeIndex          # information date used for signals


def make_calendar(index: pd.DatetimeIndex, lag_days: int, start=None, end=None) -> Calendar:
    idx = index.sort_values()
    s = pd.Series(idx, index=idx)
    last = s.groupby(idx.to_period("M")).max()
    if start is not None:
        last = last[last.index >= pd.Period(start, "M")]
    if end is not None:
        last = last[last.index <= pd.Period(end, "M")]
    pos = idx.get_indexer(pd.DatetimeIndex(last.values))
    info_pos = np.maximum(pos - lag_days, 0)
    return Calendar(pd.PeriodIndex(last.index, freq="M"), pd.DatetimeIndex(last.values), idx[info_pos])


def prices_asof(prices: pd.DataFrame, dates: pd.DatetimeIndex, max_stale_days: int = 10) -> pd.DataFrame:
    """Last valid price on or before each date, or missing if that price is older than
    `max_stale_days` trading days (a market that has stopped trading is not carried)."""
    filled = prices.ffill()
    pos = pd.DataFrame(np.where(prices.notna(), np.arange(len(prices))[:, None], np.nan),
                       index=prices.index, columns=prices.columns).ffill()
    at = prices.index.get_indexer(dates, method="pad")
    out = filled.iloc[at].copy()
    age = at[:, None] - pos.iloc[at].values
    out = out.where(age <= max_stale_days)
    out.index = dates
    return out


def monthly_returns(prices: pd.DataFrame, cal: Calendar) -> pd.DataFrame:
    """Return over each month, close of previous rebalance to close of this rebalance."""
    p = prices_asof(prices, cal.rebal)
    r = p / p.shift(1) - 1.0
    r.index = cal.months
    return r


def trend_signals(prices: pd.DataFrame, cal: Calendar, lookbacks: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    """Sign of the past k-month excess return, measured between information dates. The
    ratio-adjusted futures return is already an excess return, so no cash rate is deducted."""
    p = prices_asof(prices, cal.info)
    out = {}
    for k in lookbacks:
        past = p / p.shift(k) - 1.0
        sig = np.sign(past)
        sig.index = cal.months
        out[k] = sig
    return out


def carry_signal(carry: pd.DataFrame, cal: Calendar, lag_months: int) -> pd.DataFrame:
    """Sign of annualised carry from the month-end curve `lag_months` earlier."""
    c = carry.copy()
    c.index = c.index + lag_months   # the curve seen at month m is used from month m + lag
    return np.sign(c.reindex(cal.months))
