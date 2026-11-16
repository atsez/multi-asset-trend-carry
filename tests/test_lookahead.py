"""Look-ahead tests. These must fail loudly if any weight uses information from after the
close it is supposed to be decided on.

The core test scrambles every price after the information date i_t (and every futures curve
from month t on, which is only observed at the month-t close) and checks that all weights up
to and including month t are bit-for-bit unchanged. The canary test plants a leak and checks
that the same test catches it, so the test is known to have teeth.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from trendcarry import backtest, signals
from trendcarry.config import BASELINE


def scramble_after(ds, info_date, month, seed=0):
    rng = np.random.default_rng(seed)
    d2 = copy.deepcopy(ds)
    fut = d2.prices.index > info_date
    noise = np.exp(np.cumsum(rng.standard_normal((fut.sum(), d2.prices.shape[1])) * 0.05, axis=0))
    d2.prices.loc[fut] = d2.prices.loc[fut].values * noise * rng.uniform(0.5, 2.0)
    for g in (d2.g1, d2.g2):
        later = g.index >= month
        g.loc[later] = g.loc[later].values * rng.uniform(0.7, 1.3, size=g.loc[later].shape)
    return d2


def assert_no_lookahead(ds, params, months):
    base = backtest.backtest(ds, params).weights
    cal = backtest.prepare(ds, params).cal
    for m in months:
        j = cal.months.get_loc(pd.Period(m, "M"))
        d2 = scramble_after(ds, cal.info[j], cal.months[j], seed=j)
        w2 = backtest.backtest(d2, params).weights
        pd.testing.assert_frame_equal(base.loc[:cal.months[j]], w2.loc[:cal.months[j]],
                                      check_exact=True, obj=f"weights up to {m}")


MONTHS = ["2003-06", "2006-01", "2008-09"]


@pytest.mark.parametrize("weighting", ["equal_market", "equal_class", "erc_lw"])
def test_no_lookahead_trend(ds, weighting):
    assert_no_lookahead(ds, BASELINE.with_(weighting=weighting, end="2009-12-31"), MONTHS)


def test_no_lookahead_trend_and_carry(ds):
    assert_no_lookahead(ds, BASELINE.with_(use_carry=True, end="2009-12-31"), MONTHS)


def test_canary_planted_leak_is_caught(ds, monkeypatch):
    """Swap in a trend signal that peeks at the next month's price. The test must fail."""
    real = signals.trend_signals

    def leaky(prices, cal, lookbacks):
        out = real(prices, cal, lookbacks)
        future = signals.prices_asof(prices, cal.rebal).shift(-1)
        now = signals.prices_asof(prices, cal.info)
        peek = np.sign(future.values / now.values - 1)
        return {k: pd.DataFrame(peek, index=cal.months, columns=prices.columns) for k in out}

    monkeypatch.setattr(signals, "trend_signals", leaky)
    with pytest.raises(AssertionError):
        assert_no_lookahead(ds, BASELINE.with_(end="2009-12-31"), MONTHS[:1])


def test_weights_earn_next_month_return_only(ds):
    p = BASELINE.with_(end="2009-12-31")
    inp = backtest.prepare(ds, p)
    res = backtest.run(inp, p)
    t = pd.Period("2005-03", "M")
    expected = (res.weights.loc[t] * inp.monthly.loc[t + 1].fillna(0)).sum()
    assert res.returns.loc[t + 1, "gross"] == pytest.approx(expected, abs=1e-14)


def test_leak_free_strategy_has_no_edge_on_pure_noise(null_ds):
    """With no trend in the data, a leak-free strategy cannot earn a meaningful Sharpe."""
    r = backtest.backtest(null_ds, BASELINE).returns["gross"]
    sr = r.mean() / r.std() * np.sqrt(12)
    assert abs(sr) < 0.5, f"Sharpe {sr:.2f} on pure noise suggests a leak"
