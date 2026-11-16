import numpy as np
import pandas as pd
import pytest

from trendcarry import costs, portfolio
from trendcarry.volatility import nearest_psd, ewma_vol, daily_returns
from conftest import toy_prices


def test_erc_gives_equal_risk_contributions():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 5))
    a[:, 1] += a[:, 0]                                   # correlated pair
    corr = np.corrcoef(a.T)
    x = portfolio.erc_weights(corr)
    rc = x * (corr @ x)
    assert np.allclose(rc / rc.sum(), 0.2, atol=1e-4)
    assert x[0] < x[2] and x[1] < x[2]                  # correlated markets get less


def test_equal_class_budgets():
    elig = pd.Index(["ES", "GX", "TY", "EC", "JY", "CL", "GC", "C"])
    b = portfolio.risk_budgets(elig, "equal_class")
    assert b.sum() == pytest.approx(1.0)
    assert b[["ES", "GX"]].sum() == pytest.approx(0.25) and b["TY"] == pytest.approx(0.25)


def test_scale_hits_target_exactly_and_respects_leverage_cap():
    rng = np.random.default_rng(1)
    past = pd.DataFrame(rng.standard_normal((36, 3)) * 0.03, columns=list("ABC"))
    raw = pd.Series([1.0, -0.5, 0.2], index=list("ABC"))
    w, lev, vol = portfolio.scale_to_target(raw, past, 0.10, 24, 100.0)
    cov = past.cov().values
    assert np.sqrt(w.values @ cov @ w.values * 12) == pytest.approx(0.10, rel=1e-9)
    w2, _, vol2 = portfolio.scale_to_target(raw, past, 0.10, 24, 0.5)
    assert w2.abs().sum() == pytest.approx(0.5) and vol2 < 0.10


def test_nearest_psd_repairs_a_broken_matrix():
    m = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(m).min() < 0
    assert np.linalg.eigvalsh(nearest_psd(m)).min() >= -1e-12


def test_ewma_vol_is_annualised_and_skips_gaps():
    p = toy_prices(600, codes=("A",), seed=2)
    v = ewma_vol(daily_returns(p))["A"].dropna()
    assert 0.12 < v.iloc[-1] < 0.20                       # daily 1% is about 16% a year
    p.iloc[300:310] = np.nan
    r = daily_returns(p)["A"]
    assert r.iloc[300:310].isna().all() and np.isfinite(r.iloc[310])


def test_cost_per_unit_uses_previous_month_price():
    m = pd.period_range("2020-01", periods=3, freq="M")
    g1 = pd.DataFrame({"X": [100.0, 200.0, 400.0]}, index=m)
    specs = pd.DataFrame({"tick_size": [1.0], "cycle": ["HMUZ"]}, index=["X"])
    c = costs.cost_per_unit(specs, g1, 0.5, m)["X"]
    assert c.loc[m[1]] == pytest.approx(0.5 * 1 / 100) and c.loc[m[2]] == pytest.approx(0.5 * 1 / 200)


def test_trading_and_roll_costs():
    m = pd.period_range("2020-01", periods=2, freq="M")
    w = pd.DataFrame({"X": [1.0, -1.0]}, index=m)
    prev = pd.DataFrame({"X": [0.0, 1.0]}, index=m)
    unit = pd.DataFrame({"X": [0.001, 0.001]}, index=m)
    specs = pd.DataFrame({"cycle": ["HMUZ"]}, index=["X"])
    c = costs.monthly_costs(w, prev, unit, specs, include_rolls=True)
    assert c.trading_cost.tolist() == pytest.approx([0.001, 0.002])
    assert c.roll_cost.iloc[0] == pytest.approx(1.0 * 2 * 0.001 * 4 / 12)


def test_fees_and_high_water_mark():
    idx = pd.period_range("2020-01", periods=4, freq="M")
    x = pd.Series([0.10, -0.10, 0.05, 0.10], index=idx)
    cash = pd.Series(0.0, index=idx)
    assert costs.apply_fees(x, cash, 0, 0).equals(x)
    f = costs.apply_fees(x, cash, 0.02, 0.20)
    assert (f < x).all()
    nofee_perf = costs.apply_fees(x, cash, 0.0, 0.20)
    assert nofee_perf.iloc[2] == pytest.approx(0.05)     # still below the high-water mark: no performance fee
