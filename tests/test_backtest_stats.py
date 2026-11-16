import json

import numpy as np
import pandas as pd
import pytest

from trendcarry import analysis as A
from trendcarry import backtest, pipeline, stats
from trendcarry import config as C
from trendcarry.config import BASELINE


@pytest.fixture(scope="module")
def res(ds):
    return backtest.backtest(ds, BASELINE)


def test_realised_volatility_is_near_target(res):
    vol = res.returns["gross"].std() * np.sqrt(12)
    assert 0.06 < vol < 0.13
    assert np.allclose(res.diagnostics["exante_vol"].dropna(), 0.10)


def test_costs_reduce_returns_and_can_be_switched_off(ds, res):
    r = res.returns
    assert (r["net"] <= r["gross"] + 1e-15).all() and r["trading_cost"].sum() > 0
    off = backtest.backtest(ds, BASELINE.with_(apply_costs=False)).returns
    assert np.allclose(off["net"], off["gross"])


def test_attribution_adds_up(ds, res):
    assert np.allclose(res.speed_pnl.sum(axis=1), res.returns["gross"])
    assert np.allclose(res.market_pnl.sum(axis=1), res.returns["gross"])
    tc = backtest.backtest(ds, BASELINE.with_(use_carry=True))
    assert np.allclose(tc.sleeve_pnl.sum(axis=1), tc.returns["gross"])


def test_universe_filter_trades_only_metals(ds):
    r = backtest.backtest(ds, BASELINE.with_(markets=("GC",)))
    assert list(r.weights.columns) == ["GC"]


def test_sharpe_and_newey_west():
    x = pd.Series([0.01, -0.005] * 60)
    assert stats.sharpe(x) == pytest.approx(x.mean() / x.std() * np.sqrt(12))
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(500) * 0.02 + 0.002)
    t_plain = y.mean() / (y.std(ddof=0) / np.sqrt(len(y)))
    assert stats.newey_west_tstat(y, lags=0) == pytest.approx(t_plain, rel=1e-6)


def test_regression_recovers_beta():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.standard_normal(300) * 0.04, name="m")
    y = 0.5 * x + rng.standard_normal(300) * 0.01
    out = stats.regress(y, x.to_frame())
    assert out.loc["m", "coef"] == pytest.approx(0.5, abs=0.05) and out.loc["m", "t_nw"] > 10


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.standard_normal(240) * 0.03 + 0.004)
    b = stats.bootstrap_sharpe(x, reps=400, seed=0)
    assert b["ci_low"] < b["sharpe"] < b["ci_high"]


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.standard_normal(300) * 0.03 + 0.004)
    few = stats.deflated_sharpe(x, [0.5, 0.6])
    many = stats.deflated_sharpe(x, list(np.linspace(0.0, 1.0, 50)))
    assert many["deflated_sharpe_prob"] < few["deflated_sharpe_prob"]
    assert stats.expected_max_sharpe(100, 0.01) > stats.expected_max_sharpe(10, 0.01)


def test_drawdown_and_duration():
    x = pd.Series([0.1, -0.5, 0.2, 0.2, 1.0, 0.0])
    assert stats.drawdown(x).min() == pytest.approx(-0.5)
    assert stats.max_dd_duration(x) == 3


def test_distribution_adjustment_checks_the_price_drop():
    idx = pd.bdate_range("2014-11-03", periods=60)
    px = pd.Series(10.0, index=idx)
    px.loc["2014-12-19":] -= 0.9
    ev = pd.DataFrame({"ex_date": ["2014-12-19", "2015-01-06"], "amount": [0.9, 0.5]})
    tr, log = A.adjust_distributions(px, ev)
    assert log.applied.tolist() == [True, False]          # second payout has no matching price drop
    assert tr.iloc[-1] == pytest.approx(10.0, rel=1e-6)   # flat fund once the payout is added back


def test_overlay_with_no_trend_equals_sixty_forty(ds, res):
    tr = res.returns["net"] + res.returns["cash"]
    stats_, paths = A.overlay(tr, ds.bench, res.returns["cash"])
    base = A.sixty_forty(ds.bench).reindex(paths.index)
    assert np.allclose(paths["0% trend"], base)
    assert stats_.loc[0.2, "ann_vol"] < stats_.loc[0.0, "ann_vol"]


def test_locked_period_refuses_changed_settings(ds, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS", tmp_path)
    v = {"baseline": BASELINE}
    pipeline.check_lock(ds, v, None)
    pipeline.check_lock(ds, v, None)                                   # exact reproduction is fine
    changed = {"baseline": BASELINE.with_(target_vol=0.12)}
    with pytest.raises(pipeline.LockedPeriodError):
        pipeline.check_lock(ds, changed, None)
    rec = pipeline.check_lock(ds, changed, "testing a different target")
    assert rec["extra_trials"] == 1
    saved = json.loads((tmp_path / "locked" / "synthetic_lock.json").read_text())
    assert [r["kind"] for r in saved["runs"]] == ["first", "reproduction", "changed"]
