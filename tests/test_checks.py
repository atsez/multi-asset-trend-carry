import numpy as np
import pandas as pd

from trendcarry import checks


def test_stale_runs_found_and_cleaned_causally():
    s = pd.Series([1.0, 2, 2, 2, 2, 3, 4, 4, 5], index=pd.bdate_range("2020-01-01", periods=9))
    runs = checks.stale_runs(s, 3)
    assert len(runs) == 1 and runs.length[0] == 4
    clean, log = checks.clean_prices(s.to_frame("X"))
    # third and fourth identical closes removed; the decision uses only earlier prices
    assert clean["X"].isna().sum() == 2 and clean["X"].iloc[1] == 2.0 and clean["X"].iloc[2] == 2.0
    assert clean["X"].iloc[6] == 4.0 and clean["X"].iloc[7] == 4.0     # a pair of equal closes is kept


def test_nonpositive_prices_removed():
    s = pd.DataFrame({"X": [1.0, -2.0, 0.0, 3.0]}, index=pd.bdate_range("2020-01-01", periods=4))
    clean, log = checks.clean_prices(s)
    assert clean["X"].isna().sum() == 2 and (log.reason == "non-positive").sum() == 2


def test_manual_exception_only_touches_listed_market():
    df = pd.DataFrame({"CF": [1.0, 2.0], "Z": [1.0, 2.0]}, index=pd.to_datetime(["1999-01-01", "1999-01-04"]))
    exc = {"prices": [{"market": "CF", "dates": ["1999-01-04"], "reason": "test"}]}
    clean, _ = checks.clean_prices(df, exceptions=exc)
    assert np.isnan(clean.at[pd.Timestamp("1999-01-04"), "CF"]) and clean.at[pd.Timestamp("1999-01-04"), "Z"] == 2.0


def test_outlier_spike_is_flagged_as_reverting(sim):
    o = checks.outliers(sim["prices"])
    kc = o[(o.code == "KC")]
    assert len(kc) >= 1 and kc.reverts.iloc[0]


def test_injected_faults_are_all_detected(sim):
    p = sim["prices"]
    runs = checks.all_stale_runs(p)
    assert ((runs.code == "JB") & (runs.length >= 16)).any()
    cov = checks.coverage(p)
    assert cov.loc["LC", "longest_gap_days"] >= 12
    roll = checks.roll_consistency(p, sim["g1"])
    assert roll.loc["GC", "max_abs_step_pct"] > 10 and roll.loc["GC", "max_step_month"].startswith("2005-0")


def test_curve_cleaning_removes_untraded_and_contract_changes_but_not_commodity_spreads(sim):
    g1, g2, log = checks.clean_curves(sim["g1"], sim["g2"], {"Equity": 0.10, "Bond": 0.05, "FX": 0.15})
    assert np.isnan(g1.loc[pd.Period("2004-05", "M"), "JY"])            # g1 == g2
    assert np.isnan(g1.loc[pd.Period("1999-11", "M"), "TY"])            # 11% contract change
    assert set(log.code) <= {"JY", "TY", "ES", "GX", "NK", "RX", "JB", "EC", "AD"}  # commodities never capped
