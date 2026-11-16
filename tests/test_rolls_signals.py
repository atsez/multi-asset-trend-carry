import numpy as np
import pandas as pd
import pytest

from trendcarry import rolls, signals
from conftest import toy_prices


@pytest.mark.parametrize("cycle,month,expected", [("HMUZ", 2, 3), ("FGHJKMNQUVXZ", 7, 1), ("GJMQVZ", 1, 2),
                                                  ("FHKNQUX", 6, 1), ("FHKNQUX", 8, 2), ("HKNUZ", 11, 3)])
def test_contract_spacing(cycle, month, expected):
    assert rolls.g1_g2_spacing(cycle, month) == expected


def test_carry_sign_and_annualisation():
    m = pd.period_range("2020-01", periods=2, freq="M")
    g1 = pd.DataFrame({"CL": [50.0, 50.0]}, index=m)
    g2 = pd.DataFrame({"CL": [49.0, 51.0]}, index=m)          # backwardation then contango
    specs = pd.DataFrame({"cycle": ["FGHJKMNQUVXZ"]}, index=["CL"])
    c = rolls.annualised_carry(g1, g2, specs)["CL"]
    assert c.iloc[0] > 0 and c.iloc[1] < 0
    assert c.iloc[0] == pytest.approx(np.log(50 / 49) * 12)


def test_calendar_uses_last_trading_day_and_previous_close():
    p = toy_prices(80)
    cal = signals.make_calendar(p.index, lag_days=1)
    assert cal.rebal[0] == pd.Timestamp("2000-01-31")
    assert cal.info[0] == pd.Timestamp("2000-01-28")        # previous trading day
    assert (cal.info < cal.rebal).all()


def test_prices_asof_refuses_stale_prices():
    p = toy_prices(60)
    p.iloc[30:, 0] = np.nan                                   # market A stops trading
    out = signals.prices_asof(p, pd.DatetimeIndex([p.index[35], p.index[55]]), max_stale_days=10)
    assert out.iloc[0]["A"] == p.iloc[29, 0] and np.isnan(out.iloc[1]["A"])


def test_trend_signal_follows_direction():
    idx = pd.bdate_range("2000-01-03", periods=600)
    p = pd.DataFrame({"UP": np.linspace(100, 200, 600), "DOWN": np.linspace(200, 100, 600)}, index=idx)
    cal = signals.make_calendar(idx, 1)
    s = signals.trend_signals(p, cal, (1, 3, 12))
    for k in (1, 3, 12):
        assert (s[k]["UP"].dropna() == 1).all() and (s[k]["DOWN"].dropna() == -1).all()
    assert s[12]["UP"].isna().sum() == 12


def test_carry_signal_lags_by_calendar_month():
    m = pd.period_range("2020-01", periods=4, freq="M")
    carry = pd.DataFrame({"X": [0.1, -0.1, 0.2, -0.2]}, index=m)
    cal = signals.Calendar(m, pd.DatetimeIndex(m.to_timestamp(how="end")), pd.DatetimeIndex(m.to_timestamp(how="end")))
    s = signals.carry_signal(carry, cal, 1)["X"]
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 1 and s.iloc[2] == -1
