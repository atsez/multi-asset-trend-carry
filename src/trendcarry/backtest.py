"""Monthly backtest engine.

Timing, for rebalance month t:
  1. Signals and volatilities use prices up to the information date i_t.
  2. The covariance for the volatility target uses monthly returns up to month t-1.
  3. Weights are traded at the close of the last trading day of month t.
  4. They earn the return of month t+1. Costs of trading at t are charged to month t+1.
Returns are excess returns (futures are funded), so cash is added only for fees and the
60/40 overlay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import checks, costs, portfolio, rolls, signals
from .config import Params
from .volatility import daily_returns, ewma_vol


@dataclass
class Inputs:
    prices: pd.DataFrame
    cal: signals.Calendar
    monthly: pd.DataFrame          # return over each month, indexed by month
    vol_info: pd.DataFrame         # ex-ante vol at each information date, indexed by month
    fresh: pd.DataFrame            # True if price at i_t is recent enough to trade
    weekly: pd.DataFrame           # weekly returns for the correlation-aware scheme
    trend: dict[int, pd.DataFrame]
    carry_sig: pd.DataFrame | None
    carry: pd.DataFrame | None
    unit_cost: pd.DataFrame
    specs: pd.DataFrame
    cash: pd.Series                # monthly T-bill return, indexed by month
    cleaning_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    curve_log: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Result:
    params: Params
    weights: pd.DataFrame          # formed at end of month t (index t)
    returns: pd.DataFrame          # holding month t+1: gross, costs, net, net_after_fees
    market_pnl: pd.DataFrame       # holding month contributions by market
    speed_pnl: pd.DataFrame        # holding month contributions by trend lookback
    sleeve_pnl: pd.DataFrame       # trend versus carry
    diagnostics: pd.DataFrame      # leverage, ex-ante vol, number of markets, gross exposure


def cash_returns(tbill_yield: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Monthly cash return: T-bill yield (percent) at the previous month end over 12."""
    y = tbill_yield.dropna()
    me = y.groupby(y.index.to_period("M")).last() / 100.0 / 12.0
    me.index = me.index + 1
    return me.reindex(months).fillna(0.0)


def prepare(ds, params: Params, clean: bool = True) -> Inputs:
    prices = ds.prices.loc[:params.end]
    if params.markets is not None:
        prices = prices[[c for c in params.markets if c in prices.columns]]
    log = pd.DataFrame()
    if clean:
        exc = checks.load_exceptions() if ds.source == "real" else {}
        prices, log = checks.clean_prices(prices, exceptions=exc)
    cal = signals.make_calendar(prices.dropna(how="all").index, params.signal_lag_days, end=params.end)
    monthly = signals.monthly_returns(prices, cal)
    vol = ewma_vol(daily_returns(prices), params.vol_com_days, params.vol_min_obs, params.vol_floor_annual)
    at = vol.index.get_indexer(cal.info, method="pad")
    vol_info = vol.iloc[at].set_axis(cal.months)
    fresh = signals.prices_asof(prices, cal.info).notna().set_axis(cal.months)
    wk = prices.ffill(limit=5).resample("W-FRI").last()
    weekly = wk / wk.shift(1) - 1.0
    trend = signals.trend_signals(prices, cal, params.lookbacks_months)
    cols = prices.columns
    g1, g2, curve_log = checks.clean_curves(ds.g1[cols], ds.g2[cols], params.carry_spread_cap)
    carry = rolls.annualised_carry(g1, g2, ds.specs.loc[cols]) if g1.notna().any().any() else None
    carry_sig = signals.carry_signal(carry, cal, params.carry_lag_months) if carry is not None else None
    unit_cost = costs.cost_per_unit(ds.specs.loc[cols], ds.g1[cols], params.cost_ticks_per_trade, cal.months)
    cash = cash_returns(ds.bench["TBILL"], cal.months) if "TBILL" in ds.bench else pd.Series(0.0, index=cal.months)
    return Inputs(prices, cal, monthly, vol_info, fresh, weekly, trend, carry_sig, carry,
                  unit_cost[prices.columns], ds.specs.loc[prices.columns], cash, log, curve_log)


def run(inp: Inputs, params: Params) -> Result:
    months = inp.cal.months
    codes = inp.prices.columns
    ks = params.lookbacks_months
    mix = params.trend_carry_mix if params.use_carry else 0.0
    hist = inp.monthly.notna().cumsum()

    W = pd.DataFrame(0.0, index=months, columns=codes)
    speed_w = {k: pd.DataFrame(0.0, index=months, columns=codes) for k in ks}
    carry_w = pd.DataFrame(0.0, index=months, columns=codes)
    diag = []
    for j, t in enumerate(months):
        if j == 0:
            continue
        sig_k = {k: inp.trend[k].loc[t] for k in ks}
        ok = pd.concat(sig_k.values(), axis=1).notna().all(axis=1)
        ok &= inp.vol_info.loc[t].notna() & inp.fresh.loc[t]
        ok &= hist.iloc[j - 1] >= params.min_history_months
        elig = codes[ok.values]
        if len(elig) == 0:
            diag.append(dict(month=t, n_markets=0))
            continue
        vol = inp.vol_info.loc[t, elig]
        s_trend = sum(sig_k[k][elig] for k in ks) / len(ks)
        s_carry = pd.Series(0.0, index=elig)
        if params.use_carry and inp.carry_sig is not None:
            s_carry = inp.carry_sig.loc[t, elig].fillna(0.0)
        s_total = (1 - mix) * s_trend + mix * s_carry
        weekly = inp.weekly.loc[:inp.cal.info[j]].iloc[-params.corr_window_weeks:]
        budget = portfolio.risk_budgets(elig, params.weighting, s_total, weekly)
        raw = budget * s_total / vol
        past = inp.monthly.iloc[max(0, j - params.cov_window_months):j]
        w, lev, exante = portfolio.scale_to_target(raw, past, params.target_vol,
                                                   params.cov_min_months, params.max_gross_leverage)
        W.loc[t, w.index] = w.values
        unit = budget * lev / vol
        for k in ks:
            speed_w[k].loc[t, elig] = (unit * (1 - mix) * sig_k[k][elig] / len(ks)).values
        carry_w.loc[t, elig] = (unit * mix * s_carry).values
        diag.append(dict(month=t, n_markets=len(elig), leverage=lev, exante_vol=exante,
                         gross=w.abs().sum(), net_long=w.sum()))

    nxt = inp.monthly.shift(-1).fillna(0.0)            # month t+1 return aligned to t
    hold = months + 1

    def pnl(w):
        return (w * nxt).set_axis(hold)

    market_pnl = pnl(W)
    gross = market_pnl.sum(axis=1)
    # positions drift with returns before the next rebalance
    port_prev = pd.Series(np.r_[0.0, gross.values[:-1]], index=months)  # return over month t
    drifted = (W.shift(1).fillna(0.0) * (1 + inp.monthly.fillna(0.0))).div(1 + port_prev, axis=0)
    c = costs.monthly_costs(W, drifted, inp.unit_cost.reindex(months).fillna(0.0), inp.specs,
                            params.include_roll_costs).set_axis(hold)
    if not params.apply_costs:
        c[["trading_cost", "roll_cost"]] = 0.0
    net = gross - c["trading_cost"] - c["roll_cost"]
    cash = inp.cash.reindex(hold).fillna(0.0)
    after_fees = costs.apply_fees(net, cash, params.mgmt_fee, params.perf_fee)
    returns = pd.DataFrame({"gross": gross, **c, "net": net, "net_after_fees": after_fees, "cash": cash})

    speed_pnl = pd.DataFrame({f"{k}m": pnl(speed_w[k]).sum(axis=1) for k in ks})
    sleeve_pnl = pd.DataFrame({"trend": speed_pnl.sum(axis=1), "carry": pnl(carry_w).sum(axis=1)})
    diagnostics = pd.DataFrame(diag).set_index("month") if diag else pd.DataFrame()

    # drop months before the first live position and the final month with no return yet
    live = W.abs().sum(axis=1) > 0
    if live.any():
        first = hold[live.values.argmax()]
        last = hold[-2] if len(hold) > 1 else hold[-1]
        sl = slice(first, last)
        returns, market_pnl = returns.loc[sl], market_pnl.loc[sl]
        speed_pnl, sleeve_pnl = speed_pnl.loc[sl], sleeve_pnl.loc[sl]
    return Result(params, W, returns, market_pnl, speed_pnl, sleeve_pnl, diagnostics)


def backtest(ds, params: Params) -> Result:
    return run(prepare(ds, params), params)
