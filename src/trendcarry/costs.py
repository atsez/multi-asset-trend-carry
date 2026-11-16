"""Transaction, roll and fee costs.

Trading cost per unit of notional traded is `ticks` x tick size / price, so a half-tick cost
on a market whose tick is 1 basis point of price is 0.5 basis points per unit traded. Price is
the unadjusted first generic at the previous month end (same quoting units as the tick).
Each roll closes one contract and opens the next, so it trades twice the position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rolls import rolls_per_month


def cost_per_unit(specs: pd.DataFrame, g1: pd.DataFrame, ticks: float, months: pd.PeriodIndex) -> pd.DataFrame:
    """Cost as a fraction of notional for each market and rebalance month."""
    px = g1.reindex(g1.index.union(months)).sort_index().ffill()
    px.index = px.index + 1                       # previous month end, known before trading
    px = px.reindex(months)
    rel_tick = specs["tick_size"].astype(float) / px
    # before a market's curve data starts, fall back to its median relative tick
    rel_tick = rel_tick.fillna(rel_tick.median())
    return ticks * rel_tick


def relative_tick_check(specs: pd.DataFrame, g1: pd.DataFrame) -> pd.DataFrame:
    """Median tick as a fraction of price. Values outside 0.1 to 20 basis points usually mean
    the tick and the price are quoted in different units."""
    rel = (specs["tick_size"].astype(float) / g1).median() * 1e4
    return pd.DataFrame({"median_tick_bp": rel, "plausible": rel.between(0.1, 20)})


def monthly_costs(weights: pd.DataFrame, drifted_prev: pd.DataFrame, unit_cost: pd.DataFrame,
                  specs: pd.DataFrame, include_rolls: bool) -> pd.DataFrame:
    trade = (weights - drifted_prev).abs()
    trading = (trade * unit_cost).sum(axis=1)
    if include_rolls:
        rpm = rolls_per_month(specs).reindex(weights.columns)
        rolling = (weights.abs() * 2.0 * unit_cost * rpm).sum(axis=1)
    else:
        rolling = trading * 0.0
    return pd.DataFrame({"turnover": trade.sum(axis=1), "trading_cost": trading, "roll_cost": rolling})


def apply_fees(excess: pd.Series, cash: pd.Series, mgmt: float, perf: float) -> pd.Series:
    """Management fee each month and a performance fee on gains above the high-water mark,
    both charged on total return (excess plus cash). Returns the excess return after fees."""
    if mgmt == 0 and perf == 0:
        return excess
    cash = cash.reindex(excess.index).fillna(0.0)
    nav, hwm, out = 1.0, 1.0, []
    for r, rf in zip(excess.values, cash.values):
        gross_nav = nav * (1 + r + rf) * (1 - mgmt / 12)
        fee = perf * max(gross_nav - hwm, 0.0)
        new = gross_nav - fee
        hwm = max(hwm, new)
        out.append(new / nav - 1 - rf)
        nav = new
    return pd.Series(out, index=excess.index)
