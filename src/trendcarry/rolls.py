"""Contract cycles, carry from the futures curve and roll frequency.

FUT_GEN_MONTH gives the listed cycle as month letters (e.g. HMUZ). At a month end the first
generic is taken to be the first cycle month after the current month and the second generic
the next cycle month after that, so the spacing between them follows the cycle. This is an
approximation: a few energy contracts expire a month earlier, which does not change the
spacing for monthly cycles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_CODES = dict(zip("FGHJKMNQUVXZ", range(1, 13)))


def cycle_months(cycle: str) -> list[int]:
    months = sorted({MONTH_CODES[c] for c in str(cycle).strip().upper() if c in MONTH_CODES})
    if not months:
        raise ValueError(f"Bad contract cycle {cycle!r}")
    return months


def _next_in_cycle(month: int, months: list[int]) -> int:
    for m in months:
        if m > month:
            return m
    return months[0] + 12


def g1_g2_spacing(cycle: str, calendar_month: int) -> int:
    """Months between the first and second generic contract at the end of `calendar_month`."""
    months = cycle_months(cycle)
    first = (_next_in_cycle(calendar_month, months) - 1) % 12 + 1
    return _next_in_cycle(first, months) - first


def spacing_table(specs: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=months, columns=specs.index, dtype=float)
    for code, cyc in specs["cycle"].items():
        lookup = {m: g1_g2_spacing(cyc, m) for m in range(1, 13)}
        out[code] = [lookup[p.month] for p in months]
    return out


def log_spread(g1: pd.DataFrame, g2: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(g2 / g1)


def annualised_carry(g1: pd.DataFrame, g2: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Carry = log(g1 / g2) scaled to a year. Positive means backwardation, which a long
    position earns as the contract rolls up the curve."""
    sp = spacing_table(specs.loc[g1.columns], g1.index)
    return (-log_spread(g1, g2)) * 12.0 / sp


def rolls_per_month(specs: pd.DataFrame) -> pd.Series:
    """Average number of rolls per month for the adjusted series (one per listed contract)."""
    return specs["cycle"].map(lambda c: len(cycle_months(c)) / 12.0)
