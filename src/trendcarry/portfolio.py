"""Turn signals into notional weights.

Each market's position is signal / ex-ante volatility, so every market carries the same
risk before weighting. The weighting scheme then sets each market's risk budget:

* equal_market: every eligible market gets the same budget
* equal_class: each of the four asset classes gets the same budget, split equally inside
* erc_lw: equal risk contribution using a Ledoit-Wolf shrunk correlation of the positions
  actually held, so crowded, correlated bets are scaled down

Finally the whole book is scaled so its ex-ante volatility, from a rolling 36-month equally
weighted covariance, equals the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as C
from .volatility import monthly_cov, shrunk_correlation


def erc_weights(corr: np.ndarray) -> np.ndarray:
    """Long-only equal risk contribution weights for a correlation matrix (Spinu's convex
    formulation: minimise 0.5 x'Cx - mean(log x))."""
    n = len(corr)
    if n == 1:
        return np.ones(1)
    b = np.full(n, 1.0 / n)

    def f(x):
        cx = corr @ x
        return 0.5 * x @ cx - b @ np.log(x), cx - b / x

    res = minimize(f, np.full(n, 1.0 / np.sqrt(n)), jac=True, method="L-BFGS-B",
                   bounds=[(1e-9, None)] * n, options={"maxiter": 2000, "gtol": 1e-10})
    x = res.x / res.x.sum()
    return x


def risk_budgets(eligible: pd.Index, scheme: str, signs: pd.Series | None = None,
                 weekly_rets: pd.DataFrame | None = None) -> pd.Series:
    if len(eligible) == 0:
        return pd.Series(dtype=float)
    if scheme == "equal_market":
        return pd.Series(1.0 / len(eligible), index=eligible)
    if scheme == "equal_class":
        cls = C.UNIVERSE.loc[eligible, "asset_class"]
        n_cls = cls.nunique()
        return 1.0 / (n_cls * cls.map(cls.value_counts()))
    if scheme == "erc_lw":
        corr = shrunk_correlation(weekly_rets[eligible])
        s = np.sign(signs.reindex(eligible).fillna(1.0)).replace(0, 1.0).values
        signed = corr.values * np.outer(s, s)
        return pd.Series(erc_weights(signed), index=eligible)
    raise ValueError(f"Unknown weighting {scheme!r}")


def scale_to_target(raw: pd.Series, past_monthly: pd.DataFrame, target: float, min_obs: int,
                    max_gross: float) -> tuple[pd.Series, float, float]:
    """Scale raw notional weights so ex-ante annual volatility equals `target`."""
    raw = raw[raw != 0]
    if raw.empty:
        return raw, 0.0, np.nan
    cov = monthly_cov(past_monthly[raw.index], min_obs)
    w = raw.reindex(cov.index).fillna(0.0)
    var = float(w.values @ cov.values @ w.values) * 12.0
    if not np.isfinite(var) or var <= 0:
        return raw * 0.0, 0.0, np.nan
    lev = target / np.sqrt(var)
    scaled = raw * lev
    gross = scaled.abs().sum()
    if gross > max_gross:
        lev *= max_gross / gross
        scaled = raw * lev
    return scaled, lev, np.sqrt(var) * lev
