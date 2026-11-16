"""Ex-ante volatility, the monthly covariance used for the portfolio volatility target, and
the shrunk correlation used for correlation-aware weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns computed on each market's own valid prints, so a return across a
    gap or a removed stale print is counted once, on the day the price updates."""
    out = {}
    for c in prices.columns:
        s = prices[c].dropna()
        out[c] = s.pct_change()
    return pd.DataFrame(out).reindex(prices.index)


def ewma_vol(rets: pd.DataFrame, com: float = 60.0, min_obs: int = 60, floor: float = 0.01) -> pd.DataFrame:
    """Annualised exponentially weighted volatility, using returns up to and including each
    date. Missing days are skipped rather than counted as zero."""
    out = {}
    for c in rets.columns:
        r = rets[c].dropna()
        v = r.ewm(com=com, min_periods=min_obs).std() * np.sqrt(TRADING_DAYS)
        out[c] = v.clip(lower=floor)
    return pd.DataFrame(out).reindex(rets.index).ffill()


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    cov = (cov + cov.T) / 2
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, eps * max(w.max(), eps), None)
    return (v * w) @ v.T


def monthly_cov(monthly_rets: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Equally weighted pairwise covariance of monthly returns, repaired to be positive
    semi-definite when pairwise estimation breaks that."""
    cov = monthly_rets.cov(min_periods=min_obs)
    good = cov.index[np.isfinite(np.diag(cov.values))]
    cov = cov.loc[good, good]
    vals = cov.values.copy()
    # a pair with too little overlap gets zero correlation rather than being dropped
    vals[~np.isfinite(vals)] = 0.0
    return pd.DataFrame(nearest_psd(vals), index=good, columns=good)


def shrunk_correlation(weekly_rets: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk correlation of standardised weekly returns. Missing weeks count as
    zero after standardising, which pulls short-history pairs towards zero correlation."""
    x = weekly_rets.copy()
    x = (x - x.mean()) / x.std(ddof=1)
    x = x.fillna(0.0)
    cov = LedoitWolf(assume_centered=True).fit(x.values).covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=x.columns, columns=x.columns)
