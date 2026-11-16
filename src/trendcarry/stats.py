"""Performance statistics with honest uncertainty.

Sharpe ratios here are on monthly excess returns, annualised by sqrt(12). Confidence
intervals come from a stationary block bootstrap, which keeps the autocorrelation and
volatility clustering of monthly returns. The deflated Sharpe ratio (Bailey and Lopez de
Prado, 2014) asks whether the best variant beats what the best of N unskilled variants
would show by luck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def sharpe(x: pd.Series, periods: int = 12) -> float:
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def drawdown(x: pd.Series) -> pd.Series:
    nav = (1 + x.fillna(0)).cumprod()
    return nav / nav.cummax() - 1


def max_dd_duration(x: pd.Series) -> int:
    """Longest spell, in months, spent below a previous peak."""
    under = drawdown(x) < 0
    run = best = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def summary(x: pd.Series, periods: int = 12) -> dict:
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return {"months": len(x)}
    nav = (1 + x).cumprod()
    years = len(x) / periods
    return {
        "start": str(x.index[0]), "end": str(x.index[-1]), "months": len(x),
        "ann_return": float(x.mean() * periods),
        "cagr": float(nav.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(periods)),
        "sharpe": sharpe(x, periods),
        "sortino": float(x.mean() / x[x < 0].std(ddof=1) * np.sqrt(periods)) if (x < 0).sum() > 2 else np.nan,
        "max_drawdown": float(drawdown(x).min()),
        "max_dd_months": max_dd_duration(x),
        "skew": float(st.skew(x)), "excess_kurtosis": float(st.kurtosis(x)),
        "hit_rate": float((x > 0).mean()),
        "worst_month": float(x.min()), "best_month": float(x.max()),
        "nw_tstat": newey_west_tstat(x),
    }


def newey_west_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_se(u: np.ndarray, X: np.ndarray, lags: int) -> np.ndarray:
    """HAC standard errors for OLS coefficients with Bartlett weights."""
    n = len(u)
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        g = xu[L:].T @ xu[:-L]
        S += w * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    x = pd.Series(x).dropna().values
    lags = newey_west_lags(len(x)) if lags is None else lags
    X = np.ones((len(x), 1))
    se = newey_west_se(x - x.mean(), X, lags)[0]
    return float(x.mean() / se)


def regress(y: pd.Series, X: pd.DataFrame, lags: int | None = None, periods: int = 12) -> pd.DataFrame:
    """OLS of y on X with an intercept and Newey-West t-statistics. Alpha is annualised."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    Y = df["y"].values
    Xm = np.column_stack([np.ones(len(df)), df.drop(columns="y").values])
    beta, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
    u = Y - Xm @ beta
    lags = newey_west_lags(len(df)) if lags is None else lags
    se = newey_west_se(u, Xm, lags)
    names = ["alpha"] + list(X.columns)
    out = pd.DataFrame({"coef": beta, "t_nw": beta / se}, index=names)
    out.loc["alpha", "coef"] *= periods
    out.attrs["r2"] = 1 - u.var() / Y.var()
    out.attrs["months"] = len(df)
    return out


def stationary_bootstrap_idx(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    p = 1.0 / mean_block
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe(x: pd.Series, reps: int = 5000, mean_block: float = 12, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    x = pd.Series(x).dropna().values
    rng = np.random.default_rng(seed)
    srs = np.empty(reps)
    for r in range(reps):
        s = x[stationary_bootstrap_idx(len(x), mean_block, rng)]
        sd = s.std(ddof=1)
        srs[r] = s.mean() / sd * np.sqrt(12) if sd > 0 else np.nan
    lo, hi = np.nanpercentile(srs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(12)), "ci_low": float(lo), "ci_high": float(hi),
            "p_sharpe_le_0": float(np.nanmean(srs <= 0)), "reps": reps, "mean_block": mean_block}


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """PSR on per-period (not annualised) Sharpe; `kurt` is raw kurtosis (normal = 3)."""
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum per-period Sharpe among `n_trials` unskilled variants."""
    if n_trials < 2:
        return 0.0
    a = st.norm.ppf(1 - 1 / n_trials)
    b = st.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_trials) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(x: pd.Series, trial_sharpes_annual: list[float], periods: int = 12) -> dict:
    """Deflated Sharpe ratio of series `x` given the annualised Sharpe of every variant tried."""
    x = pd.Series(x).dropna()
    sr = x.mean() / x.std(ddof=1)
    trials = np.asarray(trial_sharpes_annual, dtype=float) / np.sqrt(periods)
    n_trials = len(trials)
    var = float(np.var(trials, ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var)
    dsr = probabilistic_sharpe(sr, len(x), float(st.skew(x)), float(st.kurtosis(x, fisher=False)), sr0)
    return {"sharpe_annual": float(sr * np.sqrt(periods)), "n_trials": n_trials,
            "expected_max_sharpe_annual": sr0 * np.sqrt(periods), "deflated_sharpe_prob": dsr,
            "psr_vs_zero": probabilistic_sharpe(sr, len(x), float(st.skew(x)),
                                                float(st.kurtosis(x, fisher=False)), 0.0)}
