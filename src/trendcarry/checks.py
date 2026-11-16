"""Data checks and cleaning rules.

Checks only report. Cleaning applies the rules written in HYPOTHESES.md plus the manual
exceptions in data/reference/data_exceptions.yaml. Missing data is never filled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config as C
from .rolls import log_spread


# ---------------------------------------------------------------- checks

def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in prices.columns:
        s = prices[code].dropna()
        if s.empty:
            rows.append(dict(code=code, n=0))
            continue
        bd = pd.bdate_range(s.index[0], s.index[-1])
        gaps = s.index.to_series().diff().dt.days
        rows.append(dict(code=code, name=C.UNIVERSE.loc[code, "name"], first=s.index[0].date(),
                         last=s.index[-1].date(), n=len(s),
                         missing_bdays_pct=round(100 * len(bd.difference(s.index)) / len(bd), 2),
                         longest_gap_days=int(gaps.max()) if len(s) > 1 else 0,
                         longest_gap_end=gaps.idxmax().date() if len(s) > 1 else None,
                         weekend_rows=int((s.index.dayofweek >= 5).sum()),
                         nonpositive=int((s <= 0).sum())))
    return pd.DataFrame(rows).set_index("code")


def stale_runs(s: pd.Series, min_len: int = 3) -> pd.DataFrame:
    """Runs of at least `min_len` identical consecutive prints."""
    s = s.dropna()
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "length", "price"])
    new_run = s.ne(s.shift())
    run_id = new_run.cumsum()
    idx = s.index.to_series(index=s.index)
    out = pd.DataFrame({"start": idx.groupby(run_id).min(), "end": idx.groupby(run_id).max(),
                        "length": s.groupby(run_id).size(), "price": s.groupby(run_id).first()})
    return out[out.length >= min_len].reset_index(drop=True)


def all_stale_runs(prices: pd.DataFrame, min_len: int = 3) -> pd.DataFrame:
    parts = []
    for code in prices.columns:
        r = stale_runs(prices[code], min_len)
        if len(r):
            parts.append(r.assign(code=code))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["start", "end", "length", "price", "code"])


def outliers(prices: pd.DataFrame, z: float = 7.0, window: int = 60) -> pd.DataFrame:
    """Daily returns larger than `z` times trailing volatility. A move that reverses the next
    day is the signature of a bad print or a roll error, so that is flagged separately."""
    rows = []
    for code in prices.columns:
        s = prices[code].dropna()
        lr = np.log(s).diff().dropna()
        vol = lr.rolling(window, min_periods=window // 2).std().shift(1)
        zz = lr / vol
        nxt = lr.shift(-1)
        for d in zz.index[zz.abs() > z]:
            rows.append(dict(code=code, date=d.date(), ret_pct=100 * np.expm1(lr[d]), z=zz[d],
                             next_ret_pct=100 * np.expm1(nxt[d]) if pd.notna(nxt[d]) else np.nan,
                             reverts=bool(pd.notna(nxt[d]) and abs(lr[d] + nxt[d]) < 0.3 * abs(lr[d]))))
    return pd.DataFrame(rows)


def roll_consistency(prices: pd.DataFrame, g1: pd.DataFrame) -> pd.DataFrame:
    """Compare the adjusted series with unadjusted generic 1 at month ends. log(adj/g1) only
    moves when a contract rolls, by the roll gap. A move with no plausible roll gap, or an
    adjusted/g1 ratio far from 1 at the latest month end, points to a broken adjustment."""
    me = prices.groupby(prices.index.to_period("M")).last()
    rows = []
    for code in prices.columns:
        j = pd.concat([me[code], g1.get(code)], axis=1, keys=["adj", "g1"]).dropna()
        if len(j) < 3:
            rows.append(dict(code=code, months=len(j)))
            continue
        step = np.log(j.adj / j.g1).diff().dropna()
        rows.append(dict(code=code, months=len(j), last_ratio=j.adj.iloc[-1] / j.g1.iloc[-1],
                         median_abs_step_pct=100 * step.abs().median(),
                         max_abs_step_pct=100 * step.abs().max(), max_step_month=str(step.abs().idxmax())))
    return pd.DataFrame(rows).set_index("code")


def curve_checks(g1: pd.DataFrame, g2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in g1.columns:
        j = pd.concat([g1[code], g2[code]], axis=1, keys=["g1", "g2"]).dropna()
        if j.empty:
            rows.append(dict(code=code, months=0))
            continue
        sp = 100 * np.log(j.g2 / j.g1)
        rows.append(dict(code=code, months=len(j), first=str(j.index[0]), equal_months=int((j.g1 == j.g2).sum()),
                         nonpositive=int((j <= 0).sum().sum()), spread_median_pct=sp.median(),
                         spread_min_pct=sp.min(), spread_min_month=str(sp.idxmin()),
                         spread_max_pct=sp.max(), spread_max_month=str(sp.idxmax())))
    return pd.DataFrame(rows).set_index("code")


def run_all(ds, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    res = {
        "coverage": coverage(ds.prices),
        "stale_runs": all_stale_runs(ds.prices),
        "outliers": outliers(ds.prices),
        "roll_consistency": roll_consistency(ds.prices, ds.g1),
        "curves": curve_checks(ds.g1, ds.g2),
        "benchmarks": _bench_coverage(ds.bench),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            v.to_csv(out_dir / f"{k}.csv")
    return res


def _bench_coverage(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in b.columns:
        s = b[c].dropna()
        rows.append(dict(series=c, first=s.index[0].date() if len(s) else None,
                         last=s.index[-1].date() if len(s) else None, n=len(s),
                         zeros=int((s == 0).sum()), negatives=int((s < 0).sum())))
    return pd.DataFrame(rows).set_index("series")


# ---------------------------------------------------------------- cleaning

def load_exceptions(path: Path | None = None) -> dict:
    path = path or (C.DATA_REFERENCE / "data_exceptions.yaml")
    if not Path(path).exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def clean_prices(prices: pd.DataFrame, min_stale: int = 3, exceptions: dict | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned prices and a log of every value removed.

    Rules: non-positive prices are removed; from the `min_stale`-th identical close in a row
    onwards, prints are removed (the rule only looks backwards, so whether today's price is
    kept never depends on later prices); manual exceptions are removed.
    """
    out = prices.copy()
    log = []
    for code in out.columns:
        s = out[code]
        bad = s <= 0
        for d in s.index[bad.fillna(False)]:
            log.append((code, d, "non-positive"))
        out.loc[bad.fillna(False), code] = np.nan
        v = out[code].dropna()
        same = v.eq(v.shift())
        streak = same.groupby((~same).cumsum()).cumsum()      # repeats so far in this run
        drop = v.index[streak >= min_stale - 1]
        out.loc[drop, code] = np.nan
        log.extend((code, d, f"stale: identical close number {int(streak[d]) + 1} in a row") for d in drop)
    for e in (exceptions or {}).get("prices", []):
        for d in e["dates"]:
            d = pd.Timestamp(d)
            if e["market"] in out.columns and d in out.index and pd.notna(out.at[d, e["market"]]):
                out.at[d, e["market"]] = np.nan
                log.append((e["market"], d, "manual: " + e["reason"]))
    return out, pd.DataFrame(log, columns=["code", "date", "reason"])


def clean_curves(g1: pd.DataFrame, g2: pd.DataFrame, caps: dict[str, float]
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove month-ends where the back month did not trade (g1 == g2), non-positive prices,
    and spreads beyond a per asset class cap. The cap removes contract changes (bond notional
    coupon switches) and bad back-month prints in markets whose true spread is small."""
    g1, g2 = g1.copy(), g2.copy()
    log = []
    sp = log_spread(g1, g2)
    for code in g1.columns:
        cls = C.UNIVERSE.loc[code, "asset_class"]
        cap = caps.get(cls)
        m = (g1[code] == g2[code]) | (g1[code] <= 0) | (g2[code] <= 0)
        reasons = {d: "g1 equals g2" for d in g1.index[m.fillna(False)]}
        if cap is not None:
            big = sp[code].abs() > cap
            reasons.update({d: f"spread beyond {cap:.0%}" for d in g1.index[big.fillna(False)]})
        for d, why in reasons.items():
            g1.at[d, code] = np.nan
            g2.at[d, code] = np.nan
            log.append((code, d, why))
    return g1, g2, pd.DataFrame(log, columns=["code", "month", "reason"])
