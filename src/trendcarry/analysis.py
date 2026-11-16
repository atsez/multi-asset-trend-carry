"""Attribution, stress tests, the 60/40 overlay and the replication comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .stats import drawdown, max_dd_duration, regress, sharpe, summary

STRESS_WINDOWS = {
    "GFC 2008": ("2008-01", "2008-12"),
    "GFC equity drawdown": ("2007-11", "2009-02"),
    "Covid crash": ("2020-02", "2020-03"),
    "2022 bond and equity sell-off": ("2022-01", "2022-12"),
}


# ---------------------------------------------------------------- attribution

def by_group(market_pnl: pd.DataFrame, field: str = "asset_class") -> pd.DataFrame:
    groups = C.UNIVERSE.loc[market_pnl.columns, field]
    g = market_pnl.T.groupby(groups).sum().T
    return contribution_table(g)


def contribution_table(parts: pd.DataFrame) -> pd.DataFrame:
    total = parts.sum(axis=1)
    out = pd.DataFrame({
        "ann_contribution": parts.mean() * 12,
        "share_of_return": parts.mean() / total.mean() if total.mean() != 0 else np.nan,
        "standalone_sharpe": parts.apply(sharpe),
        "corr_with_total": parts.corrwith(total),
    })
    out.loc["Total"] = [total.mean() * 12, 1.0, sharpe(total), 1.0]
    return out


# ---------------------------------------------------------------- stress

def stress_table(returns: dict[str, pd.Series], windows=STRESS_WINDOWS) -> pd.DataFrame:
    rows = []
    for name, (a, b) in windows.items():
        row = {"window": name, "start": a, "end": b}
        for k, x in returns.items():
            seg = x.loc[pd.Period(a, "M"):pd.Period(b, "M")]
            row[k] = float((1 + seg).prod() - 1) if len(seg) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


def trend_strength(monthly: pd.DataFrame, vol: pd.DataFrame) -> pd.Series:
    """Cross-market average of |12-month return| / annual volatility, measured after the fact.
    Low values mark quiet, trendless spells. Used only to describe results, never to trade."""
    r12 = (1 + monthly).rolling(12).apply(np.prod, raw=True) - 1
    return (r12.abs() / vol).mean(axis=1)


def quiet_spells(net: pd.Series, strength: pd.Series, quantile: float = 0.2) -> pd.DataFrame:
    s = strength.reindex(net.index)
    cut = s.quantile(quantile)
    quiet = s <= cut
    return pd.DataFrame({
        "months": [int(quiet.sum()), int((~quiet & s.notna()).sum())],
        "ann_return": [net[quiet].mean() * 12, net[~quiet & s.notna()].mean() * 12],
        "sharpe": [sharpe(net[quiet]), sharpe(net[~quiet & s.notna()])],
    }, index=[f"trendless (bottom {quantile:.0%})", "other months"])


# ---------------------------------------------------------------- 60/40 overlay

def monthly_from_daily_level(level: pd.Series) -> pd.Series:
    s = level.dropna()
    me = s.groupby(s.index.to_period("M")).last()
    return me.pct_change().dropna()


def sixty_forty(bench: pd.DataFrame) -> pd.Series:
    eq = monthly_from_daily_level(bench["SPXT"])
    bd = monthly_from_daily_level(bench["UST"])
    return (0.6 * eq + 0.4 * bd).dropna()


def overlay(trend_total: pd.Series, bench: pd.DataFrame, cash: pd.Series,
            shares=(0.0, 0.10, 0.15, 0.20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend trend (as a funded total return) into a monthly rebalanced US 60/40."""
    base = sixty_forty(bench)
    idx = base.index.intersection(trend_total.dropna().index)
    base, tr, rf = base.loc[idx], trend_total.loc[idx], cash.reindex(idx).fillna(0.0)
    eq = monthly_from_daily_level(bench["SPXT"]).reindex(idx)
    selloff = eq <= eq.quantile(0.1)
    rows, paths = [], {}
    for w in shares:
        p = (1 - w) * base + w * tr
        paths[f"{int(w * 100)}% trend"] = p
        rows.append({"trend_share": w, "ann_return": p.mean() * 12, "ann_vol": p.std() * np.sqrt(12),
                     "sharpe": sharpe(p - rf), "max_drawdown": drawdown(p).min(),
                     "max_dd_months": max_dd_duration(p),
                     "return_in_worst_equity_decile_months": p[selloff].mean() * 12})
    stats = pd.DataFrame(rows).set_index("trend_share")
    stats.attrs["corr_trend_equity_all"] = float(tr.corr(eq))
    stats.attrs["corr_trend_equity_selloffs"] = float(tr[selloff].corr(eq[selloff]))
    stats.attrs["months"] = len(idx)
    return stats, pd.DataFrame(paths)


def worst_drawdowns(x: pd.Series, n: int = 10) -> pd.DataFrame:
    """Peak-to-trough drawdown episodes of `x`, largest first."""
    nav = (1 + x.fillna(0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    eps, in_dd, start = [], False, None
    for t, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, t
        elif v == 0 and in_dd:
            seg = dd.loc[start:t]
            eps.append((start, seg.idxmin(), seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        eps.append((start, seg.idxmin(), seg.min()))
    df = pd.DataFrame(eps, columns=["start", "trough", "depth"]).sort_values("depth").head(n)
    return df.reset_index(drop=True)


def crisis_table(trend_total: pd.Series, bench: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """HOP-style table: the worst 60/40 drawdowns and trend's return over the same months.
    Episode start is the first month below the previous peak."""
    base = sixty_forty(bench)
    idx = base.index.intersection(trend_total.dropna().index)
    eps = worst_drawdowns(base.loc[idx], n)
    eps["trend_return"] = [float((1 + trend_total.loc[a:b]).prod() - 1) for a, b in zip(eps.start, eps.trough)]
    eps["sixty_forty_return"] = [float((1 + base.loc[a:b]).prod() - 1) for a, b in zip(eps.start, eps.trough)]
    eps.attrs["trend_positive"] = int((eps["trend_return"] > 0).sum())
    return eps


# ---------------------------------------------------------------- replication

def compare_aqr(ours: pd.Series, aqr: pd.DataFrame) -> pd.DataFrame:
    """Correlation and regression of our trend returns on AQR's published TSMOM factor."""
    idx = ours.dropna().index.intersection(aqr.index)
    rows = []
    for col in aqr.columns:
        a = aqr[col].loc[idx]
        reg = regress(ours.loc[idx], a.to_frame(col))
        rows.append({"factor": col, "months": len(idx), "correlation": float(ours.loc[idx].corr(a)),
                     "beta": float(reg.loc[col, "coef"]), "beta_t_nw": float(reg.loc[col, "t_nw"]),
                     "our_sharpe": sharpe(ours.loc[idx]), "aqr_sharpe": sharpe(a)})
    return pd.DataFrame(rows).set_index("factor")


def by_period(x: pd.Series, edges: list[str]) -> pd.DataFrame:
    """Summary statistics for consecutive sub-periods, e.g. decades."""
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x.loc[pd.Period(a, "M"):pd.Period(b, "M") - 1]
        if len(seg) > 12:
            s = summary(seg)
            rows.append({"period": f"{a[:4]}-{int(b[:4]) - 1}", "months": len(seg), "ann_return": s["ann_return"],
                         "ann_vol": s["ann_vol"], "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"]})
    return pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame()


# ---------------------------------------------------------------- fund distributions

def adjust_distributions(price: pd.Series, events: pd.DataFrame, tolerance: float = 0.5
                         ) -> tuple[pd.Series, pd.DataFrame]:
    """Rebuild a total return series from a price series and a list of ex-dates and amounts.

    An event is applied only if the price fell on the ex-date by something close to the
    amount (within `tolerance` of the amount, after allowing for that day's market move as
    measured by a 1.5 standard deviation band). Each event is logged with the check result.
    """
    p = price.dropna()
    r = p.pct_change()
    sd = r.rolling(60, min_periods=20).std().shift(1)
    rows = []
    add = pd.Series(0.0, index=p.index)
    for _, e in events.iterrows():
        d = p.index[p.index >= pd.Timestamp(e["ex_date"])]
        if len(d) == 0:
            continue
        d = d[0]
        prev = p.shift(1).loc[d]
        drop = prev - p.loc[d]
        band = 1.5 * (sd.loc[d] if pd.notna(sd.loc[d]) else 0.01) * prev
        ok = abs(drop - e["amount"]) <= tolerance * e["amount"] + band
        rows.append({"ex_date": d.date(), "amount": e["amount"], "price_drop": drop, "applied": bool(ok)})
        if ok:
            add.loc[d] = e["amount"] / prev        # (P_d + D) / P_{d-1} - 1
    tr = (1 + r.fillna(0) + add).cumprod() * p.iloc[0]
    return tr, pd.DataFrame(rows, columns=["ex_date", "amount", "price_drop", "applied"])
