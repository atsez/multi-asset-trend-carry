"""Synthetic futures data written in exactly the Bloomberg workbook layout, so the real
pipeline (loader, checks, cleaning, backtest) runs end to end without licensed data.

Each market follows a random walk with fat tails, slowly varying volatility, sector
factors, a persistent trend component of tunable strength and a carry premium. The
unadjusted first and second generics are built from the adjusted series and the carry,
with a jump at every roll. Known faults are injected so the data checks can be tested:
a stale run, a spike that reverses, a gap, a broken roll adjustment, a curve that does not
trade, a contract change and a fund distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from . import config as C

# Public contract details: listed cycle, tick size, rough current price (synthetic only)
SPEC = {
    "ES": ("HMUZ", 0.25, 6000), "NQ": ("HMUZ", 0.25, 21000), "VG": ("HMUZ", 1, 5000),
    "GX": ("HMUZ", 1, 23000), "CF": ("FGHJKMNQUVXZ", 0.5, 7800), "Z": ("HMUZ", 0.5, 9000),
    "NK": ("HMUZ", 10, 45000), "HI": ("FGHJKMNQUVXZ", 1, 25000), "XP": ("HMUZ", 1, 8500),
    "PT": ("HMUZ", 0.1, 1700), "KM": ("HMUZ", 0.05, 400), "TU": ("HMUZ", 0.00390625, 104),
    "FV": ("HMUZ", 0.0078125, 108), "TY": ("HMUZ", 0.015625, 112), "US": ("HMUZ", 0.03125, 117),
    "DU": ("HMUZ", 0.005, 107), "OE": ("HMUZ", 0.01, 117), "RX": ("HMUZ", 0.01, 128),
    "OAT": ("HMUZ", 0.01, 123), "IK": ("HMUZ", 0.01, 120), "G": ("HMUZ", 0.01, 92),
    "JB": ("HMUZ", 0.01, 136), "CN": ("HMUZ", 0.01, 110), "EC": ("HMUZ", 5e-05, 1.16),
    "JY": ("HMUZ", 0.005, 64), "BP": ("HMUZ", 0.01, 133), "SF": ("HMUZ", 0.005, 125),
    "AD": ("HMUZ", 0.005, 65), "CD": ("HMUZ", 0.005, 72), "NV": ("HMUZ", 0.005, 58),
    "PE": ("HMUZ", 0.001, 5.4), "CL": ("FGHJKMNQUVXZ", 0.01, 60), "CO": ("FGHJKMNQUVXZ", 0.01, 64),
    "HO": ("FGHJKMNQUVXZ", 0.01, 230), "XB": ("FGHJKMNQUVXZ", 0.01, 200),
    "NG": ("FGHJKMNQUVXZ", 0.001, 3.5), "QS": ("FGHJKMNQUVXZ", 0.25, 650), "GC": ("GJMQVZ", 0.1, 4000),
    "SI": ("HKNUZ", 0.005, 50), "HG": ("HKNUZ", 0.05, 500), "PL": ("FJNV", 0.1, 1500),
    "C": ("HKNUZ", 0.25, 430), "S": ("FHKNQUX", 0.25, 1100), "W": ("HKNUZ", 0.25, 540),
    "KW": ("HKNUZ", 0.25, 530), "SM": ("FHKNQUVZ", 0.1, 300), "BO": ("FHKNQUVZ", 0.01, 50),
    "SB": ("HKNV", 0.01, 15), "KC": ("HKNUZ", 0.05, 380), "CC": ("HKNUZ", 1, 6000),
    "CT": ("HKNVZ", 0.01, 65), "LC": ("GJMQVZ", 0.025, 230), "LH": ("GJKMNQVZ", 0.025, 85),
    "FC": ("FHJKQUVX", 0.025, 330),
}
START = {"ES": "1997-09-10", "NQ": "1999-06-22", "VG": "1998-06-23", "GX": "1990-11-26",
         "HI": "1992-04-02", "XP": "2000-05-03", "PT": "1999-09-08", "KM": "1996-05-06",
         "TU": "1990-06-26", "DU": "1997-03-10", "OE": "1991-10-07", "RX": "1990-11-26",
         "OAT": "2012-04-17", "IK": "2009-09-15", "EC": "1998-05-20", "NV": "1997-05-08",
         "PE": "1995-04-26", "XB": "2005-10-04", "NG": "1990-04-04"}
VOL = {"Equity": 0.20, "Bond": 0.06, "FX": 0.10, "Energy": 0.35, "Metals": 0.22,
       "Agriculture": 0.25, "Softs": 0.30, "Livestock": 0.16}
FACTOR_LOAD = {"Equity": 0.75, "Bond": 0.75, "FX": 0.55, "Energy": 0.7, "Metals": 0.45,
               "Agriculture": 0.55, "Softs": 0.2, "Livestock": 0.4}
CARRY_SD = {"Equity": 0.01, "Bond": 0.01, "FX": 0.02, "Energy": 0.10, "Metals": 0.02,
            "Agriculture": 0.08, "Softs": 0.08, "Livestock": 0.10}
SMALL = ["ES", "GX", "NK", "TY", "RX", "JB", "EC", "JY", "AD", "CL", "GC", "C", "KC", "LC"]


def _market_paths(rng, days, codes, trend_strength, carry_premium):
    n = len(days)
    sectors = C.UNIVERSE.loc[codes, "sector"]
    f = {s: rng.standard_normal(n) for s in VOL}
    common = rng.standard_normal(n)
    zc = {s: _ar1(rng, n, 1 - 1 / 126) for s in VOL}
    rets, carries = {}, {}
    for code in codes:
        sec = sectors[code]
        b = FACTOR_LOAD[sec]
        eps = rng.standard_t(5, n) / np.sqrt(5 / 3)
        shock = np.sqrt(b) * f[sec] + np.sqrt(1 - b) * (0.2 * common + np.sqrt(1 - 0.04) * eps)
        h = _ar1(rng, n, 0.99) * 0.2
        sig = VOL[sec] / np.sqrt(252) * np.exp(h - 0.02)
        z = 0.6 * zc[sec] + 0.8 * _ar1(rng, n, 1 - 1 / 126)
        carry = CARRY_SD[sec] * _ar1(rng, n, 1 - 1 / 250)
        mu = trend_strength * sig * z + carry_premium * carry / 252
        rets[code] = mu + sig * shock
        carries[code] = carry
    return pd.DataFrame(rets, index=days), pd.DataFrame(carries, index=days)


def _ar1(rng, n, phi):
    e = rng.standard_normal(n) * np.sqrt(1 - phi ** 2)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate(seed: int = 7, small: bool = False, trend_strength: float = 0.045,
             carry_premium: float = 0.5, start: str = "1990-01-02", end: str = "2026-11-13") -> dict:
    rng = np.random.default_rng(seed)
    codes = SMALL if small else list(C.UNIVERSE.index)
    if small:
        start = "1998-01-02"
    days = pd.bdate_range(start, end)
    rets, carry = _market_paths(rng, days, codes, trend_strength, carry_premium)
    prices, g1, g2, truth = {}, {}, {}, {"injected": []}
    months = days.to_period("M")
    month_end = pd.Series(days, index=days).groupby(months).max()
    for code in codes:
        cyc, tick, level = SPEC[code]
        live = days >= pd.Timestamp(START.get(code, start))
        r = rets[code].where(live)
        adj = np.exp(r.fillna(0).cumsum()).where(live)
        adj = adj / adj.dropna().iloc[-1] * level
        # holidays: drop about 3% of days at random
        drop = rng.random(len(days)) < 0.03
        adj[drop] = np.nan
        # unadjusted curve: jump by exp(-carry * spacing / 12) at each roll month end
        cmonths = {"FGHJKMNQUVXZ".index(ch) + 1 for ch in cyc}
        me = month_end[month_end >= adj.first_valid_index()]
        c_me = carry[code].reindex(me.values).values
        spacing = np.array([_spacing(cmonths, p.month) for p in me.index])
        roll = np.array([((p.month % 12) + 1) in cmonths for p in me.index])
        jumps = np.where(roll, -c_me * spacing / 12, 0.0)
        cum = np.cumsum(jumps) - jumps.sum()
        a_me = adj.reindex(me.values).ffill().values
        u1 = a_me * np.exp(cum)
        u2 = u1 * np.exp(-c_me * spacing / 12) * (1 + 0.0005 * rng.standard_normal(len(me)))
        prices[code] = adj
        g1[code] = pd.Series(u1, index=me.index)
        g2[code] = pd.Series(u2, index=me.index)
    prices = pd.DataFrame(prices)
    g1, g2 = pd.DataFrame(g1), pd.DataFrame(g2)
    _inject(prices, g1, g2, truth, codes)
    bench = _benchmarks(rng, days, rets, codes, truth)
    return {"prices": prices, "g1": g1, "g2": g2, "bench": bench, "truth": truth, "codes": codes}


def _spacing(cmonths, month):
    ms = sorted(cmonths)
    nxt = lambda m: next((x for x in ms if x > m), ms[0] + 12)
    first = (nxt(month) - 1) % 12 + 1
    return nxt(first) - first


def _inject(prices, g1, g2, truth, codes):
    inj = truth["injected"]
    last = prices.index[-1]

    def has(code, date):
        return code in prices.columns and pd.Timestamp(date) < last - pd.Timedelta(days=40) \
            and prices[code].loc[date:].notna().any()

    if has("JB", "1999-11-01"):  # stale run of 16 identical closes
        idx = prices["JB"].loc["1999-11-01":"1999-11-30"].dropna().index[:16]
        prices.loc[idx, "JB"] = prices.loc[idx[0], "JB"]
        inj.append(dict(kind="stale_run", code="JB", start=str(idx[0].date()), length=len(idx)))
    if has("KC", "2003-10-13"):  # one-day spike that reverses
        d = prices["KC"].loc["2003-10-13":].dropna().index[0]
        prices.loc[d, "KC"] *= 1.35
        inj.append(dict(kind="spike", code="KC", date=str(d.date())))
    if has("LC", "2001-11-12"):  # a 10 trading day gap
        idx = prices.index[(prices.index >= "2001-11-12")][:10]
        prices.loc[idx, "LC"] = np.nan
        inj.append(dict(kind="gap", code="LC", start=str(idx[0].date()), days=10))
    if has("GC", "2005-06-30"):  # broken back-adjustment: permanent 15% level shift on a roll date
        d = prices["GC"].loc["2005-06-30":].dropna().index[0]
        prices.loc[:d - pd.Timedelta(days=1), "GC"] /= 1.15
        inj.append(dict(kind="roll_break", code="GC", date=str(d.date())))
    if has("JY", "2004-06-01"):  # back month that did not trade
        m = pd.Period("2004-05", "M")
        g2.loc[m, "JY"] = g1.loc[m, "JY"]
        inj.append(dict(kind="g1_equals_g2", code="JY", month=str(m)))
    if has("TY", "1999-12-01"):  # contract change: 11% curve gap
        m = pd.Period("1999-11", "M")
        g2.loc[m, "TY"] = g1.loc[m, "TY"] * 0.89
        inj.append(dict(kind="contract_change", code="TY", month=str(m)))


def _benchmarks(rng, days, rets, codes, truth):
    n = len(days)
    eq = rets[[c for c in codes if C.UNIVERSE.loc[c, "sector"] == "Equity"]].mean(axis=1).fillna(0)
    bd = rets[[c for c in codes if C.UNIVERSE.loc[c, "sector"] == "Bond"]].mean(axis=1).fillna(0)
    tb = np.clip(4 + np.cumsum(rng.standard_normal(n) * 0.03), 0, 8)
    spxt = 400 * np.exp(np.cumsum(eq.values + 0.07 / 252))
    ust = 450 * np.exp(np.cumsum(bd.values + tb / 100 / 252))
    out = pd.DataFrame({"SPXT": spxt, "UST": ust, "TBILL": tb}, index=days)
    ust_monthly = days < pd.Timestamp("1994-03-01")
    me = pd.Series(days, index=days).groupby(days.to_period("M")).transform("max") == days
    out.loc[ust_monthly & ~me, "UST"] = np.nan
    for name, start, px0, dist in (("AQMIX", "2010-01-05", 10.0, [("2014-12-19", 0.9)]),
                                   ("DBMF", "2019-05-08", 25.0, [("2022-12-28", 2.2)])):
        live = days >= pd.Timestamp(start)
        if live.sum() < 300:
            out[name] = np.nan
            continue
        r = pd.Series(rng.standard_normal(n) * 0.10 / np.sqrt(252), index=days)[live]
        px = px0 * np.exp(r.cumsum())
        for d, amt in dist:
            d = px.index[px.index >= pd.Timestamp(d)][0]
            px.loc[d:] -= amt
            truth["injected"].append(dict(kind="distribution", code=name, date=str(d.date()), amount=amt))
        out[name] = px.reindex(days)
    return out


# ---------------------------------------------------------------- writers (Bloomberg layout)

def _write_blocks(ws, blocks):
    """blocks: list of (name, ticker, series). Row 1 name, row 2 ticker, row 3 headers."""
    cols = []
    for name, tick, s in blocks:
        cols.append((name, tick, list(s.index), list(s.values)))
    nrows = max(len(c[2]) for c in cols)
    header = [[], [], []]
    for name, tick, _, _ in cols:
        header[0] += [name, None, None]
        header[1] += [tick, None, None]
        header[2] += ["Date", "PX_LAST", None]
    for h in header:
        ws.append(h)
    for i in range(nrows):
        row = []
        for _, _, idx, vals in cols:
            if i < len(idx):
                v = vals[i]
                row += [idx[i].to_pydatetime() if hasattr(idx[i], "to_pydatetime") else idx[i],
                        "#N/A N/A" if pd.isna(v) else float(v), None]
            else:
                row += [None, None, None]
        ws.append(row)


def _universe_sheet(wb, codes):
    ws = wb.create_sheet("Universe")
    ws.append(["Market", "Asset class", "Adjusted ticker (signals and returns)", "Generic 1 (unadjusted)",
               "Generic 2 (unadjusted)", "Roll rule"])
    for c in codes:
        u = C.UNIVERSE.loc[c]
        ws.append([u["name"], u["sector"], u["adj_ticker"], u["g1"], u["g2"], "synthetic"])


def write_workbooks(sim: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    codes = sim["codes"]
    fin = [c for c in codes if C.UNIVERSE.loc[c, "sector"] in ("Equity", "Bond", "FX")]
    com = [c for c in codes if c not in fin]
    for fname, group in (("TrendPull_1_Daily_Financials_final.xlsx", fin),
                         ("TrendPull_2_Daily_Commodities_final.xlsx", com)):
        wb = openpyxl.Workbook(write_only=True)
        ws = wb.create_sheet("Daily_Adjusted")
        _write_blocks(ws, [(C.UNIVERSE.loc[c, "name"], C.UNIVERSE.loc[c, "adj_ticker"],
                            sim["prices"][c].dropna()) for c in group])
        _universe_sheet(wb, group)
        wb.save(folder / fname)
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("Carry_Monthly")
    blocks = []
    for c in codes:
        for gen, df in (("g1", sim["g1"]), ("g2", sim["g2"])):
            s = df[c].copy()
            s.index = s.index.to_timestamp(how="end").normalize()
            first = sim["prices"][c].first_valid_index()
            full = pd.Series(np.nan, index=pd.period_range("1990-01", s.index[-1], freq="M")
                             .to_timestamp(how="end").normalize())
            full.loc[s.index] = s.values
            full[full.index < first] = np.nan       # pre-launch '#N/A N/A', as Bloomberg returns
            blocks.append((f"{C.UNIVERSE.loc[c, 'name']} {gen.upper()}", C.UNIVERSE.loc[c, gen], full))
    _write_blocks(ws, blocks)
    ws = wb.create_sheet("Benchmarks_Daily")
    b = sim["bench"]
    _write_blocks(ws, [(k, C.BENCHMARKS[k], b[k].dropna()) for k in C.BENCHMARK_BLOCK_ORDER])
    ws = wb.create_sheet("Specs")
    ws.append(["Market", "Generic 1 ticker", "NAME", "CRNCY", "FUT_GEN_MONTH", "FUT_TICK_SIZE",
               "FUT_TICK_VAL", "FUT_CONT_SIZE"])
    for c in codes:
        cyc, tick, _ = SPEC[c]
        ws.append([C.UNIVERSE.loc[c, "name"], C.UNIVERSE.loc[c, "g1"], "synthetic", "USD", cyc, tick, 10, 1000])
    _universe_sheet(wb, codes)
    wb.save(folder / "TrendPull_3_Carry_Benchmarks_Specs_final.xlsx")
    with open(folder / "injected_faults.json", "w") as fh:
        json.dump(sim["truth"], fh, indent=2)


def build(folder: Path | None = None, seed: int = 7, small: bool = False, **kw) -> Path:
    folder = Path(folder or C.DATA_SYNTHETIC)
    write_workbooks(generate(seed=seed, small=small, **kw), folder)
    return folder


def to_dataset(sim: dict):
    """Build a Dataset straight from a simulation, skipping the workbook round trip (fast,
    for tests). The workbook path is tested separately."""
    from .data import Dataset
    codes = sim["codes"]
    specs = pd.DataFrame({c: {"currency": "USD", "cycle": SPEC[c][0], "tick_size": SPEC[c][1],
                              "tick_value": 10, "contract_size": 1000} for c in codes}).T
    return Dataset(prices=sim["prices"][codes], g1=sim["g1"][codes], g2=sim["g2"][codes], specs=specs,
                   bench=sim["bench"], source="synthetic")
