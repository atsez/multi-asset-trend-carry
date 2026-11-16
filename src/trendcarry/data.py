"""Load the Bloomberg pull workbooks into one tidy `Dataset`.

Workbook layout (every data sheet): row 1 market name, row 2 ticker, row 3 headers, values
from row 4. Each market is a (date, value) block with one blank column between blocks, so
blocks start at columns A, D, G and so on. Text such as '#N/A N/A' is kept as missing and is
never filled.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from . import config as C

EXCEL_EPOCH = dt.datetime(1899, 12, 30)


@dataclass
class Block:
    name: str
    ticker: str
    series: pd.Series
    text_cells: list[str] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return int(self.series.notna().sum())


@dataclass
class Dataset:
    prices: pd.DataFrame            # daily ratio-adjusted generic 1, columns = market codes
    g1: pd.DataFrame                # month-end unadjusted generic 1
    g2: pd.DataFrame                # month-end unadjusted generic 2
    specs: pd.DataFrame             # per market contract details
    bench: pd.DataFrame             # daily benchmarks
    aqr: pd.DataFrame | None = None  # AQR monthly TSMOM factors
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def universe(self) -> pd.DataFrame:
        return C.UNIVERSE.loc[self.prices.columns]


def _to_date(x):
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day)
    if isinstance(x, (int, float)) and not isinstance(x, bool) and 20000 < x < 80000:
        return EXCEL_EPOCH + dt.timedelta(days=float(x))
    return None


def read_blocks(path: Path | str, sheet: str) -> list[Block]:
    """Parse every (date, value) block on a sheet."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb[sheet].iter_rows(values_only=True))
    finally:
        wb.close()
    if len(rows) < 3:
        return []
    names, tickers = rows[0], rows[1]
    width = max(len(r) for r in rows)
    blocks = []
    for c in range(0, width, 3):
        name = names[c] if c < len(names) else None
        tick = tickers[c] if c < len(tickers) else None
        if name is None and tick is None:
            continue
        dates, vals, text = [], [], []
        for r in rows[3:]:
            a = r[c] if c < len(r) else None
            b = r[c + 1] if c + 1 < len(r) else None
            if a is None and b is None:
                continue
            d = _to_date(a)
            if d is None:
                text.append(str(a))
                continue
            if isinstance(b, (int, float)) and not isinstance(b, bool) and np.isfinite(b):
                vals.append(float(b))
            else:
                vals.append(np.nan)
                text.append(str(b))
            dates.append(d)
        s = pd.Series(vals, index=pd.DatetimeIndex(dates, name="date"), dtype=float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        blocks.append(Block(str(name), str(tick).strip() if tick else "", s, text))
    return blocks


def read_table(path: Path | str, sheet: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = [r for r in wb[sheet].iter_rows(values_only=True) if any(v is not None for v in r)]
    finally:
        wb.close()
    return pd.DataFrame(rows[1:], columns=[str(h) for h in rows[0]])


def first_sheet(path: Path | str) -> str:
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        return wb.sheetnames[0]
    finally:
        wb.close()


def find_files(folder: Path) -> dict[str, Path | None]:
    out = {}
    for key, pat in C.FILE_PATTERNS.items():
        hits = sorted(glob.glob(str(folder / pat)))
        out[key] = Path(hits[-1]) if hits else None
    return out


def _by_ticker(blocks: list[Block]) -> dict[str, Block]:
    return {b.ticker: b for b in blocks}


def load_daily(paths: list[Path], require_all: bool = True) -> pd.DataFrame:
    tick_to_code = {t: c for c, t in C.UNIVERSE["adj_ticker"].items()}
    cols = {}
    for p in paths:
        for b in read_blocks(p, "Daily_Adjusted"):
            code = tick_to_code.get(b.ticker)
            if code is None:
                raise ValueError(f"Unknown adjusted ticker {b.ticker!r} in {p.name}")
            cols[code] = b.series
    missing = [c for c in C.UNIVERSE.index if c not in cols]
    if missing and require_all:
        raise ValueError(f"Daily data missing for {missing}")
    df = pd.DataFrame(cols).sort_index()
    return df[[c for c in C.UNIVERSE.index if c in cols]]


def load_carry(main: Path, revised: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Month-end generic 1 and 2. Where a ticker appears in both files, keep the block
    with more valid values (the revised file fixed 15 blocks but broke KC2)."""
    notes = []
    blocks = _by_ticker(read_blocks(main, "Carry_Monthly"))
    if revised is not None:
        rev = _by_ticker(read_blocks(revised, first_sheet(revised)))
        for t, b in rev.items():
            old = blocks.get(t)
            if old is None or b.n_valid > old.n_valid:
                blocks[t] = b
                notes.append(f"carry {t}: taken from {revised.name} ({b.n_valid} values)")
    g1, g2 = {}, {}
    for code, row in C.UNIVERSE.iterrows():
        if row.g1 not in blocks and row.g2 not in blocks:
            continue
        for tick, store in ((row.g1, g1), (row.g2, g2)):
            b = blocks.get(tick)
            store[code] = b.series if b is not None else pd.Series(dtype=float)
            if b is None or b.n_valid == 0:
                notes.append(f"carry {tick}: no data")
    g1, g2 = _monthly(pd.DataFrame(g1)), _monthly(pd.DataFrame(g2))
    return g1, g2, notes


def _monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Index month-end observations by calendar month (some blocks end on a Saturday)."""
    df = df.sort_index()
    df.index = df.index.to_period("M")
    return df.groupby(level=0).last()


def load_benchmarks(path: Path) -> pd.DataFrame:
    blocks = read_blocks(path, "Benchmarks_Daily")
    if len(blocks) < len(C.BENCHMARK_BLOCK_ORDER):
        raise ValueError("Benchmarks_Daily has fewer blocks than expected")
    return pd.DataFrame({k: b.series for k, b in zip(C.BENCHMARK_BLOCK_ORDER, blocks)}).sort_index()


def load_specs(path: Path) -> pd.DataFrame:
    t = read_table(path, "Specs")
    g1_to_code = {v: k for k, v in C.UNIVERSE["g1"].items()}
    t["code"] = t["Generic 1 ticker"].map(g1_to_code)
    if t["code"].isna().any():
        raise ValueError(f"Unmapped specs rows: {t.loc[t.code.isna(), 'Generic 1 ticker'].tolist()}")
    t = t.set_index("code").rename(columns={
        "CRNCY": "currency", "FUT_GEN_MONTH": "cycle", "FUT_TICK_SIZE": "tick_size",
        "FUT_TICK_VAL": "tick_value", "FUT_CONT_SIZE": "contract_size"})
    return t[["currency", "cycle", "tick_size", "tick_value", "contract_size"]]


def load_aqr_factors(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb["TSMOM Factors"].iter_rows(values_only=True))
    finally:
        wb.close()
    hdr_i = next(i for i, r in enumerate(rows) if r and "TSMOM" in [str(x) for x in r[1:6]])
    hdr = [str(x) for x in rows[hdr_i][1:6]]
    recs = [(r[0], *r[1:6]) for r in rows[hdr_i + 1:]
            if r and _to_date(r[0]) is not None and isinstance(r[1], (int, float))]
    df = pd.DataFrame(recs, columns=["date", *hdr])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").astype(float)
    df.index = df.index.to_period("M")
    return df.rename(columns=lambda c: c.replace("TSMOM^", "TSMOM_"))


def _fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def load_folder(folder: Path, source: str, use_cache: bool = True) -> Dataset:
    files = find_files(folder)
    need = ["daily_financials", "daily_commodities", "carry_benchmarks_specs"]
    missing = [k for k in need if files[k] is None]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {folder}. Expected patterns: "
                                f"{[C.FILE_PATTERNS[k] for k in missing]}")
    present = [p for p in files.values() if p is not None]
    C.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache = C.DATA_CACHE / f"{source}_{_fingerprint(present)}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    prices = load_daily([files["daily_financials"], files["daily_commodities"]], require_all=source == "real")
    g1, g2, notes = load_carry(files["carry_benchmarks_specs"], files["carry_revised"])
    g1, g2 = g1.reindex(columns=prices.columns), g2.reindex(columns=prices.columns)
    ds = Dataset(
        prices=prices, g1=g1, g2=g2,
        specs=load_specs(files["carry_benchmarks_specs"]).reindex(prices.columns),
        bench=load_benchmarks(files["carry_benchmarks_specs"]),
        aqr=load_aqr_factors(files["aqr_tsmom"]) if files["aqr_tsmom"] else None,
        source=source, notes=notes)
    with open(cache, "wb") as fh:
        pickle.dump(ds, fh)
    return ds


def load(dataset: str = "real", folder: Path | None = None, use_cache: bool = True) -> Dataset:
    if folder is None:
        folder = C.DATA_RAW if dataset == "real" else C.DATA_SYNTHETIC
    return load_folder(Path(folder), dataset, use_cache=use_cache)
