import datetime as dt

import numpy as np
import openpyxl
import pandas as pd
import pytest

from trendcarry import config as C
from trendcarry import data, synthetic


def _sheet(path, sheet, blocks):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    for b, (name, tick, rows) in enumerate(blocks):
        c = 1 + 3 * b
        ws.cell(1, c, name)
        ws.cell(2, c, tick)
        ws.cell(3, c, "Date")
        ws.cell(3, c + 1, "PX_LAST")
        for i, (d, v) in enumerate(rows):
            ws.cell(4 + i, c, d)
            ws.cell(4 + i, c + 1, v)
    wb.save(path)


def test_universe_is_complete_and_consistent():
    u = C.UNIVERSE
    assert len(u) == 54
    assert set(u.asset_class) == {"Equity", "Bond", "FX", "Commodity"}
    assert u.adj_ticker.is_unique and u.g1.is_unique and u.g2.is_unique
    assert (u.asset_class.value_counts() == pd.Series({"Commodity": 23, "Bond": 12, "Equity": 11, "FX": 8})).all()


def test_read_blocks_handles_text_serial_dates_and_blank_blocks(tmp_path):
    p = tmp_path / "x.xlsx"
    _sheet(p, "S", [("A", "AA Index", [(dt.datetime(2020, 1, 31), "#N/A N/A"), (dt.datetime(2020, 2, 28), 5.0)]),
                    ("B", "BB Index", [(43861, 10.0), (43889, 11.0)]),           # Excel serial dates
                    ("C", "CC Index", [("#N/A Requesting Data...", None)])])
    b = data.read_blocks(p, "S")
    assert [x.ticker for x in b] == ["AA Index", "BB Index", "CC Index"]
    assert np.isnan(b[0].series.iloc[0]) and b[0].series.iloc[1] == 5.0      # text kept as missing, not filled
    assert b[1].series.index[0] == pd.Timestamp("2020-01-31")
    assert b[2].n_valid == 0 and "Requesting" in b[2].text_cells[0]


def test_carry_merge_keeps_the_block_with_more_data(tmp_path):
    d = [dt.datetime(2020, m, 28) for m in range(1, 4)]
    main, rev = tmp_path / "TrendPull_3_x_final.xlsx", tmp_path / "carry_revised.xlsx"
    _sheet(main, "Carry_Monthly", [("ES G1", "ES1 Index", [("#N/A Requesting Data...", None)]),
                                   ("ES G2", "ES2 Index", list(zip(d, [1.0, 2.0, 3.0])))])
    _sheet(rev, "carrey", [("ES G1", "ES1 Index", list(zip(d, [4.0, 5.0, 6.0]))),
                           ("ES G2", "ES2 Index", [(d[0], "#VALUE!")])])
    g1, g2, notes = data.load_carry(main, rev)
    assert g1["ES"].tolist() == [4.0, 5.0, 6.0]       # fixed block taken from the revised file
    assert g2["ES"].tolist() == [1.0, 2.0, 3.0]       # broken revised block ignored
    assert isinstance(g1.index, pd.PeriodIndex)


def test_benchmarks_are_mapped_by_position_not_header(tmp_path):
    p = tmp_path / "b.xlsx"
    rows = [(dt.datetime(2020, 1, 2), 1.0)]
    _sheet(p, "Benchmarks_Daily", [("x", "SPXT Index", rows), ("x", "LUATTRUU Index", rows), ("x", "USGG3M Index", rows),
                                   ("SG Trend Index", "NEIXCTAT Index", rows), ("SG CTA Index", "NEIXCTA Index", rows)])
    b = data.load_benchmarks(p)
    assert list(b.columns) == ["SPXT", "UST", "TBILL", "AQMIX", "DBMF"]


def test_synthetic_workbooks_round_trip_through_the_loader(tmp_path):
    sim = synthetic.generate(seed=5, small=True, end="2000-06-30")
    synthetic.write_workbooks(sim, tmp_path)
    ds = data.load_folder(tmp_path, "synthetic_rt", use_cache=False)
    assert list(ds.prices.columns) == sim["codes"]
    pd.testing.assert_frame_equal(ds.prices.dropna(how="all"), sim["prices"].dropna(how="all"),
                                  check_freq=False, check_names=False, atol=1e-9)
    assert ds.specs.loc["ES", "cycle"] == "HMUZ"
    assert ds.g1.loc[:, "TY"].notna().sum() > 10


def test_aqr_factor_loader_finds_header(tmp_path):
    p = tmp_path / "Time_Series_Momentum_Factors_Monthly.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TSMOM Factors"
    ws.append(["AQR blurb"])
    ws.append([])
    ws.append([None, "TSMOM", "TSMOM^CM", "TSMOM^EQ", "TSMOM^FI", "TSMOM^FX"])
    ws.append([dt.datetime(1985, 1, 31), 0.01, 0.02, 0.03, 0.04, 0.05])
    wb.save(p)
    f = data.load_aqr_factors(p)
    assert list(f.columns) == ["TSMOM", "TSMOM_CM", "TSMOM_EQ", "TSMOM_FI", "TSMOM_FX"]
    assert f.index[0] == pd.Period("1985-01", "M")


def test_missing_files_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_folder(tmp_path, "real", use_cache=False)
