"""Universe, parameters and paths.

Every research choice that could move a result lives in `Params`, so HYPOTHESES.md can
point at one place and the deflated Sharpe ratio can count how many variants were tried.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
DATA_REFERENCE = ROOT / "data" / "reference"
RESULTS = ROOT / "results"

# code, market name (as in the Universe sheets), sector, adjusted ticker, generic 1, generic 2
_UNIVERSE_ROWS = [
    ("ES", "S&P 500 E-mini", "Equity", "ES1 R:05_0_R Index", "ES1 Index", "ES2 Index"),
    ("NQ", "Nasdaq 100 E-mini", "Equity", "NQ1 R:05_0_R Index", "NQ1 Index", "NQ2 Index"),
    ("VG", "Euro Stoxx 50", "Equity", "VG1 R:05_0_R Index", "VG1 Index", "VG2 Index"),
    ("GX", "DAX", "Equity", "GX1 R:05_0_R Index", "GX1 Index", "GX2 Index"),
    ("CF", "CAC 40", "Equity", "CF1 R:05_0_R Index", "CF1 Index", "CF2 Index"),
    ("Z", "FTSE 100", "Equity", "Z 1 R:05_0_R Index", "Z 1 Index", "Z 2 Index"),
    ("NK", "Nikkei 225", "Equity", "NK1 R:05_0_R Index", "NK1 Index", "NK2 Index"),
    ("HI", "Hang Seng", "Equity", "HI1 R:05_0_R Index", "HI1 Index", "HI2 Index"),
    ("XP", "SPI 200", "Equity", "XP1 R:05_0_R Index", "XP1 Index", "XP2 Index"),
    ("PT", "S&P/TSX 60", "Equity", "PT1 R:05_0_R Index", "PT1 Index", "PT2 Index"),
    ("KM", "KOSPI 200", "Equity", "KM1 R:05_0_R Index", "KM1 Index", "KM2 Index"),
    ("TU", "US 2y Note", "Bond", "TU1 N:05_0_R Comdty", "TU1 Comdty", "TU2 Comdty"),
    ("FV", "US 5y Note", "Bond", "FV1 N:05_0_R Comdty", "FV1 Comdty", "FV2 Comdty"),
    ("TY", "US 10y Note", "Bond", "TY1 N:05_0_R Comdty", "TY1 Comdty", "TY2 Comdty"),
    ("US", "US Long Bond", "Bond", "US1 N:05_0_R Comdty", "US1 Comdty", "US2 Comdty"),
    ("DU", "Euro Schatz", "Bond", "DU1 N:05_0_R Comdty", "DU1 Comdty", "DU2 Comdty"),
    ("OE", "Euro Bobl", "Bond", "OE1 N:05_0_R Comdty", "OE1 Comdty", "OE2 Comdty"),
    ("RX", "Euro Bund", "Bond", "RX1 N:05_0_R Comdty", "RX1 Comdty", "RX2 Comdty"),
    ("OAT", "Euro OAT", "Bond", "OAT1 N:05_0_R Comdty", "OAT1 Comdty", "OAT2 Comdty"),
    ("IK", "Euro BTP", "Bond", "IK1 N:05_0_R Comdty", "IK1 Comdty", "IK2 Comdty"),
    ("G", "Long Gilt", "Bond", "G 1 N:05_0_R Comdty", "G 1 Comdty", "G 2 Comdty"),
    ("JB", "JGB 10y", "Bond", "JB1 N:05_0_R Comdty", "JB1 Comdty", "JB2 Comdty"),
    ("CN", "Canada 10y", "Bond", "CN1 N:05_0_R Comdty", "CN1 Comdty", "CN2 Comdty"),
    ("EC", "EUR/USD", "FX", "EC1 R:05_0_R Curncy", "EC1 Curncy", "EC2 Curncy"),
    ("JY", "JPY/USD", "FX", "JY1 R:05_0_R Curncy", "JY1 Curncy", "JY2 Curncy"),
    ("BP", "GBP/USD", "FX", "BP1 R:05_0_R Curncy", "BP1 Curncy", "BP2 Curncy"),
    ("SF", "CHF/USD", "FX", "SF1 R:05_0_R Curncy", "SF1 Curncy", "SF2 Curncy"),
    ("AD", "AUD/USD", "FX", "AD1 R:05_0_R Curncy", "AD1 Curncy", "AD2 Curncy"),
    ("CD", "CAD/USD", "FX", "CD1 R:05_0_R Curncy", "CD1 Curncy", "CD2 Curncy"),
    ("NV", "NZD/USD", "FX", "NV1 R:05_0_R Curncy", "NV1 Curncy", "NV2 Curncy"),
    ("PE", "MXN/USD", "FX", "PE1 R:05_0_R Curncy", "PE1 Curncy", "PE2 Curncy"),
    ("CL", "WTI Crude", "Energy", "CL1 R:05_0_R Comdty", "CL1 Comdty", "CL2 Comdty"),
    ("CO", "Brent Crude", "Energy", "CO1 R:05_0_R Comdty", "CO1 Comdty", "CO2 Comdty"),
    ("HO", "Heating Oil", "Energy", "HO1 R:05_0_R Comdty", "HO1 Comdty", "HO2 Comdty"),
    ("XB", "RBOB Gasoline", "Energy", "XB1 R:05_0_R Comdty", "XB1 Comdty", "XB2 Comdty"),
    ("NG", "Natural Gas", "Energy", "NG1 R:05_0_R Comdty", "NG1 Comdty", "NG2 Comdty"),
    ("QS", "ICE Gasoil", "Energy", "QS1 R:05_0_R Comdty", "QS1 Comdty", "QS2 Comdty"),
    ("GC", "Gold", "Metals", "GC1 N:05_0_R Comdty", "GC1 Comdty", "GC2 Comdty"),
    ("SI", "Silver", "Metals", "SI1 N:05_0_R Comdty", "SI1 Comdty", "SI2 Comdty"),
    ("HG", "Copper", "Metals", "HG1 N:05_0_R Comdty", "HG1 Comdty", "HG2 Comdty"),
    ("PL", "Platinum", "Metals", "PL1 N:05_0_R Comdty", "PL1 Comdty", "PL2 Comdty"),
    ("C", "Corn", "Agriculture", "C 1 N:05_0_R Comdty", "C 1 Comdty", "C 2 Comdty"),
    ("S", "Soybeans", "Agriculture", "S 1 N:05_0_R Comdty", "S 1 Comdty", "S 2 Comdty"),
    ("W", "Chicago Wheat", "Agriculture", "W 1 N:05_0_R Comdty", "W 1 Comdty", "W 2 Comdty"),
    ("KW", "KC Wheat", "Agriculture", "KW1 N:05_0_R Comdty", "KW1 Comdty", "KW2 Comdty"),
    ("SM", "Soybean Meal", "Agriculture", "SM1 N:05_0_R Comdty", "SM1 Comdty", "SM2 Comdty"),
    ("BO", "Soybean Oil", "Agriculture", "BO1 N:05_0_R Comdty", "BO1 Comdty", "BO2 Comdty"),
    ("SB", "Sugar No.11", "Softs", "SB1 N:05_0_R Comdty", "SB1 Comdty", "SB2 Comdty"),
    ("KC", "Coffee", "Softs", "KC1 N:05_0_R Comdty", "KC1 Comdty", "KC2 Comdty"),
    ("CC", "Cocoa", "Softs", "CC1 N:05_0_R Comdty", "CC1 Comdty", "CC2 Comdty"),
    ("CT", "Cotton", "Softs", "CT1 N:05_0_R Comdty", "CT1 Comdty", "CT2 Comdty"),
    ("LC", "Live Cattle", "Livestock", "LC1 N:05_0_R Comdty", "LC1 Comdty", "LC2 Comdty"),
    ("LH", "Lean Hogs", "Livestock", "LH1 R:05_0_R Comdty", "LH1 Comdty", "LH2 Comdty"),
    ("FC", "Feeder Cattle", "Livestock", "FC1 R:05_0_R Comdty", "FC1 Comdty", "FC2 Comdty"),
]

# The four asset classes used by Hurst, Ooi and Pedersen
HOP_CLASS = {"Equity": "Equity", "Bond": "Bond", "FX": "FX", "Energy": "Commodity",
             "Metals": "Commodity", "Agriculture": "Commodity", "Softs": "Commodity",
             "Livestock": "Commodity"}

UNIVERSE = pd.DataFrame(_UNIVERSE_ROWS, columns=["code", "name", "sector", "adj_ticker", "g1", "g2"])
UNIVERSE["asset_class"] = UNIVERSE["sector"].map(HOP_CLASS)
UNIVERSE = UNIVERSE.set_index("code")

METALS_CHECK = ["GC", "SI", "HG", "PL"]  # trend on mining-related metals versus the earlier pairs study

BENCHMARKS = {
    "SPXT": "SPXT Index",          # S&P 500 total return
    "UST": "LUATTRUU Index",       # Bloomberg US Treasury total return
    "TBILL": "USGG3M Index",       # 3m T-bill yield, percent
    "AQMIX": "AQMIX US Equity",    # AQR Managed Futures fund, price only, after fees
    "DBMF": "DBMF US Equity",      # iMGP DBi Managed Futures ETF, price only, after fees
}
# In Benchmarks_Daily the AQMIX and DBMF blocks still carry the old SG headers, so blocks
# are mapped by position (block 0 = column A, block 3 = column J, block 4 = column M).
BENCHMARK_BLOCK_ORDER = ["SPXT", "UST", "TBILL", "AQMIX", "DBMF"]

FILE_PATTERNS = {
    "daily_financials": "TrendPull_1*final*.xlsx",
    "daily_commodities": "TrendPull_2*final*.xlsx",
    "carry_benchmarks_specs": "TrendPull_3*final*.xlsx",
    "carry_revised": "carry_revised*.xlsx",
    "aqr_tsmom": "Time_Series_Momentum_Factors_Monthly*.xlsx",
}


@dataclass(frozen=True)
class Params:
    """Pre-registered research choices. Changing any field counts as a new variant."""
    # trend
    lookbacks_months: tuple[int, ...] = (1, 3, 12)
    # information lag: signals use prices up to the close this many trading days before the
    # rebalance close. 1 means every input was known at the previous close.
    signal_lag_days: int = 1
    carry_lag_months: int = 1          # month-end curves are only seen at that close
    # volatility
    vol_com_days: float = 60.0         # EWMA centre of mass in days, as in MOP and HOP
    vol_min_obs: int = 60
    vol_floor_annual: float = 0.01
    # portfolio
    weighting: str = "equal_market"    # equal_market | equal_class | erc_lw
    target_vol: float = 0.10
    cov_window_months: int = 36
    cov_min_months: int = 24
    corr_window_weeks: int = 156       # for the Ledoit-Wolf correlation-aware weights
    max_gross_leverage: float = 10.0   # safety cap on the sum of absolute notional weights
    markets: tuple[str, ...] | None = None  # restrict the universe (e.g. the metals check)
    min_history_months: int = 24       # market must have this many monthly returns to trade
    # carry
    use_carry: bool = False
    trend_carry_mix: float = 0.5       # share of risk budget given to carry when use_carry
    # max |log(G2/G1)| kept: bonds catch notional coupon changes, equity and FX only bad prints
    carry_spread_cap: dict = field(default_factory=lambda: {"Equity": 0.10, "Bond": 0.05, "FX": 0.15})
    # costs
    cost_ticks_per_trade: float = 0.5  # half a tick per unit traded, base case
    include_roll_costs: bool = True
    apply_costs: bool = True
    mgmt_fee: float = 0.0              # set to 0.02 and 0.20 for the HOP 2/20 comparison
    perf_fee: float = 0.0
    # periods
    start: str = "1990-01-01"
    insample_end: str = "2016-12-31"
    test_start: str = "2017-01-01"
    end: str = "2026-10-31"

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# Published HOP-style specification: same-close execution, equal risk per market, trend only.
HOP_REPLICATION = Params(signal_lag_days=0, weighting="equal_market", use_carry=False)
BASELINE = Params()

