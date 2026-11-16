"""One command that rebuilds every number, table and chart.

Periods:
  insample  data is cut at the end of 2016 before anything runs, so the test years are never
            touched. Use this for all development.
  full      runs through the latest month and reports the locked 2017-2026 period. For real
            data the first full run writes a lock record; any later run with different
            settings is refused unless a reason is given, and that reason is logged as an
            extra trial for the deflated Sharpe ratio.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from . import checks, reporting
from . import config as C
from .backtest import Inputs, Result, prepare, run
from .costs import apply_fees, relative_tick_check
from .data import Dataset, load
from .stats import bootstrap_sharpe, deflated_sharpe, regress, sharpe, summary

B = C.BASELINE
VARIANTS: dict[str, C.Params] = {
    "hop_replication": C.HOP_REPLICATION,
    "baseline": B,
    "equal_class": B.with_(weighting="equal_class"),
    "erc_lw": B.with_(weighting="erc_lw"),
    "trend_carry": B.with_(use_carry=True),
    "carry_only": B.with_(use_carry=True, trend_carry_mix=1.0),
    "trend_1m": B.with_(lookbacks_months=(1,)),
    "trend_3m": B.with_(lookbacks_months=(3,)),
    "trend_12m": B.with_(lookbacks_months=(12,)),
    "cost_quarter_tick": B.with_(cost_ticks_per_trade=0.25),
    "cost_one_tick": B.with_(cost_ticks_per_trade=1.0),
    "metals_only": B.with_(markets=tuple(C.METALS_CHECK)),
}
# Variants that compete to be "the strategy" and so count as trials for the deflated Sharpe
STRATEGY_TRIALS = ["hop_replication", "baseline", "equal_class", "erc_lw", "trend_carry", "carry_only",
                   "trend_1m", "trend_3m", "trend_12m"]


class LockedPeriodError(RuntimeError):
    pass


def data_fingerprint(ds: Dataset) -> str:
    h = pd.util.hash_pandas_object(ds.prices.fillna(-1.0), index=True).sum()
    return f"{int(h) & 0xFFFFFFFFFFFF:012x}"


def check_lock(ds: Dataset, variants: dict[str, C.Params], rerun_reason: str | None) -> dict:
    C.RESULTS.joinpath("locked").mkdir(parents=True, exist_ok=True)
    path = C.RESULTS / "locked" / f"{ds.source}_lock.json"
    current = {"data": data_fingerprint(ds), "params": {k: v.fingerprint() for k, v in variants.items()}}
    now = dt.datetime.now().isoformat(timespec="seconds")
    if not path.exists():
        rec = {"first_run": now, **current, "runs": [{"time": now, "kind": "first"}], "extra_trials": 0}
    else:
        rec = json.loads(path.read_text())
        same = rec["data"] == current["data"] and rec["params"] == current["params"]
        if same:
            rec["runs"].append({"time": now, "kind": "reproduction"})
        elif rerun_reason:
            rec["runs"].append({"time": now, "kind": "changed", "reason": rerun_reason, **current})
            rec["extra_trials"] += 1
        else:
            raise LockedPeriodError(
                f"The locked test period was already run on {rec['first_run']} with different settings "
                f"or data. Re-running needs --rerun-locked 'reason', which is logged and counted as a trial.")
    path.write_text(json.dumps(rec, indent=2))
    return rec


def fund_events(ds: Dataset) -> pd.DataFrame:
    if ds.source == "synthetic":
        f = C.DATA_SYNTHETIC / "injected_faults.json"
        if f.exists():
            inj = json.loads(f.read_text())["injected"]
            return pd.DataFrame([{"fund": i["code"], "ex_date": i["date"], "amount": i["amount"]}
                                 for i in inj if i["kind"] == "distribution"])
    f = C.DATA_REFERENCE / "fund_distributions.csv"
    return pd.read_csv(f, comment="#") if f.exists() else pd.DataFrame(columns=["fund", "ex_date", "amount"])


def run_all(dataset: str = "synthetic", period: str = "insample", rerun_reason: str | None = None,
            bootstrap_reps: int = 5000, folder: Path | None = None) -> Path:
    ds = load(dataset, folder=folder)
    out = C.RESULTS / dataset / period
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    T, F = out / "tables", out / "figures"

    end = B.insample_end if period == "insample" else B.end
    variants = {k: v.with_(end=end) for k, v in VARIANTS.items()}
    lock = None
    if period == "full" and dataset == "real":
        lock = check_lock(ds, variants, rerun_reason)

    # ---------------------------------------------------- data checks
    chk = checks.run_all(ds, out / "data_checks")
    relative_tick_check(ds.specs, ds.g1).to_csv(out / "data_checks" / "relative_tick.csv")

    # ---------------------------------------------------- backtests
    results: dict[str, Result] = {}
    inputs: dict[str, Inputs] = {}
    for name, p in variants.items():
        inputs[name] = inp = prepare(ds, p)
        results[name] = run(inp, p)
    inp0, base = inputs["baseline"], results["baseline"]
    inp0.cleaning_log.to_csv(out / "data_checks" / "cleaning_log.csv", index=False)
    inp0.curve_log.to_csv(out / "data_checks" / "curve_cleaning_log.csv", index=False)

    def seg(x: pd.Series, which: str) -> pd.Series:
        if which == "insample":
            return x.loc[:pd.Period(B.insample_end, "M")]
        if which == "test":
            return x.loc[pd.Period(B.test_start, "M"):]
        return x

    segments = ["insample"] if period == "insample" else ["insample", "test", "full"]

    # headline table
    rows = []
    for name, r in results.items():
        for sname in segments:
            for col in ("gross", "net"):
                s = summary(seg(r.returns[col], sname))
                rows.append({"variant": name, "segment": sname, "returns": col, **s})
    headline = pd.DataFrame(rows).set_index(["variant", "segment", "returns"])
    reporting.write_table(headline, T / "headline")

    # HOP net of 2/20 fees on top of costs
    hop = results["hop_replication"].returns
    hop_fees = apply_fees(hop["net"], hop["cash"], 0.02, 0.20)

    # ---------------------------------------------------- statistics
    stats_rows = {}
    for name in ("baseline", "hop_replication", "trend_carry", "erc_lw", "equal_class"):
        for sname in segments:
            x = seg(results[name].returns["net"], sname)
            stats_rows[(name, sname)] = bootstrap_sharpe(x, reps=bootstrap_reps, seed=1)
    boot = pd.DataFrame(stats_rows).T
    reporting.write_table(boot, T / "bootstrap_sharpe")

    insample_sr = [sharpe(seg(results[k].returns["net"], "insample")) for k in STRATEGY_TRIALS]
    extra = lock["extra_trials"] if lock else 0
    insample_sr += [np.mean(insample_sr)] * extra  # each logged re-run counts as one more trial
    best = STRATEGY_TRIALS[int(np.nanargmax(insample_sr[:len(STRATEGY_TRIALS)]))]
    dsr = {k: deflated_sharpe(seg(results[k].returns["net"], "insample"), insample_sr)
           for k in ("baseline", best)}
    reporting.write_table(pd.DataFrame(dsr).T, T / "deflated_sharpe")

    # realised versus target volatility, leverage, turnover, costs
    diag_rows = {}
    for name, r in results.items():
        x = r.returns
        diag_rows[name] = {"target_vol": r.params.target_vol, "realised_vol_gross": x["gross"].std() * np.sqrt(12),
                           "avg_markets": r.diagnostics["n_markets"].mean() if len(r.diagnostics) else np.nan,
                           "avg_gross_leverage": r.diagnostics["gross"].mean() if len(r.diagnostics) else np.nan,
                           "annual_turnover": x["turnover"].mean() * 12,
                           "trading_cost_pa": x["trading_cost"].mean() * 12, "roll_cost_pa": x["roll_cost"].mean() * 12,
                           "sharpe_gross": sharpe(x["gross"]), "sharpe_net": sharpe(x["net"])}
    reporting.write_table(pd.DataFrame(diag_rows).T, T / "diagnostics")

    # ---------------------------------------------------- benchmarks and factor exposure
    net = base.returns["net"]
    cash = base.returns["cash"]
    eq = A.monthly_from_daily_level(ds.bench["SPXT"])
    reg = regress(net, (eq - cash.reindex(eq.index).fillna(0)).rename("SPX_excess").to_frame())
    reg.to_csv(T / "equity_regression.csv")
    events = fund_events(ds)
    fund_rows = []
    for fund in ("AQMIX", "DBMF"):
        if fund not in ds.bench or ds.bench[fund].dropna().empty:
            continue
        ev = events[events.fund == fund] if len(events) else events
        adj, check = A.adjust_distributions(ds.bench[fund], ev)
        check.to_csv(T / f"{fund}_distribution_check.csv", index=False)
        fr = A.monthly_from_daily_level(adj)
        fr_ex = fr - cash.reindex(fr.index).fillna(0)
        idx = net.index.intersection(fr_ex.index)
        if len(idx) > 12:
            r = regress(net.loc[idx], fr_ex.loc[idx].rename(fund).to_frame())
            fund_rows.append({"fund": fund, "months": len(idx), "correlation": net.loc[idx].corr(fr_ex.loc[idx]),
                              "beta": r.loc[fund, "coef"], "beta_t_nw": r.loc[fund, "t_nw"],
                              "events_applied": int(check["applied"].sum()) if len(check) else 0})
    if fund_rows:
        reporting.write_table(pd.DataFrame(fund_rows).set_index("fund"), T / "fund_comparison")

    # ---------------------------------------------------- attribution
    reporting.write_table(A.by_group(base.market_pnl, "asset_class"), T / "attribution_asset_class")
    reporting.write_table(A.by_group(base.market_pnl, "sector"), T / "attribution_sector")
    reporting.write_table(A.contribution_table(base.speed_pnl), T / "attribution_speed")
    tc = results["trend_carry"]
    reporting.write_table(A.contribution_table(tc.sleeve_pnl), T / "attribution_trend_vs_carry")
    corr_tc = results["baseline"].returns["net"].corr(results["carry_only"].returns["net"])
    tc_reg = regress(results["trend_carry"].returns["net"],
                     results["baseline"].returns["net"].rename("trend").to_frame())

    # ---------------------------------------------------- stress, overlay, crises
    trend_total = net + cash
    sixty = A.sixty_forty(ds.bench)
    stress = A.stress_table({"baseline_net": net, "trend_carry_net": results["trend_carry"].returns["net"],
                             "sixty_forty": sixty, "spx": eq})
    reporting.write_table(stress, T / "stress_windows")
    strength = A.trend_strength(inp0.monthly, inp0.vol_info)
    reporting.write_table(A.quiet_spells(net, strength), T / "quiet_spells")
    ov, paths = A.overlay(trend_total, ds.bench, cash)
    reporting.write_table(ov, T / "overlay_6040")
    crisis = A.crisis_table(trend_total, ds.bench)
    reporting.write_table(crisis, T / "crisis_table")

    # ---------------------------------------------------- replication
    rep = {"hop_gross": hop["gross"], "hop_net_costs": hop["net"], "hop_net_costs_fees": hop_fees}
    dec_edges = ["1990-01", "2000-01", "2010-01", "2017-01", "2027-01"]
    decades = pd.concat({k: A.by_period(v, dec_edges) for k, v in rep.items()}, axis=0)
    reporting.write_table(decades, T / "replication_by_period")
    aqr_tab = None
    if ds.aqr is not None:
        aqr_tab = A.compare_aqr(hop["gross"], ds.aqr)
        reporting.write_table(aqr_tab, T / "replication_vs_aqr_factor")
    hop_ref = C.DATA_REFERENCE / "hop_published.csv"
    if hop_ref.exists():
        pub = pd.read_csv(hop_ref, comment="#")
        if len(pub):
            pub.to_csv(T / "hop_published_reference.csv", index=False)

    # metals check (gold, silver, copper, platinum)
    metals = results["metals_only"].returns["net"]
    reporting.write_table(pd.DataFrame({"metals_trend_net": summary(metals)}), T / "metals_check")

    # ---------------------------------------------------- figures
    test_start = B.test_start if period == "full" else None
    series = {"Trend, net of costs": net}
    if ds.aqr is not None:
        a = ds.aqr["TSMOM"].reindex(net.index)
        a = a * (net.std() / a.std())
        series["AQR TSMOM factor, same volatility"] = a
    reporting.cumulative(series, F / "cumulative.png", "Trend portfolio, 10% volatility target", test_start)
    reporting.drawdowns({"Trend, net": net}, F / "drawdown.png", "Drawdown")
    reporting.rolling_sharpe(net, F / "rolling_sharpe.png")
    reporting.bars(A.by_group(base.market_pnl, "sector")["ann_contribution"].drop("Total"),
                   F / "attribution_sector.png", "Annual return contribution by sector")
    reporting.drawdowns({"60/40": paths["0% trend"], "80% 60/40 + 20% trend": paths["20% trend"]},
                        F / "overlay_drawdown.png", "60/40 with and without a 20% trend allocation")
    if len(crisis):
        lab = crisis.assign(label=[f"{a} to {b}" for a, b in zip(crisis.start, crisis.trough)]).set_index("label")
        reporting.grouped_bars(lab[["sixty_forty_return", "trend_return"]].rename(
            columns={"sixty_forty_return": "60/40", "trend_return": "Trend"}),
            F / "crisis_table.png", "Worst 60/40 drawdowns and trend over the same months")

    # ---------------------------------------------------- summary
    def s(x):
        return summary(x)

    summ = {
        "dataset": dataset, "period": period, "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "params_baseline": B.with_(end=end).to_dict(),
        "data": {"markets": int(ds.prices.shape[1]), "first_date": str(ds.prices.index[0].date()),
                 "last_date": str(ds.prices.index[-1].date()), "notes": ds.notes,
                 "stale_runs": int(len(chk["stale_runs"])), "outliers_z7": int(len(chk["outliers"])),
                 "outliers_reverting": int(chk["outliers"]["reverts"].sum()) if len(chk["outliers"]) else 0,
                 "prices_removed_by_cleaning": int(len(inp0.cleaning_log)),
                 "curve_months_removed": int(len(inp0.curve_log))},
        "baseline_net": {k: s(seg(net, k)) for k in segments},
        "hop_replication": {"gross": s(hop["gross"]), "net_costs": s(hop["net"]), "net_costs_fees": s(hop_fees)},
        "trend_vs_carry": {"corr_trend_carry": corr_tc, "trend_carry_on_trend": tc_reg.to_dict()},
        "equity_regression": reg.to_dict(),
        "deflated_sharpe": dsr, "best_insample_variant": best,
        "overlay": {"corr_trend_equity": ov.attrs["corr_trend_equity_all"],
                    "corr_trend_equity_selloffs": ov.attrs["corr_trend_equity_selloffs"]},
        "crisis_trend_positive": f"{crisis.attrs.get('trend_positive')} of {len(crisis)}",
        "aqr_correlation": aqr_tab["correlation"].to_dict() if aqr_tab is not None else None,
        "metals": s(metals),
        "lock": lock,
    }
    reporting.write_json(summ, out / "summary.json")
    write_report(out, summ, headline, segments)
    return out


def write_report(out: Path, summ: dict, headline: pd.DataFrame, segments: list[str]) -> None:
    """Numbers-only report generated from the outputs. Prose for the note is written by hand."""
    lines = [f"# Results: {summ['dataset']} data, {summ['period']} run", "",
             f"Generated {summ['generated']}. Every figure below is read from the tables folder.", ""]
    hl = headline.xs("net", level="returns")[["months", "ann_return", "ann_vol", "sharpe", "max_drawdown",
                                              "nw_tstat"]]
    lines += ["## Net of costs by variant", "", reporting._plain_markdown(hl, ".3f"), ""]
    b = summ["baseline_net"]
    for k in segments:
        x = b[k]
        lines.append(f"- Baseline {k}: Sharpe {x['sharpe']:.2f}, return {x['ann_return']:.1%}, "
                     f"vol {x['ann_vol']:.1%}, max drawdown {x['max_drawdown']:.1%} over {x['months']} months")
    d = summ["deflated_sharpe"]["baseline"]
    lines += ["", f"- Deflated Sharpe probability (baseline, {d['n_trials']} trials): {d['deflated_sharpe_prob']:.2f}",
              f"- Trend and carry correlation: {summ['trend_vs_carry']['corr_trend_carry']:.2f}",
              f"- Correlation with S&P 500: {summ['overlay']['corr_trend_equity']:.2f} overall, "
              f"{summ['overlay']['corr_trend_equity_selloffs']:.2f} in the worst 10% of equity months",
              f"- Trend positive in {summ['crisis_trend_positive']} worst 60/40 drawdowns"]
    if summ["aqr_correlation"]:
        lines.append(f"- Correlation of replication (gross) with AQR TSMOM factor: "
                     f"{summ['aqr_correlation'].get('TSMOM', float('nan')):.2f}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
