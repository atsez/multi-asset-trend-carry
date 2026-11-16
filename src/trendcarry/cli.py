"""Command line entry point.

    trendcarry synthetic [--small]             write synthetic workbooks to data/synthetic
    trendcarry checks --dataset real           data checks only
    trendcarry run --dataset real              in-sample run (1990-2016), the default
    trendcarry run --dataset real --period full   locked run including 2017-2026
"""
from __future__ import annotations

import argparse

from . import checks, config as C, synthetic
from .data import load
from .pipeline import run_all


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="trendcarry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--small", action="store_true")
    s.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("checks")
    c.add_argument("--dataset", default="real", choices=["real", "synthetic"])
    r = sub.add_parser("run")
    r.add_argument("--dataset", default="synthetic", choices=["real", "synthetic"])
    r.add_argument("--period", default="insample", choices=["insample", "full"])
    r.add_argument("--rerun-locked", default=None, metavar="REASON")
    r.add_argument("--small", action="store_true", help="fewer bootstrap draws, for CI")
    a = ap.parse_args(argv)

    if a.cmd == "synthetic":
        print("wrote", synthetic.build(seed=a.seed, small=a.small))
    elif a.cmd == "checks":
        out = C.RESULTS / a.dataset / "data_checks"
        res = checks.run_all(load(a.dataset), out)
        for k, v in res.items():
            print(f"{k}: {len(v)} rows -> {out / (k + '.csv')}")
    elif a.cmd == "run":
        out = run_all(a.dataset, a.period, a.rerun_locked, bootstrap_reps=500 if a.small else 5000)
        print("results in", out)


if __name__ == "__main__":
    main()
