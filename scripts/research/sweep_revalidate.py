#!/usr/bin/env python3
"""Post-process the per-ETF module sweep (user 2026-06-08: "best methodology through all modules").

For each ETF, find the CURRENT-TRUTH best DEPLOYABLE config from TODAY'S fresh re-runs, compare to the
stored per_etf_best, and classify: REPRODUCES/IMPROVED, DECAYED-SOME, or STALE (fresh << stored). The
sweep only re-ran each champion's own labeler across sizer x reduce — so a STALE verdict means the
LABELER decayed and the ETF needs a deep_sweep labeler re-sweep (sizer/reduce can't fix a dead mechanism).

Usage: python3 scripts/sweep_revalidate.py [YYYY-MM-DD]   (default today's sweep date 2026-06-09)
"""
import json
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-06-09"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    k = json.load(open(os.path.join(HERE, "knowledge.json")))
    pe = k.get("per_etf_best", {})
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "round_results.csv"))))
    print(f"Module-sweep re-validation (fresh runs on/after {DATE})\n")
    print(f"{'ETF':6s} {'stored':>7s} {'fresh':>7s} {'tr':>4s}  {'best_sizer':>10s}/{'reduce':<6s}  verdict")
    stale, repro = [], []
    for tk, v in sorted(pe.items(), key=lambda x: -(x[1].get("real_calmar") or 0)):
        cfg = v.get("config") or {}
        lab = cfg.get("labeler")
        if not lab or lab == "always_long":
            continue
        stored = v.get("real_calmar") or 0.0
        # today's fresh rows for this ETF + its champion labeler
        fr = [r for r in rows
              if r.get("ticker") == tk and r.get("labeler") == lab
              and str(r.get("timestamp", "")).startswith(DATE)]
        dep = [(f(r.get("real_calmar")), int(f(r.get("trades")) or 0), r.get("sizing"))
               for r in fr if (f(r.get("trades")) or 0) >= 80 and f(r.get("real_calmar")) is not None]
        if not fr:
            print(f"{tk:6s} {stored:7.2f} {'--':>7s} {'--':>4s}  {'(not yet swept)':>17s}")
            continue
        if not dep:
            print(f"{tk:6s} {stored:7.2f} {'none':>7s} {0:4d}  {'NO-DEPLOYABLE-FRESH':>17s}  STALE")
            stale.append(tk)
            continue
        best, bt, bsz = max(dep, key=lambda x: x[0])
        if best < 0.5 * stored:
            verdict, lst = f"STALE ({best:.2f}<<{stored:.2f})", stale
        elif best >= 0.95 * stored:
            verdict, lst = "REPRODUCES/IMPROVED", repro
        else:
            verdict, lst = "DECAYED-SOME", repro
        lst.append(tk)
        print(f"{tk:6s} {stored:7.2f} {best:7.2f} {bt:4d}  {str(bsz):>17s}  {verdict}")
    print(f"\nREPRODUCES/holds ({len(repro)}): {repro}")
    print(f"STALE — need labeler re-sweep ({len(stale)}): {stale}")


if __name__ == "__main__":
    main()
