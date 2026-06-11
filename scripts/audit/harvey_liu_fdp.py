#!/usr/bin/env python3
"""Harvey-Liu reduced-form false-discovery + power reckoning over the session's trials
(report deep-v3 lever T5 — the most decision-relevant honesty question: of all the
configs raced, what fraction of "winners" are FALSE, and is the loop discarding REAL
edges?). Pure host-side over results/round_results.csv (real_sharpe + n_days columns).

Per-trial t = SR_annual * sqrt(n_days/252). True-null fraction pi0 via Storey (lambda=.5);
FDP_hat = pi0*m*alpha / #discoveries; gray zone 2<|t|<=3 = candidate MISSED edges under a
strict |t|>3 bar. Cash/near-zero-vol ETFs (Sharpe-inflating artifacts) are excluded — they
otherwise dominate the t-ranking (BIL/SHV |t|~25) without being tradeable edges.

CAVEATS (honest): configs are correlated (knob-variations of the same names), so pi0/FDP
under independence are APPROXIMATE and the gray-zone count OVERSTATES independent missed
edges; t assumes ~iid returns (Lo-corrected Sharpe round showed the weekly autocorrelation
is minor, eta<=1.10).

    python3 scripts/harvey_liu_fdp.py
"""
import csv
import math
import os

from lb.paths import ROUND_RESULTS_CSV as _RR_CSV

CSV = str(_RR_CSV)
CASH = {"BIL", "SHV", "GSY", "SHM", "BIV", "SHY", "ICSH", "NEAR", "MINT", "JPST",
        "FLOT", "BSV", "VGSH", "SGOV"}


def _p2(t):
    return math.erfc(abs(t) / math.sqrt(2.0))   # two-sided normal p-value


def analyze(alpha=0.05, t_cash_cap=10.0):
    best = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if str(r.get("deployable")).lower() != "true" or r["ticker"] in CASH:
                continue
            try:
                sr, nd, tr = float(r["real_sharpe"]), float(r["n_days"]), float(r["trades"])
            except (ValueError, TypeError, KeyError):
                continue
            if nd < 60 or tr < 80:
                continue
            t = sr * math.sqrt(nd / 252.0)
            if abs(t) > t_cash_cap:               # belt-and-suspenders cash filter
                continue
            key = f"{r['ticker']}_{r['axis']}_{r['labeler']}_{r['thresh']}_{r['sizing']}"
            if key not in best or abs(t) > best[key][0]:
                best[key] = (abs(t), t, _p2(t), sr, int(nd), r["ticker"])
    m = len(best)
    ps = [v[2] for v in best.values()]
    pi0 = min(1.0, sum(1 for p in ps if p > 0.5) / (0.5 * m)) if m else 1.0
    ndisc = sum(1 for p in ps if p < alpha)
    fdp = (pi0 * m * alpha) / ndisc if ndisc else 0.0
    nt3 = sum(1 for v in best.values() if v[0] > 3.0)
    gray = sum(1 for v in best.values() if 2.0 < v[0] <= 3.0)
    pos_top = sorted((v for v in best.values() if v[1] > 0), key=lambda v: -v[0])[:8]
    return dict(m=m, pi0=pi0, ndisc=ndisc, fdp=fdp, nt3=nt3, gray=gray, pos_top=pos_top)


if __name__ == "__main__":
    a = analyze()
    print(f"tradeable distinct configs (trials): {a['m']}")
    print(f"  pi0 (true-null fraction)   = {a['pi0']:.3f}  (~{int(a['pi0']*a['m'])} noise)")
    print(f"  discoveries @ p<0.05       = {a['ndisc']}  ->  FDP_hat = {a['fdp']:.3f}")
    print(f"  strict |t|>3               = {a['nt3']}   gray 2<|t|<=3 (candidate missed) = {a['gray']}")
    print("  strongest POSITIVE tradeable edges by autocorrelation-robust |t|:")
    for v in a["pos_top"]:
        print(f"    |t|={v[0]:5.2f}  SR={v[3]:5.2f}  ndays={v[4]}  {v[5]}")
