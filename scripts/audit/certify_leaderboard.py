#!/usr/bin/env python3
"""CERTIFY the book candidates with Hansen SPA (deep-v3 T1, 2026-06-09).

Does the BEST cached strategy's edge survive the data-snooping null, given the K candidates searched?
DSR/PBO/LOND don't pose this joint question; Hansen SPA does — stationary-bootstrap null preserving
serial + cross dependence. Uses series_cache.json (strategy OOS equity curves), aligns to common
timestamps, runs hansen_spa on the per-period return matrix (benchmark = 0 / risk-free).

Usage: python3 scripts/certify_leaderboard.py
"""
import json
import os
import sys

import numpy as np
from stats_rigor import hansen_spa
from lb.paths import ROOT as _LBROOT


def main():
    cache = json.load(open(str(_LBROOT / "results" / "series_cache.json")))
    members = [m for m, s in cache.items() if isinstance(s, dict) and len(s) >= 100]
    tssets = [set(cache[m].keys()) for m in members]
    common = sorted(set.intersection(*tssets), key=lambda x: int(x))
    R, names = [], []
    for m in members:
        eq = [float(cache[m][t]) for t in common]
        rets = [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq)) if eq[i - 1] > 0]
        if len(rets) == len(common) - 1:
            R.append(rets)
            names.append(m)
    D = np.asarray(R).T   # (n, K): per-period returns = outperformance vs risk-free
    n, K = D.shape
    print(f"Hansen SPA certification — {K} book candidates, {n} common periods\n")
    p, T, tk = hansen_spa(D, avg_block=20, B=3000, seed=42)
    order = np.argsort(-tk)
    print(f"{'rank':>4s} {'name':6s} {'t-stat':>7s} {'ann_Sharpe':>10s}")
    for r, i in enumerate(order):
        sh = float(np.mean(D[:, i]) / (np.std(D[:, i]) + 1e-12) * np.sqrt(252.0))
        print(f"{r + 1:4d} {names[i]:6s} {tk[i]:7.2f} {sh:10.2f}")
    print(f"\nHansen SPA p-value (best beats risk-free, accounting for {K} searched trials + dependence): {p:.4f}")
    print("=> BEST edge is REAL / data-snooping-robust" if p < 0.05
          else "=> best edge NOT distinguishable from data-snooping at 5%")


if __name__ == "__main__":
    main()
