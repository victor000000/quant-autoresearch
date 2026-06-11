#!/usr/bin/env python3
"""Stationary block-bootstrap Calmar confidence interval for a deployed book member
(report deep-v3 lever T4 — the autocorrelation-AWARE uncertainty band the iid permute
control cannot give: the permute destroys serial structure, so it can't say whether a
Calmar is robust to resampling that PRESERVES momentum/drawdown autocorrelation).

Hypothesis tested per member: does its OOS Calmar 95%-CI LOWER bound still exceed its
buy-and-hold Calmar? A YES means the edge survives an autocorrelation-preserving
Monte-Carlo, not just the iid label-permute.

Pure host-side (no QC, no new data): reads results/series_cache.json (deployed member
equity curves on the common 224-pt OOS grid) + knowledge.json buy-hold Calmars.

    python3 scripts/bootstrap_calmar_ci.py            # GLD (+ default members)
    python3 scripts/bootstrap_calmar_ci.py GLD UUP IWM
"""
import json
import os
import sys

import numpy as np

from stats_rigor import stationary_bootstrap  # noqa: E402
from lb.paths import ROOT as _LBROOT

ROOT = str(_LBROOT)
SERIES = os.path.join(ROOT, "results", "series_cache.json")
KJ = os.path.join(ROOT, "knowledge.json")
SECS_YR = 365.25 * 86400.0


def _load_equity(ticker):
    d = json.load(open(SERIES))[ticker]
    ts = sorted(d.keys(), key=lambda k: int(k))
    eq = np.array([float(d[k]) for k in ts], dtype=float)
    years = (int(ts[-1]) - int(ts[0])) / SECS_YR
    return eq, years


def _calmar_factory(years):
    """Annualized Calmar of a return series, computed over the SAME calendar span
    (so a length-n resample maps to the original window)."""
    def calmar(r):
        eq = np.cumprod(1.0 + r)
        if eq[-1] <= 0:
            return -1.0
        cagr = eq[-1] ** (1.0 / years) - 1.0
        rm = np.maximum.accumulate(eq)
        mdd = float(np.max((rm - eq) / rm))
        return (cagr / mdd) if mdd > 1e-9 else 0.0
    return calmar


def run(ticker="GLD", B=8000, blocks=(8, 12, 20), seed=42):
    eq, years = _load_equity(ticker)
    r = np.diff(eq) / eq[:-1]
    n = len(r)
    calmar = _calmar_factory(years)
    point = calmar(r)
    bh = json.load(open(KJ)).get("buyhold", {}).get(ticker, {}).get("calmar")
    print(f"{ticker}: n={n} weekly OOS pts · {years:.2f}y · point Calmar={point:.3f}"
          + (f" · buy-hold={bh:.3f}" if bh is not None else " · buy-hold=?"))
    verdicts = []
    for ab in blocks:
        dist = stationary_bootstrap(r, avg_block=ab, B=B, seed=seed, statfn=calmar)
        lo, med, hi = np.percentile(dist, [2.5, 50, 97.5])
        excl = (bh is not None) and (lo > bh)
        frac = float(np.mean(dist > bh)) if bh is not None else None
        verdicts.append(excl)
        fr = f" · P(>BH)={frac:.3f}" if frac is not None else ""
        print(f"  block~{ab:>2}: 95% CI [{lo:6.3f}, {hi:6.3f}] · median {med:6.3f}"
              + (f" · CI-lower>BH? {excl}" if bh is not None else "") + fr)
    if bh is not None:
        robust = all(verdicts)
        print(f"  => GLD-style edge robust to autocorrelation-aware resampling: "
              f"{'YES (CI excludes buy-hold at every block size)' if robust else 'NO (CI overlaps buy-hold)'}")
    return point, bh


if __name__ == "__main__":
    tks = sys.argv[1:] or ["GLD", "UUP", "IWM", "USO"]
    for t in tks:
        try:
            run(t)
        except Exception as e:
            print(f"{t}: skipped ({type(e).__name__}: {e})")
        print()
