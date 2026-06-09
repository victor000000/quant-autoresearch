#!/usr/bin/env python3
"""SPEC-CURVE / Non-Standard-Errors robustness (deep-v2 A3, Menkveld 2024 / Simonsohn spec-curve).

The DSR haircut deflates best-of-N but never asks: is a crown's Calmar TYPICAL of its defensible
specs, or a lucky OUTLIER (knife-edge)? This reads the ledger (round_results.csv) and, per champion,
computes the Calmar DISPERSION across its defensible spec subset:
    defensible = carrying axes {logdollar, imbalance} x sizings {cdf_overlay, dd_overlay, cdf_plain}
                 x deployable trials (trades > 80, finite real_calmar)
A crown near the MEDIAN of this set (low dispersion) is spec-ROBUST; a crown that is the lone MAX
(others fail) is spec-FRAGILE — which argues for the conservative book over stacking that member.

Diagnostic only; no model/leak/deploy surface. Informs the human/Opus aggressive-vs-conservative crown.
"""
import csv
import os
import statistics as st

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "round_results.csv")
AXES = {"logdollar", "imbalance"}
SIZERS = {"cdf_overlay", "dd_overlay", "cdf_plain"}
# (ticker, the crown's deployed Calmar for reference)
CROWNS = [("GLD", 4.02), ("USO", 3.85), ("UUP", 0.44), ("IWM", 0.47),
          ("IXP", 1.97), ("AAXJ", 2.43), ("EWL", 1.98), ("DJP", 2.01)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(open(LEDGER)))
    print(f"{'crown':5s} {'n_spec':>6s} {'median':>7s} {'std(NSE)':>8s} {'min':>6s} {'max':>6s} "
          f"{'crown_C':>7s} {'pctile':>6s}  verdict")
    out = []
    for tk, crown_cal in CROWNS:
        cals = []
        for r in rows:
            if r.get("ticker") != tk:
                continue
            if r.get("axis") not in AXES or r.get("sizing") not in SIZERS:
                continue
            tr = _f(r.get("trades"))
            c = _f(r.get("real_calmar"))
            if c is None or tr is None or tr < 80:
                continue
            cals.append(c)
        n = len(cals)
        if n < 3:
            print(f"{tk:5s} {n:6d}   (too few defensible specs to assess)")
            out.append((tk, n, None))
            continue
        med = st.median(cals)
        sd = st.pstdev(cals)
        lo, hi = min(cals), max(cals)
        pct = 100.0 * sum(1 for c in cals if c <= crown_cal) / n
        # robust: crown near/below median AND dispersion modest relative to level; fragile: crown is the lone max.
        rng = hi - lo if hi > lo else 1e-9
        crown_is_topish = (crown_cal >= hi - 0.10 * rng)
        if crown_is_topish and (med < 0.5 * crown_cal):
            verdict = "FRAGILE (crown=outlier max; others fail)"
        elif med >= 0.6 * crown_cal and sd <= 0.5 * max(med, 1e-9):
            verdict = "ROBUST (crown typical of defensible set)"
        else:
            verdict = "MIXED"
        print(f"{tk:5s} {n:6d} {med:7.2f} {sd:8.2f} {lo:6.2f} {hi:6.2f} {crown_cal:7.2f} {pct:5.0f}%  {verdict}")
        out.append((tk, n, verdict))
    print("\nRead: spec-robustness across defensible specs (carrying axes x main sizers, deployable).")
    print("ROBUST -> trustworthy for the aggressive book; FRAGILE -> argues for the conservative core.")
    return out


if __name__ == "__main__":
    main()
