#!/usr/bin/env python3
"""Universe-screen FIT report — parses results/round_results.csv and lists genuine FITs from the
trend-vs-buyhold screen (screen_etfs.py), correctly pairing each ETF's 5-mechanism method panel
(trend_leg+regime_gmm / bgm+ker / ker / changepoint / sadf_explosive) against its OWN always_long
buy-hold baseline. FIT = best method Calmar > buy-hold AND val_auc>0.55 AND trades>80 (program.md:
matching buy-hold is NO-FIT). Pure CSV analysis (no QC), so it never competes with the coordinator.

  python3 scripts/screen_fit_report.py            # ranked fit table (latest pass per ETF)
"""
import csv
import os
from lb.paths import ROUND_RESULTS_CSV as _RR_CSV

CSV = str(_RR_CSV)
# fixed-12 + current book are not screen targets (their edges are separately tracked)
EXCLUDE = set("DBC EEM EFA FXY GLD HYG ITB IWM KRE QQQ SLV SMH SOXX SPY TIP TLT UUP VIXY XBI XLE "
              "XME USO".split())
VAL_AUC_MIN, TRADES_MIN = 0.55, 80


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.reader(open(CSV)))
    # per ETF: latest buy-hold Calmar + best method (Calmar, labeler, trades, val_auc), scanning in
    # order so later (re-screened) rows overwrite earlier ones.
    bh, best = {}, {}
    for r in rows:
        if len(r) < 18:
            continue
        etf, lab = r[5], r[7]
        cal, va, tr = _f(r[10]), _f(r[16]), _f(r[12])
        if cal is None:
            continue
        if lab == "always_long":
            bh[etf] = cal                                  # latest buy-hold baseline
        else:
            cur = best.get(etf)
            if cur is None or cal > cur[0]:
                best[etf] = (cal, lab, int(tr or 0), va or 0.0)
    fits = []
    for etf, (cal, lab, tr, va) in best.items():
        if etf in EXCLUDE or etf not in bh:
            continue
        b = bh[etf]
        if cal > b and va > VAL_AUC_MIN and tr > TRADES_MIN:
            fits.append((etf, cal, b, cal - b, lab, tr, va))
    fits.sort(key=lambda x: -x[3])                          # rank by edge over buy-hold
    print(f"Universe-screen FITs (method > own buy-hold, val_auc>{VAL_AUC_MIN}, trades>{TRADES_MIN}):")
    print(f"  {'ETF':5s} {'Calmar':>7s} {'BuyHold':>7s} {'Edge':>6s} {'val_auc':>7s} {'trades':>6s}  method")
    for etf, cal, b, edge, lab, tr, va in fits:
        print(f"  {etf:5s} {cal:7.3f} {b:7.3f} {edge:+6.2f} {va:7.2f} {tr:6d}  {lab}")
    print(f"\n{len(fits)} fits. Screen-strong != book seat: each must still clear permute + decay + "
          f"deflation in the deep-sweep before any book consideration. Edge>0.3 + Calmar>1.3 = worth "
          f"validating first.")
    return fits


if __name__ == "__main__":
    main()
