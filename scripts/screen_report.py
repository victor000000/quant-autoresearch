#!/usr/bin/env python3
"""Classify the universe screen: methodology (trend OR regime leg) vs always_long (buy-hold) per ETF.
Writes results/etf_screen.csv (the persistent FIT map) and prints a tally. FIT = methodology BEATS
buy-hold AND val_auc>0.55 AND deployable; STRONG = also Calmar>1.0 and edge>0.3."""
import csv, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROG = os.path.join(HERE, "results", "etf_screen_progress.log")
RR = os.path.join(HERE, "results", "round_results.csv")
UNI = os.path.join(HERE, "results", "etf_qc_confirmed_pre2009.csv")
OUT = os.path.join(HERE, "results", "etf_screen.csv")

done = set()
if os.path.exists(PROG):
    for l in open(PROG):
        if l.startswith("DONE "):
            done.add(l.split()[1])
ac = {r["Ticker"].strip(): r.get("Asset_Class", "?") for r in csv.DictReader(open(UNI))}
name = {r["Ticker"].strip(): r.get("Name", "") for r in csv.DictReader(open(UNI))}

rows = list(csv.reader(open(RR)))
hdr = rows[0]
def col(n): return hdr.index(n) if n in hdr else None
ti, ci, vi, tri, li, tsi = (col("ticker"), col("real_calmar") or col("calmar"),
                            col("val_auc"), col("trades"), col("labeler"), col("timestamp"))
meth, bh = {}, {}
for r in rows[1:]:
    try:
        tk = r[ti]
        if tk not in done:
            continue
        lab, cal, va, tr, ts = r[li], float(r[ci]), float(r[vi]), int(float(r[tri])), r[tsi]
    except Exception:
        continue
    if lab.startswith("always_long"):
        if tk not in bh or ts > bh[tk][1]:
            bh[tk] = (cal, ts)
    else:
        if tk not in meth or ts > meth[tk][4]:
            meth[tk] = (cal, va, tr, lab, ts)

out = []
for tk in done:
    if tk not in meth or tk not in bh:
        continue
    mc, va, tr, lab, _ = meth[tk]
    bc = bh[tk][0]
    edge = mc - bc
    rec = "regime" if "bgm" in lab else "trend"
    # ARTIFACT guard: Calmar = CAGR/MaxDD blows up for cash-like / near-flat assets (T-bills, ultra-short
    # duration) whose MaxDD ~ 0 -> a "Calmar 30" is a division artifact, not a tradeable edge. No real ETF
    # edge exceeds ~4 (GLD). Flag implausibly high Calmar OR a flat buy-hold (cash) with a big "edge".
    artifact = (mc >= 8.0) or (abs(bc) < 0.05 and mc > 3.0)
    if artifact:
        verdict = "ARTIFACT(cash)"
    elif mc > bc and va == va and va > 0.55 and tr > 80:
        verdict = "STRONG" if (mc > 1.0 and edge > 0.3) else "marginal"
    else:
        verdict = "NO-FIT"
    out.append(dict(ticker=tk, asset_class=ac.get(tk, "?"), recipe=rec,
                    method_calmar=round(mc, 3), buyhold_calmar=round(bc, 3),
                    edge=round(edge, 3), val_auc=round(va, 3) if va == va else "nan",
                    trades=tr, verdict=verdict, name=name.get(tk, "")[:40]))
out.sort(key=lambda d: -d["edge"])
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker", "asset_class", "recipe", "method_calmar",
                                      "buyhold_calmar", "edge", "val_auc", "trades", "verdict", "name"])
    w.writeheader()
    w.writerows(out)

strong = [d for d in out if d["verdict"] == "STRONG"]
marg = [d for d in out if d["verdict"] == "marginal"]
nofit = [d for d in out if d["verdict"] == "NO-FIT"]
artifact = [d for d in out if d["verdict"] == "ARTIFACT(cash)"]
from collections import Counter
print(f"=== {len(done)} screened, {len(out)} classified -> {OUT} ===")
print("STRONG FITS (methodology beats buy-hold, deployable, plausible Calmar):")
for d in strong:
    print(f"  {d['ticker']:6s} {d['asset_class'][:13]:13s} {d['recipe']:6s} "
          f"{d['method_calmar']:+.2f} vs B&H {d['buyhold_calmar']:+.2f}  edge +{d['edge']:.2f}  {d['name'][:28]}")
print(f"marginal: {[d['ticker'] for d in marg]}")
if artifact:
    print(f"ARTIFACTS (cash-like, Calmar inflated by ~0 MaxDD — NOT real edges): {[(d['ticker'], d['method_calmar']) for d in artifact]}")
print(f">>> {len(strong)} STRONG, {len(marg)} marginal, {len(artifact)} artifact, {len(nofit)} no-fit | classes: {dict(Counter(d['asset_class'][:8] for d in out))}")
