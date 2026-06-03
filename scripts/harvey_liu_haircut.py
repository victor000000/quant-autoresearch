#!/usr/bin/env python3
"""HARVEY-LIU (2015) HAIRCUT SHARPE — an INDEPENDENT multiple-testing cross-check of the DSR audit.

DSR (honest_audit.py) deflates via the extreme-value E[max] benchmark, which depends on the VARIANCE
(dispersion) of the trial Sharpes. Harvey-Liu instead adjusts each champion's OWN Sharpe t-statistic
p-value for the number of tests M (Bonferroni FWER, Benjamini-Hochberg FDR), then maps the adjusted
p back to a 'haircut' Sharpe = how much of the Sharpe survives multiple testing. The two methods
answer the same honest question by DIFFERENT mechanisms, so AGREEMENT hardens a verdict and
DISAGREEMENT is itself a finding (absolute-significance vs search-dispersion-luck).

t-stat of an annualized Sharpe over n obs: t = SR_ann * sqrt(years), years = n_days/252 (daily P&L).
p (one-sided) = 1 - Phi(t).  Bonferroni: p*M.  BH: q-value across the asset's M trial p-values.
haircut_t = Phinv(1 - p_adj);  haircut_SR = SR_ann * max(0, haircut_t)/t.
"""
import csv, math, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_rigor import Phi, Phinv

PPY = 252.0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "autoresearch", "results", "round_results.csv")
OUT = os.path.join(HERE, "autoresearch", "HONEST_AUDIT.md")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def t_stat(sr_ann, n):
    return sr_ann * math.sqrt(max(1e-9, n / PPY))


def p_one_sided(t):
    return min(1.0, max(1e-16, 1.0 - Phi(t)))


def bh_qvalues(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values), returned in original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        q[i] = val
        prev = val
    return q


def haircut_sr(sr_ann, t, p_adj):
    if p_adj >= 1.0:
        return 0.0
    t_adj = Phinv(1.0 - p_adj)
    return sr_ann * max(0.0, t_adj) / t if t > 0 else 0.0


def main():
    rows = list(csv.DictReader(open(CSV)))
    by = {}
    for r in rows:
        if (r.get("labeler") or "") == "always_long":
            continue
        sr = fnum(r.get("real_sharpe"))
        nd = fnum(r.get("n_days"))
        cal = fnum(r.get("real_calmar"))
        if sr is None or nd is None or nd <= 1:
            continue
        by.setdefault(r["ticker"], []).append({"sr": sr, "n": int(nd), "cal": cal, "lab": r.get("labeler")})

    audited = []
    for tk, trials in by.items():
        cals = [t for t in trials if t["cal"] is not None]
        if len(trials) < 5 or not cals:
            continue
        champ = max(cals, key=lambda t: t["cal"])
        if champ["cal"] <= 0:
            continue
        M = len(trials)
        pvals = [p_one_sided(t_stat(t["sr"], t["n"])) for t in trials]
        qvals = bh_qvalues(pvals)
        ci = trials.index(champ)
        t0 = t_stat(champ["sr"], champ["n"])
        p0 = pvals[ci]
        p_bonf = min(1.0, p0 * M)
        q_bh = qvals[ci]
        audited.append({
            "tk": tk, "N": M, "cal": champ["cal"], "sr": champ["sr"], "t": t0, "p": p0,
            "hc_bonf": haircut_sr(champ["sr"], t0, p_bonf),
            "hc_bh": haircut_sr(champ["sr"], t0, q_bh),
            "p_bonf": p_bonf, "q_bh": q_bh,
        })

    audited.sort(key=lambda a: -a["cal"])
    print(f"{'ETF':5s} {'N':>4s} {'Calmar':>7s} {'SR':>6s} {'t':>6s} {'p1':>9s} {'HC_Bonf':>8s} {'HC_BH':>7s}  verdict")
    lines = ["", "## Harvey-Liu haircut Sharpe (independent multiple-testing cross-check)", "",
             "Adjusts each champion's Sharpe t-stat p-value for its true trial count M (Bonferroni FWER, BH FDR);",
             "haircut SR = Sharpe surviving the correction. Independent of DSR's extreme-value mechanism.", "",
             "```",
             f"{'ETF':5s} {'N':>4s} {'Calmar':>7s} {'SR':>6s} {'t':>6s} {'p1':>9s} {'HC_Bonf':>8s} {'HC_BH':>7s}  verdict"]
    for a in audited:
        v = ("SURVIVES (HC>0.5 both)" if min(a["hc_bonf"], a["hc_bh"]) > 0.5 else
             ("partial (HC_BH>0)" if a["hc_bh"] > 0.05 else "FAILS (haircut ~0)"))
        row = (f"{a['tk']:5s} {a['N']:4d} {a['cal']:7.3f} {a['sr']:6.3f} {a['t']:6.2f} {a['p']:9.2e} "
               f"{a['hc_bonf']:8.3f} {a['hc_bh']:7.3f}  {v}")
        print(row)
        lines.append(row)
    lines.append("```")
    lines.append("")
    # append to HONEST_AUDIT.md (keep the DSR section)
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    marker = "## Harvey-Liu haircut"
    if marker in prev:
        prev = prev[:prev.index(marker)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print("\nappended to:", OUT)


if __name__ == "__main__":
    main()
