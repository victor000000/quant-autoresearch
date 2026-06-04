#!/usr/bin/env python3
"""SESSION-SCALE HONEST AUDIT — does each crowned edge survive the FULL multiple-testing burden?

The deep-research review's #1 gap was a FORMAL Deflated-Sharpe / multiple-testing control at the TRUE
session-wide trial count (not best-of-5). We have logged 528 trials in results/round_results.csv, each
with its OOS annualized Sharpe + skew + (non-excess) kurtosis + n_days. This applies the real
stats_rigor tools at full scale:

  per asset:  champion = max real_calmar model trial (excl. always_long baselines)
              N_trials  = ALL model trials on that asset this session (the real search size)
              PSR0      = P(true Sharpe > 0)                         [Bailey-LdP 2012]
              E[max]    = best-of-N-noise Sharpe benchmark           [Bailey-LdP 2014]
              DSR       = PSR against E[max] = P(edge real | search)
  then across the audited champions:  Holm-Bonferroni (FWER<=.05) + Benjamini-Hochberg (FDR<=.10).

PSR/DSR need PER-OBSERVATION Sharpe; the log stores ANNUALIZED, so divide by sqrt(252) (daily P&L);
trial-Sharpe variance is converted the same way. The kurt term is tiny at these per-obs Sharpes, so
the verdict is robust to the excess/non-excess kurtosis convention.
"""
import csv, math, json, os, statistics as st
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_rigor import (probabilistic_sharpe_ratio, expected_max_sharpe,
                         deflated_sharpe_ratio, holm_bonferroni, benjamini_hochberg)

PPY = 252.0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "round_results.csv")
KNOW = os.path.join(HERE, "knowledge.json")
OUT = os.path.join(HERE, "HONEST_AUDIT.md")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(open(CSV)))
    by = {}
    for r in rows:
        lab = (r.get("labeler") or "")
        sr = fnum(r.get("real_sharpe"))
        if sr is None or lab == "always_long":      # baselines carry no selection bias
            continue
        by.setdefault(r["ticker"], []).append({
            "sr": sr, "cal": fnum(r.get("real_calmar")), "skew": fnum(r.get("real_skew")) or 0.0,
            "kurt": fnum(r.get("real_kurt")) or 3.0, "n": int(fnum(r.get("n_days")) or 0),
            "lab": lab, "trades": fnum(r.get("trades")) or 0,
        })

    audited = []
    for tk, trials in by.items():
        cals = [t for t in trials if t["cal"] is not None]
        if len(trials) < 5 or not cals:
            continue
        champ = max(cals, key=lambda t: t["cal"])
        if champ["cal"] <= 0 or champ["n"] <= 1:
            continue
        N = len(trials)                                   # TRUE session-wide trial count
        sr_obs = [t["sr"] / math.sqrt(PPY) for t in trials]
        var_obs = st.variance(sr_obs) if len(sr_obs) > 1 else 0.0
        c_obs = champ["sr"] / math.sqrt(PPY)
        psr0 = probabilistic_sharpe_ratio(c_obs, champ["n"], champ["skew"], champ["kurt"], 0.0)
        emax = expected_max_sharpe(var_obs, N)
        dsr = deflated_sharpe_ratio(c_obs, champ["n"], champ["skew"], champ["kurt"], N, var_obs)
        audited.append({"tk": tk, "N": N, "cal": champ["cal"], "sr_ann": champ["sr"],
                        "emax_ann": emax * math.sqrt(PPY), "psr0": psr0, "dsr": dsr,
                        "p": max(0.0, 1.0 - dsr), "lab": champ["lab"]})

    audited.sort(key=lambda a: -a["cal"])
    pvals = [a["p"] for a in audited]
    holm = holm_bonferroni(pvals, alpha=0.05)
    bh = benjamini_hochberg(pvals, q=0.10)

    lines = ["# Session-scale honest audit (Deflated Sharpe at true trial counts)", ""]
    lines.append(f"Audited {sum(len(v) for v in by.values())} model trials across {len(audited)} assets "
                 f"with >=5 trials & positive champion. Holm-Bonferroni FWER<=0.05, BH FDR<=0.10.")
    lines.append("")
    hdr = f"{'ETF':5s} {'N':>4s} {'Calmar':>7s} {'SR_ann':>7s} {'Emax_ann':>9s} {'PSR>0':>6s} {'DSR':>6s} {'Holm':>5s} {'BH':>4s}  verdict"
    lines.append("```")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    print(hdr)
    for a, hj, bj in zip(audited, holm, bh):
        verdict = "REAL (survives best-of-N)" if a["dsr"] >= 0.95 else (
            "marginal" if a["dsr"] >= 0.90 else "FAILS deflation")
        flag = ("Holm+BH" if hj else ("BH-only" if bj else "neither"))
        row = (f"{a['tk']:5s} {a['N']:4d} {a['cal']:7.3f} {a['sr_ann']:7.3f} {a['emax_ann']:9.3f} "
               f"{a['psr0']:6.3f} {a['dsr']:6.3f} {str(hj):>5s} {str(bj):>4s}  {verdict} [{flag}]")
        lines.append(row)
        print(row)
    lines.append("```")
    lines.append("")
    surv = [a["tk"] for a, hj in zip(audited, holm) if hj]
    lines.append(f"**Survive Holm-Bonferroni (FWER<=.05): {', '.join(surv) if surv else 'NONE'}.** "
                 f"DSR>=.95 = edge clears the best-of-N-trials noise at the real session search size.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\nwritten:", OUT)

    # persist DSR fields into per_etf_best (non-destructive)
    try:
        K = json.load(open(KNOW))
        pe = K.get("per_etf_best", {})
        for a, hj in zip(audited, holm):
            if a["tk"] in pe:
                pe[a["tk"]]["dsr"] = round(a["dsr"], 4)
                pe[a["tk"]]["dsr_n_trials"] = a["N"]
                pe[a["tk"]]["dsr_survives_holm"] = bool(hj)
        json.dump(K, open(KNOW, "w"), indent=2)
        print("updated per_etf_best DSR fields in knowledge.json")
    except Exception as e:
        print("knowledge.json update skipped:", e)


if __name__ == "__main__":
    main()
