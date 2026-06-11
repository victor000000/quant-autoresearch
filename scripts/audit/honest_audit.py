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
from stats_rigor import (probabilistic_sharpe_ratio, expected_max_sharpe,
                         deflated_sharpe_ratio, holm_bonferroni, benjamini_hochberg,
                         min_backtest_length)

PPY = 252.0
# Paths from lb.paths (single source of truth) — the old dirname(dirname(__file__))
# resolved to scripts/ after this file moved to scripts/audit/ in the restructure.
from lb.paths import ROOT, ROUND_RESULTS_CSV, KNOWLEDGE_JSON
CSV = str(ROUND_RESULTS_CSV)
KNOW = str(KNOWLEDGE_JSON)
OUT = str(ROOT / "docs" / "analysis" / "HONEST_AUDIT.md")


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
            "axis": (r.get("axis") or ""), "thresh": fnum(r.get("thresh")),
            "sizing": (r.get("sizing") or ""),
        })

    # DEPLOYED-CONFIG certification (2026-06-10 leak-review MEDIUM fix): the audited
    # row must be the trial that MATCHES per_etf_best.config — the per-ticker
    # max-Calmar row can be a RETRACTED pre-leak-fix trial, and stamping its DSR
    # onto the deployed config certified the wrong strategy. N stays ALL trials
    # (the true search burden). Falls back to max-Calmar ONLY with a loud proxy flag.
    try:
        _pe_cfg = {tk: (v.get("config") or {}) for tk, v in
                   json.load(open(KNOW)).get("per_etf_best", {}).items()}
    except Exception:
        _pe_cfg = {}

    audited = []
    for tk, trials in by.items():
        # REPLICATIONS ARE NOT SEARCHES (2026-06-10): re-validations / phase-cert /
        # permute-gate re-runs of the same config produce near-identical rows; counting
        # them as independent trials inflates N and trial variance, overstating E[max].
        # Dedupe on (axis, labeler, thresh, sizing, Calmar@3dp) — distinct strategies
        # keep distinct rows, bit-exact replications collapse to one.
        seen = set()
        dedup = []
        for t in trials:
            k = (t["axis"], t["lab"], t["thresh"], t["sizing"],
                 None if t["cal"] is None else round(t["cal"], 3))
            if k in seen:
                continue
            seen.add(k)
            dedup.append(t)
        trials = dedup
        cals = [t for t in trials if t["cal"] is not None]
        if len(trials) < 5 or not cals:
            continue
        cfg = _pe_cfg.get(tk) or {}
        match = [t for t in cals
                 if cfg and t["axis"] == cfg.get("axis", "") and t["lab"] == cfg.get("labeler", "")
                 and t["sizing"] == cfg.get("sizing", "")
                 and t["thresh"] is not None and abs(t["thresh"] - float(cfg.get("thresh", -1))) < 1e-9]
        # The CSV does not record features/reduce, so the quadruple alone can hit a
        # DIFFERENT strategy sharing it (e.g. GLD wangrich rows). Within the quadruple,
        # prefer rows whose Calmar equals the stored deployed number — that pins the
        # deployed strategy's row (the number is already fixed by deployment, this is
        # row identification, not selection). Fall back to latest-quadruple (stored
        # number stale, e.g. post-decay UUP), then loudly to max-Calmar.
        _pe_cal = None
        try:
            _pe_cal = float(json.load(open(KNOW)).get("per_etf_best", {}).get(tk, {}).get("real_calmar"))
        except Exception:
            pass
        exact = [t for t in match if _pe_cal is not None and t["cal"] is not None
                 and abs(t["cal"] - _pe_cal) < 1e-3]
        if exact:
            champ = dict(exact[-1], proxy=False)
        elif match:
            champ = dict(match[-1], proxy=False)   # latest config row (stored number stale)
        else:
            champ = dict(max(cals, key=lambda t: t["cal"]), proxy=True)   # loud fallback
        if champ["cal"] <= 0 or champ["n"] <= 1:
            continue
        N = len(trials)                                   # TRUE session-wide trial count
        sr_obs = [t["sr"] / math.sqrt(PPY) for t in trials]
        var_obs = st.variance(sr_obs) if len(sr_obs) > 1 else 0.0
        c_obs = champ["sr"] / math.sqrt(PPY)
        psr0 = probabilistic_sharpe_ratio(c_obs, champ["n"], champ["skew"], champ["kurt"], 0.0)
        emax = expected_max_sharpe(var_obs, N)
        dsr = deflated_sharpe_ratio(c_obs, champ["n"], champ["skew"], champ["kurt"], N, var_obs)
        oos_years = champ["n"] / PPY                       # actual OOS track length (n_days -> years)
        minbtl = min_backtest_length(champ["sr"], N)       # years needed to clear best-of-N at this search size
        audited.append({"tk": tk, "N": N, "cal": champ["cal"], "sr_ann": champ["sr"],
                        "emax_ann": emax * math.sqrt(PPY), "psr0": psr0, "dsr": dsr,
                        "minbtl": minbtl, "oos_years": oos_years, "suff": minbtl <= oos_years,
                        "p": max(0.0, 1.0 - dsr), "lab": champ["lab"],
                        "proxy": champ.get("proxy", False)})

    audited.sort(key=lambda a: -a["cal"])
    pvals = [a["p"] for a in audited]
    holm = holm_bonferroni(pvals, alpha=0.05)
    bh = benjamini_hochberg(pvals, q=0.10)

    lines = ["# Session-scale honest audit (Deflated Sharpe at true trial counts)", ""]
    lines.append(f"Audited {sum(len(v) for v in by.values())} model trials across {len(audited)} assets "
                 f"with >=5 trials & positive champion. Holm-Bonferroni FWER<=0.05, BH FDR<=0.10.")
    lines.append("")
    hdr = f"{'ETF':5s} {'N':>4s} {'Calmar':>7s} {'SR_ann':>7s} {'Emax_ann':>9s} {'PSR>0':>6s} {'DSR':>6s} {'MinBTL':>7s} {'Suff':>5s} {'Holm':>5s} {'BH':>4s}  verdict"
    lines.append("```")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    print(hdr)
    for a, hj, bj in zip(audited, holm, bh):
        verdict = "REAL (survives best-of-N)" if a["dsr"] >= 0.95 else (
            "marginal" if a["dsr"] >= 0.90 else "FAILS deflation")
        if a.get("proxy"):
            verdict += " [MAX-CAL PROXY — no CSV row matches deployed config]"
        flag = ("Holm+BH" if hj else ("BH-only" if bj else "neither"))
        _mb = f"{a['minbtl']:.2f}y" if a['minbtl'] != float('inf') else "inf"
        row = (f"{a['tk']:5s} {a['N']:4d} {a['cal']:7.3f} {a['sr_ann']:7.3f} {a['emax_ann']:9.3f} "
               f"{a['psr0']:6.3f} {a['dsr']:6.3f} {_mb:>7s} {('Y' if a['suff'] else 'n'):>5s} "
               f"{str(hj):>5s} {str(bj):>4s}  {verdict} [{flag}]")
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
                pe[a["tk"]]["minbtl_years"] = round(a["minbtl"], 3) if a["minbtl"] != float("inf") else None
                pe[a["tk"]]["minbtl_sufficient"] = bool(a["suff"])
        json.dump(K, open(KNOW, "w"), indent=2)
        print("updated per_etf_best DSR fields in knowledge.json")
    except Exception as e:
        print("knowledge.json update skipped:", e)


if __name__ == "__main__":
    main()
