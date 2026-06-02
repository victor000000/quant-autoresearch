#!/usr/bin/env python3
"""Multiple-testing / selection-bias assessment of the 7 champions (Wang review #1).

Re-runs each champion's infer (its cell was kept) to get OOS daily Sharpe + skew +
kurtosis + N, computes the Probabilistic Sharpe Ratio PSR(SR>0), and compares it to a
Bonferroni-adjusted threshold 1 - 0.05/N_trials (N_trials = # configs tried for that ETF
from round_results.csv). A champion is "significant" only if its edge survives the
trials adjustment. Writes sharpe/psr/n_trials/significant into per_etf_best and prints
the verdict. (A full Deflated-Sharpe needs per-trial Sharpes, now being captured going
forward; PSR+Bonferroni is the rigorous control available from the data we have.)"""
import os, sys, json, csv, collections
from math import erf, sqrt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_autoresearch_round as R


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


trials = collections.Counter()
try:
    for row in csv.DictReader(open(R.ROUND_RESULTS_CSV)):
        trials[row["ticker"]] += 1
except Exception as e:
    print("warn: round_results.csv:", e)

K = json.load(open(R.KNOWLEDGE_JSON))
pe = K["per_etf_best"]


def cell_of(c):
    return (f"{c['axis']}_{c['labeler'].replace('+', '_x_')}_{c['sizing']}"
            f"_t{int(round(float(c['thresh']) * 100))}")


jobs = []
for etf, v in pe.items():
    c = v.get("config", {})
    if c:
        jobs.append((f"psr_{etf}", R.render_infer_cell(etf, cell_of(c))))
print(f"[dsr] running {len(jobs)} champion infers for OOS Sharpe ...")
res = R.run_pool(jobs)

print(f"\n=== Multiple-testing assessment (PSR + Bonferroni by N_trials) ===")
print(f"{'ETF':5s} {'Calmar':>7s} {'Sharpe':>7s} {'PSR>0':>8s} {'N_tr':>5s} {'Bonf_thr':>9s}  verdict")
for etf, v in sorted(pe.items(), key=lambda kv: -(kv[1].get("real_calmar") or 0)):
    bt = res.get(f"psr_{etf}", {})
    rt = (bt.get("runtimeStatistics", {}) or {}) if isinstance(bt, dict) else {}
    sh, sk, ku, N = R._f(rt.get("sharpe_oos")), R._f(rt.get("skew_oos")), R._f(rt.get("kurt_oos")), int(R._f(rt.get("n_days")))
    if not (N > 3 and sh):
        print(f"{etf:5s} {(v.get('real_calmar') or 0):7.3f}  (no Sharpe — infer stat missing)")
        continue
    srd = sh / sqrt(252.0)                                   # per-observation (daily) Sharpe
    denom = sqrt(max(1e-9, 1.0 - sk * srd + (ku - 1.0) / 4.0 * srd * srd))
    psr = Phi(srd * sqrt(N - 1) / denom)
    nt = trials.get(etf, 1)
    thr = 1.0 - 0.05 / max(1, nt)
    sig = psr > thr
    pe[etf].update({"sharpe": round(sh, 3), "psr": round(psr, 4), "n_trials": nt, "significant": bool(sig)})
    print(f"{etf:5s} {(v.get('real_calmar') or 0):7.3f} {sh:7.2f} {psr:8.4f} {nt:5d} {thr:9.5f}  "
          f"{'SIGNIFICANT' if sig else 'not significant (selection bias)'}")
json.dump(K, open(R.KNOWLEDGE_JSON, "w"), indent=2)
print("\n[dsr] wrote sharpe/psr/n_trials/significant into per_etf_best")
