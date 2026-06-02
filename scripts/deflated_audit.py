#!/usr/bin/env python3
"""Deflated-Calmar selection-bias audit (RESEARCH_REVIEW.md Tier 1, feasible-now variant).

The Deflated Sharpe Ratio (Bailey & López de Prado 2014) asks: given that you tried N
configs and kept the MAX, is that max above what the best of N *noise* trials would
produce? We apply the same extreme-value deflation to CALMAR (our objective metric),
using the per-asset dispersion of trial Calmars already logged in round_results.csv.

For each asset's trials {C_1..C_N}:
  var      = sample variance of trial Calmars
  E_max0   = sqrt(var) * [ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)) ]     (γ=Euler-Mascheroni)
           = expected max of N iid N(0,var) trials   (the "best of N noise" benchmark)
A champion's Calmar that does NOT exceed E_max0 is statistically indistinguishable from
luck given how many configs were tried — a selection-bias red flag.

Caveats (honest): (1) DSR is defined on Sharpe; Calmar is heavier-tailed, so E_max0 is an
approximation, not an exact null. (2) Deflation only applies to SEARCHED/model-driven
edges — an `always_long` (buy-hold) champion is the *baseline*, not selected from a search,
so it carries no selection bias and is reported as N/A. (3) True per-trial-Sharpe DSR +
PBO/CSCV need per-trial OOS return series (logged going forward).
"""
import os, sys, json, csv, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_autoresearch_round as R
from stats_rigor import expected_max_sharpe as expected_max_of_N  # shared rigor lib (Bailey-LdP)


# per-asset trial Calmars + (when logged) Sharpes from the round log
trials = collections.defaultdict(list)
trial_sharpe = collections.defaultdict(list)
for row in csv.DictReader(open(R.ROUND_RESULTS_CSV)):
    try:
        trials[row["ticker"]].append(float(row["real_calmar"]))
    except (ValueError, KeyError):
        pass
    try:
        s = row.get("real_sharpe", "")
        if s not in ("", None):
            trial_sharpe[row["ticker"]].append(float(s))
    except (ValueError, KeyError):
        pass

_n_sharpe = sum(len(v) for v in trial_sharpe.values())
print(f"[per-trial Sharpe logged so far: {_n_sharpe} trials across {len(trial_sharpe)} assets — "
      f"exact Deflated-Sharpe auto-activates per asset once ≥5 Sharpe-trials accrue; Calmar-deflation used meanwhile]\n")

K = json.load(open(R.KNOWLEDGE_JSON))
pe = K["per_etf_best"]

print("=== DEFLATED-CALMAR SELECTION-BIAS AUDIT ===")
print(f"{'ETF':5s} {'champ':>7s} {'labeler':>14s} {'N_tr':>5s} {'sd_tr':>6s} {'E_max0':>7s} {'margin':>7s}  verdict")
for etf, v in sorted(pe.items(), key=lambda kv: -(kv[1].get("real_calmar") or 0)):
    cfg = v.get("config", {})
    lab = cfg.get("labeler", "?")
    champ = float(v.get("real_calmar") or 0.0)
    ct = trials.get(etf, [])
    N = len(ct)
    if lab == "always_long":
        print(f"{etf:5s} {champ:7.3f} {lab:>14s} {N:5d}     —       —       —   N/A (buy-hold baseline, no search selection bias)")
        pe[etf]["deflation"] = "N/A_buyhold"
        continue
    if N < 3:
        print(f"{etf:5s} {champ:7.3f} {lab:>14s} {N:5d}   (too few trials to deflate)")
        continue
    # Prefer EXACT Sharpe-based DSR once enough per-trial Sharpes have accrued; else Calmar proxy.
    st = trial_sharpe.get(etf, [])
    if len(st) >= 5:
        metric, champ_m, ts, Nm = "Sharpe", max(st), st, len(st)
    else:
        metric, champ_m, ts, Nm = "Calmar", champ, ct, N
    mean = sum(ts) / Nm
    var = sum((c - mean) ** 2 for c in ts) / (Nm - 1)
    emax = expected_max_of_N(var, Nm)
    margin = champ_m - emax
    verdict = f"SURVIVES ({metric}, above best-of-N noise)" if margin > 0 else f"FAILS — selection-bias artifact ({metric})"
    print(f"{etf:5s} {champ_m:7.3f} {lab:>14s} {Nm:5d} {math.sqrt(var):6.3f} {emax:7.3f} {margin:+7.3f}  {verdict}")
    pe[etf]["deflation_metric"] = metric
    pe[etf]["deflated_benchmark"] = round(emax, 4)
    pe[etf]["deflation_margin"] = round(margin, 4)
    pe[etf]["survives_deflation"] = bool(margin > 0)

json.dump(K, open(R.KNOWLEDGE_JSON, "w"), indent=1)
print("\nNote: deflation applies to SEARCHED model edges only; buy-hold members are the baseline (no selection bias).")
print("written: deflated_calmar_benchmark / survives_deflation into per_etf_best.")
