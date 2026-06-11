#!/usr/bin/env python3
"""Multiple-testing / selection-bias assessment of the 7 champions (Wang review #1).

Re-runs each champion's infer (its cell was kept) to get OOS daily Sharpe + skew +
kurtosis + N, computes the Probabilistic Sharpe Ratio PSR(SR>0), and compares it to a
multiple-testing-adjusted threshold.

DSR OVER-DEFLATION FIX (311-plan step 3, quick-win): the threshold is NO LONGER
1 - 0.05/n_trials. Raw n_trials treats a correlated 21x27 sweep as n_trials INDEPENDENT
bets and over-penalises — it can kill real borderline fits. Instead we deflate by
N_eff = eigenvalue PARTICIPATION RATIO (Σλ)^2/Σλ^2 of the per-ETF TRIAL correlation
matrix (scripts/audit/stats_rigor.py::effective_n_trials), so highly-redundant configs count once.
Per-trial OOS daily PnL is not on disk for the deleted cells, so the correlation matrix
is the documented CONFIG-AFFINITY proxy (shared axis/labeler/sizing => correlated PnL);
when real per-trial PnL is captured, feed stats_rigor.corr_from_returns instead — the
N_eff plumbing is identical. A persistent GLOBAL trial counter (results/global_trials.json)
records the durable cross-session search burden.

Writes sharpe/psr/n_trials/n_eff/significant into per_etf_best and prints the verdict.
(A full Deflated-Sharpe needs per-trial Sharpes, now being captured going forward;
PSR + N_eff-adjusted threshold is the rigorous control available from the data we have.)"""
import os, sys, json, csv, collections
from math import erf, sqrt
import importlib.util as _ilu
from lb.paths import ROOT as _ROOT
_spec = _ilu.spec_from_file_location("run_round", str(_ROOT / "scripts" / "run_round.py"))
R = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)  # driver-internal helpers (run_pool, _f, _cagr_from_stats, ...)
import stats_rigor as SR


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def load_trials_by_etf():
    """Per ETF: list of trial CONFIG dicts (axis/labeler/sizing) from round_results.csv.
    Used to build the config-affinity correlation matrix -> N_eff."""
    by = collections.defaultdict(list)
    try:
        for row in csv.DictReader(open(R.ROUND_RESULTS_CSV)):
            by[row["ticker"]].append({"axis": row.get("axis"),
                                      "labeler": row.get("labeler"),
                                      "sizing": row.get("sizing")})
    except Exception as e:
        print("warn: round_results.csv:", e)
    return by


def etf_neff(trial_cfgs):
    """N_eff (effective independent trials) for one ETF from its trial config list."""
    if not trial_cfgs:
        return 1.0
    C = SR.config_affinity_corr(trial_cfgs)
    return SR.effective_n_trials(C)


def main():
    by_etf = load_trials_by_etf()
    trials = {etf: len(v) for etf, v in by_etf.items()}
    total_trials = sum(trials.values())
    global_total = SR.bump_global_trials(total_trials, tag="assess_dsr")

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
    print(f"[dsr] global trial burden now {global_total} (this run +{total_trials})")
    res = R.run_pool(jobs)

    print(f"\n=== Multiple-testing assessment (PSR + N_eff-deflated threshold) ===")
    print(f"{'ETF':5s} {'Calmar':>7s} {'Sharpe':>7s} {'PSR>0':>8s} {'N_tr':>5s} {'N_eff':>6s} {'thr':>9s}  verdict")
    for etf, v in sorted(pe.items(), key=lambda kv: -(kv[1].get("real_calmar") or 0)):
        bt = res.get(f"psr_{etf}", {})
        rt = (bt.get("runtimeStatistics", {}) or {}) if isinstance(bt, dict) else {}
        sh, sk, ku, N = R._f(rt.get("sharpe_oos")), R._f(rt.get("skew_oos")), R._f(rt.get("kurt_oos")), int(R._f(rt.get("n_days")))
        nt = trials.get(etf, 1)
        neff = etf_neff(by_etf.get(etf, []))
        if not (N > 3 and sh):
            print(f"{etf:5s} {(v.get('real_calmar') or 0):7.3f}  (no Sharpe — infer stat missing)  N_tr={nt} N_eff={neff:.2f}")
            pe[etf].update({"n_trials": nt, "n_eff": round(neff, 3)})
            continue
        srd = sh / sqrt(252.0)                                   # per-observation (daily) Sharpe
        denom = sqrt(max(1e-9, 1.0 - sk * srd + (ku - 1.0) / 4.0 * srd * srd))
        psr = Phi(srd * sqrt(N - 1) / denom)
        # N_eff-deflated FWER threshold (was 1 - 0.05/nt). Round up so a single residual
        # effective trial still applies a (tiny) haircut.
        m_eff = max(1, int(round(neff)))
        thr = 1.0 - 0.05 / m_eff
        sig = psr > thr
        pe[etf].update({"sharpe": round(sh, 3), "psr": round(psr, 4), "n_trials": nt,
                        "n_eff": round(neff, 3), "significant": bool(sig)})
        print(f"{etf:5s} {(v.get('real_calmar') or 0):7.3f} {sh:7.2f} {psr:8.4f} {nt:5d} {neff:6.2f} {thr:9.5f}  "
              f"{'SIGNIFICANT' if sig else 'not significant (selection bias)'}")
    json.dump(K, open(R.KNOWLEDGE_JSON, "w"), indent=2)
    print("\n[dsr] wrote sharpe/psr/n_trials/n_eff/significant into per_etf_best")


if __name__ == "__main__":
    main()
