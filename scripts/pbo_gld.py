#!/usr/bin/env python3
"""PBO-via-CSCV for GLD — the gold-standard backtest-overfitting metric (Bailey-Borwein-LdP-Zhu 2014),
the last honesty gap, now feasible because per-config OOS return series are extractable read-only.

DSR flagged GLD's 4.55 point estimate as dispersion-inflated; PBO answers the sharper question: when
you pick the IS-best config from GLD's search space, how often is it OOS-BELOW-MEDIAN? Low PBO (<0.3)
= the selection generalizes (4.55 is not an overfit artifact); high PBO (>0.5) = config-overfit.

Method: sweep N comparable GLD configs (labeler variants, the dominant lever; axis/sizing/thresh/ncomp
fixed at champion), train+infer each (parallel via run_pool), extract each OOS equity series read-only,
align on common timestamps -> T x N returns matrix -> pbo_cscv (CSCV over 16 time-blocks).
"""
import sys, os, math
sys.path.insert(0, ".")
sys.path.insert(0, "autoresearch")
sys.path.insert(0, "autoresearch/harness")
sys.path.insert(0, "scripts")
from harness.orchestrator import render_train_config, render_infer_cell
from harness.qc_client import _qc_post
from harness.constants import QC_PROJECT_ID
import run_autoresearch_round as R
from stats_rigor import pbo_cscv, _sharpe

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "autoresearch", "HONEST_AUDIT.md")

BASE = dict(ticker="GLD", axis="logdollar", thresh=0.40, sizing="dd_overlay", n_components=15)
LABELERS = ["ker+regime_gmm", "ker+trend_scan", "ker", "trend_scan", "regime_gmm",
            "ker+accel", "ker+regime_gmm+accel", "accel", "ker+bgm"]


def cell_key(cfg):
    lab = cfg["labeler"].replace("+", "_x_")
    suf = "" if int(cfg.get("n_components", 20)) == 20 else f"_n{int(cfg['n_components'])}"
    return f"{cfg['axis']}_{lab}_{cfg['sizing']}_t{int(round(float(cfg['thresh']) * 100))}{suf}"


def equity_map(bid):
    r = _qc_post("/backtests/chart/read",
                 {"projectId": QC_PROJECT_ID, "backtestId": bid, "name": "Strategy Equity",
                  "count": 5000, "start": 0, "end": 2000000000})
    if not r.get("success"):
        return {}
    vals = ((r.get("chart") or {}).get("series") or {}).get("Equity", {}).get("values") or []
    return {int(row[0]): float(row[-1]) for row in vals if isinstance(row, list) and len(row) >= 2 and row[-1] > 0}


def main():
    cfgs = [dict(BASE, labeler=l) for l in LABELERS]
    # Phase 1: train all (parallel)
    print(f"[PBO] training {len(cfgs)} GLD configs ...", flush=True)
    train_jobs = []
    for c in cfgs:
        code, extra = render_train_config(c)
        train_jobs.append((f"pbo_tr_{c['labeler']}", code, extra))
    tr = R.run_pool(train_jobs)
    ok = [c for c in cfgs if str(tr.get(f"pbo_tr_{c['labeler']}", {}).get("status", "")).startswith("Completed")]
    print(f"[PBO] {len(ok)}/{len(cfgs)} trained OK", flush=True)
    # Phase 2: infer all (parallel)
    infer_jobs = [(f"pbo_in_{c['labeler']}", render_infer_cell("GLD", cell_key(c))) for c in ok]
    inf = R.run_pool(infer_jobs)
    # Phase 3: extract series
    series = {}
    for c in ok:
        bt = inf.get(f"pbo_in_{c['labeler']}", {})
        if not str(bt.get("status", "")).startswith("Completed"):
            continue
        em = equity_map(bt.get("backtestId"))
        if len(em) >= 30:
            series[c["labeler"]] = em
    print(f"[PBO] {len(series)} configs with usable series: {list(series.keys())}", flush=True)
    if len(series) < 4:
        print("[PBO] too few series for a meaningful CSCV"); return
    # Align on common timestamps
    common = set.intersection(*[set(s.keys()) for s in series.values()])
    ts = sorted(common)
    names = list(series.keys())
    print(f"[PBO] {len(ts)} common timestamps across {len(names)} configs", flush=True)
    # returns matrix: T-1 rows x N cols
    matrix = []
    for i in range(1, len(ts)):
        row = []
        for nm in names:
            c1, c0 = series[nm][ts[i]], series[nm][ts[i - 1]]
            row.append(c1 / c0 - 1.0 if c0 > 0 else 0.0)
        matrix.append(row)
    res = pbo_cscv(matrix, n_splits=16)
    # per-config full-sample OOS Sharpe (for context)
    full_sr = {nm: _sharpe([matrix[r][j] for r in range(len(matrix))]) for j, nm in enumerate(names)}
    champ_rank = sorted(names, key=lambda n: -full_sr[n])
    pbo = res.get("pbo")
    print(f"\n[PBO] GLD PBO = {pbo:.3f}  over {res.get('n_combinations')} CSCV partitions, N={len(names)} configs")
    print("[PBO] full-sample OOS Sharpe by config:")
    for nm in champ_rank:
        print(f"    {nm:24s} {full_sr[nm]:+.4f}")
    verdict = ("LOW PBO -> selection GENERALIZES (champion not an overfit artifact)" if pbo < 0.3 else
               "MODERATE PBO" if pbo < 0.5 else "HIGH PBO -> config-OVERFIT (IS-best tends OOS-below-median)")
    lines = ["", "## PBO-via-CSCV (GLD config-overfitting, Bailey-Borwein-LdP-Zhu 2014)", "",
             f"Swept {len(names)} comparable GLD configs (labeler variants; axis/sizing/thresh/ncomp fixed at "
             f"champion), extracted each OOS return series, CSCV over 16 time-blocks ({res.get('n_combinations')} "
             f"partitions).", "", "```",
             f"GLD PBO = {pbo:.3f}   N_configs={len(names)}   T={len(matrix)}  -> {verdict}",
             "full-sample OOS Sharpe by config (IS-best candidate = top):"]
    for nm in champ_rank:
        lines.append(f"  {nm:24s} {full_sr[nm]:+.4f}")
    lines.append("```")
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    marker = "## PBO-via-CSCV"
    if marker in prev:
        prev = prev[:prev.index(marker)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print("\nappended to:", OUT)


if __name__ == "__main__":
    main()
