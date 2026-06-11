#!/usr/bin/env python3
"""Prove ONLINE == BATCH custom-bar generation in the OOS window for the champion
axes. For each (ticker, axis): run a train (footer saves oosbars_{axis}.json with the
frozen threshold + batch OOS bars), then run the verify backtest (rebuilds the OOS bars
ONLINE from that frozen threshold, asserts byte-identical). Reports bars_match per axis."""
import os, sys, json
import importlib.util as _ilu
from lb.paths import ROOT as _ROOT
_spec = _ilu.spec_from_file_location("run_round", str(_ROOT / "scripts" / "run_round.py"))
R = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)  # driver-internal helpers (run_pool, _f, _cagr_from_stats, ...)
from lb.harness.orchestrator import render_verify

PAIRS = [
    ("GLD", {"ticker": "GLD", "axis": "logdollar", "labeler": "always_long", "thresh": 0.45, "sizing": "cdf_overlay"}),
    ("IWM", {"ticker": "IWM", "axis": "imbalance", "labeler": "triple_barrier+bgm", "thresh": 0.45, "sizing": "cdf_overlay"}),
    ("TLT", {"ticker": "TLT", "axis": "range", "labeler": "triple_barrier", "thresh": 0.50, "sizing": "ls_overlay"}),
]

train_jobs = [(f"vbtrain_{tk}", R.render_train_config(cfg)) for tk, cfg in PAIRS]
print(f"[verify] training {len(train_jobs)} cells (footer saves oosbars) ...")
tr = R.run_pool(train_jobs)

verify_jobs = []
for tk, cfg in PAIRS:
    if str(tr.get(f"vbtrain_{tk}", {}).get("status", "")).startswith("Completed"):
        verify_jobs.append((f"vb_{tk}", render_verify(tk, cfg["axis"])))
    else:
        print(f"[verify] {tk} train not completed — skip")
print(f"[verify] running {len(verify_jobs)} online-bar verifications ...")
vr = R.run_pool(verify_jobs) if verify_jobs else {}

print("\n=== ONLINE vs BATCH OOS-bar consistency ===")
for tk, cfg in PAIRS:
    bt = vr.get(f"vb_{tk}", {})
    rt = (bt.get("runtimeStatistics", {}) or {}) if isinstance(bt, dict) else {}
    print(f"  {tk:5s} {cfg['axis']:10s} bars_match={rt.get('bars_match','—')} "
          f"n_online={rt.get('n_online','—')} n_batch={rt.get('n_batch','—')} "
          f"first_mismatch={rt.get('first_mismatch','—')} max_lc_diff={rt.get('max_lc_diff','—')} "
          f"{('ERR:'+rt['verify_error']) if rt.get('verify_error') else ''}")
