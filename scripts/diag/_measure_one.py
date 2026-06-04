#!/usr/bin/env python3
"""One-shot measurement: how long does a single GLD/vol TRAIN backtest take on QC,
and does it complete within the 5-minute (300s) wall? Auto-deletes on timeout.
This answers the per-axis-vs-per-cell architecture question before we build more."""
import os, sys, time, json
sys.path.insert(0, "/home/ubuntu/lb")
from harness.qc_client import submit_and_wait
from harness.orchestrator import render_script

ticker = sys.argv[1] if len(sys.argv) > 1 else "GLD"
axis   = sys.argv[2] if len(sys.argv) > 2 else "vol"

code = render_script(ticker, axis=axis)
print(f"[measure] ticker={ticker} axis={axis} rendered_chars={len(code)}", flush=True)
t0 = time.time()
bt, status = submit_and_wait(code, f"measure_{ticker}_{axis}", timeout_s=300)
elapsed = time.time() - t0
rt = (bt.get("runtimeStatistics") or {}) if isinstance(bt, dict) else {}
print(f"[measure] STATUS={status}  ELAPSED={elapsed:.1f}s", flush=True)
print("[measure] runtime stats:", json.dumps({k: rt.get(k) for k in
      ["best_cal","train_auc","val_auc","n_cells","bar_type","best_cfg"]}), flush=True)
if status != "completed":
    detail = {k: bt.get(k) for k in ["status","timeout_s","error"]} if isinstance(bt, dict) else {}
    logs = (bt.get("logs") or "")[:1200] if isinstance(bt, dict) else ""
    print("[measure] NON-COMPLETED detail:", json.dumps(detail), flush=True)
    if logs:
        print("[measure] logs head:\n", logs, flush=True)
