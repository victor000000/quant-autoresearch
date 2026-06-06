#!/usr/bin/env python3
"""Alpha-decay check for the oil mean-reversion candidates (UCO/USO revert), modeled on
champion_series.py. Reuses the audited train->infer->read-equity path; passes CELL='__CELL__'
so infer resolves the cell via latest_key.json (the cell the train job just saved) — no cell-key
guesswork. Reports early/late half-window Sharpe + Page-Hinkley + CUSUM on the REAL OOS equity
curve. Question: is the oil-reversion edge front-loaded (fragile, like UUP decayed 2.67->0.74) or
persistent (like GLD)?  Run: python3 scripts/decay_oil.py
"""
import sys, os, math
sys.path.insert(0, ".")
sys.path.insert(0, "harness")
sys.path.insert(0, "scripts")
from harness.orchestrator import render_train_config, render_infer_cell
from harness.qc_client import submit_and_wait, _qc_post
from harness.constants import QC_PROJECT_ID
from decay_monitor import flag_decay, page_hinkley, cusum_meanshift


def equity_series(bid):
    r = _qc_post("/backtests/chart/read",
                 {"projectId": QC_PROJECT_ID, "backtestId": bid, "name": "Strategy Equity",
                  "count": 5000, "start": 0, "end": 2000000000})
    if not r.get("success"):
        return None
    ser = (r.get("chart") or {}).get("series") or {}
    vals = (ser.get("Equity") or {}).get("values") or []
    return [(row[0], row[-1]) for row in vals if isinstance(row, list) and len(row) >= 2 and row[-1] > 0]


def ann_sharpe(rets, ppy):
    if len(rets) < 5:
        return float("nan")
    m = sum(rets) / len(rets)
    sd = (sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
    return (m / sd * math.sqrt(ppy)) if sd > 1e-12 else float("nan")


CANDS = [
    {"ticker": "UCO", "axis": "logdollar", "labeler": "revert", "thresh": 0.45, "sizing": "cdf_overlay"},
    {"ticker": "USO", "axis": "logdollar", "labeler": "revert", "thresh": 0.45, "sizing": "cdf_overlay"},
]

for cfg in CANDS:
    tk = cfg["ticker"]
    print(f"[{tk}] train ...", flush=True)
    tcode, extra = render_train_config(cfg)
    bt_tr, st = submit_and_wait(tcode, f"decayoil_{tk}_tr", timeout_s=540, extra_files=extra)
    if st != "completed":
        print(f"[{tk}] TRAIN failed: {st}", flush=True)
        continue
    print(f"[{tk}] infer ...", flush=True)
    bt_in, st2 = submit_and_wait(render_infer_cell(tk, "__CELL__"), f"decayoil_{tk}_in", timeout_s=300)
    if st2 != "completed":
        print(f"[{tk}] INFER failed: {st2}", flush=True)
        continue
    eq = equity_series(bt_in.get("backtestId"))
    if not eq or len(eq) < 20:
        print(f"[{tk}] no equity series (n={len(eq) if eq else 0})", flush=True)
        continue
    closes = [c for _, c in eq]
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
    span = max(1e-6, (eq[-1][0] - eq[0][0]) / (365.25 * 86400.0))
    ppy = len(rets) / span
    half = len(rets) // 2
    se = ann_sharpe(rets[:half], ppy)
    sl = ann_sharpe(rets[half:], ppy)
    stale = flag_decay(se, sl) if (se == se and sl == sl) else {"stale": None, "reason": "nan"}
    ph = page_hinkley(rets)
    cu = cusum_meanshift(rets)
    fr = lambda i: "--" if i is None else f"{i / len(rets):.0%} in"
    print(f"[{tk}] RESULT npts={len(rets)} ppy~{ppy:.1f} earlySR={se:.2f} lateSR={sl:.2f} "
          f"PageHinkley={fr(ph)} CUSUM={fr(cu)} -> {'STALE' if stale.get('stale') else 'HOLDING'} "
          f"| {stale.get('reason')}", flush=True)
