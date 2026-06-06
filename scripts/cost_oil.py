#!/usr/bin/env python3
"""Transaction-cost stress for the oil-reversion candidates (UCO/USO revert) — the last honesty gate
before a book seat, and it decides the VEHICLE. Replays each cell's infer (same decisions, pure
ObjectStore replay) with an explicit ConstantSlippageModel @5/10bp and measures Calmar erosion.
UCO (2x lev, 552 trades) is the high-turnover case; USO (2.18, fewer effective costs) may win net.
Uses CELL='__CELL__' -> latest_key.json = the revert cell trained in the R1199 decay run. The printed
order count (~552 UCO / ~601 USO) confirms the right cell is being replayed.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "harness")
from harness.orchestrator import render_infer_cell
from harness.qc_client import submit_and_wait

ANCHOR = 'self.sym = self.add_equity(TICKER, Resolution.MINUTE).symbol'
INJECT = ANCHOR + '\n        self.securities[self.sym].set_slippage_model(ConstantSlippageModel(%s))'
CANDS = [("UCO", 3.506), ("USO", 2.175)]   # (ticker, no-cost Calmar baseline from R1197/R1196)
SLIP = [0.0005, 0.0010]                     # 5 bp, 10 bp per fill


def calmar(bt):
    st = bt.get("statistics", {}) or {}
    try:
        cagr = float(st.get("Compounding Annual Return", "0%").replace("%", ""))
        mdd = float(st.get("Drawdown", "0%").replace("%", ""))
        orders = int(st.get("Total Orders", "0"))
        return (cagr / mdd if abs(mdd) > 0.01 else 0.0), cagr, mdd, orders
    except (ValueError, TypeError):
        return 0.0, 0.0, 0.0, 0


for tk, base_cal in CANDS:
    base = render_infer_cell(tk, "__CELL__")
    if ANCHOR not in base:
        print(f"[{tk}] anchor not found — skip", flush=True)
        continue
    print(f"[{tk}] no-cost baseline Calmar = {base_cal:.3f}", flush=True)
    for slip in SLIP:
        code = base.replace(ANCHOR, INJECT % slip, 1)
        print(f"[{tk}] infer @ {slip*1e4:.0f}bp ...", flush=True)
        bt, status = submit_and_wait(code, f"costoil_{tk}_{int(slip*1e4)}bp", timeout_s=300)
        if status != "completed":
            print(f"[{tk}] {slip*1e4:.0f}bp FAILED: {status}", flush=True)
            continue
        c, cagr, mdd, orders = calmar(bt)
        ero = (1 - c / base_cal) * 100 if base_cal else 0.0
        print(f"[{tk}] {slip*1e4:.0f}bp -> Calmar {c:.3f} (CAGR {cagr:.2f}% MDD {mdd:.2f}% {orders} ord) "
              f"erosion {ero:.0f}% vs {base_cal:.3f}", flush=True)
