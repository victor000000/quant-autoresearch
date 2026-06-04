#!/usr/bin/env python3
"""TRANSACTION-COST STRESS — the one real-world honesty dimension untested. The pipeline sets NO
explicit slippage/fee model, so reported Calmars use QC defaults (often optimistic). This re-runs each
crown's infer (pure ObjectStore replay — SAME decisions) with an explicit ConstantSlippageModel at
realistic levels (5bp, 10bp per fill) and measures Calmar erosion. Liquid ETFs (~1-2bp spreads) at
~2 orders/day should survive; high-frequency GLD (1546 trades) is the cost-sensitive case to watch.

Does NOT modify the production infer template (audited-clean) — inserts the slippage line into a
rendered copy only.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, ".")
sys.path.insert(0, "harness")
from harness.orchestrator import render_infer_cell
from harness.qc_client import submit_and_wait

ANCHOR = 'self.sym = self.add_equity(TICKER, Resolution.MINUTE).symbol'
INJECT = ANCHOR + '\n        self.securities[self.sym].set_slippage_model(ConstantSlippageModel(%s))'

CROWNS = [   # the 2026-06 leak-free CURRENT crowns (updated 2026-06-04: GLD IG crown 4.02, IWM provisional)
    ("GLD", "logdollar_trend_leg_x_regime_gmm_dd_overlay_t40_n15_b3_ig", 4.022),
    ("UUP", "imbalance_bgm_x_sadf_explosive_x_ker_cdf_overlay_t50", 1.847),
    ("IWM", "logdollar_trend_leg_cdf_overlay_t45_ig", 0.665),
]
SLIP = [0.0005, 0.0010]  # 5 bp, 10 bp per fill


def calmar(bt):
    st = bt.get("statistics", {}) or {}
    try:
        cagr = float(st.get("Compounding Annual Return", "0%").replace("%", ""))
        mdd = float(st.get("Drawdown", "0%").replace("%", ""))
        orders = int(st.get("Total Orders", "0"))
        return (cagr / mdd if abs(mdd) > 0.01 else 0.0), cagr, mdd, orders
    except (ValueError, TypeError):
        return 0.0, 0.0, 0.0, 0


def main():
    rows = []
    for tk, cell, default_cal in CROWNS:
        base = render_infer_cell(tk, cell)
        if ANCHOR not in base:
            print(f"[{tk}] anchor not found — skip"); continue
        res = {"tk": tk, "default": default_cal}
        for slip in SLIP:
            code = base.replace(ANCHOR, INJECT % slip, 1)
            print(f"[{tk}] infer @ {slip*1e4:.0f}bp ...", flush=True)
            bt, status = submit_and_wait(code, f"cost_{tk}_{int(slip*1e4)}bp", timeout_s=300)
            if status != "completed":
                print(f"[{tk}] {slip*1e4:.0f}bp failed: {status}"); res[slip] = None; continue
            c, cagr, mdd, orders = calmar(bt)
            res[slip] = c
            res[f"{slip}_o"] = orders
            print(f"[{tk}] {slip*1e4:.0f}bp -> Calmar {c:.3f} (CAGR {cagr:.2f}%, MDD {mdd:.2f}%, {orders} orders)", flush=True)
        rows.append(res)

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "analysis", "HONEST_AUDIT.md")
    lines = ["", "## Transaction-cost stress (explicit slippage; pipeline default = none)", "",
             "Calmar after re-running each crown's infer (same decisions) with explicit per-fill slippage:", "", "```",
             f"{'crown':5s} {'default':>8s} {'5bp':>8s} {'10bp':>8s}   erosion@10bp"]
    print("\n" + lines[-1])
    for r in rows:
        c5 = r.get(0.0005); c10 = r.get(0.0010)
        ero = f"{(1 - c10/r['default'])*100:4.0f}%" if c10 and r['default'] else "—"
        row = (f"{r['tk']:5s} {r['default']:8.3f} {(c5 if c5 is not None else float('nan')):8.3f} "
               f"{(c10 if c10 is not None else float('nan')):8.3f}   {ero}")
        print(row); lines.append(row)
    lines.append("```")
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    m = "## Transaction-cost stress"
    if m in prev:
        prev = prev[:prev.index(m)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
