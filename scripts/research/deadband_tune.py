#!/usr/bin/env python3
"""Tune GLD's rebalance DEAD-BAND against the NET-OF-COST objective (the real deployment metric).
GLD is cost-fragile (1546 trades → −24% Calmar at 5bp). A wider dead-band rebalances less often →
fewer trades → less cost drag, at the cost of tracking the target weight less tightly. This finds the
band that maximizes net-of-5bp-cost Calmar. Pure replay (same saved predictions); only EXECUTION
changes — no leak. Injects into a rendered infer COPY; production template untouched.
"""
import sys, os
from lb.harness.orchestrator import render_infer_cell
from lb.harness.qc_client import submit_and_wait
from lb.paths import ROOT as _LBROOT

_DEFAULTS = {"GLD": "logdollar_ker_x_regime_gmm_dd_overlay_t40_n15",
             "SOXX": "logdollar_ker_x_trend_scan_x_bgm_cdf_overlay_t50",
             "UUP": "imbalance_bgm_x_ker_cdf_overlay_t50"}
TK = sys.argv[1] if len(sys.argv) > 1 else "GLD"
CELL = sys.argv[2] if len(sys.argv) > 2 else _DEFAULTS[TK]
SLIP = 0.0005  # 5 bp realistic slippage — optimize the NET objective
BANDS = [0.01, 0.02, 0.03, 0.05, 0.08]   # 0.01 = current (control)

ANCHOR = 'self.sym = self.add_equity(TICKER, Resolution.MINUTE).symbol'
BAND_SRC = 'abs(w - self._cur_w) > 0.01'


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
    base = render_infer_cell(TK, CELL)
    assert ANCHOR in base and BAND_SRC in base, "anchor/band source not found"
    base = base.replace(ANCHOR, ANCHOR + f'\n        self.securities[self.sym].set_slippage_model(ConstantSlippageModel({SLIP}))', 1)
    rows = []
    for band in BANDS:
        code = base.replace(BAND_SRC, f'abs(w - self._cur_w) > {band}', 1)
        print(f"[{TK}] dead-band {band} @ 5bp ...", flush=True)
        bt, status = submit_and_wait(code, f"band_{int(band*1000)}", timeout_s=300)
        if status != "completed":
            print(f"  band {band} failed: {status}"); rows.append((band, None, None, None)); continue
        c, cagr, mdd, orders = calmar(bt)
        rows.append((band, c, orders, cagr))
        print(f"  band {band}: net Calmar {c:.3f}, orders {orders}, CAGR {cagr:.2f}%", flush=True)

    best = max((r for r in rows if r[1] is not None), key=lambda r: r[1], default=None)
    OUT = str(_LBROOT / "docs" / "analysis" / "HONEST_AUDIT.md")
    lines = ["", f"## {TK} rebalance dead-band tuning (net-of-5bp-cost objective)", "",
             "Wider band → fewer trades → less cost drag. Net Calmar @ 5bp slippage (current band = 0.01):", "", "```",
             f"{'band':>6s} {'netCalmar':>10s} {'orders':>7s} {'CAGR%':>6s}"]
    print("\n" + lines[-1])
    for band, c, orders, cagr in rows:
        mark = "  <- best" if best and band == best[0] else ""
        row = (f"{band:6.2f} {(c if c is not None else float('nan')):10.3f} "
               f"{(orders if orders is not None else 0):7d} {(cagr if cagr is not None else 0):6.2f}{mark}")
        print(row); lines.append(row)
    lines.append("```")
    if best:
        base5 = next((r for r in rows if r[0] == 0.01), None)
        gain = (best[1] / base5[1] - 1) * 100 if base5 and base5[1] else 0
        lines.append(f"\nBest net band = {best[0]} → net Calmar {best[1]:.3f} ({best[2]} orders); "
                     f"{'+' if gain>=0 else ''}{gain:.0f}% vs current 0.01 band at 5bp. "
                     f"{'WORTH widening (cuts cost drag).' if gain > 3 else 'current band ~optimal; cost drag is intrinsic.'}")
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    m = f"## {TK} rebalance dead-band tuning"
    if m in prev:
        prev = prev[:prev.index(m)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
