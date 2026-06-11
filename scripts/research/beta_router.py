#!/usr/bin/env python3
"""beta_n ROUTER (Wang's mechanism diagnostic, from docs/analysis/WANG_INVESTIGATION.md).

Temporarily injects a LEAK-SAFE beta_n emission into the footer template (restored in finally),
renders+runs a normal train job per ticker, and reads beta50/beta100/beta200 = fraction of TRAIN
bars whose n-bar forward return is positive. Wang's predictor of which mechanism an asset admits:
  beta ~ 0.5  -> symmetric  -> TREND/HMM wins (GLD 0.53 -> Wang 5.72)
  beta >> 0.5 -> drift      -> BUY-HOLD ("always-1" already wins; QQQ 0.74 -> 1.42)
  beta < 0.5  -> mean-rev   -> REVERSION (where our oil edge lives)
Validates the lens on OUR data. Injection is computed on tr_m & fv only (TRAIN, leak-safe) and
wrapped in try/except so it can never break a train.  Run: python3 scripts/beta_router.py
"""
import sys, os
from lb.harness.orchestrator import render_train_config
from lb.harness.qc_client import submit_and_wait

FOOTER = "templates/footer.py.tmpl"
TARGET = "fwd_ret, fwd_vol = compute_forward_metrics(lc, lr)\n"
INJECT = TARGET + (
    "            try:\n"
    "                for _bn in (50, 100, 200):\n"
    "                    _fr = fwd_ret[_bn] if (isinstance(fwd_ret, dict) and _bn in fwd_ret) else None\n"
    "                    if _fr is not None:\n"
    "                        _bm = tr_m & fv & ~np.isnan(_fr)\n"
    "                        if int(_bm.sum()) > 0:\n"
    "                            self.set_runtime_statistic('beta' + str(_bn), str(round(float(np.mean(_fr[_bm] > 0)), 4)))\n"
    "            except Exception:\n"
    "                pass\n"
)

TICKERS = ["IXP", "AAXJ", "EWL", "DJP", "VTI", "IVV"]   # 2026-06-08 hypothesis: sleeve fits (IXP/AAXJ/EWL/DJP) ~0.5 (timing-exploitable) vs US-equity NO-FITs (VTI/IVV) >>0.6 (buy-hold drift)


def route(b200):
    try:
        b = float(b200)
    except (TypeError, ValueError):
        return "?"
    return "REVERSION" if b < 0.47 else ("BUY-HOLD" if b > 0.62 else "TREND")


orig = open(FOOTER).read()
n = orig.count(TARGET)
if n != 1:
    print(f"ABORT: anchor appears {n}x (need exactly 1)")
    sys.exit(1)

try:
    open(FOOTER, "w").write(orig.replace(TARGET, INJECT, 1))
    print(f"injected beta-emission; running {TICKERS}", flush=True)
    print(f"{'ETF':5s} {'beta50':>7s} {'beta100':>7s} {'beta200':>7s}  route", flush=True)
    for tk in TICKERS:
        cfg = {"ticker": tk, "axis": "logdollar", "labeler": "trend_leg", "thresh": 0.45, "sizing": "cdf_overlay"}
        tcode, extra = render_train_config(cfg)
        bt, st = submit_and_wait(tcode, f"beta_{tk}", timeout_s=300, extra_files=extra)
        rt = bt.get("runtimeStatistics", {}) or {}
        b50, b100, b200 = rt.get("beta50", "?"), rt.get("beta100", "?"), rt.get("beta200", "?")
        print(f"{tk:5s} {str(b50):>7s} {str(b100):>7s} {str(b200):>7s}  {route(b200)}  ({st})", flush=True)
finally:
    open(FOOTER, "w").write(orig)
    print("restored original footer", flush=True)
