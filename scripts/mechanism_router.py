#!/usr/bin/env python3
"""MECHANISM ROUTER (completes scripts/beta_router.py). Emits, per ticker, via a leak-safe footer
injection (restored in finally): beta_n (buy-hold filter) AND bar-return autocorrelation rho1/2/3 on
TRAIN bars. Wang (Q1) named lag-2 autocorr as an axis diagnostic; here it's the trend-vs-reversion
discriminator the β200 lens lacks. Hypothesis: among symmetric (β≈0.5) names, TREND (GLD/IWM) shows
rho > REVERSION (USO/UCO). All stats on tr_m only (TRAIN, leak-safe), try/except-guarded so it can
never break a train.  Run: python3 scripts/mechanism_router.py
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "harness")
from harness.orchestrator import render_train_config
from harness.qc_client import submit_and_wait

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
    "                _lrt = lr[tr_m]\n"
    "                _lrt = _lrt[np.isfinite(_lrt)]\n"
    "                if len(_lrt) > 100:\n"
    "                    _lrt = _lrt - np.mean(_lrt)\n"
    "                    _vv = float(np.mean(_lrt * _lrt))\n"
    "                    if _vv > 1e-300:\n"
    "                        for _lag in (1, 2, 3):\n"
    "                            _ac = float(np.mean(_lrt[_lag:] * _lrt[:-_lag])) / _vv\n"
    "                            self.set_runtime_statistic('rho' + str(_lag), str(round(_ac, 4)))\n"
    "            except Exception:\n"
    "                pass\n"
)

# 2 TREND (logdollar/trend champions) vs 2 REVERSION (oil revert) — the discriminating contrast
TICKERS = ["GLD", "IWM", "USO", "UCO"]


def main():
    orig = open(FOOTER).read()
    n = orig.count(TARGET)
    if n != 1:
        print(f"ABORT: anchor appears {n}x (need 1)")
        return
    try:
        open(FOOTER, "w").write(orig.replace(TARGET, INJECT, 1))
        print(f"injected beta+rho emission; running {TICKERS}", flush=True)
        print(f"{'ETF':5s} {'mech':>9s} {'beta200':>7s} {'rho1':>7s} {'rho2':>7s} {'rho3':>7s}", flush=True)
        known = {"GLD": "trend", "IWM": "trend", "USO": "revert", "UCO": "revert"}
        for tk in TICKERS:
            cfg = {"ticker": tk, "axis": "logdollar", "labeler": "trend_leg", "thresh": 0.45, "sizing": "cdf_overlay"}
            tcode, extra = render_train_config(cfg)
            bt, st = submit_and_wait(tcode, f"mech_{tk}", timeout_s=300, extra_files=extra)
            rt = bt.get("runtimeStatistics", {}) or {}
            print(f"{tk:5s} {known.get(tk,'?'):>9s} {str(rt.get('beta200','?')):>7s} "
                  f"{str(rt.get('rho1','?')):>7s} {str(rt.get('rho2','?')):>7s} {str(rt.get('rho3','?')):>7s}  ({st})",
                  flush=True)
    finally:
        open(FOOTER, "w").write(orig)
        print("restored original footer", flush=True)


main()
