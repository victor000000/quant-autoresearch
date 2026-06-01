#!/usr/bin/env python3
"""Systematic ETF Deployment — Output of ~163 Experiments.

Applies the optimal (axis, pipeline, split, labels) per ETF.
Usage: python3 qc/deploy_optimal.py [--fast] [ETF...]"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"
PID = 31338454

OPTIMAL = {
    "XLP": ("v385", "Range Bars", 0.60), "XLU": ("v385", "Range Bars", 0.49),
    "LQD": ("v385", "Range Bars", 1.09),
    "QQQ": ("v400", "Multi-Order", 2.12), "XLI": ("v400", "Multi-Order", 0.60),
    "EEM": ("v400", "Multi-Order", 1.22),
    "GLD": ("v370", "Vol+Cary", 1.59), "IAU": ("v370", "Vol+KMeans", 1.60),
    "GDX": ("v371", "Vol+KMeans+Ridge", 1.51), "GDXJ": ("v372", "Vol+BGM", 1.60),
    "SLV": ("v372", "Vol+Ensemble", 1.10), "XME": ("v372", "Vol+Cary", 1.13),
    "MTUM": ("v372", "Vol+BGM", 1.35), "EMB": ("v372", "Vol+Cary", 1.25),
    "SPY": ("v381", "Vol+KMeans", 1.04),
    "SMH": ("v246", "Dollar 2019", 1.78), "SOXX": ("v246", "Dollar 2019", 1.34),
    "VGT": ("v246", "Dollar 2019", 1.32), "HYG": ("v246", "Dollar 2019", 1.63),
    "SIL": ("v246", "Dollar 2019", 1.39), "DIA": ("v246", "Dollar 2019", 1.48),
    "EWJ": ("v246", "Dollar 2019", 1.08), "EWT": ("v246", "Dollar 2019", 1.29),
    "EZA": ("v246", "Dollar 2019", 0.86), "XLV": ("v246", "Dollar 2019", 0.26),
    "SHY": ("v274", "Ridge+MR", 1.90),
    "XLF": ("v501", "Split-Opt", 0.85), "IWM": ("v501", "Split-Opt", 0.12),
    "XRT": ("v501", "Split-Opt", 0.39), "MUB": ("v501", "Split-Opt", 0.36),
}
RESISTANT = ["FXY","TLT","TAN","IBB","RSX","XTN"]

def deploy_one(ticker, pipeline, known, fast=False):
    train_p = f"{BASE}/_pipeline_{pipeline}_train/main.py"
    infer_p = f"{BASE}/_pipeline_{pipeline}_infer/main.py"
    if not os.path.exists(train_p): return {"error": f"no {train_p}"}
    r = {"ticker": ticker, "pipeline": pipeline, "known": known}
    if not fast:
        with open(train_p) as f: code = f.read().replace("__TICKER__", ticker)
        try: bid = submit(PID, code, f"{pipeline}_{ticker}_dep")
        except Exception as e: r["error"] = str(e); return r
        for _ in range(90):
            time.sleep(10); s, rt, _ = read_bt_status(PID, bid)
            if s.startswith("Completed"): r["best"] = rt.get("best_cfg","?"); break
            if "Error" in s: r["error"] = s; return r
    with open(infer_p) as f: code2 = f.read().replace("__TICKER__", ticker)
    try: tbid = submit(PID, code2, f"{pipeline}_td_{ticker}")
    except Exception as e: r["error"] = str(e); return r
    time.sleep(70)
    bt = read_bt(PID, tbid)
    s = bt.get("statistics", {}) or {}
    eq = float(s.get("Total Net Profit","$100,000.00").replace("$","").replace(",",""))
    cagr = float(s.get("Compounding Annual Return","0%").replace("%",""))
    mdd = float(s.get("Drawdown","0%").replace("%",""))
    r["calmar"] = cagr/mdd if mdd>0 else 0
    r["return"] = (eq/100000-1)*100
    r["orders"] = int(s.get("Total Orders","0"))
    r["delta"] = r["calmar"] - known
    return r

def main():
    fast = "--fast" in sys.argv
    tickers = [a for a in sys.argv[1:] if not a.startswith("--") and a in OPTIMAL] or list(OPTIMAL.keys())
    print(f"Deploying {len(tickers)} ETFs | Skip: {RESISTANT}")
    results = []
    for i, t in enumerate(tickers):
        p, desc, k = OPTIMAL[t]
        print(f"[{i+1}/{len(tickers)}] {t}→{p}({desc})", end=" ", flush=True)
        r = deploy_one(t, p, k, fast)
        if "error" in r: print(f"❌{r['error']}")
        else:
            d = r["delta"]; f = "🎉" if d>0.05 else ("✅" if d>-0.05 else "⚠")
            print(f"{f} Cal={r['calmar']:.2f} Δ={d:+.2f} Ret={r['return']:+.1f}%")
        results.append(r)
    valid = [r for r in results if "error" not in r]
    print(f"\nDone: {len(valid)}/{len(tickers)} deployed")
    improved = [r for r in valid if r.get("delta",0)>0]
    if improved: print(f"Improved: {len(improved)} — {', '.join(r['ticker'] for r in improved)}")

if __name__ == "__main__": main()
