#!/usr/bin/env python3
"""Auto-ETF V2: Tests v246, v264, v270 pipelines. Selects best by REAL backtest Calmar.
Usage: python3 auto_etf_v2.py TICKER"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"
PIPELINES = {
    "v246": {"desc": "Dollar, KMeans, XGBoost, 2019 split", "pid": 31338454},
    "v264": {"desc": "Dollar, KMeans, Entropy, 2021 split", "pid": 31338454},
    "v270": {"desc": "Tick, KMeans, Entropy, 2021 split", "pid": 31338454},
}

def run_backtest(ticker, pipeline):
    """Train + test for one pipeline. Returns (equity, cagr, mdd, orders, config) or None."""
    train_path = f"{BASE}/_pipeline_{pipeline}_train/main.py"
    infer_path = f"{BASE}/_pipeline_{pipeline}_infer/main.py"

    if not os.path.exists(train_path) or not os.path.exists(infer_path):
        print(f"  SKIP {pipeline}: missing files")
        return None

    # Submit training
    with open(train_path) as f:
        code = f.read().replace("__TICKER__", ticker)

    try:
        bid = submit(PIPELINES[pipeline]["pid"], code, f"{pipeline}_{ticker}")
    except Exception as e:
        print(f"  {pipeline} submit error: {e}")
        return None

    # Wait for training
    for _ in range(90):
        time.sleep(10)
        status, rt, _ = read_bt_status(PIPELINES[pipeline]["pid"], bid)
        if status.startswith("Completed"):
            break
        if "Error" in status:
            print(f"  {pipeline} train error: {status}")
            return None

    # Submit inference
    with open(infer_path) as f:
        code2 = f.read().replace("__TICKER__", ticker)

    try:
        tbid = submit(PIPELINES[pipeline]["pid"], code2, f"{pipeline}_test_{ticker}")
    except:
        return None

    time.sleep(60)

    try:
        bt = read_bt(PIPELINES[pipeline]["pid"], tbid)
    except:
        return None

    s = bt.get("statistics", {}) or {}
    rt = bt.get("runtimeStatistics", {}) or {}
    eq_str = rt.get("Equity", s.get("Total Net Profit", "$100,000.00"))
    # Parse equity: "$123,456.78" -> 123456.78
    try:
        eq = float(eq_str.replace("$","").replace(",",""))
    except:
        eq = 100000.0

    cagr_str = s.get("Compounding Annual Return", "0%")
    try:
        cagr = float(cagr_str.replace("%",""))
    except:
        cagr = 0.0

    mdd_str = s.get("Drawdown", "0%")
    try:
        mdd = float(mdd_str.replace("%",""))
    except:
        mdd = 0.0

    orders_str = s.get("Total Orders", "0")
    try:
        orders = int(orders_str)
    except:
        orders = 0

    calmar = cagr / mdd if mdd > 0 else 0
    config = rt.get("best_cfg", "?")

    return {"equity": eq, "cagr": cagr, "mdd": mdd, "calmar": calmar,
            "orders": orders, "config": config, "pipeline": pipeline}

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(f"\n{'='*60}")
    print(f"Auto-ETF V2: {ticker} — Testing 3 pipelines with REAL backtests")
    print(f"{'='*60}")

    best = None

    for pipeline, info in PIPELINES.items():
        print(f"\n[{pipeline}] {info['desc']}...")
        result = run_backtest(ticker, pipeline)
        if result is None:
            continue

        print(f"  Equity: ${result['equity']:,.2f} ({((result['equity']/100000)-1)*100:+.1f}%)")
        print(f"  CAGR: {result['cagr']:.2f}%  MDD: {result['mdd']:.1f}%")
        print(f"  Calmar: {result['calmar']:.2f}  Orders: {result['orders']}")
        print(f"  Config: {result['config']}")

        # Select best by Calmar, but require >5 orders for valid trading
        if result['orders'] >= 5:
            if best is None or result['calmar'] > best['calmar']:
                best = result

    if best is None:
        print(f"\n{ticker}: No valid pipeline found (all had <5 orders)")
        return

    print(f"\n{'='*60}")
    print(f"BEST: {best['pipeline']} — {PIPELINES[best['pipeline']]['desc']}")
    print(f"  Equity: ${best['equity']:,.2f} ({((best['equity']/100000)-1)*100:+.1f}%)")
    print(f"  CAGR: {best['cagr']:.2f}%  MDD: {best['mdd']:.1f}%")
    print(f"  REAL Calmar: {best['calmar']:.2f}  Orders: {best['orders']}")
    print(f"  Config: {best['config']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
