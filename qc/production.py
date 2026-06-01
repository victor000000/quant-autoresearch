#!/usr/bin/env python3
"""One-command production deployment of all 7 ETFs with optimal configs."""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

OPTIMAL = {
    "QQQ": (31338454, "v246", "Dollar, KMeans, 2019 split"),
    "GLD": (31338453, "v270", "Tick, KMeans, Entropy, 2021 split"),
    "EEM": (31338455, "v264", "Dollar, KMeans, Entropy, 2021 split"),
    "IWM": (31338456, "v286", "Dollar, KMeans, Entropy, 2018 split"),
    "XLE": (31338458, "v270", "Tick, KMeans, Entropy, 2021 split"),
    "HYG": (31338457, "v246", "Dollar, KMeans, 2019 split"),
    "TLT": (31338460, "v274", "Dollar, MR, Ridge a=1, 2021 split"),
}
BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"

def load(ver, kind):
    path = f"{BASE}/_pipeline_{ver}_{kind}/main.py"
    return open(path).read() if os.path.exists(path) else None

def main():
    jobs = {}
    print("Training 7 ETFs with optimal configs...\n")
    for ticker, (pid, ver, desc) in OPTIMAL.items():
        code = load(ver, "train")
        if not code: print(f"  SKIP {ticker}: no code"); continue
        try:
            bid = submit(pid, code.replace("__TICKER__", ticker), f"prod_{ticker}")
            jobs[ticker] = (pid, bid, ver, desc)
            print(f"  {ticker}: {bid} | {desc}")
        except Exception as e:
            print(f"  {ticker}: {e}")
        time.sleep(1)

    print(f"\nWaiting ({len(jobs)} training)...")
    pending = [(t, p, b) for t, (p, b, *_) in jobs.items()]
    for _ in range(80):
        time.sleep(15)
        still = []
        for ticker, pid, bid in pending:
            s, rt, _ = read_bt_status(pid, bid)
            if s.startswith("Completed") or "Error" in s:
                print(f"  {ticker}: {s} cfg={rt.get('best_cfg','?')}")
            else:
                still.append((ticker, pid, bid))
        pending = still
        if not pending: break

    print(f"\nTesting with real SetHoldings...")
    results = {}
    for ticker, (pid, bid, ver, desc) in jobs.items():
        code = load(ver, "infer")
        if not code: continue
        c = code.replace("__TICKER__", ticker)
        try:
            tbid = submit(pid, c, f"prod_test_{ticker}")
            results[ticker] = (pid, tbid, desc)
            print(f"  {ticker}: {tbid}")
        except Exception as e:
            print(f"  {ticker}: {e}")
        time.sleep(2)

    time.sleep(90)
    print(f"\n{'='*60}")
    print("FINAL RESULTS (Real SetHoldings Orders)")
    print(f"{'='*60}")
    for ticker, (pid, bid, desc) in results.items():
        bt = read_bt(pid, bid)
        s = bt.get("statistics", {}) or {}
        rt = bt.get("runtimeStatistics", {}) or {}
        eq = rt.get("Equity", s.get("Total Net Profit", "$0"))
        cagr = s.get("Compounding Annual Return", "0%")
        mdd = s.get("Drawdown", "0%")
        orders = s.get("Total Orders", "0")
        print(f"  {ticker:<6} {eq:<14} {cagr:<10} {mdd:<10} {orders:<6} {desc}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
