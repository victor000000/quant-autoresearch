#!/usr/bin/env python3
"""Production monitoring: Run all 7 ETFs with optimal configs and report results."""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

CONFIG = {
    "QQQ": (31338454, "v246", "v246_infer", "Dollar, KMeans, 2019"),
    "GLD": (31338453, "v270", "v270_infer", "Tick, KMeans, Entropy, 2021"),
    "EEM": (31338455, "v324", "v264_infer", "Dollar, KMeans, Entropy, 2021 (fixed)"),
    "IWM": (31338456, "v325", "v264_infer", "Dollar, KMeans, Entropy, 2018 (fixed)"),
    "XLE": (31338458, "v270", "v270_infer", "Tick, KMeans, Entropy, 2021"),
    "HYG": (31338457, "v246", "v246_infer", "Dollar, KMeans, 2019"),
    "TLT": (31338460, "v274", "v274_infer", "Dollar, MR, Ridge a=1, 2021"),
}

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"

def load(ver, suffix):
    path = f"{BASE}/_pipeline_{ver}_{suffix}/main.py"
    return open(path).read() if os.path.exists(path) else None

def main():
    jobs = {}
    # Submit training
    for ticker, (pid, train_ver, infer_ver, desc) in CONFIG.items():
        code = load(train_ver, "train")
        if not code:
            print(f"SKIP {ticker}: code not found")
            continue
        try:
            bid = submit(pid, code.replace("__TICKER__", ticker), f"prod_{ticker}")
            jobs[ticker] = (pid, bid, desc, infer_ver)
            print(f"TRAIN {ticker}: {bid} | {desc}")
        except Exception as e:
            print(f"FAIL {ticker}: {e}")
        time.sleep(1)

    # Wait for training
    print(f"\nTraining {len(jobs)} ETFs...")
    pending = [(t, p, b) for t, (p, b, _, _) in jobs.items()]
    for _ in range(80):
        time.sleep(15)
        still = []
        for ticker, pid, bid in pending:
            status, rt, _ = read_bt_status(pid, bid)
            if status.startswith("Completed") or "Error" in status:
                print(f"  {ticker}: {status} cfg={rt.get('best_cfg','?')}")
            else:
                still.append((ticker, pid, bid))
        pending = still
        if not pending:
            break

    # Submit inference
    print(f"\nTesting with real SetHoldings...")
    results = {}
    for ticker, (pid, bid, desc, infer_ver) in jobs.items():
        code = load(infer_ver, "infer")
        if not code:
            continue
        c = code.replace("__TICKER__", ticker)
        # Fix path for EEM/IWM which use different train versions
        train_ver = CONFIG[ticker][1]
        if train_ver != infer_ver.replace("_infer", ""):
            c = c.replace(f'{infer_ver.replace("_infer","")}/', f'{train_ver}/')
        # Fix start date for 2018 split
        if ticker == "IWM":
            c = c.replace("datetime(2023, 8, 1)", "datetime(2021, 8, 1)")
        try:
            tbid = submit(pid, c, f"prod_test_{ticker}")
            results[ticker] = (pid, tbid, desc)
            print(f"TEST {ticker}: {tbid}")
        except Exception as e:
            print(f"FAIL {ticker}: {e}")
        time.sleep(2)

    # Report
    time.sleep(90)
    print(f"\n{'='*65}")
    print(f"ETF PERFORMANCE REPORT (Real SetHoldings Orders)")
    print(f"{'='*65}")
    print(f"{'ETF':<6} {'Equity':<14} {'Return':<10} {'CAGR':<10} {'MDD':<10} {'Orders':<8}")
    print(f"{'-'*58}")
    for ticker, (pid, bid, desc) in results.items():
        bt = read_bt(pid, bid)
        s = bt.get("statistics", {}) or {}
        rt = bt.get("runtimeStatistics", {}) or {}
        eq = rt.get("Equity", "$0")
        cagr = s.get("Compounding Annual Return", "0%")
        mdd = s.get("Drawdown", "0%")
        orders = s.get("Total Orders", "0")
        try:
            ret = f"{float(eq.replace('$','').replace(',',''))/1000 - 100:.1f}%"
        except:
            ret = "?"
        print(f"{ticker:<6} {eq:<14} {ret:<10} {cagr:<10} {mdd:<10} {orders:<8}")
    print(f"{'='*65}")
    print(f"Report complete. {len(results)}/7 ETFs.")

if __name__ == "__main__":
    main()
