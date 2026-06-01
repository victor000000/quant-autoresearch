#!/usr/bin/env python3
"""Auto-ETF: Submit any ticker with auto-diagnosed optimal configuration.
Usage: python3 auto_etf.py TICKER"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt
import time

# Decision rules from 141 experiments
RULES = {
    "TLT": ("mean-reverting", "v274", "Dollar, MR, Ridge a=1, 2021"),
    "HYG": ("credit", "v246", "Dollar, KMeans, XGBoost, 2019"),
    "LQD": ("credit", "v246", "Dollar, KMeans, XGBoost, 2019"),
    "IEF": ("rates", "v274", "Dollar, MR, Ridge a=1, 2021"),
    "SHY": ("rates", "v274", "Dollar, MR, Ridge a=1, 2021"),
    "GLD": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
    "SLV": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
    "XLE": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
    "USO": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
    "IWM": ("noisy-trend", "v286", "Dollar, KMeans, Entropy, 2018"),
    "EEM": ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021"),
    "QQQ": ("trend-driven", "v246", "Dollar, KMeans, XGBoost, 2019"),
    "SPY": ("trend-driven", "v246", "Dollar, KMeans, XGBoost, 2019"),
    "DIA": ("trend-driven", "v246", "Dollar, KMeans, XGBoost, 2019"),
    "SMH": ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021"),
    "SOXX": ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021"),
    "XLF": ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021"),
    "XLI": ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021"),
    "XLB": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
    "XLV": ("trend-driven", "v246", "Dollar, KMeans, XGBoost, 2019"),
    "XLP": ("defensive", "v264", "Dollar, KMeans, Entropy, 2021"),
    "XLU": ("defensive", "v264", "Dollar, KMeans, Entropy, 2021"),
    "XRT": ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021"),
    "XHB": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
    "IBB": ("defensive", "v264", "Dollar, KMeans, Entropy, 2021"),
    "TAN": ("event-driven", "v270", "Tick, KMeans, Entropy, 2021"),
}

DEFAULT = ("trend-driven", "v264", "Dollar, KMeans, Entropy, 2021 (default)")

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"
POOLS = {
    "trend-driven": 31338454,
    "event-driven": 31338453,
    "noisy-trend": 31338456,
    "credit": 31338457,
    "rates": 31338460,
    "mean-reverting": 31338460,
}

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    category, version, desc = RULES.get(ticker, DEFAULT)
    pid = POOLS.get(category, 31338454)

    print(f"Auto-ETF: {ticker}")
    print(f"  Category: {category}")
    print(f"  Pipeline: {version}")
    print(f"  Config: {desc}")
    print(f"  Project: {pid}")

    # Submit training
    train_path = f"{BASE}/_pipeline_{version}_train/main.py"
    if not os.path.exists(train_path):
        print(f"  ERROR: {train_path} not found")
        return

    with open(train_path) as f:
        code = f.read().replace("__TICKER__", ticker)

    try:
        bid = submit(pid, code, f"auto_{ticker}")
        print(f"\nTraining: {bid}")
        for _ in range(30):
            time.sleep(10)
            status, rt, _ = read_bt_status(pid, bid)
            if status.startswith("Completed") or "Error" in status:
                print(f"  Status: {status} cfg={rt.get('best_cfg','?')}")
                break

        # Submit inference
        infer_path = f"{BASE}/_pipeline_{version}_infer/main.py"
        if not os.path.exists(infer_path):
            infer_path = f"{BASE}/_pipeline_v264_infer/main.py"

        with open(infer_path) as f:
            code2 = f.read().replace("__TICKER__", ticker)
            code2 = code2.replace("v264/", f"{version}/")

        tbid = submit(pid, code2, f"auto_test_{ticker}")
        print(f"Testing: {tbid}")
        time.sleep(60)

        bt = read_bt(pid, tbid)
        s = bt.get("statistics", {}) or {}
        rt = bt.get("runtimeStatistics", {}) or {}
        print(f"\n=== {ticker} Results (Real SetHoldings) ===")
        print(f"  Equity: {rt.get('Equity', s.get('Total Net Profit', '$?'))}")
        print(f"  CAGR: {s.get('Compounding Annual Return', '?')}")
        print(f"  MDD: {s.get('Drawdown', '?')}")
        print(f"  Orders: {s.get('Total Orders', '?')}")
        print(f"  Category: {category} | {desc}")

    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    main()
