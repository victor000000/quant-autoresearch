"""Deploy all 7 ETFs with optimal configs. Usage: python3 deploy_all.py"""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

DEPLOY = {
    "QQQ": {"pid": 31338454, "train": "v246", "desc": "Dollar+KMeans, 2019 split"},
    "GLD": {"pid": 31338453, "train": "v270", "desc": "Tick+KMeans+Entropy, 2021 split"},
    "EEM": {"pid": 31338455, "train": "v264", "desc": "Dollar+KMeans+Entropy, 2021 split"},
    "IWM": {"pid": 31338456, "train": "v286", "desc": "Dollar+KMeans+Entropy, 2018 split"},
    "XLE": {"pid": 31338458, "train": "v270", "desc": "Tick+KMeans+Entropy, 2021 split"},
    "HYG": {"pid": 31338457, "train": "v246", "desc": "Dollar+KMeans, 2019 split"},
    "TLT": {"pid": 31338460, "train": "v274", "desc": "Dollar+MR+Ridge a=1, 2021 split"},
}

def load_code(ver, suffix):
    path = f"/Users/liyuanjun/ai_work/lb/lean_workspace/_pipeline_{ver}_{suffix}/main.py"
    return open(path).read() if os.path.exists(path) else None

def deploy():
    # Phase 1: Training
    jobs = {}
    for ticker, cfg in DEPLOY.items():
        code = load_code(cfg["train"], "train")
        if not code:
            print(f"SKIP {ticker}: code not found")
            continue
        try:
            bid = submit(cfg["pid"], code.replace("__TICKER__", ticker), f"prod_{ticker}")
            jobs[ticker] = (cfg["pid"], bid, cfg)
            print(f"TRAIN {ticker}: {bid} | {cfg['desc']}")
        except Exception as e:
            print(f"FAIL {ticker}: {e}")
        time.sleep(2)

    # Phase 2: Wait
    print(f"\nWaiting for {len(jobs)} training jobs...")
    pending = [(t, p, b) for t, (p, b, _) in jobs.items()]
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
        if not pending: break

    # Phase 3: Inference
    print(f"\nSubmitting inference...")
    results = {}
    for ticker, (pid, bid, cfg) in jobs.items():
        code = load_code(cfg["train"], "infer")
        if not code: continue
        try:
            tbid = submit(pid, code.replace("__TICKER__", ticker), f"prod_test_{ticker}")
            results[ticker] = (pid, tbid)
            print(f"TEST {ticker}: {tbid}")
        except Exception as e:
            print(f"FAIL {ticker}: {e}")
        time.sleep(2)

    # Phase 4: Report
    time.sleep(90)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Real SetHoldings Orders)")
    print(f"{'='*60}")
    print(f"{'ETF':<6} {'Equity':<14} {'CAGR':<10} {'MDD':<10} {'Calmar':<8} {'Orders':<8}")
    print(f"{'-'*56}")
    for ticker, (pid, bid) in results.items():
        bt = read_bt(pid, bid)
        s = bt.get("statistics", {}) or {}
        rt = bt.get("runtimeStatistics", {}) or {}
        eq = rt.get("Equity", s.get("Total Net Profit", "$0"))
        cagr = s.get("Compounding Annual Return", "0%")
        mdd = s.get("Drawdown", "0%")
        orders = s.get("Total Orders", "0")
        try:
            c = float(cagr.replace("%",""))
            d = float(mdd.replace("%",""))
            cal = f"{c/d:.2f}" if d > 0.01 else "-"
        except:
            cal = "-"
        print(f"{ticker:<6} {eq:<14} {cagr:<10} {mdd:<10} {cal:<8} {orders:<8}")
    print(f"\nDeployment complete. {len(results)}/7 ETFs.")

if __name__ == "__main__":
    deploy()
