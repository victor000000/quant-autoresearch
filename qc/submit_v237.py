"""Submit v237 Trend Scanning pipeline for all 7 ETFs."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from api_curl import qc_post, submit, read_bt, is_done

ETFS = ["GLD", "QQQ", "EEM", "IWM", "HYG", "XLE", "TLT"]
POOL_PIDS = [31338453, 31338454, 31338455, 31338456, 31338457, 31338458, 31338460, 31338461]

def load_code(version, suffix="train"):
    path = Path(f"/Users/liyuanjun/ai_work/lb/lean_workspace/_pipeline_v{version}_{suffix}/main.py")
    with open(path) as f:
        return f.read()

def main():
    train_code = load_code(237, "train")
    infer_code = load_code(237, "infer")

    # Phase 1: Submit training for all 7 ETFs
    print("=== Phase 1: Training ===")
    train_jobs = []
    for etf, pid in zip(ETFS, POOL_PIDS):
        code = train_code.replace("__TICKER__", etf)
        name = f"v237_train_{etf}"
        print(f"Submitting {name} to pid={pid}...")
        try:
            bid = submit(pid, code, name)
            train_jobs.append((pid, bid, etf))
            print(f"  bid={bid}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nSubmitted {len(train_jobs)} training jobs. Waiting...")

    # Poll training
    t0 = time.time()
    results = {}
    pending = train_jobs[:]
    while pending and time.time() - t0 < 3600:
        time.sleep(15)
        still = []
        for pid, bid, etf in pending:
            bt = read_bt(pid, bid)
            if is_done(bt):
                results[etf] = bt
                s = bt.get("statistics", {})
                rt = bt.get("runtimeStatistics", {})
                best = rt.get("best_cfg", "?")
                n_bars = rt.get("n_bars", "?")
                print(f"[{int(time.time()-t0)}s] DONE {etf}: best_cfg={best} n_bars={n_bars} status={bt.get('status','')}")
            else:
                still.append((pid, bid, etf))
                if len(still) <= 3:
                    print(f"[{int(time.time()-t0)}s] WAIT {etf}: progress={bt.get('progress',0):.1f}")
        pending = still

    print(f"\n=== Training Complete: {len(results)}/{len(ETFS)} ===")

    # Save training results
    train_summary = {}
    for etf, bt in results.items():
        rt = bt.get("runtimeStatistics", {})
        train_summary[etf] = {
            "best_cfg": rt.get("best_cfg", ""),
            "n_bars": rt.get("n_bars", ""),
            "status": bt.get("status", ""),
        }
    with open(Path(__file__).parent / "v237_train_results.json", "w") as f:
        json.dump(train_summary, f, indent=2)

    # Phase 2: Submit inference (TEST period)
    print("\n=== Phase 2: TEST Inference ===")
    test_jobs = []
    for etf, pid in zip(ETFS, POOL_PIDS):
        code = infer_code.replace("__TICKER__", etf)
        name = f"v237_test_{etf}"
        print(f"Submitting {name}...")
        try:
            bid = submit(pid, code, name)
            test_jobs.append((pid, bid, etf))
            print(f"  bid={bid}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Poll test
    t0 = time.time()
    test_results = {}
    pending = test_jobs[:]
    while pending and time.time() - t0 < 1800:
        time.sleep(15)
        still = []
        for pid, bid, etf in pending:
            bt = read_bt(pid, bid)
            if is_done(bt):
                test_results[etf] = bt
                s = bt.get("statistics", {})
                cagr = s.get("Compounding Annual Return", "0%")
                mdd = s.get("Drawdown", "0%")
                net = s.get("Total Net Profit", "0%")
                try:
                    c = float(cagr.replace("%",""))
                    d = float(mdd.replace("%",""))
                    cal = round(c/d, 2) if d > 0.01 else 0.0
                except: cal = 0.0
                print(f"[{int(time.time()-t0)}s] DONE {etf}: NET={net} Cal={cal} orders={s.get('Total Orders','-')}")
            else:
                still.append((pid, bid, etf))
        pending = still

    # Print final summary
    print("\n=== v237 Final Results ===")
    for etf in ETFS:
        tr = train_summary.get(etf, {})
        bt = test_results.get(etf, {})
        s = bt.get("statistics", {})
        net = s.get("Total Net Profit", "$0")
        cagr = s.get("Compounding Annual Return", "0%")
        mdd = s.get("Drawdown", "0%")
        orders = s.get("Total Orders", "-")
        print(f"  {etf}: {net} Cal={cagr}/{mdd} orders={orders} cfg={tr.get('best_cfg','?')}")

if __name__ == "__main__":
    main()
