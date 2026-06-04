#!/usr/bin/env python3
"""Focused experiment runner — target the ETFs with highest improvement potential.

Based on empirical per_ticker_optimal.csv analysis:
- Near 1.0 Calmar: DBA(0.99), AGG(0.97), XLI(0.93) → need small boost
- Stuck 0.5-0.8: USO(0.90), DBC(0.87), XHB(0.78), XLF(0.76), KRE(0.75)
- Near-zero: PFF(0.57), VNQ(0.46), XOP(0.20)

Strategy: Run v389 auto-select or v392 calibrated ensemble on borderline ETFs
to find optimal per-ETF configuration.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"
PID = 31338454

# Focused experiments: (ETF, Pipeline, Strategy Reason)
EXPERIMENTS = [
    # Near 1.0 — try v392 calibrated ensemble for better position sizing
    ("DBA", "v392", "Calibrated ensemble — boost from 0.99 to 1.0+"),
    ("AGG", "v392", "Calibrated ensemble — boost from 0.97 to 1.0+"),
    ("XLI", "v392", "Calibrated ensemble — boost from 0.93 to 1.0+"),

    # Stuck 0.5-0.8 — try different axis
    ("USO", "v389", "Auto-select — find better axis than vol bars"),
    ("DBC", "v389", "Auto-select — find better axis than vol bars"),
    ("XHB", "v395", "Multi-order — stronger trend signals"),
    ("XLF", "v395", "Multi-order — financials may respond to order combo"),
    ("KRE", "v394", "Change-point labels — banks are regime-sensitive"),

    # Near-zero — try most sophisticated
    ("XOP", "v394", "Change-point — oil E&P is highly cyclical"),
    ("VNQ", "v392", "Calibrated ensemble — REITs need calibrated sizing"),
    ("XRT", "v395", "Multi-order — retail is mean-reverting + trending"),
    ("PFF", "v392", "Calibrated ensemble — preferred shares need calibration"),

    # Verify new pipelines on known winners
    ("GLD", "v395", "Multi-order on proven gold — compare to v370's 1.59"),
    ("QQQ", "v395", "Multi-order on proven tech — compare to v246's 1.29"),
]


def run_experiment(ticker, pipeline, reason):
    """Train + test one ETF with real SetHoldings. Returns result dict."""
    train_path = f"{BASE}/_pipeline_{pipeline}_train/main.py"
    infer_path = f"{BASE}/_pipeline_{pipeline}_infer/main.py"

    if not os.path.exists(train_path):
        return {"ticker": ticker, "pipeline": pipeline, "error": f"missing train file"}
    if not os.path.exists(infer_path):
        return {"ticker": ticker, "pipeline": pipeline, "error": f"missing infer file"}

    # Submit training
    with open(train_path) as f:
        code = f.read().replace("__TICKER__", ticker)

    try:
        bid = submit(PID, code, f"{pipeline}_{ticker}_exp")
    except Exception as e:
        return {"ticker": ticker, "pipeline": pipeline, "error": f"submit: {e}"}

    print(f"  Training... (bid={bid})")
    for attempt in range(120):
        time.sleep(10)
        status, rt, _ = read_bt_status(PID, bid)
        if status.startswith("Completed"):
            rt_stats = rt or {}
            best_cfg = rt_stats.get("best_cfg", "?")
            val_cal = rt_stats.get("val_calmar", "?")
            print(f"  ✓ Trained | best={best_cfg} val_cal={val_cal}")
            break
        if "Error" in status:
            return {"ticker": ticker, "pipeline": pipeline, "error": f"train: {status}"}
    else:
        return {"ticker": ticker, "pipeline": pipeline, "error": "train timeout"}

    # Submit inference (real backtest)
    with open(infer_path) as f:
        code2 = f.read().replace("__TICKER__", ticker)

    try:
        tbid = submit(PID, code2, f"{pipeline}_test_{ticker}_exp")
    except Exception as e:
        return {"ticker": ticker, "pipeline": pipeline, "error": f"test submit: {e}"}

    print(f"  Backtesting... (tbid={tbid})")
    time.sleep(70)

    try:
        bt = read_bt(PID, tbid)
    except:
        return {"ticker": ticker, "pipeline": pipeline, "error": "read bt failed"}

    s = bt.get("statistics", {}) or {}
    rt = bt.get("runtimeStatistics", {}) or {}
    eq_str = rt.get("Equity", s.get("Total Net Profit", "$100,000.00"))
    try: eq = float(eq_str.replace("$","").replace(",",""))
    except: eq = 100000.0

    cagr_str = s.get("Compounding Annual Return", "0%")
    try: cagr = float(cagr_str.replace("%",""))
    except: cagr = 0.0

    mdd_str = s.get("Drawdown", "0%")
    try: mdd = float(mdd_str.replace("%",""))
    except: mdd = 0.0

    orders_str = s.get("Total Orders", "0")
    try: orders = int(orders_str)
    except: orders = 0

    calmar = cagr / mdd if mdd > 0 else 0
    ret_pct = (eq / 100000 - 1) * 100

    return {
        "ticker": ticker, "pipeline": pipeline,
        "equity": eq, "return_pct": ret_pct,
        "cagr": cagr, "mdd": mdd, "calmar": calmar,
        "orders": orders, "reason": reason,
    }


def main():
    fast = "--fast" in sys.argv
    tickers_filter = [a for a in sys.argv[1:] if not a.startswith("--")]

    experiments = EXPERIMENTS
    if tickers_filter:
        experiments = [e for e in EXPERIMENTS if e[0] in tickers_filter]

    print(f"Focused Experiments: {len(experiments)} ETF×Pipeline combinations")
    print(f"{'='*70}")
    print(f"{'ETF':<6} {'Pipeline':<8} {'Reason'}")
    for ticker, pipeline, reason in experiments:
        print(f"{ticker:<6} {pipeline:<8} {reason}")
    print(f"{'='*70}")

    results = []
    for i, (ticker, pipeline, reason) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {ticker} → {pipeline}: {reason}")
        result = run_experiment(ticker, pipeline, reason)
        results.append(result)

        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            print(f"  ✅ ${result['equity']:,.0f} ({result['return_pct']:+.1f}%) "
                  f"Calmar={result['calmar']:.2f} Orders={result['orders']}")

    # Leaderboard
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["calmar"], reverse=True)

    print(f"\n{'='*70}")
    print(f"RESULTS ({len(valid)}/{len(experiments)} successful)")
    print(f"{'='*70}")
    print(f"{'ETF':<6} {'Pipeline':<8} {'Return':>8} {'Calmar':>8} {'Orders':>6} {'Reason'}")
    print(f"{'-'*80}")
    for r in valid:
        print(f"{r['ticker']:<6} {r['pipeline']:<8} {r['return_pct']:>7.1f}% "
              f"{r['calmar']:>7.2f} {r['orders']:>6} {r.get('reason','')[:30]}")

    # Compare with previous best
    print(f"\n{'='*70}")
    print("COMPARISON WITH PREVIOUS BEST")
    print(f"{'='*70}")
    prev_best = {
        "DBA": (0.99, "v372"), "AGG": (0.97, "v372"), "XLI": (0.93, "v372"),
        "USO": (0.90, "v372"), "DBC": (0.87, "v372"), "XHB": (0.78, "v372"),
        "XLF": (0.76, "v264"), "KRE": (0.75, "v264"), "XOP": (0.20, "v372"),
        "VNQ": (0.46, "v372"), "XRT": (0.68, "v372"), "PFF": (0.57, "v372"),
        "GLD": (1.59, "v370"), "QQQ": (1.29, "v246"),
    }
    for r in valid:
        t = r['ticker']
        prev = prev_best.get(t, (0, "?"))
        delta = r['calmar'] - prev[0]
        sign = "+" if delta > 0 else ""
        print(f"{t:<6}: {prev[1]} Cal {prev[0]:.2f} → {r['pipeline']} Cal {r['calmar']:.2f} "
              f"({sign}{delta:.2f}) {'🎉 NEW BEST!' if delta > 0.01 else ''}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "focused_experiment_results.json")
    with open(out_path, 'w') as f:
        json.dump({"n_experiments": len(experiments), "n_valid": len(valid),
                   "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
