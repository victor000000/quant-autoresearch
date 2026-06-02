#!/usr/bin/env python3
"""Run ALL ETFs through their optimal pipeline. Real SetHoldings backtests.
Usage: python3 run_all_etfs.py [--fast] (--fast skips training, uses cached predictions)"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"
PID = 31338454

# All ETFs mapped to their optimal pipeline (from comprehensive testing)
ETF_PIPELINE = {
    # Tech (v246 - 2019 split): Best for QQQ, XLK, SMH, SOXX, VGT, SPY, EWT
    "QQQ": "v246", "XLK": "v246", "SMH": "v246", "SOXX": "v246",
    "VGT": "v246", "SPY": "v246", "EWT": "v246", "DIA": "v246",
    # Commodity (v270 - Tick bars): Best for GLD, GDXJ, SIL, SLV, XME, XLE
    "GLD": "v270", "GDXJ": "v270", "SIL": "v270", "SLV": "v270",
    "XME": "v270", "XLE": "v270", "XLB": "v270", "USO": "v270",
    # General Trend (v264 - 2021 split): Best for GDX, KRE, XTN, XLI, XLY, XRT, XHB
    "GDX": "v264", "KRE": "v264", "XTN": "v264", "XLI": "v264",
    "XLY": "v264", "XRT": "v264", "XHB": "v264", "ITB": "v264",
    "XLF": "v264", "KBE": "v264", "XOP": "v264", "FCG": "v264",
    # Mean-Revert (v274 - Ridge + MR): Best for TLT, SHY, IEF
    "TLT": "v274", "SHY": "v274", "IEF": "v274",
    # Credit (v246): Best for HYG, LQD
    "HYG": "v246", "LQD": "v246",
    # Defensive (v264): XLP, XLU, XLV, XBI, IBB, TAN
    "XLP": "v264", "XLU": "v264", "XLV": "v264", "XBI": "v264",
    "IBB": "v264", "TAN": "v264",
    # Indices/International
    "IWM": "v286", "EEM": "v264", "EWH": "v264", "VNQ": "v264",
    "REM": "v264", "DBC": "v270",
}

def run_one(ticker, pipeline, fast=False):
    """Train + test one ETF. Returns result dict."""
    train_path = f"{BASE}/_pipeline_{pipeline}_train/main.py"
    infer_path = f"{BASE}/_pipeline_{pipeline}_infer/main.py"

    if not fast:
        if not os.path.exists(train_path):
            return {"ticker": ticker, "error": "missing train file"}

        with open(train_path) as f:
            code = f.read().replace("__TICKER__", ticker)

        try:
            bid = submit(PID, code, f"{pipeline}_{ticker}")
        except Exception as e:
            return {"ticker": ticker, "error": f"submit: {e}"}

        for _ in range(90):
            time.sleep(10)
            status, rt, _ = read_bt_status(PID, bid)
            if status.startswith("Completed"): break
            if "Error" in status:
                return {"ticker": ticker, "error": f"train: {status}"}

    # Test with real SetHoldings
    if not os.path.exists(infer_path):
        return {"ticker": ticker, "error": "missing infer file"}

    with open(infer_path) as f:
        code2 = f.read().replace("__TICKER__", ticker)

    try:
        tbid = submit(PID, code2, f"{pipeline}_test_{ticker}")
    except Exception as e:
        return {"ticker": ticker, "error": f"test submit: {e}"}

    time.sleep(70)

    try:
        bt = read_bt(PID, tbid)
    except:
        return {"ticker": ticker, "error": "read bt failed"}

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
        "orders": orders
    }

def main():
    fast = "--fast" in sys.argv
    tickers = list(ETF_PIPELINE.keys())
    if len(sys.argv) > 1 and sys.argv[1] != "--fast":
        tickers = [t for t in sys.argv[1:] if t in ETF_PIPELINE]

    print(f"Running {len(tickers)} ETFs {'(fast mode)' if fast else ''}")
    print(f"{'='*70}")

    results = []
    for i, ticker in enumerate(tickers):
        pipeline = ETF_PIPELINE[ticker]
        print(f"\n[{i+1}/{len(tickers)}] {ticker} → {pipeline}")

        result = run_one(ticker, pipeline, fast=fast)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  REAL: ${result['equity']:,.0f} ({result['return_pct']:+.1f}%) "
                  f"CAGR={result['cagr']:.1f}% MDD={result['mdd']:.1f}% "
                  f"Calmar={result['calmar']:.2f} Ord={result['orders']}")

    # Print leaderboard
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["calmar"], reverse=True)

    print(f"\n{'='*70}")
    print(f"LEADERBOARD ({len(valid)} ETFs)")
    print(f"{'='*70}")
    print(f"{'ETF':<8} {'Pipeline':<8} {'Return':>8} {'Calmar':>8} {'Orders':>8}")
    print(f"{'-'*45}")

    tiers = {"S": [], "A": [], "B": [], "C": [], "D": [], "F": []}
    for r in valid:
        print(f"{r['ticker']:<8} {r['pipeline']:<8} {r['return_pct']:>7.1f}% {r['calmar']:>7.2f} {r['orders']:>8}")
        if r['calmar'] > 1.0: tiers["S"].append(r['ticker'])
        elif r['calmar'] > 0.5: tiers["A"].append(r['ticker'])
        elif r['calmar'] > 0.3: tiers["B"].append(r['ticker'])
        elif r['calmar'] > 0.1: tiers["C"].append(r['ticker'])
        elif r['calmar'] > 0: tiers["D"].append(r['ticker'])
        else: tiers["F"].append(r['ticker'])

    print(f"\nTiers: S={len(tiers['S'])} A={len(tiers['A'])} B={len(tiers['B'])} "
          f"C={len(tiers['C'])} D={len(tiers['D'])} F={len(tiers['F'])}")
    for tier in ["S","A","B","C","D","F"]:
        if tiers[tier]:
            print(f"  {tier}: {', '.join(tiers[tier])}")

if __name__ == "__main__":
    main()
