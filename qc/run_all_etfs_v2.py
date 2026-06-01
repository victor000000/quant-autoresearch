#!/usr/bin/env python3
"""Run ALL ETFs through their optimal pipeline v2 — includes v385-v389.

New pipeline variants:
  v385: Range Bars + Enhanced Features + Multi-Label (NEW AXIS)
  v386: Dollar Bars + Quantile Tertile Labels + Enhanced Features (NEW LABELS)
  v387: Meta-Labeling + Confidence Gating (NEW ARCHITECTURE)
  v388: Tick Bars + GMM Soft Labels + Enhanced Features (NEW LABELS)
  v389: Per-ETF Comprehensive Auto-Select (ALL combined)

Usage:
  python3 run_all_etfs_v2.py              # Run all ETFs with optimal pipeline
  python3 run_all_etfs_v2.py --fast        # Skip training, use cached predictions
  python3 run_all_etfs_v2.py QQQ GLD       # Run specific ETFs
  python3 run_all_etfs_v2.py --pipeline v389  # Run all ETFs with specific pipeline
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from api_curl import submit, read_bt_status, read_bt

BASE = "/Users/liyuanjun/ai_work/lb/lean_workspace"
PID = 31338454

# Expanded ETF → Pipeline mapping with new v385-v389 options
ETF_PIPELINE = {
    # === Tech / Growth (Dollar bars, 2019 split) ===
    "QQQ": "v389",    # Auto-select (was v246)
    "XLK": "v386",    # Quantile labels (NEW)
    "SMH": "v389",    # Auto-select
    "SOXX": "v389",   # Auto-select
    "VGT": "v386",    # Quantile labels
    "SPY": "v389",    # Auto-select
    "EWT": "v387",    # Meta-labeling (NEW)
    "DIA": "v389",    # Auto-select

    # === Commodities (Tick bars) ===
    "GLD": "v388",    # GMM soft labels (NEW, was v270)
    "GDXJ": "v388",   # GMM soft labels
    "SIL": "v388",    # GMM soft labels
    "SLV": "v388",    # GMM soft labels
    "XME": "v388",    # GMM soft labels
    "XLE": "v388",    # GMM soft labels
    "XLB": "v388",    # GMM soft labels
    "USO": "v388",    # GMM soft labels
    "DBC": "v388",    # GMM soft labels

    # === General Trend (Dollar bars, 2021 split) ===
    "GDX": "v389",    # Auto-select (was v264)
    "KRE": "v389",    # Auto-select
    "XTN": "v386",    # Quantile labels
    "XLI": "v386",    # Quantile labels
    "XLY": "v386",    # Quantile labels
    "XRT": "v387",    # Meta-labeling
    "XHB": "v387",    # Meta-labeling
    "ITB": "v387",    # Meta-labeling
    "XLF": "v389",    # Auto-select
    "KBE": "v389",    # Auto-select
    "XOP": "v386",    # Quantile labels
    "FCG": "v386",    # Quantile labels

    # === Mean-Revert (Ridge + MR labels) ===
    "TLT": "v389",    # Auto-select (was v274)
    "SHY": "v389",    # Auto-select
    "IEF": "v387",    # Meta-labeling

    # === Credit ===
    "HYG": "v386",    # Quantile labels (was v246)
    "LQD": "v387",    # Meta-labeling

    # === Defensive ===
    "XLP": "v386",    # Quantile labels
    "XLU": "v386",    # Quantile labels
    "XLV": "v387",    # Meta-labeling
    "XBI": "v387",    # Meta-labeling
    "IBB": "v387",    # Meta-labeling
    "TAN": "v386",    # Quantile labels

    # === International / Indices ===
    "IWM": "v389",    # Auto-select (was v286)
    "EEM": "v389",    # Auto-select
    "EWH": "v389",    # Auto-select
    "VNQ": "v387",    # Meta-labeling
    "REM": "v387",    # Meta-labeling

    # === NEW: Range Bars exploration ===
    "EZA": "v385",    # Range bars (South Africa — volatile EM)
    "EWZ": "v385",    # Range bars (Brazil — volatile EM)
    "FXI": "v385",    # Range bars (China — trending)
    "INDA": "v385",   # Range bars (India — trending)
    "RSX": "v385",    # Range bars (Russia — volatile)
    "EWW": "v385",    # Range bars (Mexico — trending)

    # === NEW: Info Bars exploration (v390) ===
    "UUP": "v390",    # Info bars (Dollar — regime-driven)
    "FXE": "v390",    # Info bars (Euro — regime-driven)
    "FXY": "v390",    # Info bars (Yen — safe-haven regimes)

    # === NEW: Spectral Labels (v391) ===
    "EMB": "v391",    # Spectral (EM bonds — complex manifold)
    "HYG": "v391",    # Spectral (HY credit — non-convex clusters)
    "MUB": "v391",    # Spectral (Muni bonds — complex structure)

    # === NEW: Calibrated Ensemble (v392) — universal enhancement ===
    "SPY": "v392",    # Calibrated ensemble (broad market)
    "IWM": "v392",    # Calibrated ensemble (small caps)
    "DIA": "v392",    # Calibrated ensemble (blue chips)
    "EEM": "v392",    # Calibrated ensemble (emerging markets)
}

# Pipeline descriptions
PIPELINE_INFO = {
    "v246": "Dollar + KMeans 2-stage + 2019 split (Tech/Consumer)",
    "v264": "Dollar + KMeans 2-stage + Entropy + 2021 split (General Trend)",
    "v270": "Tick + KMeans 2-stage + Entropy + 2021 split (Commodities)",
    "v274": "Dollar + MR labels + Ridge + 2021 split (Rates)",
    "v286": "Dollar + Long horizons + Entropy + 2018 split",
    "v385": "RANGE BARS + Enhanced Features + Multi-Label (NEW AXIS)",
    "v386": "Dollar + QUANTILE TERTILE Labels + Enhanced Features (NEW LABELS)",
    "v387": "META-LABELING + Confidence Gating + Enhanced Features (NEW ARCH)",
    "v388": "Tick + GMM SOFT Labels + Enhanced Features (NEW LABELS)",
    "v389": "COMPREHENSIVE AUTO-SELECT — All labels × splits × depths (MEGA)",
}


def run_one(ticker, pipeline, fast=False):
    """Train + test one ETF. Returns result dict."""
    train_path = f"{BASE}/_pipeline_{pipeline}_train/main.py"
    infer_path = f"{BASE}/_pipeline_{pipeline}_infer/main.py"

    if not fast:
        if not os.path.exists(train_path):
            return {"ticker": ticker, "pipeline": pipeline, "error": f"missing train: {train_path}"}

        with open(train_path) as f:
            code = f.read().replace("__TICKER__", ticker)

        try:
            bid = submit(PID, code, f"{pipeline}_{ticker}")
        except Exception as e:
            return {"ticker": ticker, "pipeline": pipeline, "error": f"submit: {e}"}

        for attempt in range(120):
            time.sleep(10)
            status, rt, _ = read_bt_status(PID, bid)
            if status.startswith("Completed"): break
            if "Error" in status:
                return {"ticker": ticker, "pipeline": pipeline, "error": f"train: {status}"}
        else:
            return {"ticker": ticker, "pipeline": pipeline, "error": "train timeout"}

    # Test with real SetHoldings
    if not os.path.exists(infer_path):
        return {"ticker": ticker, "pipeline": pipeline, "error": f"missing infer: {infer_path}"}

    with open(infer_path) as f:
        code2 = f.read().replace("__TICKER__", ticker)

    try:
        tbid = submit(PID, code2, f"{pipeline}_test_{ticker}")
    except Exception as e:
        return {"ticker": ticker, "pipeline": pipeline, "error": f"test submit: {e}"}

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
        "orders": orders
    }


def main():
    fast = "--fast" in sys.argv
    target_pipeline = None

    # Parse --pipeline flag
    for i, arg in enumerate(sys.argv):
        if arg == "--pipeline" and i + 1 < len(sys.argv):
            target_pipeline = sys.argv[i + 1]

    tickers = list(ETF_PIPELINE.keys())
    # Filter by command-line args (skip flags)
    cli_tickers = [a for a in sys.argv[1:] if not a.startswith("--") and a != target_pipeline]
    if cli_tickers:
        tickers = [t for t in cli_tickers if t in ETF_PIPELINE]
    if target_pipeline:
        # Override all to use target pipeline
        pass

    pipe = target_pipeline or "per-ETF optimal"
    print(f"Running {len(tickers)} ETFs | Pipeline: {pipe} | {'Fast mode' if fast else 'Full train+test'}")
    print(f"{'='*70}")

    results = []
    for i, ticker in enumerate(tickers):
        pipeline = target_pipeline or ETF_PIPELINE[ticker]
        info = PIPELINE_INFO.get(pipeline, "?")
        print(f"\n[{i+1}/{len(tickers)}] {ticker} → {pipeline} ({info})")

        result = run_one(ticker, pipeline, fast=fast)
        results.append(result)

        if "error" in result:
            print(f"  ❌ ERROR: {result['error']}")
        else:
            print(f"  ✅ ${result['equity']:,.0f} ({result['return_pct']:+.1f}%) "
                  f"CAGR={result['cagr']:.1f}% MDD={result['mdd']:.1f}% "
                  f"Calmar={result['calmar']:.2f} Orders={result['orders']}")

    # Leaderboard
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["calmar"], reverse=True)

    print(f"\n{'='*70}")
    print(f"LEADERBOARD ({len(valid)}/{len(tickers)} ETFs)")
    print(f"{'='*70}")
    print(f"{'ETF':<8} {'Pipeline':<10} {'Return':>8} {'Calmar':>8} {'Orders':>8}")
    print(f"{'-'*50}")

    tiers = {"S": [], "A": [], "B": [], "C": [], "D": [], "F": []}
    total_return = 0
    for r in valid:
        print(f"{r['ticker']:<8} {r['pipeline']:<10} {r['return_pct']:>7.1f}% {r['calmar']:>7.2f} {r['orders']:>8}")
        total_return += r['return_pct']
        if r['calmar'] > 1.0: tiers["S"].append(r['ticker'])
        elif r['calmar'] > 0.5: tiers["A"].append(r['ticker'])
        elif r['calmar'] > 0.3: tiers["B"].append(r['ticker'])
        elif r['calmar'] > 0.1: tiers["C"].append(r['ticker'])
        elif r['calmar'] > 0: tiers["D"].append(r['ticker'])
        else: tiers["F"].append(r['ticker'])

    avg_ret = total_return / len(valid) if valid else 0
    print(f"\nSummary: {len(valid)} ETFs | Avg Return: {avg_ret:+.1f}%")
    print(f"Tiers: S={len(tiers['S'])} A={len(tiers['A'])} B={len(tiers['B'])} "
          f"C={len(tiers['C'])} D={len(tiers['D'])} F={len(tiers['F'])}")
    for tier in ["S","A","B","C","D","F"]:
        if tiers[tier]:
            print(f"  {tier}: {', '.join(tiers[tier])}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), f"v385_389_results.json")
    with open(out_path, 'w') as f:
        json.dump({
            "pipeline": pipe, "fast_mode": fast,
            "n_etfs": len(tickers), "n_valid": len(valid),
            "avg_return_pct": avg_ret,
            "tiers": {t: tiers[t] for t in ["S","A","B","C","D","F"] if tiers[t]},
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
