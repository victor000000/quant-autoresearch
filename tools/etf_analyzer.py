#!/usr/bin/env python3
"""Systematic ETF Statistical Analysis → Strategy Classification → Pipeline Mapping.

Wang's principle: Understand the DATA first, then design the pipeline.
Different ETFs have different statistical properties → different optimal strategies.

Computes for each ETF:
  1. Hurst exponent (trend-following vs mean-reverting)
  2. Variance ratio test (random walk vs predictable)
  3. ADF stationarity test
  4. Volatility clustering (ARCH effects)
  5. Return distribution characteristics
  6. Autocorrelation profile
  7. Regime-switching tendency

Then classifies into strategy type and maps to optimal pipeline.
"""
import json, math, os, sys
import numpy as np
import pandas as pd
from io import BytesIO
from collections import defaultdict

# 19 ETFs in our core universe (from the original optimization)
CORE_ETFS = [
    "QQQ", "SPY", "IWM", "DIA",  # Broad equity
    "GLD", "GDX", "GDXJ", "SLV", "IAU",  # Precious metals
    "SMH", "SOXX", "XLK",  # Tech/Semis
    "XLE", "XME", "XOP", "USO",  # Energy/Materials
    "XLF", "KRE",  # Financials
    "XLI", "XLY", "XLP", "XLU", "XLV", "XBI", "XHB", "XRT", "ITB",  # Sectors
    "TLT", "SHY", "IEF", "HYG", "LQD", "EMB",  # Fixed Income
    "EEM", "EWJ", "EWH", "VNQ", "REM", "DBC",  # International/Other
    "XLK", "VGT", "SOXX", "SMH",  # Tech variants
    "TAN", "IBB", "PFF", "AGG", "MTUM", "DBA",  # Thematic
    "UUP", "FXE", "FXY",  # Currency
    "EZA", "EWZ", "FXI", "INDA", "RSX", "EWW",  # EM
]

# Extended universe from run_all_etfs_v2.py
ALL_ETFS = [
    "QQQ", "XLK", "SMH", "SOXX", "VGT", "SPY", "EWT", "DIA",
    "GLD", "GDXJ", "SIL", "SLV", "XME", "XLE", "XLB", "USO", "DBC",
    "GDX", "KRE", "XTN", "XLI", "XLY", "XRT", "XHB", "ITB", "XLF", "KBE", "XOP", "FCG",
    "TLT", "SHY", "IEF",
    "HYG", "LQD",
    "XLP", "XLU", "XLV", "XBI", "IBB", "TAN",
    "IWM", "EEM", "EWH", "VNQ", "REM",
    "EZA", "EWZ", "FXI", "INDA", "RSX", "EWW",
    "UUP", "FXE", "FXY", "EMB", "MUB",
    "PFF", "AGG", "MTUM", "DBA", "IAU",
]


def compute_statistical_properties(prices, returns):
    """Compute comprehensive statistical properties for an ETF.

    Returns dict with all metrics needed for strategy classification.
    """
    N = len(prices)
    if N < 200:
        return None

    props = {}

    # === 1. Hurst Exponent (R/S analysis) ===
    def hurst_rs(x, min_window=50, max_lag=20):
        n = len(x); lags = np.arange(min_window, min(n//4, 200), max(1, n//50))
        if len(lags) < 4: return 0.5
        rs_vals = []
        for lag in lags:
            n_parts = n // lag
            if n_parts < 4: continue
            rs_sum = 0.0
            for p in range(n_parts):
                part = x[p*lag:(p+1)*lag]
                z = part - part.mean()
                R = z.max() - z.min()
                S = np.std(part) + 1e-12
                rs_sum += R / S
            rs_vals.append(rs_sum / n_parts)
        if len(rs_vals) < 4: return 0.5
        log_rs = np.log(rs_vals); log_lag = np.log(lags[:len(rs_vals)])
        slope = np.polyfit(log_lag, log_rs, 1)[0]
        return float(np.clip(slope, 0.1, 1.0))

    props["hurst"] = hurst_rs(prices)
    props["hurst_returns"] = hurst_rs(returns)
    props["hurst_abs_returns"] = hurst_rs(np.abs(returns))

    # === 2. Variance Ratio Test ===
    # VR(k) = Var(r_k) / (k * Var(r_1)). Under random walk, VR ≈ 1
    def variance_ratio(x, k=10):
        n = len(x)
        r1 = x[1:] - x[:-1]
        rk = x[k:] - x[:-k]
        var1 = np.var(r1) + 1e-12
        vark = np.var(rk[:len(r1)-k+1]) + 1e-12
        return float(vark / (k * var1))

    for k in [5, 10, 20, 50]:
        props[f"vr_{k}"] = variance_ratio(prices, k=k)
    # Average deviation from 1 (random walk)
    vr_deviations = [abs(props[f"vr_{k}"] - 1.0) for k in [5, 10, 20, 50]]
    props["vr_predictability"] = float(np.mean(vr_deviations))

    # === 3. Autocorrelation Profile ===
    for lag in [1, 5, 10, 20, 50]:
        if len(returns) > lag + 10:
            ac = np.corrcoef(returns[lag:], returns[:-lag])[0, 1]
            props[f"ac_{lag}"] = float(ac) if not np.isnan(ac) else 0.0
        else:
            props[f"ac_{lag}"] = 0.0
    # First-order autocorrelation (key for mean-reversion detection)
    props["ac1"] = props.get("ac_1", 0.0)

    # === 4. Volatility Clustering (ARCH effect) ===
    # Ljung-Box test on squared returns
    sq_ret = returns ** 2
    lb_stats = []
    for lag in [5, 10, 20]:
        if len(sq_ret) > lag + 10:
            acf = [np.corrcoef(sq_ret[i:], sq_ret[:-i])[0, 1] if i < len(sq_ret) else 0
                   for i in range(1, lag+1)]
            lb = len(returns) * (len(returns)+2) * sum(ac**2 / (len(returns)-k)
                   for k, ac in enumerate(acf, 1))
            lb_stats.append(float(lb))
    props["arch_lb_mean"] = float(np.mean(lb_stats)) if lb_stats else 0.0
    props["arch_strength"] = min(1.0, props["arch_lb_mean"] / 100.0)  # Normalize

    # === 5. Return Distribution ===
    props["ret_mean"] = float(np.mean(returns))
    props["ret_std"] = float(np.std(returns))
    props["ret_skew"] = float(pd.Series(returns).skew())
    props["ret_kurt"] = float(pd.Series(returns).kurtosis())

    # Sharpe-like ratio (annualized)
    ann_factor = 252 * 6.5  # ETF trading hours
    props["sharpe"] = float(props["ret_mean"] / (props["ret_std"] + 1e-12) * np.sqrt(ann_factor))

    # === 6. Trend Strength ===
    # Ratio of directional movement to total movement
    cum_ret = np.cumsum(returns)
    total_move = np.sum(np.abs(returns))
    net_move = abs(cum_ret[-1] - cum_ret[0]) if len(cum_ret) > 0 else 0
    props["trend_efficiency"] = float(net_move / (total_move + 1e-12))

    # Rolling trend consistency: % of time above 200-period MA
    ma200 = pd.Series(prices).rolling(200, min_periods=200).mean().to_numpy()
    props["above_ma200_pct"] = float(np.mean(prices > ma200))

    # === 7. Volatility Regime Detection ===
    # How much does volatility vary over time? (higher = more regime-switching)
    rolling_vol = pd.Series(returns).rolling(50, min_periods=50).std().to_numpy()
    vol_of_vol = np.std(rolling_vol[~np.isnan(rolling_vol)]) if len(rolling_vol) > 0 else 0
    props["vol_of_vol"] = float(vol_of_vol)
    props["vol_regime_ratio"] = float(vol_of_vol / (props["ret_std"] + 1e-12))

    # === 8. Mean-Reversion Score ===
    # Combine: negative AC1 + Hurst < 0.5 + high VR predictability → mean-reverting
    mr_score = 0.0
    mr_score += max(0, -props["ac1"]) * 2.0  # Negative autocorrelation
    mr_score += max(0, 0.5 - props["hurst"]) * 3.0  # Low Hurst
    mr_score += props["vr_predictability"] * 1.0  # Deviation from random walk
    mr_score += max(0, props["ret_skew"]) * 0.5 if props["ret_skew"] > 0 else 0
    props["mr_score"] = float(mr_score)

    # === 9. Trend-Following Score ===
    tf_score = 0.0
    tf_score += props["trend_efficiency"] * 2.0
    tf_score += max(0, props["hurst"] - 0.55) * 3.0
    tf_score += props["above_ma200_pct"] * 1.0
    tf_score += max(0, -props["ret_skew"]) * 0.5 if props["ret_skew"] < 0 else 0
    props["tf_score"] = float(tf_score)

    # === 10. Volatility Strategy Score ===
    vol_score = 0.0
    vol_score += props["arch_strength"] * 2.0
    vol_score += props["vol_regime_ratio"] * 2.0
    vol_score += abs(props["ret_skew"]) * 0.5
    vol_score += min(1.0, props["ret_kurt"] / 10.0) if props["ret_kurt"] > 0 else 0
    props["vol_score"] = float(vol_score)

    return props


def classify_strategy(props):
    """Classify ETF into optimal strategy type based on statistical properties.

    Returns: (primary_strategy, secondary_strategy, confidence, explanation)
    """
    if props is None:
        return "unknown", "unknown", 0.0, "insufficient data"

    tf = props["tf_score"]
    mr = props["mr_score"]
    vol = props["vol_score"]

    strategies = [
        ("trend_following", tf, f"H={props['hurst']:.3f}, trend_eff={props['trend_efficiency']:.3f}"),
        ("mean_reversion", mr, f"AC1={props['ac1']:.3f}, H={props['hurst']:.3f}, VR={props['vr_predictability']:.3f}"),
        ("volatility_regime", vol, f"ARCH={props['arch_strength']:.3f}, VolReg={props['vol_regime_ratio']:.3f}"),
    ]

    strategies.sort(key=lambda x: x[1], reverse=True)
    primary = strategies[0]
    secondary = strategies[1]

    # Confidence: how much better is primary vs secondary?
    if secondary[1] > 0:
        confidence = (primary[1] - secondary[1]) / (primary[1] + secondary[1] + 1e-12)
    else:
        confidence = 1.0

    return primary[0], secondary[0], float(confidence), primary[2]


def map_to_pipeline(strategy_type, props):
    """Map strategy type → optimal pipeline configuration.

    Based on Wang's framework + our experiment results.
    """
    if strategy_type == "trend_following":
        if props["hurst"] > 0.6:
            # Strong trend: dollar bars + multi-order combination
            return {
                "axis": "dollar", "label": "km2_or_multiorder",
                "pipeline": "v395",  # Multi-order combination
                "alt_pipeline": "v389",  # Auto-select
                "rationale": f"Strong trend (H={props['hurst']:.3f}) → multi-order labels for conviction"
            }
        else:
            # Moderate trend: dollar bars + tertile labels
            return {
                "axis": "dollar", "label": "tertile_or_km2",
                "pipeline": "v386",  # Quantile tertile
                "alt_pipeline": "v389",
                "rationale": f"Moderate trend (H={props['hurst']:.3f}) → tertile labels with noise filter"
            }
    elif strategy_type == "mean_reversion":
        if abs(props["ac1"]) > 0.02 and props["hurst"] < 0.52:
            # Strong mean-reversion: range bars + MR labels
            return {
                "axis": "range", "label": "mr_or_cusum",
                "pipeline": "v394",  # Change-point detection
                "alt_pipeline": "v385",  # Range bars
                "rationale": f"Mean-reverting (AC1={props['ac1']:.3f}, H={props['hurst']:.3f}) → change-point labels"
            }
        else:
            return {
                "axis": "dollar", "label": "mr_ridge",
                "pipeline": "v392",  # Calibrated ensemble (Ridge included)
                "alt_pipeline": "v274",  # Original MR
                "rationale": f"Weak MR (AC1={props['ac1']:.3f}) → Ridge + calibrated ensemble"
            }
    elif strategy_type == "volatility_regime":
        return {
            "axis": "vol", "label": "carry_or_km2",
            "pipeline": "v392",  # Calibrated ensemble
            "alt_pipeline": "v388",  # GMM soft on tick bars
            "rationale": f"Vol regime (ARCH={props['arch_strength']:.3f}) → vol bars + carry/ensemble"
        }
    else:
        return {
            "axis": "dollar", "label": "auto",
            "pipeline": "v389",
            "alt_pipeline": "v392",
            "rationale": "Universal auto-select"
        }


def analyze_etf_from_bars(ticker, bar_data_path=None):
    """Analyze an ETF from its dollar bar data.

    If bar_data_path is provided, reads from local parquet files.
    Otherwise, analyzes from QC object store (requires API access).
    """
    # Try to load from local cache first
    bars = None
    search_paths = [
        f"/Users/liyuanjun/ai_work/lb/data_cache/{ticker}_bars.parquet",
        f"/Users/liyuanjun/ai_work/lb/data_cache/{ticker}_daily.csv",
    ]

    for path in search_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.parquet'):
                    bars = pd.read_parquet(path)
                else:
                    bars = pd.read_csv(path, parse_dates=['date'], index_col='date')
                break
            except Exception:
                continue

    if bars is None:
        return None

    # Extract prices
    if 'close' in bars.columns:
        prices = bars['close'].values.astype(float)
    elif 'log_close' in bars.columns:
        prices = np.exp(bars['log_close'].values.astype(float))
    else:
        return None

    prices = prices[~np.isnan(prices)]
    if len(prices) < 200:
        return None

    log_prices = np.log(prices)
    returns = np.diff(log_prices)

    props = compute_statistical_properties(log_prices, returns)
    if props is None:
        return None

    strategy, alt_strategy, confidence, explanation = classify_strategy(props)
    pipeline = map_to_pipeline(strategy, props)

    return {
        "ticker": ticker,
        "n_bars": len(prices),
        "properties": props,
        "primary_strategy": strategy,
        "secondary_strategy": alt_strategy,
        "strategy_confidence": confidence,
        "strategy_explanation": explanation,
        "recommended_pipeline": pipeline["pipeline"],
        "alt_pipeline": pipeline["alt_pipeline"],
        "recommended_axis": pipeline["axis"],
        "recommended_label": pipeline["label"],
        "rationale": pipeline["rationale"],
    }


def analyze_etf_from_yahoo(ticker):
    """Analyze an ETF using Yahoo Finance data (for quick local analysis)."""
    try:
        import yfinance as yf
        data = yf.download(ticker, start="2009-08-01", end="2026-05-30", progress=False)
        if len(data) < 200:
            return None
        prices = data['Close'].values.astype(float)
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        props = compute_statistical_properties(log_prices, returns)
        if props is None: return None
        strategy, alt_strategy, confidence, explanation = classify_strategy(props)
        pipeline = map_to_pipeline(strategy, props)
        return {
            "ticker": ticker, "n_bars": len(prices),
            "properties": props,
            "primary_strategy": strategy,
            "secondary_strategy": alt_strategy,
            "strategy_confidence": confidence,
            "recommended_pipeline": pipeline["pipeline"],
            "recommended_axis": pipeline["axis"],
            "rationale": pipeline["rationale"],
        }
    except ImportError:
        return None
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def generate_strategy_summary(results):
    """Generate a strategy-based ETF classification summary."""
    strategies = defaultdict(list)
    for r in results:
        if r and "error" not in r:
            strategies[r.get("primary_strategy", "unknown")].append(r)

    print(f"\n{'='*70}")
    print(f"ETF STRATEGY CLASSIFICATION")
    print(f"{'='*70}")

    for strat_name in ["trend_following", "mean_reversion", "volatility_regime"]:
        etfs = strategies.get(strat_name, [])
        etfs.sort(key=lambda x: x.get("strategy_confidence", 0), reverse=True)
        print(f"\n## {strat_name.upper()} ({len(etfs)} ETFs)")
        print(f"{'ETF':<8} {'Hurst':>7} {'AC1':>7} {'MR':>6} {'TF':>6} {'VOL':>6} {'Pipeline':<8} {'Rationale'}")
        print(f"{'-'*90}")
        for e in etfs[:15]:
            p = e["properties"]
            print(f"{e['ticker']:<8} {p['hurst']:7.3f} {p['ac1']:7.3f} "
                  f"{p['mr_score']:6.2f} {p['tf_score']:6.2f} {p['vol_score']:6.2f} "
                  f"{e['recommended_pipeline']:<8} {e['rationale'][:40]}")

    return strategies


def main():
    print("Systematic ETF Statistical Analysis — Wang's Framework")
    print("=" * 70)

    results = []
    for ticker in CORE_ETFS[:30]:  # Test first 30
        print(f"\nAnalyzing {ticker}...", end=" ")
        result = analyze_etf_from_yahoo(ticker)
        if result and "error" not in result:
            p = result["properties"]
            print(f"H={p['hurst']:.3f} AC1={p['ac1']:.3f} "
                  f"→ {result['primary_strategy']} (conf={result['strategy_confidence']:.2f}) "
                  f"→ {result['recommended_pipeline']}")
            results.append(result)
        elif result and "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print("FAILED")

    # Generate summary
    strategies = generate_strategy_summary(results)

    # Save results
    out = {
        "n_etfs_analyzed": len(results),
        "by_strategy": {s: [r["ticker"] for r in etfs]
                       for s, etfs in strategies.items()},
        "detailed_results": results,
    }
    with open("experiment_summary/etf_classification.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to experiment_summary/etf_classification.json")

    # Print pipeline recommendations
    print(f"\n{'='*70}")
    print("PIPELINE RECOMMENDATIONS")
    print(f"{'='*70}")
    pipeline_etfs = defaultdict(list)
    for r in results:
        pipeline_etfs[r.get("recommended_pipeline", "unknown")].append(r["ticker"])

    pipeline_descriptions = {
        "v395": "Multi-Order Combination (Wang's signature — best for strong trends)",
        "v394": "Change-Point Detection Labels (Wang recommended — superior to TB)",
        "v393": "FracDiff Primary Features (Wang recommended — outperforms integer diff)",
        "v392": "Calibrated Ensemble (universal — stacking + isotonic calibration)",
        "v389": "Auto-Select (288 combos — per-ETF exhaustive sweep)",
        "v388": "GMM Soft Labels (Bayesian probabilistic — best for commodities)",
        "v387": "Meta-Labeling (2-model confidence gating — reduces false signals)",
        "v386": "Quantile Tertile Labels (noise-filtered — middle tertile skipped)",
        "v385": "Range Bars (price-driven sampling — best for volatile/EM)",
        "v390": "Information-Driven Bars (entropy-based — adapts to uncertainty)",
    }

    for pipeline, etfs in sorted(pipeline_etfs.items()):
        desc = pipeline_descriptions.get(pipeline, "?")
        print(f"\n{pipeline}: {desc}")
        print(f"  ETFs: {', '.join(sorted(etfs))}")

    return results


if __name__ == "__main__":
    main()
