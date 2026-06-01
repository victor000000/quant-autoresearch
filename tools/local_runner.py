#!/usr/bin/env python3
"""Local Pipeline Runner — Fast iteration without QC cloud.

Simulates the full pipeline locally using daily data:
  bars → labels → features → model → evaluation → synthetic Calmar

This enables rapid experimentation: test 100s of configs locally,
then submit only the BEST to QC for real backtest validation.

Usage:
  python3 tools/local_runner.py GLD              # Test all configs for GLD
  python3 tools/local_runner.py GLD --pipeline v386  # Specific pipeline
  python3 tools/local_runner.py --all            # Test all ETFs
  python3 tools/local_runner.py --sweep          # Full sweep mode
"""
import json, math, os, sys, time
import numpy as np
import pandas as pd
from io import BytesIO

# ═══════════════ CORE ALGORITHMS ═══════════════

class DollarBarBuilder:
    def __init__(self, threshold):
        self.thresh = threshold; self.cum = 0.0; self.close_lc = None
    def update(self, close, vol):
        if close <= 0 or vol <= 0: return None
        lc = math.log(close); self.cum += close * vol; self.close_lc = lc
        if self.cum >= self.thresh: self.cum = 0.0; return {"log_close": self.close_lc}
        return None

class TickBarBuilder:
    def __init__(self, threshold):
        self.thresh = threshold; self.cum = 0; self.close_lc = None
    def update(self, close):
        if close <= 0: return None
        lc = math.log(close); self.cum += 1; self.close_lc = lc
        if self.cum >= self.thresh: self.cum = 0; return {"log_close": self.close_lc}
        return None

class RangeBarBuilder:
    def __init__(self, threshold=0.005):
        self.thresh = threshold; self.ref = None
    def update(self, close):
        if close <= 0: return None
        lc = math.log(close)
        if self.ref is None: self.ref = lc; return None
        if abs(lc - self.ref) >= self.thresh: self.ref = lc; return {"log_close": lc}
        return None


def sample_entropy_range(x, m=2, r_factor=0.2, max_comp=40):
    """Wang's method: use RANGE not STD for tolerance."""
    N = len(x); r = r_factor * (np.max(x) - np.min(x)) + 1e-12
    if N < m + 3 or r == 0: return 0.0
    def cm(tlen):
        cnt, tot = 0, 0
        step = max(1, (N - tlen) // 200)
        for i in range(0, N - tlen - 1, step):
            max_j = min(i + max_comp + 1, N - tlen)
            for j in range(i + 1, max_j):
                d = 0.0
                for k in range(tlen): d = max(d, abs(x[i+k] - x[j+k]));
                if d < r: cnt += 1
                tot += 1
        return cnt, tot
    A, tA = cm(m+1); B, tB = cm(m)
    if B == 0 or A == 0: return 0.0
    return -math.log((A/tA)/(B/tB)) if tA>0 and tB>0 else 0.0


def build_features(lc, lr):
    """Build ~100 features: integer diff + rolling stats + entropy + microstructure."""
    N = len(lc); feats = []
    # Integer diff lags
    for k in range(1, 21):
        r = np.full(N, np.nan); r[k:] = lc[k:] - lc[:-k]; feats.append(r.astype(np.float32))
    # Z-scored lags
    W_Z = 100
    for k in range(1, 21):
        r = np.full(N, np.nan); r[k:] = lc[k:] - lc[:-k]
        rs = pd.Series(r); m = rs.rolling(W_Z, min_periods=W_Z).mean()
        s = rs.rolling(W_Z, min_periods=W_Z).std()
        feats.append(((rs - m) / (s + 1e-12)).astype(np.float32).to_numpy())
    # Rolling stats
    slr = pd.Series(lr)
    for W in [5, 10, 20, 50, 100, 200, 400, 800]:
        feats.append(slr.rolling(W, min_periods=W).std().astype(np.float32).to_numpy())
        feats.append(slr.rolling(W, min_periods=W).mean().astype(np.float32).to_numpy())
    # Kurtosis + vol ratios + price vs MA
    for W in [50, 100, 200, 400]:
        feats.append(slr.rolling(W, min_periods=W).kurt().astype(np.float32).to_numpy())
    for W in [5, 10, 20, 50]:
        feats.append((slr.rolling(W, min_periods=W).std() / (slr.rolling(200, min_periods=200).std()+1e-9)).astype(np.float32).to_numpy())
    for W in [25, 50, 100, 200]:
        ma = pd.Series(lc).rolling(W, min_periods=W).mean().to_numpy()
        feats.append((lc > ma).astype(np.float32))
    # Sample Entropy (range-based, Wang fix)
    lr_arr = np.diff(lc, prepend=lc[0])
    for W in [30, 50, 100, 200]:
        for r_f in [0.1, 0.2]:
            se = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W//5)
            for i in range(W, N, stride): se[i] = sample_entropy_range(lc[i-W:i], m=2, r_factor=r_f)
            feats.append(pd.Series(se).fillna(method='ffill').fillna(0.0).astype(np.float32).to_numpy())

    return np.column_stack(feats).astype(np.float32)


def cstats(rets):
    if len(rets) < 2: return 0, 0, 0
    cum = np.cumsum(rets); peak = np.maximum.accumulate(cum); dd = cum - peak
    mdd = abs(float(np.min(dd))) + 1e-9
    ann = float(np.mean(rets)) * 252  # daily
    return ann / mdd, ann, mdd


def evaluate_config(lc, lr, bar_indices, features, fv, train_end_idx, val_end_idx,
                    label_type, fk, nc, ma_period, suffix=""):
    """Evaluate one pipeline configuration. Returns (calmar, config_name)."""
    N = len(lc)
    tr_m = bar_indices < train_end_idx
    va_m = (bar_indices >= train_end_idx) & (bar_indices < val_end_idx)

    # Build labels
    fwd_ret = np.full(N, np.nan)
    for t in range(N - fk):
        fwd_ret[t] = lc[t + fk] - lc[t]
    fvd = ~np.isnan(fwd_ret)

    if label_type == "km2":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler as FS
        fwd_vol = np.full(N, np.nan)
        for t in range(N - fk):
            wr = lr[t+1:t+fk+1]
            if len(wr) >= 2: fwd_vol[t] = float(np.std(wr))
        fwd_abs = np.abs(fwd_ret)
        # Stage 1: vol clustering
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(fwd_vol[tr_m & fvd].reshape(-1,1))
        cv_vol = km_vol.predict(fwd_vol[fvd].reshape(-1,1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low_vol = np.full(N, False); is_low_vol[fvd] = (cv_vol == lo_vol)
        # Stage 2: direction clustering
        stage2 = tr_m & fvd & is_low_vol
        vf_tr = np.column_stack([fwd_ret[stage2], fwd_abs[stage2]])
        if len(vf_tr) < 60: return -999, ""
        fs = FS().fit(vf_tr)
        vf_all = np.column_stack([fwd_ret[fvd & is_low_vol], fwd_abs[fvd & is_low_vol]])
        km_dir = KMeans(n_clusters=nc, random_state=42, n_init=5).fit(fs.transform(vf_tr))
        cv_dir = km_dir.predict(fs.transform(vf_all))
        up_c = int(np.argmax(km_dir.cluster_centers_[:,0]))
        y = np.full(N, -1, dtype=int)
        y_dir = np.full(cv_dir.shape[0], -1, dtype=int)
        y_dir[cv_dir == up_c] = 1; y_dir[cv_dir != up_c] = 0
        y[fvd & is_low_vol] = y_dir
        cfg = f"km2_f{fk}_c{nc}_ma{ma_period}{suffix}"

    elif label_type == "tertile":
        fr_train = fwd_ret[tr_m & fvd]
        if len(fr_train) < 100: return -999, ""
        t1 = float(np.percentile(fr_train, 33.3))
        t2 = float(np.percentile(fr_train, 66.7))
        y = np.full(N, -1, dtype=int)
        y[fvd & (fwd_ret >= t2)] = 1
        y[fvd & (fwd_ret <= t1)] = 0
        cfg = f"tertile_f{fk}_ma{ma_period}{suffix}"

    elif label_type == "carry":
        fwd_vol = np.full(N, np.nan)
        for t in range(N - fk):
            wr = lr[t+1:t+fk+1]
            if len(wr) >= 2: fwd_vol[t] = float(np.std(wr))
        fvd_v = ~np.isnan(fwd_vol)
        med_v = float(np.median(fwd_vol[tr_m & fvd_v]))
        y = np.full(N, -1, dtype=int)
        y[fvd_v & (fwd_vol <= med_v)] = 1
        y[fvd_v & (fwd_vol > med_v)] = 0
        cfg = f"carry_f{fk}_ma{ma_period}{suffix}"

    else:
        return -999, ""

    ly = y >= 0
    tx = fv & ly & tr_m; vx = fv & ly & va_m
    if tx.sum() < 200 or vx.sum() < 30: return -999, ""

    # Train XGBoost
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    sc = StandardScaler(); Xt = sc.fit_transform(features[tx]); Xv = sc.transform(features[vx])
    m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
        reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=42, n_jobs=1, early_stopping_rounds=30, base_score=0.5)
    m.fit(Xt, y[tx], eval_set=[(Xv, y[vx])], verbose=False)
    pv = m.predict_proba(Xv)[:, 1]

    # Evaluate with MA filter
    vi = np.where(vx)[0]
    ma = pd.Series(lc).rolling(ma_period, min_periods=ma_period).mean().to_numpy()
    pos_v = ((pv[:-1] > 0.5) & (lc[vi][:-1] > ma[vi][:-1])).astype(float)
    calmar, ann_ret, mdd = cstats(pos_v * lr[vi][1:])

    return calmar, cfg


def run_local_sweep(ticker, bar_type="dollar"):
    """Run a local sweep of label configurations for one ETF."""
    print(f"\n{'='*60}")
    print(f"Local Sweep: {ticker} ({bar_type} bars)")
    print(f"{'='*60}")

    # Download data
    try:
        import yfinance as yf
        data = yf.download(ticker, start="2009-08-01", end="2026-05-30", progress=False)
        if len(data) < 500:
            print(f"  Insufficient data: {len(data)} rows")
            return None
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

    close = data['Close'].values.astype(float)
    vol = data['Volume'].values.astype(float) if 'Volume' in data.columns else np.ones_like(close)
    ts_idx = np.arange(len(close))

    # Build bars
    if bar_type == "dollar":
        td = float(np.sum(close * vol))
        bld = DollarBarBuilder(td / 15000)
        bars = []
        for i in range(len(close)):
            b = bld.update(close[i], vol[i])
            if b is not None: bars.append(b)
    elif bar_type == "tick":
        bld = TickBarBuilder(max(1, len(close) // 15000))
        bars = []
        for i in range(len(close)):
            b = bld.update(close[i])
            if b is not None: bars.append(b)
    elif bar_type == "range":
        bld = RangeBarBuilder(0.005)
        bars = []
        for i in range(len(close)):
            b = bld.update(close[i])
            if b is not None: bars.append(b)
    else:
        print(f"  Unknown bar type: {bar_type}")
        return None

    if len(bars) < 500:
        print(f"  Too few bars: {len(bars)}")
        return None

    N = len(bars)
    lc = np.array([x["log_close"] for x in bars])
    lr = np.zeros(N); lr[1:] = lc[1:] - lc[:-1]
    bar_idx = np.arange(N)

    # Split: train=70%, val=15%, test=15%
    train_end = int(N * 0.70)
    val_end = int(N * 0.85)

    print(f"  Bars: {N} | Train: {train_end} | Val: {val_end-train_end} | Test: {N-val_end}")

    # Build features
    t0 = time.time()
    features = build_features(lc, lr)
    fv = ~np.isnan(features).any(axis=1)
    print(f"  Features: {features.shape[1]} cols | Built in {time.time()-t0:.1f}s")

    # Sweep configurations
    configs = []
    horizons = [50, 100, 200]
    ma_periods = [50, 100, 200]
    n_clusters = [2, 3]
    label_types = ["km2", "tertile", "carry"]
    suffixes = ["", "_inv"]

    total = len(horizons) * len(label_types) * len(ma_periods) * len(n_clusters) * len(suffixes)
    print(f"  Sweeping {total} configurations...")

    t0 = time.time()
    count = 0
    for fk in horizons:
        for lt in label_types:
            ncs = n_clusters if lt == "km2" else [0]
            for nc in ncs:
                for ma in ma_periods:
                    for suffix in suffixes:
                        calmar, cfg = evaluate_config(lc, lr, bar_idx, features, fv, train_end, val_end, lt, fk, nc, ma, suffix)
                        if calmar > -900:
                            configs.append((calmar, cfg))
                        count += 1
                        if count % 100 == 0:
                            elapsed = time.time() - t0
                            rate = count / elapsed if elapsed > 0 else 0
                            eta = (total - count) / rate if rate > 0 else 0
                            print(f"    [{count}/{total}] {elapsed:.0f}s | Best so far: Calmar={max(c[0] for c in configs):.3f} | ETA: {eta:.0f}s")

    # Results
    configs.sort(key=lambda x: x[0], reverse=True)
    print(f"\n  ✓ Completed in {time.time()-t0:.1f}s")
    print(f"\n  Top 10 Configurations:")
    print(f"  {'Rank':<5} {'Calmar':>8} {'Configuration'}")
    print(f"  {'-'*50}")
    for i, (calmar, cfg) in enumerate(configs[:10]):
        print(f"  {i+1:<5} {calmar:>8.3f} {cfg}")

    return {
        "ticker": ticker, "bar_type": bar_type,
        "n_bars": N, "n_features": features.shape[1],
        "n_configs": total, "n_valid": len(configs),
        "top_configs": [{"calmar": float(c), "config": cfg} for c, cfg in configs[:10]],
    }


def main():
    tickers = ["GLD", "QQQ", "SPY", "IWM"]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] != "--all":
        tickers = [a.upper() for a in args if a.upper() != "--SWEEP"]

    all_results = []
    for ticker in tickers:
        for bar_type in ["dollar"]:  # Start with dollar, fastest to test
            result = run_local_sweep(ticker, bar_type)
            if result: all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        top = r["top_configs"][0] if r["top_configs"] else {"calmar": 0, "config": "none"}
        print(f"{r['ticker']:<6} {r['bar_type']:<8} {r['n_bars']:>6} bars | "
              f"Best: Calmar={top['calmar']:.3f} ({top['config']})")

    # Save
    with open("experiment_summary/local_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to experiment_summary/local_sweep_results.json")


if __name__ == "__main__":
    main()
