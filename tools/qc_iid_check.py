#!/usr/bin/env python3
"""
IID diagnostic for Wang ZN-axis bars.

For each ETF in /bars_zn_v1/<TICKER>/*.parquet, computes:
  - excess kurtosis, skewness (IID-Gaussian ⇒ both ≈ 0)
  - ACF at lags 1, 5, 10 on bar log-returns
  - ACF at lags 1, 5, 10 on squared returns (ARCH/volatility-clustering check)
  - Ljung-Box Q(10) p-values on returns and squared returns
  - Variance-ratio (Lo-MacKinlay) at k=2, 5, 10
  - Runs test
  - Jarque-Bera normality p-value

Also reports the same on "axis returns" = diff(cum_z), which is Wang's
gaussianized series.

Outputs a wide table to stdout and a parquet to /home/txy/lb/data_cache/iid_zn.parquet.
"""
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from scipy.stats import kurtosis, skew, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.sandbox.stats.runs import runstest_1samp

sys.path.insert(0, "/home/txy/lb/tools")
from qc_objstore import load_creds, list_path, auth_headers  # noqa: E402

API_BASE = "https://www.quantconnect.com/api/v2"
ETFS_EXISTING = ["QQQ", "SPY", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "XLE"]
ETFS_NEW = ["VNQ", "HYG", "TIP", "DBC", "XLU", "XLP", "SHY", "UUP", "EWJ", "VXX"]
ALL_ETFS = ETFS_EXISTING + ETFS_NEW


def fetch_bytes(uid, tok, org, key):
    r = requests.post(
        f"{API_BASE}/object/get",
        headers=auth_headers(uid, tok),
        json={"organizationId": org, "keys": [key]},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(f"get {key} failed: {j.get('errors')}")
    url = j["objects"][0]["url"]
    rr = requests.get(url, timeout=60)
    rr.raise_for_status()
    return rr.content


def load_etf_bars(uid, tok, org, ticker):
    items = list_path(uid, tok, org, f"/bars_zn_v1/{ticker}")
    if not items:
        return None
    dfs = []
    for it in sorted(items, key=lambda x: x.get("key", "")):
        key = it.get("key", "")
        if not key.endswith(".parquet"):
            continue
        try:
            data = fetch_bytes(uid, tok, org, key)
            df = pd.read_parquet(io.BytesIO(data))
            dfs.append(df)
        except Exception as e:
            print(f"  [{ticker}] fetch {key} failed: {e}", file=sys.stderr)
    if not dfs:
        return None
    bars = pd.concat(dfs, ignore_index=True)
    bars["ts_close"] = pd.to_datetime(bars["ts_close"])
    bars = bars.sort_values("ts_close").drop_duplicates("ts_close").reset_index(drop=True)
    return bars


def variance_ratio(r, k):
    """Lo-MacKinlay variance ratio at lag k (heteroskedasticity-robust z-stat).
    VR(k) ≈ 1 for IID series. Returns (VR, z, p-two-sided)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < k + 10:
        return np.nan, np.nan, np.nan
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    # k-period variance
    rk = np.array([r[i:i+k].sum() for i in range(n - k + 1)])
    muk = rk.mean()
    vark = ((rk - muk) ** 2).sum() / (len(rk) - 1) / k  # per-period
    vr = vark / var1 if var1 > 0 else np.nan
    # Heteroskedasticity-robust standard error (Lo-MacKinlay 1988 eq 14)
    delta = 0.0
    for j in range(1, k):
        w = 2.0 * (k - j) / k
        num = 0.0
        den = ((r - mu) ** 2).sum() ** 2
        for t in range(j, n):
            num += ((r[t] - mu) ** 2) * ((r[t - j] - mu) ** 2)
        delta_j = (n * num) / den if den > 0 else 0.0
        delta += (w ** 2) * delta_j
    se = np.sqrt(delta / n) if delta > 0 else np.nan
    z = (vr - 1.0) / se if se and not np.isnan(se) else np.nan
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    return float(vr), float(z), float(p)


def iid_stats(r, label):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 100:
        return {"label": label, "n": n}

    out = {"label": label, "n": n}
    out["mean"] = r.mean()
    out["std"] = r.std()
    out["skew"] = skew(r)
    out["kurt_excess"] = kurtosis(r, fisher=True)

    # ACF on returns
    a = acf(r, nlags=20, fft=True)
    out["acf_1"] = a[1]
    out["acf_5"] = a[5]
    out["acf_10"] = a[10]

    # ACF on squared returns (volatility clustering)
    a2 = acf(r ** 2, nlags=20, fft=True)
    out["acf2_1"] = a2[1]
    out["acf2_5"] = a2[5]
    out["acf2_10"] = a2[10]

    # Ljung-Box on returns and squared returns
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_r2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True)
    out["lb_p_r"] = float(lb_r["lb_pvalue"].iloc[0])
    out["lb_p_r2"] = float(lb_r2["lb_pvalue"].iloc[0])

    # Variance ratio
    vr2, _, p2 = variance_ratio(r, 2)
    vr5, _, p5 = variance_ratio(r, 5)
    vr10, _, p10 = variance_ratio(r, 10)
    out["vr2"] = vr2
    out["vr5"] = vr5
    out["vr10"] = vr10
    out["vr2_p"] = p2
    out["vr10_p"] = p10

    # Runs test (sign of returns vs median)
    try:
        z_runs, p_runs = runstest_1samp(r, correction=False)
        out["runs_p"] = float(p_runs)
    except Exception:
        out["runs_p"] = np.nan

    # Jarque-Bera normality
    jb_stat, jb_p = jarque_bera(r)
    out["jb_p"] = float(jb_p)

    return out


def main():
    uid, tok, org = load_creds()
    cache_dir = "/home/txy/lb/data_cache"
    os.makedirs(cache_dir, exist_ok=True)

    rows = []
    for t in ALL_ETFS:
        cache_path = f"{cache_dir}/bars_zn_{t}.parquet"
        if os.path.exists(cache_path):
            bars = pd.read_parquet(cache_path)
        else:
            print(f"# fetching {t}…", file=sys.stderr)
            bars = load_etf_bars(uid, tok, org, t)
            if bars is None:
                print(f"  {t}: no bars", file=sys.stderr)
                continue
            bars.to_parquet(cache_path, compression="snappy")
        n_bars = len(bars)
        log_ret = np.diff(np.log(bars["close"].values.astype(float)))
        axis_ret = np.diff(bars["cum_z"].values.astype(float))
        n_min_mean = bars["n_minutes"].astype(float).mean() if "n_minutes" in bars.columns else np.nan

        for series_name, r in [("close_ret", log_ret), ("axis_ret", axis_ret)]:
            s = iid_stats(r, f"{t}/{series_name}")
            s["ticker"] = t
            s["series"] = series_name
            s["n_bars_total"] = n_bars
            s["n_min_mean"] = n_min_mean
            rows.append(s)

    df = pd.DataFrame(rows)
    out = f"{cache_dir}/iid_zn.parquet"
    df.to_parquet(out, index=False)

    cols_close = ["ticker", "n_bars_total", "n_min_mean", "kurt_excess", "skew",
                  "acf_1", "acf_5", "acf_10", "acf2_1", "acf2_5", "acf2_10",
                  "lb_p_r", "lb_p_r2", "vr2", "vr5", "vr10", "vr2_p", "vr10_p",
                  "runs_p", "jb_p"]

    print("\n=== ZN-bar close-price log-returns: IID diagnostics ===")
    print(df[df.series == "close_ret"][cols_close].to_string(index=False, float_format=lambda x: f"{x:>+.4f}" if abs(x) > 1e-4 else f"{x:>+.2e}"))
    print("\n=== Axis-cum_z returns (Wang gaussianized) ===")
    print(df[df.series == "axis_ret"][cols_close].to_string(index=False, float_format=lambda x: f"{x:>+.4f}" if abs(x) > 1e-4 else f"{x:>+.2e}"))
    print(f"\n# saved {out}")


if __name__ == "__main__":
    main()
