"""Feature engineering. Edit freely — any features, any dimensionality.

Currently: 72 base features (momentum, z-score, rolling stats, kurtosis, vol ratio,
price vs MA) + 8 entropy features = 80 total.
"""
import math
import numpy as np
import pandas as pd


def sample_entropy(x, m=2, r_factor=0.2, max_comp=40):
    """Approximate sample entropy with bounded computation."""
    N = len(x)
    r = r_factor * np.std(x) + 1e-12
    if N < m + 3 or r == 0:
        return 0.0

    def count_matches(tlen):
        cnt, tot = 0, 0
        step = max(1, (N - tlen) // 200)
        for i in range(0, N - tlen - 1, step):
            max_j = min(i + max_comp + 1, N - tlen)
            for j in range(i + 1, max_j):
                if max(abs(x[i + k] - x[j + k]) for k in range(tlen)) < r:
                    cnt += 1
                tot += 1
        return cnt, tot

    A, tA = count_matches(m + 1)
    B, tB = count_matches(m)
    if B == 0 or A == 0:
        return 0.0
    return -math.log((A / tA) / (B / tB)) if tA > 0 and tB > 0 else 0.0


def build_feats(lc, lr, spy_lc=None, spy_lr=None):
    """Build feature matrix from log-close and log-return arrays.

    Args:
        lc, lr: ETF log-close and log-return arrays
        spy_lc, spy_lr: optional SPY arrays (reserved for future cross-asset features)

    Returns: np.array of shape (N, F) with float32 dtype. F = 80.
    NOTE: Cross-asset features temporarily disabled — they crowded out ETF-specific
    features in correlation filter, reducing trades from 521→1 (QQQ) and 77→1 (EEM).
    """
    N = len(lc)
    feats = []

    # Raw momentum: k-bar log-returns (k = 1..20)
    for k in range(1, 21):
        r = np.full(N, np.nan)
        r[k:] = lc[k:] - lc[:-k]
        feats.append(r.astype(np.float32))

    # Z-scored momentum (100-bar rolling window)
    W_Z = 100
    for k in range(1, 21):
        r = np.full(N, np.nan)
        r[k:] = lc[k:] - lc[:-k]
        rs = pd.Series(r)
        m = rs.rolling(W_Z, min_periods=W_Z).mean()
        s = rs.rolling(W_Z, min_periods=W_Z).std()
        feats.append(((rs - m) / (s + 1e-12)).astype(np.float32).to_numpy())

    # Rolling std and mean of log-returns
    slr = pd.Series(lr)
    for W in [5, 10, 20, 50, 100, 200, 400, 800]:
        feats.append(slr.rolling(W, min_periods=W).std().astype(np.float32).to_numpy())
        feats.append(slr.rolling(W, min_periods=W).mean().astype(np.float32).to_numpy())

    # Rolling kurtosis (returns and absolute returns)
    sa = pd.Series(np.abs(lr))
    for W in [50, 100, 200, 400]:
        feats.append(slr.rolling(W, min_periods=W).kurt().astype(np.float32).to_numpy())
        feats.append(sa.rolling(W, min_periods=W).kurt().astype(np.float32).to_numpy())

    # Volatility ratio: short-term / long-term std
    for W in [5, 10, 20, 50]:
        ratio = slr.rolling(W, min_periods=W).std() / (slr.rolling(200, min_periods=200).std() + 1e-9)
        feats.append(ratio.astype(np.float32).to_numpy())

    # Price vs moving average (binary)
    for W in [25, 50, 100, 200]:
        ma = pd.Series(lc).rolling(W, min_periods=W).mean().to_numpy()
        feats.append((lc > ma).astype(np.float32))

    # Sample entropy features
    lr_arr = np.diff(lc, prepend=lc[0])
    lr_abs = np.abs(lr_arr)
    for W in [50, 100, 200]:
        for r_f in [0.1, 0.2]:
            se = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W // 5)
            for i in range(W, N, stride):
                se[i] = sample_entropy(lc[i - W:i], m=2, r_factor=r_f)
            feats.append(pd.Series(se).ffill().fillna(0.0).astype(np.float32).to_numpy())
    for W in [50, 100]:
        se = np.full(N, np.nan, dtype=np.float32)
        stride = max(1, W // 5)
        for i in range(W, N, stride):
            se[i] = sample_entropy(lr_abs[i - W:i], m=2, r_factor=0.2)
        feats.append(pd.Series(se).ffill().fillna(0.0).astype(np.float32).to_numpy())

    # Cross-asset (SPY-relative) features — CAUSAL, economically motivated for
    # flight-to-quality assets (e.g. TLT rises when equities sell off). Kept as
    # low-variance RELATIVE measures (differences / bounded correlation) so they
    # don't dominate the correlation dim-reduce the way raw SPY levels did.
    # Only added when SPY is provided and valid.
    if (spy_lc is not None and spy_lr is not None
            and len(spy_lc) == N and float(np.nanstd(spy_lc)) > 1e-9):
        slc = np.asarray(spy_lc, dtype=float)
        slr = np.asarray(spy_lr, dtype=float)
        for k in [5, 20, 60]:                       # relative momentum: own − SPY k-bar return
            r = np.full(N, np.nan)
            r[k:] = (lc[k:] - lc[:-k]) - (slc[k:] - slc[:-k])
            feats.append(r.astype(np.float32))
        corr = pd.Series(lr).rolling(60, min_periods=60).corr(pd.Series(slr))  # risk-on/off regime
        feats.append(corr.astype(np.float32).to_numpy())

    return np.column_stack(feats).astype(np.float32)
