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


def build_feats(lc, lr, spy_lc=None, spy_lr=None, abs_start=0):
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
    # Entropy is sampled on a stride grid anchored at the FULL-series index (abs_start
    # = absolute index of lc[0]). With abs_start=0 (train, full series) this is exactly
    # range(W, N, stride). Online infer passes abs_start so a TRAILING window reproduces
    # the SAME grid points -> byte-identical entropy as the full-series build.
    for W in [50, 100, 200]:
        for r_f in [0.1, 0.2]:
            se = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W // 5)
            for i in range(W, N):
                if (abs_start + i - W) % stride == 0:
                    se[i] = sample_entropy(lc[i - W:i], m=2, r_factor=r_f)
            feats.append(pd.Series(se).ffill().fillna(0.0).astype(np.float32).to_numpy())
    for W in [50, 100]:
        se = np.full(N, np.nan, dtype=np.float32)
        stride = max(1, W // 5)
        for i in range(W, N):
            if (abs_start + i - W) % stride == 0:
                se[i] = sample_entropy(lr_abs[i - W:i], m=2, r_factor=0.2)
        feats.append(pd.Series(se).ffill().fillna(0.0).astype(np.float32).to_numpy())

    # Mean-reversion oscillators (③, causal) — for assets whose edge is in reversals
    # (e.g. TLT around rate regimes): RSI and Bollinger %b over a few windows. All use
    # only past/current bars (rolling, no forward window).
    slr2 = pd.Series(lr)
    up = slr2.clip(lower=0.0)
    dn = (-slr2).clip(lower=0.0)
    for W in [14, 30, 60]:
        rs = up.rolling(W, min_periods=W).mean() / (dn.rolling(W, min_periods=W).mean() + 1e-12)
        rsi = 1.0 - 1.0 / (1.0 + rs)                  # in [0,1]; <0.5 oversold, >0.5 overbought
        feats.append((rsi - 0.5).astype(np.float32).to_numpy())
    slc2 = pd.Series(lc)
    for W in [20, 50]:
        ma = slc2.rolling(W, min_periods=W).mean()
        sd = slc2.rolling(W, min_periods=W).std()
        pctb = (slc2 - ma) / (2.0 * sd + 1e-12)       # Bollinger %b (centered): >0 above band mid
        feats.append(pctb.astype(np.float32).to_numpy())

    # NOTE: SPY-relative cross-asset features were tested (round 34) and REGRESSED
    # TLT (0.7679 -> 0.677); a real cross-asset edge needs a 2-symbol PAIRS strategy,
    # not SPY features in a single-asset model. Reverted; left disabled. See f_crossasset.

    # Fractional-difference (FFD) features (③, Wang-canon, memory-preserving): for a few
    # orders d, a CAUSAL fractional difference of log-close (binomial weights, truncated at
    # |w|<1e-4, width<=200). Appended as BOUNDED-VARIANCE transforms — first-difference + a
    # trailing z-score — so the near-non-stationary FFD level cannot crowd the correlation-
    # select (R34 crowding). All trailing/causal (no .shift(-N)) -> byte-identical online
    # (the causal conv + rolling-100 fit inside the infer trailing window). F: 80 -> 86.
    def _ffd_w(d, thresh=1e-4, max_w=200):
        w = [1.0]
        k = 1
        while k < max_w:
            wk = -w[-1] * (d - k + 1) / k
            if abs(wk) < thresh:
                break
            w.append(wk)
            k += 1
        return np.array(w, dtype=float)
    for d in [0.3, 0.5, 0.7]:
        w = _ffd_w(d)
        L = len(w)
        fd = np.convolve(lc, w)[:N]            # causal FFD: fd[i] = sum_k w[k]*lc[i-k]
        fd[:L - 1] = np.nan                     # warm-up (incomplete window)
        sfd = pd.Series(fd)
        feats.append(sfd.diff().astype(np.float32).to_numpy())                          # FFD first-difference
        m = sfd.rolling(100, min_periods=100).mean()
        s = sfd.rolling(100, min_periods=100).std()
        feats.append(((sfd - m) / (s + 1e-12)).astype(np.float32).to_numpy())           # FFD trailing z-score

    return np.column_stack(feats).astype(np.float32)
