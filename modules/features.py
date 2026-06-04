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


def build_feats(lc, lr, spy_lc=None, spy_lr=None, abs_start=0, rich=False, termstruct=False):
    """Build feature matrix from log-close and log-return arrays.

    Args:
        lc, lr: ETF log-close and log-return arrays
        spy_lc, spy_lr: optional SPY arrays (reserved for future cross-asset features)
        rich: if True, APPEND variance-ratio trend-persistence features (Lo-MacKinlay). These
            HURT under the correlation filter (label-agnostic crowding, GLD 3.20->2.60) but are
            label-relevant trend-persistence signal that the INFORMATION-GAIN reducer can select
            WITHOUT crowding (reduce=infogain picks top-K by MI). Tests the program hypothesis
            "IG fixes the crowding that made VR features hurt." Causal (rolling, past-only).

    Returns: np.array of shape (N, F) with float32 dtype. F = 80 (base), or 88 with rich=True.
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

    # NOTE: variance-ratio trend-persistence features were tested (2026-06-03) and HURT GLD
    # (3.20 -> 2.60) by crowding the correlation-select — same failure mode as the FFD features.
    # Reverted. The optimized edges are FEATURE-OPTIMAL with the existing 80 features.

    # NOTE: SPY-relative cross-asset features were tested (round 34) and REGRESSED
    # TLT (0.7679 -> 0.677); a real cross-asset edge needs a 2-symbol PAIRS strategy,
    # not SPY features in a single-asset model. Reverted; left disabled. See f_crossasset.

    # RICH feature set (opt-in, reduce=infogain only): variance-ratio trend-persistence (Lo-MacKinlay
    # 1988). VR(k) = Var(k-bar logret)/(k*Var(1-bar logret)); VR>1 = persistent/trending, VR<1 =
    # mean-reverting. Tested under the corr-filter where they HURT (crowding); re-enabled here for the
    # label-relevant IG reducer to select. 8 features (2 base windows x 4 horizons). Causal rolling.
    if rich:
        s1 = pd.Series(lr)
        for W in [200, 400]:
            var_1 = s1.rolling(W, min_periods=W).var()
            for k in [5, 10, 20, 40]:
                kret = np.full(N, np.nan)
                kret[k:] = lc[k:] - lc[:-k]
                var_k = pd.Series(kret).rolling(W, min_periods=W).var()
                vr = var_k / (k * var_1 + 1e-12)
                feats.append((vr - 1.0).astype(np.float32).to_numpy())

    # CROSS-ASSET TERM-STRUCTURE features (opt-in, single-ticker: we still trade only the primary asset; spy_lc here
    # is a SECOND maturity of the SAME underlying — e.g. VIXY(short) vs VIXM(mid) VIX futures). The log-ratio
    # lc-spy_lc tracks contango (<0, vol decay) vs backwardation (>0, vol stress/spike) = the vol-REGIME signal that
    # is EXOGENOUS to the asset's own price (the wall the price-only vol probes hit). Offset-invariant transforms
    # (z-score, change) carry the signal; the raw level has an arbitrary share-price offset (IG drops it). Causal.
    if termstruct and spy_lc is not None and len(spy_lc) == N:
        ratio = pd.Series(lc - spy_lc)                    # log(primary/secondary) ~ term-structure slope
        feats.append((lc - spy_lc).astype(np.float32))    # raw level (offset-confounded; IG may drop)
        for W in [20, 60, 200]:
            m = ratio.rolling(W, min_periods=W).mean()
            s = ratio.rolling(W, min_periods=W).std()
            feats.append(((ratio - m) / (s + 1e-9)).astype(np.float32).to_numpy())   # z-scored term structure (regime)
        for W in [10, 40]:
            feats.append((ratio - ratio.shift(W)).astype(np.float32).to_numpy())     # term-structure CHANGE (roll dynamics)

    return np.column_stack(feats).astype(np.float32)
