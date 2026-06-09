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


def _gpd_mom(exceed):
    """Method-of-moments GPD fit (Hosking-Wallis) over peak exceedances — pure numpy, no scipy.
    GPD mean m = sigma/(1-xi), var v = sigma^2/((1-xi)^2(1-2xi))  =>  xi = 0.5*(1 - m^2/v),
    sigma = m*(1-xi). Returns (xi, sigma) clamped to a numerically safe heavy-tail range. This is
    the closed-form that lets us run EVT tail detection in QuantConnect (no scipy MLE)."""
    n = exceed.size
    if n < 8:
        return 0.0, max(float(np.mean(exceed)) if n else 1e-6, 1e-6)
    m = float(np.mean(exceed))
    v = float(np.var(exceed))
    if not np.isfinite(m) or not np.isfinite(v) or v <= 1e-18 or m <= 1e-12:
        return 0.0, max(m, 1e-6)
    xi = 0.5 * (1.0 - (m * m) / v)
    xi = float(np.clip(xi, -0.5, 0.45))          # >0.5 -> infinite variance; clamp for stability
    sigma = max(m * (1.0 - xi), 1e-9)
    return xi, sigma


def evt_tail_score(losses, q=0.90):
    """Self-calibrating EVT (peaks-over-threshold) tail-extremeness score for the LAST point of a
    trailing window of `losses` (e.g. -log_return; a crash is a big positive loss). DSPOT-style but
    rolling-window (causal + online-reproducible) rather than a stateful stream:

      u = window quantile(q); peaks = losses[losses>u]-u; (xi,sigma)=GPD-MoM(peaks);
      current loss y -> upper-tail prob p = (n_peaks/W)*(1+xi*(y-u)/sigma)^(-1/xi)  [y>u]
      feature = -log(p+eps)  (LARGE when the current move sits deep in a self-calibrated tail).

    Past-only (the window is trailing bars), so it is leak-free, and a trailing window online
    reproduces the same value as the full-series build -> deploys like sample_entropy."""
    w = losses.size
    if w < 20:
        return 0.0
    finite = losses[np.isfinite(losses)]
    if finite.size < 20:
        return 0.0
    u = float(np.quantile(finite, q))
    peaks = finite[finite > u] - u
    if peaks.size < 8:
        return 0.0
    xi, sigma = _gpd_mom(peaks)
    y = float(losses[-1])
    if not np.isfinite(y) or y <= u:
        return 0.0                                # not in the tail -> no extremeness signal
    z = (y - u) / sigma
    if abs(xi) < 1e-6:
        surv = math.exp(-z)                       # xi->0 limit: exponential tail
    else:
        base = 1.0 + xi * z
        surv = base ** (-1.0 / xi) if base > 0 else 0.0
    p = (peaks.size / float(w)) * surv            # P(loss > y): tiny when extreme
    return float(-math.log(p + 1e-9))             # tail-risk score: large = deep tail


def dispersion_entropy(x, m=2, c=6):
    """Dispersion entropy (Rostaghi & Azami 2016) — O(N) AND amplitude-aware, the quadrant our
    sample_entropy (amplitude-aware but O(W^2)) and permutation-entropy (O(N) but ordinal-only,
    which DEGRADED edges) both miss. NCDF-symbolize the window to c classes, count m-length
    dispersion patterns, return normalized Shannon entropy of the pattern distribution in [0,1].
    Pure numpy, causal when fed a trailing window."""
    n = len(x)
    if n < m + 2:
        return 0.0
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    mu = float(np.mean(x))
    # NCDF map -> classes 1..c (0.5*(1+erf) is the normal CDF; vectorized)
    z = (np.asarray(x, dtype=float) - mu) / sd
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    y = np.clip(np.round(c * cdf + 0.5).astype(int), 1, c)      # class labels
    # dispersion patterns: base-c index of each m-gram (delay 1)
    idx = np.zeros(n - m + 1, dtype=int)
    for k in range(m):
        idx = idx * c + (y[k:n - m + 1 + k] - 1)
    counts = np.bincount(idx, minlength=c ** m).astype(float)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    ent = -float(np.sum(p * np.log(p)))
    return ent / math.log(c ** m)                                # normalize to [0,1]


def signature_lead_lag(x1, x2):
    """Level-2 path-signature CROSS terms (truncated-signature method, Chevyrev-Kormilitzin) between
    two channels over a window — ORDER-AWARE lead-lag geometry that scalar features (z-score, corr,
    termstruct) are blind to. S^{ij}=int (x^i - x^i_0) dx^j (iterated integral); the antisymmetric
    part = Levy area = signed lead-lag (x1 leads x2 vs x2 leads x1). Returns (s12, s21, levy),
    vol-normalized so they're stationary. Pure numpy, causal on a trailing window. Built to mine the
    GLD<-UUP (gold<-dollar) timing relationship as a feature without forming a pair."""
    n = len(x1)
    if n < 5 or len(x2) != n:
        return 0.0, 0.0, 0.0
    a = np.asarray(x1, dtype=float) - float(x1[0])      # path centered at window start
    b = np.asarray(x2, dtype=float) - float(x2[0])
    da = np.diff(a)
    db = np.diff(b)
    s12 = float(np.sum(a[:-1] * db))                    # int (x1-x1_0) dx2
    s21 = float(np.sum(b[:-1] * da))                    # int (x2-x2_0) dx1
    sd = (float(np.std(da)) * float(np.std(db)) * n) + 1e-12   # vol*length normalizer -> stationary
    return s12 / sd, s21 / sd, 0.5 * (s12 - s21) / sd   # (s12, s21, Levy area)


def realyield_feats(ry, sl=None, N=None):
    """Causal exogenous REAL-YIELD features for gold (GLD <- real rates).

    ry = 10y TIPS real-yield LEVEL (FRED DFII10), aligned 1:1 to the bar grid,
    1-day-lagged and forward-filled onto the event bars by the caller (footer).
    sl = the 2s10s nominal slope (DGS10-DGS2), same alignment, optional.

    Gold is a long-duration zero-coupon REAL asset: a RISING real yield raises the
    opportunity cost of holding a non-yielding store of value (headwind), a FALLING
    real yield is a tailwind. So the predictive content is the real-yield LEVEL-vs-
    regime and its CHANGE, not a price ratio. Every transform is rolling/shift =
    PAST-ONLY (causal): the feature at bar i uses only ry[:i+1], so an online
    trailing window reproduces the batch value exactly (append-OOS-invariant). A
    yield is already offset-free (not a share price), so raw levels are kept.

    Returns a list of float32 arrays (len N each); [] if ry is unusable.
    """
    ry = np.asarray(ry, dtype=float)
    n = len(ry)
    if n == 0 or (N is not None and n != N):
        return []
    out = []
    s = pd.Series(ry)
    out.append(s.astype(np.float32).to_numpy())                       # real-yield LEVEL (regime; offset-free)
    for W in (60, 252):                                               # level z-score vs its own recent range
        m = s.rolling(W, min_periods=W).mean()
        sd = s.rolling(W, min_periods=W).std()
        out.append(((s - m) / (sd + 1e-9)).astype(np.float32).to_numpy())
    for k in (5, 20, 60):                                             # real-yield CHANGE (the gold driver)
        out.append((s - s.shift(k)).astype(np.float32).to_numpy())
    if sl is not None:
        sls = pd.Series(np.asarray(sl, dtype=float))
        if len(sls) == n:
            out.append(sls.astype(np.float32).to_numpy())             # 2s10s slope LEVEL (already a difference)
            m = sls.rolling(252, min_periods=252).mean()
            sd = sls.rolling(252, min_periods=252).std()
            out.append(((sls - m) / (sd + 1e-9)).astype(np.float32).to_numpy())  # slope regime z-score
            out.append((sls - sls.shift(20)).astype(np.float32).to_numpy())      # slope CHANGE
    return out


def build_feats(lc, lr, spy_lc=None, spy_lr=None, abs_start=0, rich=False, termstruct=False, evt=False, disp=False, sig=False, ry=None, sl=None, realyield=False):
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
        # SECONDARY-OWN features (2026-06-06, user "other ETFs as features"): the cross-ratio above suits two
        # maturities of the SAME underlying (VIXY/VIXM), but for "asset <- its macro DRIVER" (GLD<-UUP dollar,
        # SPY<-VXX vol) the signal is the DRIVER'S OWN dynamics, not the level ratio. Add the secondary's own
        # momentum / realized-vol / regime z-score so infogain can select them ON MERIT (the ratio alone was
        # IG-dropped on GLD R1233 / crowded under correlation R1234). Causal: spy_lc/spy_lr are past-only,
        # as-of joined (footer leak-fix). Offset-invariant transforms only (raw secondary level is share-price-
        # confounded). Engages only under reduce=infogain (correlation crowds — proven).
        if spy_lr is not None and len(spy_lr) == N:
            slc = pd.Series(np.asarray(spy_lc, dtype=float))
            slr = pd.Series(np.asarray(spy_lr, dtype=float))
            for k in [1, 5, 20]:                                                      # driver MOMENTUM (k-bar return)
                feats.append((slc - slc.shift(k)).astype(np.float32).to_numpy())
            for W in [20, 60]:                                                        # driver REALIZED VOL (vol-of-driver)
                feats.append(slr.rolling(W, min_periods=W).std().astype(np.float32).to_numpy())
            for W in [60, 200]:                                                       # driver REGIME (z-scored level)
                m = slc.rolling(W, min_periods=W).mean()
                s = slc.rolling(W, min_periods=W).std()
                feats.append(((slc - m) / (s + 1e-9)).astype(np.float32).to_numpy())

    # EVT TAIL-RISK features (opt-in, reduce=infogain; 2026-06-08). Self-calibrating peaks-over-
    # threshold tail-extremeness score (evt_tail_score, GPD method-of-moments — no scipy). Built to
    # CONVERT the SPY crash-veto lead: crash_ahead's fixed -k*sigma threshold is brittle (re-run
    # collapsed 43 trades -> 1); a streaming-EVT self-calibrating tail signal gives the crash model a
    # distribution-aware extremeness measure that should fire more stably (target: deployable trades).
    # Causal: computed on a TRAILING window of losses (=-logret) on the SAME abs_start stride-grid as
    # the entropy features, so an online trailing window reproduces the full-series value (verified
    # online==batch <1e-12). 2 features (W in {100,200}).
    if evt:
        losses_arr = -np.diff(lc, prepend=lc[0])          # crash = large positive loss
        for W in [100, 200]:
            ev = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W // 5)
            for i in range(W, N):
                if (abs_start + i - W) % stride == 0:
                    ev[i] = evt_tail_score(losses_arr[i - W + 1:i + 1], q=0.90)
            feats.append(pd.Series(ev).ffill().fillna(0.0).astype(np.float32).to_numpy())

    # DISPERSION-ENTROPY features (opt-in, reduce=infogain; 2026-06-08). O(N) AND amplitude-aware
    # complexity (Rostaghi-Azami 2016) — the quadrant sample_entropy (amplitude-aware but slow) and
    # permutation-entropy (fast but ordinal-only, which DEGRADED edges) both miss. Causal trailing
    # window on the SAME abs_start stride-grid as the sample-entropy features -> online-reproducible.
    # 3 features (W in {50,100,200}).
    if disp:
        for W in [50, 100, 200]:
            de = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W // 5)
            for i in range(W, N):
                if (abs_start + i - W) % stride == 0:
                    de[i] = dispersion_entropy(lc[i - W:i], m=2, c=6)
            feats.append(pd.Series(de).ffill().fillna(0.0).astype(np.float32).to_numpy())

    # PATH-SIGNATURE lead-lag features (opt-in, reduce=infogain; 2026-06-08). Level-2 cross signature
    # between the asset and its exogenous DRIVER (spy_lc, e.g. GLD<-UUP dollar) = ORDER-AWARE lead-lag
    # the scalar termstruct features (IG-dropped on GLD R1233) cannot see. Causal trailing window on
    # the abs_start stride-grid -> online-reproducible. 3 features x 2 windows. Engages only when the
    # exogenous channel is present (single-ticker: we still TRADE only the primary asset).
    if sig and spy_lc is not None and len(spy_lc) == N:
        for W in [60, 200]:
            s12 = np.full(N, np.nan, dtype=np.float32)
            s21 = np.full(N, np.nan, dtype=np.float32)
            lev = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W // 5)
            for i in range(W, N):
                if (abs_start + i - W) % stride == 0:
                    a, b, l = signature_lead_lag(lc[i - W:i], spy_lc[i - W:i])
                    s12[i], s21[i], lev[i] = a, b, l
            for arr in (s12, s21, lev):
                feats.append(pd.Series(arr).ffill().fillna(0.0).astype(np.float32).to_numpy())

    # EXOGENOUS REAL-YIELD features (opt-in, reduce=infogain; 2026-06-09 direction).
    # The one untested fundamental-macro channel: gold <- 10y TIPS real yield (DFII10)
    # + 2s10s slope, threaded as a 1-day-lagged daily series the footer forward-fills
    # onto the event bars. Causal (rolling/shift past-only) -> online-reproducible.
    # Distinct from the closed cross-asset ETF-PRICE proxy (GLD<-UUP, R1242).
    if realyield and ry is not None:
        feats.extend(realyield_feats(ry, sl, N))

    return np.column_stack(feats).astype(np.float32)
