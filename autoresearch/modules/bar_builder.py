"""Module (1) Custom Axis — information-driven bar builders.

Each "axis" is a rule for emitting a bar from a minute stream. Different axes
sample the price path on a different clock (transactions, dollars, volatility,
price-range, log-dollars, information surprise). The downstream pipeline is
identical across axes, so the orchestrator can sweep axes generically.

UNIFORM BUILDER INTERFACE
-------------------------
Every Builder class exposes:
    __init__(threshold)
    update(ts, close, vol) -> bar_dict | None
where bar_dict == {"ts_close": ts, "log_close": <log price at emission>}.

DISPATCH
--------
    build_bars(close, vol, ts_arr, bar_type=..., target_bars=15000)
        -> (lc, lr, N, bar_ts)
    AXES = {name: BuilderClass}   # registry, see _AXES_ORDER below.

CONCATENATION / NAMESPACE CONTRACT
----------------------------------
This file is concatenated (header + modules + footer) into ONE QC script that
shares a single global namespace. Therefore:
  * Do NOT import sibling modules.
  * At top level import ONLY: numpy as np, pandas as pd, math.
  * The following are guaranteed present at runtime (provided by header/other
    modules): np, pd, math, json, datetime, and the constants TICKER,
    CROSS_ASSET, TARGET_BARS, TRAIN_END, VAL_END, TEST_END.
  * The file must always pass `python3 -m py_compile`, including on machines
    where heavy ML libs are absent — so no heavy imports live here at all.

CAUSALITY
---------
Any fitted parameter (a threshold, a frozen distribution) is fit on the TRAIN
mask only. The only axis that fits anything is `entropy`: its 5-bucket
return-sign distribution and surprise threshold T are estimated from TRAIN
minutes only (minute timestamp < TRAIN_END), then applied causally going
forward. No bfill, no reversed series, no forward-looking statistics.
"""
import math

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Builders (one per axis). All share the uniform (threshold) / update() API.
# ----------------------------------------------------------------------------
class DollarBarBuilder:
    """Dollar bar (Wang axis): accumulate close * volume; emit at threshold.

    Proven existing math: cum += close * vol. Each bar carries roughly equal
    traded notional, which de-clocks bursts of activity.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.cum = 0.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        self.close_lc = math.log(close)
        self.cum += close * vol
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class TickBarBuilder:
    """Tick bar (Wang axis): count observations; emit every N ticks.

    Proven existing math: count += 1. Pure transaction clock.
    """

    def __init__(self, threshold):
        self.thresh = int(max(1, round(threshold)))
        self.count = 0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        self.close_lc = math.log(close)
        self.count += 1
        if self.count >= self.thresh:
            self.count = 0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class VolBarBuilder:
    """Volatility bar (Wang's 3rd axis): accumulate (delta log-close)^2 * sqrt(vol).

    Proven existing math: contrib = ret^2 * sqrt(vol). Each bar carries roughly
    equal realized-variance-weighted-by-liquidity, which whitens the return
    series and is the workhorse information-driven axis.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.cum = 0.0
        self.last_lc = None
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        lc = math.log(close)
        if self.last_lc is not None:
            ret = lc - self.last_lc
            contrib = (ret * ret) * math.sqrt(vol)
            if contrib > 0:
                self.cum += contrib
        self.last_lc = lc
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class RangeBarBuilder:
    """Range bar: emit when price moves >= threshold % from the last emission.

    Proven existing math: pct_change = |close - last_sample| / last_sample. Each
    bar represents an equal *price* move, which is ideal for volatile assets
    where dollar/time bars space irregularly.
    """

    def __init__(self, threshold):
        self.thresh_pct = float(threshold)
        self.last_sample_close = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        if self.last_sample_close is None:
            self.last_sample_close = close
            return None
        pct_change = abs(close - self.last_sample_close) / self.last_sample_close
        if pct_change >= self.thresh_pct:
            self.last_sample_close = close
            return {"ts_close": ts, "log_close": math.log(close)}
        return None


class LogDollarBarBuilder:
    """Log-dollar bar (NEW, information-driven): accumulate log(1 + close*vol).

    Canonical definition: cumulative sum of log1p(close * vol); emit when the
    accumulator reaches the threshold. Parameter-free — the only knob is the
    threshold, auto-computed like dollar bars (total / target_bars). Compresses
    the heavy right tail of notional, so a single huge print no longer dominates
    a bar the way it can with plain dollar bars.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.cum = 0.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        self.close_lc = math.log(close)
        # log1p is monotone and parameter-free; close*vol >= 0 here.
        self.cum += math.log1p(close * vol)
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class EntropyBarBuilder:
    """Entropy / information-surprise bar (NEW, information-driven).

    Canonical definition: a TRAIN-frozen 5-bucket distribution over the SIGN and
    MAGNITUDE of minute returns. Each incoming minute is mapped to a bucket; its
    Shannon surprise is -log(p_bucket) (NATURAL log). The accumulator sums
    surprise and emits a bar once it reaches threshold T. Rare moves (fat-tail
    minutes) carry more information and therefore close bars faster, so the
    sampling clock speeds up exactly when the market is informative.

    CAUSALITY: both the bucket probabilities `probs` and the threshold `T` are
    estimated from TRAIN minutes only (see build_bars / _fit_entropy_axis) and
    then applied unchanged going forward. This builder consumes those frozen
    parameters; it fits nothing itself.

    The 5 buckets (frozen at fit time as the edges over signed returns):
        0: strong down    1: mild down    2: flat    3: mild up    4: strong up
    Bucketing uses TRAIN return quantiles so that, on TRAIN, the distribution is
    roughly uniform; out-of-sample shifts in the distribution are exactly the
    "surprise" the axis is meant to react to.
    """

    N_BUCKETS = 5
    # Floor so an empty TRAIN bucket cannot produce infinite surprise.
    _PROB_FLOOR = 1e-6

    def __init__(self, threshold, edges=None, probs=None):
        self.thresh = float(threshold)
        # edges: 4 interior cut points over signed minute returns (len N_BUCKETS-1).
        self.edges = None if edges is None else np.asarray(edges, dtype=float)
        # probs: per-bucket probability mass from TRAIN (len N_BUCKETS).
        if probs is None:
            self.probs = None
            self.surprise = None
        else:
            p = np.asarray(probs, dtype=float)
            p = np.clip(p, self._PROB_FLOOR, None)
            self.probs = p
            self.surprise = -np.log(p)  # natural-log Shannon surprise per bucket
        self.cum = 0.0
        self.last_lc = None
        self.close_lc = None

    def _bucket(self, ret):
        # np.searchsorted on the frozen interior edges -> bucket index in [0, N_BUCKETS-1].
        return int(np.searchsorted(self.edges, ret, side="right"))

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        if self.last_lc is None or self.edges is None or self.surprise is None:
            # First observation, or builder not yet fitted: prime state, no emit.
            self.last_lc = lc
            self.close_lc = lc
            return None
        ret = lc - self.last_lc
        b = self._bucket(ret)
        if b < 0:
            b = 0
        elif b >= self.N_BUCKETS:
            b = self.N_BUCKETS - 1
        self.cum += float(self.surprise[b])
        self.last_lc = lc
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


# ----------------------------------------------------------------------------
# Registry — names must be EXACTLY these and in this order.
# ----------------------------------------------------------------------------
class DollarImbalanceBarBuilder:
    """Custom axis (2): signed-dollar IMBALANCE bars (de Prado, information-driven).

    Tick rule gives each minute a sign b_t (+1 up / -1 down / carry-forward on a
    flat print). Accumulate signed dollar flow theta += b_t * close * vol; emit a
    bar when |theta| >= threshold, then reset. Unlike the magnitude axes
    (dollar/logdollar) that sample symmetrically on traded notional, this clock
    fires on DIRECTIONAL runs and shocks — concentrating bars exactly where a
    two-sided asset (e.g. TLT's rate moves) carries its directional edge.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.theta = 0.0
        self.last_lc = None
        self.last_sign = 1.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            self.last_lc = None      # break the tick chain across invalid prints
            return None
        lc = math.log(close)
        if self.last_lc is not None:
            d = lc - self.last_lc
            if d > 0:
                self.last_sign = 1.0
            elif d < 0:
                self.last_sign = -1.0
            # d == 0 -> carry forward last_sign (tick rule)
        self.last_lc = lc
        self.close_lc = lc
        self.theta += self.last_sign * (close * vol)
        if abs(self.theta) >= self.thresh:
            self.theta = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


class TickImbalanceBarBuilder:
    """Custom axis (2b): TICK-imbalance bars (sign-only, de Prado). Accumulate the
    tick-rule sign b_t (+1 up / -1 down / carry-forward on flat); emit when the net
    directional run |Σ b| >= threshold, then reset. A PURER directional clock than
    dollar-imbalance — it ignores notional magnitude, firing purely on persistent
    directional RUNS. Threshold ~ sqrt(minutes_per_bar) (±1 random-walk scaling);
    no fitted distribution, so trivially causal.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.theta = 0.0
        self.last_lc = None
        self.last_sign = 1.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            self.last_lc = None
            return None
        lc = math.log(close)
        if self.last_lc is not None:
            d = lc - self.last_lc
            if d > 0:
                self.last_sign = 1.0
            elif d < 0:
                self.last_sign = -1.0
        self.last_lc = lc
        self.close_lc = lc
        self.theta += self.last_sign
        if abs(self.theta) >= self.thresh:
            self.theta = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


class VolumeImbalanceBarBuilder:
    """Custom axis (2c): VOLUME-imbalance bars (de Prado, information-driven).

    Tick rule gives each minute a sign b_t (+1 up / -1 down / carry-forward on a
    flat print). Accumulate signed *share volume* theta += b_t * vol; emit a bar
    when |theta| >= threshold, then reset. This is the de Prado volume-imbalance
    clock: it sits BETWEEN the two existing directional axes —
        * `imbalance`  (signed-dollar) weights each minute by close*vol (notional),
        * `tickimb`    (sign-only)     weights each minute by 1,
        * `volumeimb`  (this axis)     weights each minute by raw share volume.
    Weighting by SHARES rather than notional removes the price level from the
    accumulator, so a single high-priced print no longer dominates a run; what
    drives emission is *how many shares* trade with the trend. For a two-sided
    bond proxy like TLT, institutional/rate-shock flow shows up as bursts of
    one-sided VOLUME (asset reallocation) more cleanly than as notional, so the
    clock fires on the directional volume runs that carry the rate-move edge,
    rather than on whichever minutes happen to print at a high dollar value.

    Threshold = TRAIN signed-volume volatility, random-walk scaled to
    ~target_bars (|theta| after k minutes ~ sigma*sqrt(k)); see _make_builder.
    Fit on TRAIN minutes only (causal), applied forward unchanged. The builder
    itself fits nothing — it only consumes the frozen threshold.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.theta = 0.0
        self.last_lc = None
        self.last_sign = 1.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            self.last_lc = None      # break the tick chain across invalid prints
            return None
        lc = math.log(close)
        if self.last_lc is not None:
            d = lc - self.last_lc
            if d > 0:
                self.last_sign = 1.0
            elif d < 0:
                self.last_sign = -1.0
            # d == 0 -> carry forward last_sign (tick rule)
        self.last_lc = lc
        self.close_lc = lc
        self.theta += self.last_sign * vol      # signed SHARE volume (not notional)
        if abs(self.theta) >= self.thresh:
            self.theta = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


class FracDiffBarBuilder:
    """Custom axis (3): FRACTIONAL-DIFFERENCE / memory-preserving bars (de Prado,
    Ch.5 "Fractionally Differentiated Features"; novel as a SAMPLING CLOCK).

    Motivation. Integer differencing (the log-RETURN, d=1) makes the series
    stationary but ERASES memory — every level/trend signal is wiped out. The raw
    log-PRICE (d=0) keeps all memory but is non-stationary (a unit root), so any
    threshold defined on it drifts with the price level. Fractional differencing
    with d in (0,1) is the minimal transform that stationarises the series while
    PRESERVING the maximum amount of long memory. Sampling bars on the increments
    of that fractionally-differenced log-price gives a clock that ticks on
    *memory-bearing, stationarised* price information — neither pure notional
    (dollar) nor pure realised variance (vol), but the persistent directional
    structure that a two-sided asset like TLT carries in its level.

    Mechanics. A fixed-width FFD (fixed-width-window fractional differentiation)
    filter with frozen weights w (w[0] applies to the NEWEST log-price) is
    convolved causally over the trailing log-price window. Let
        fd_t = sum_j w[j] * log_price[t-j].
    The accumulator sums the absolute first-difference of the FFD series,
        cum += |fd_t - fd_{t-1}|,
    and a bar is emitted when cum >= threshold (a "runs of FFD movement" clock),
    then cum resets. Because fd is stationary, a single fixed threshold spaces
    bars consistently across calm and shock regimes (unlike a level threshold on
    raw price, which would drift).

    CAUSALITY (G3). Both fitted parameters are estimated on TRAIN minutes ONLY
    and then applied forward unchanged (see _fit_fracdiff_axis):
      * d  — the smallest d on a grid whose TRAIN FFD-log-price series passes an
             ADF unit-root test (ADF t-stat <= critical value), i.e. de Prado's
             "minimum d that achieves stationarity". Smallest d = most memory kept.
      * threshold — TRAIN total-variation of the FFD series per minute, scaled by
             the full-series length to hit ~target_bars (same scaling philosophy
             as the entropy/imbalance axes).
    The FFD weights are a deterministic function of d (the binomial expansion of
    (1-B)^d), so freezing d freezes the filter. The builder fits NOTHING itself;
    it only consumes the frozen (threshold, weights). The convolution at bar t
    uses only log-prices at t, t-1, ..., t-width+1 — strictly past/current. Each
    PER-BAR observation is a pure causal feature of past prices (no forward look).
    """

    def __init__(self, threshold, weights=None):
        self.thresh = float(threshold)
        # weights: frozen FFD weights (len = window width); w[0] -> newest sample.
        if weights is None:
            self.w = None
            self.width = 0
        else:
            self.w = np.asarray(weights, dtype=float)
            self.width = int(len(self.w))
        # Trailing log-price ring buffer (newest appended last). Plain list +
        # manual trim keeps the module import-light (no collections import needed).
        self._buf = []
        self.cum = 0.0
        self.last_fd = None
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        self.close_lc = lc
        if self.w is None or self.width == 0:
            return None
        self._buf.append(lc)
        if len(self._buf) > self.width:
            # keep only the trailing `width` log-prices
            del self._buf[0:len(self._buf) - self.width]
        if len(self._buf) < self.width:
            return None
        # FFD convolution: w[0] applies to the NEWEST sample (buf[-1]).
        fd = 0.0
        for j in range(self.width):
            fd += self.w[j] * self._buf[self.width - 1 - j]
        if self.last_fd is not None:
            self.cum += abs(fd - self.last_fd)
        self.last_fd = fd
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


# ----------------------------------------------------------------------------
class DirectionalChangeBarBuilder:
    """Directional-Change / intrinsic-time axis (Glattfelder-Olsen). Track the trend
    extreme; when log-price reverses by >= delta from that extreme, CONFIRM a directional
    change, flip the trend mode, and emit a bar. Samples the path at REVERSALS (intrinsic
    time) — dense in volatile two-sided action, sparse in calm trends. Built for two-sided
    assets (e.g. TLT's rate regimes) whose edge lives at turning points, not in magnitude.
    delta fit on TRAIN; purely causal (past prices + delta).
    """

    def __init__(self, threshold):
        self.delta = float(threshold)   # reversal threshold in log-price
        self.mode = 1                   # +1 up-trend, -1 down-trend
        self.ext = None                 # running extreme log-price
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        self.close_lc = lc
        if self.ext is None:
            self.ext = lc
            return None
        if self.mode == 1:
            if lc > self.ext:
                self.ext = lc                       # extend up-trend
            elif lc <= self.ext - self.delta:       # downward reversal confirmed
                self.mode = -1
                self.ext = lc
                return {"ts_close": ts, "log_close": lc}
        else:
            if lc < self.ext:
                self.ext = lc                       # extend down-trend
            elif lc >= self.ext + self.delta:       # upward reversal confirmed
                self.mode = 1
                self.ext = lc
                return {"ts_close": ts, "log_close": lc}
        return None


# Registry — names must be EXACTLY these and in this order.
# ----------------------------------------------------------------------------
_AXES_ORDER = ["dollar", "tick", "vol", "range", "logdollar", "entropy", "imbalance", "tickimb", "volumeimb", "fracdiff", "dc"]
AXES = {
    "dollar": DollarBarBuilder,
    "tick": TickBarBuilder,
    "vol": VolBarBuilder,
    "range": RangeBarBuilder,
    "logdollar": LogDollarBarBuilder,
    "entropy": EntropyBarBuilder,
    "imbalance": DollarImbalanceBarBuilder,
    "tickimb": TickImbalanceBarBuilder,
    "volumeimb": VolumeImbalanceBarBuilder,
    "fracdiff": FracDiffBarBuilder,
    "dc": DirectionalChangeBarBuilder,
}


# ----------------------------------------------------------------------------
# Threshold auto-calibration helpers (run on the minute stream, pre-build).
# ----------------------------------------------------------------------------
def _safe_thresh(total, target_bars):
    """Threshold that yields ~target_bars bars from a positive accumulator total."""
    tb = max(1, int(target_bars))
    return (total / tb) if total > 0 else 1e-9


def _train_minute_mask(ts_arr):
    """Boolean mask of minutes strictly before TRAIN_END.

    Uses the TRAIN_END global (a datetime) supplied by the header at runtime.
    Falls back to "all minutes are TRAIN" if TRAIN_END is unavailable (e.g. a
    bare unit harness) so the entropy axis still builds rather than crashing.
    """
    n = len(ts_arr)
    try:
        cutoff = np.datetime64(TRAIN_END)  # noqa: F821 — global from header
    except Exception:
        return np.ones(n, dtype=bool)
    try:
        ts64 = np.array([np.datetime64(str(t)) for t in ts_arr])
        return ts64 < cutoff
    except Exception:
        return np.ones(n, dtype=bool)


def _minute_log_returns(close):
    """Signed minute log-returns over valid (positive) closes; NaN where invalid.

    Strictly causal: r[i] = log(close[i]) - log(close[i-1]); r[0] = NaN. No
    shift(-N), no reversal — only past/current minutes are used.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    ret = np.full(n, np.nan)
    last_lc = None
    for i in range(n):
        if close[i] <= 0:
            last_lc = None  # break the chain across invalid prints
            continue
        lc = math.log(close[i])
        if last_lc is not None:
            ret[i] = lc - last_lc
        last_lc = lc
    return ret


def _fit_entropy_axis(close, vol, ts_arr, target_bars):
    """Fit the entropy axis on TRAIN minutes only and return (edges, probs, T).

    1. Compute signed minute log-returns.
    2. Restrict to TRAIN minutes (causality rule 1).
    3. Freeze 4 interior bucket edges = TRAIN return quantiles -> 5 buckets.
    4. Freeze per-bucket probabilities from the TRAIN bucket histogram.
    5. Binary-search the surprise threshold T (on TRAIN minutes only) so that
       the entropy clock emits ~target_bars bars over the FULL series. Because
       average TRAIN surprise per minute is stable, T*(target_bars) approximates
       total surprise; we refine with a bisection on the TRAIN simulation.
    """
    ret = _minute_log_returns(close)
    tr_mask = _train_minute_mask(ts_arr)
    tr_ret = ret[tr_mask & ~np.isnan(ret)]

    nb = EntropyBarBuilder.N_BUCKETS
    if len(tr_ret) < (nb * 10):
        # Too little TRAIN data to fit a meaningful distribution.
        return None, None, None

    # 4 interior edges -> 5 buckets, frozen from TRAIN quantiles.
    qs = np.linspace(0.0, 1.0, nb + 1)[1:-1]  # e.g. 0.2,0.4,0.6,0.8
    edges = np.quantile(tr_ret, qs)
    # De-duplicate degenerate edges (constant-ish TRAIN returns) to keep buckets ordered.
    edges = np.maximum.accumulate(edges)
    eps = 1e-12
    for j in range(1, len(edges)):
        if edges[j] <= edges[j - 1]:
            edges[j] = edges[j - 1] + eps

    # Per-bucket TRAIN probabilities via the same searchsorted rule the builder uses.
    buckets = np.searchsorted(edges, tr_ret, side="right")
    buckets = np.clip(buckets, 0, nb - 1)
    counts = np.bincount(buckets, minlength=nb).astype(float)
    probs = counts / counts.sum()
    probs = np.clip(probs, EntropyBarBuilder._PROB_FLOOR, None)
    probs = probs / probs.sum()
    surprise = -np.log(probs)  # natural log

    # Total TRAIN surprise and the TRAIN minute fraction, used to scale to the
    # full series and to size the binary search target.
    total_tr_surprise = float(np.sum(surprise[buckets]))
    n_tr_min = int(np.count_nonzero(tr_mask))
    n_all_min = max(1, len(close))
    # Expected total surprise over the full series, assuming TRAIN's average
    # surprise-per-minute holds out-of-sample.
    avg_surprise = total_tr_surprise / max(1, n_tr_min)
    expected_total = avg_surprise * n_all_min
    tb = max(1, int(target_bars))

    # Analytic seed; then bisection on the TRAIN stream for ~ (target_bars on
    # TRAIN scaled by TRAIN fraction) so the full-series count lands near target.
    tr_frac = n_tr_min / n_all_min
    target_tr_bars = max(1, int(round(tb * tr_frac)))
    seed_T = expected_total / tb if expected_total > 0 else avg_surprise
    if seed_T <= 0:
        seed_T = 1e-6

    def count_bars_on_train(T):
        # Simulate the entropy accumulator over TRAIN minutes only.
        c = 0.0
        n = 0
        for s in surprise[buckets]:
            c += s
            if c >= T:
                c = 0.0
                n += 1
        return n

    # Bracket around the analytic seed, then bisect.
    lo = seed_T / 8.0
    hi = seed_T * 8.0
    # Ensure the bracket actually straddles target_tr_bars.
    # Larger T -> fewer bars (monotone decreasing).
    for _ in range(6):
        if count_bars_on_train(lo) >= target_tr_bars:
            lo /= 4.0
        else:
            break
    for _ in range(6):
        if count_bars_on_train(hi) <= target_tr_bars:
            hi *= 4.0
        else:
            break
    T = seed_T
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n_bars = count_bars_on_train(mid)
        T = mid
        if n_bars == target_tr_bars:
            break
        if n_bars > target_tr_bars:
            lo = mid  # too many bars -> raise T
        else:
            hi = mid  # too few bars -> lower T
    if T <= 0:
        T = seed_T if seed_T > 0 else 1e-6
    return edges, probs, T


def _ffd_weights(d, w_thresh=1e-4, max_width=2000):
    """Fixed-width-window fractional-difference weights for order d (de Prado).

    The weights are the binomial expansion of (1 - B)^d:
        w_0 = 1,  w_k = -w_{k-1} * (d - k + 1) / k.
    The series is truncated once |w_k| < w_thresh (fixed-width window), bounding
    the convolution cost and the warm-up length. Returned newest-first: w[0]
    multiplies the most recent log-price. Deterministic in d only -> freezing d
    (on TRAIN) freezes the entire filter, which is what keeps the axis causal.
    """
    w = [1.0]
    k = 1
    while k < int(max_width):
        nw = -w[-1] * (d - k + 1) / k
        if abs(nw) < w_thresh:
            break
        w.append(nw)
        k += 1
    return np.asarray(w, dtype=float)


def _ffd_series_from_lc(lc, w):
    """Causal FFD convolution of a log-price array with weights w.

    out[i] = sum_j w[j] * lc[i-j] for i >= width-1, else NaN (warm-up). Uses only
    past/current samples. lc must already be a clean (gap-free) log-price series.
    """
    lc = np.asarray(lc, dtype=float)
    width = len(w)
    n = len(lc)
    out = np.full(n, np.nan)
    if width == 0 or n < width:
        return out
    wr = w[::-1]  # so that a forward dot with the trailing window applies w[0] to newest
    for i in range(width - 1, n):
        out[i] = float(np.dot(wr, lc[i - width + 1:i + 1]))
    return out


def _adf_tstat(x):
    """Augmented Dickey-Fuller t-stat with zero lags (the DF core), numpy-only.

    Regress dY_t = a + rho * Y_{t-1} + e and return rho / se(rho). A more NEGATIVE
    value => stronger mean reversion => more stationary (closer to rejecting the
    unit root). Self-contained so the module never needs statsmodels.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return 0.0
    Y = x[:-1]
    dY = np.diff(x)
    n = len(Y)
    mx = Y.mean()
    my = dY.mean()
    sxx = float(np.dot(Y - mx, Y - mx))
    if sxx <= 0:
        return 0.0
    sxy = float(np.dot(Y - mx, dY - my))
    rho = sxy / sxx
    a = my - rho * mx
    resid = dY - a - rho * Y
    sse = float(np.dot(resid, resid))
    s2 = sse / max(1, (n - 2))
    se = math.sqrt(s2 / sxx) if sxx > 0 else 1e9
    return (rho / se) if se > 0 else 0.0


# de Prado's ~5% ADF critical value (constant, no trend). Smallest d whose TRAIN
# FFD series clears this is selected — minimum differencing => maximum memory.
_FRACDIFF_ADF_CRIT = -2.9
_FRACDIFF_D_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _clean_log_price(close):
    """Gap-free causal log-price series for the FFD convolution.

    log(close) where close>0; invalid prints are forward-filled with the last
    valid log-price (carry-forward — strictly causal, no bfill) so the fixed-width
    window stays contiguous. Leading invalid prints (no prior valid price) are
    returned as NaN and trimmed by the caller.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    lc = np.full(n, np.nan)
    last = None
    for i in range(n):
        if close[i] > 0:
            last = math.log(close[i])
        if last is not None:
            lc[i] = last
    return lc


def _fit_fracdiff_axis(close, vol, ts_arr, target_bars):
    """Fit the fractional-difference axis on TRAIN minutes only.

    Returns (threshold, weights) or (None, None) if uncalibratable.

    1. Build a gap-free causal log-price series; trim leading NaNs.
    2. For each d on the grid (ascending), build the TRAIN FFD series and its ADF
       t-stat. Pick the SMALLEST d whose TRAIN ADF t-stat <= _FRACDIFF_ADF_CRIT
       (most memory while stationary). If none clears it, keep the most-stationary
       (most-negative-t) d as a fallback. d is fit on TRAIN only.
    3. Threshold = TRAIN total-variation of the FFD series per valid minute,
       scaled by the full-series valid length to hit ~target_bars. TRAIN-only fit.
    """
    lc_full = _clean_log_price(close)
    first = int(np.argmax(np.isfinite(lc_full))) if np.any(np.isfinite(lc_full)) else 0
    lc_full = lc_full[first:]
    if len(lc_full) < 200 or not np.all(np.isfinite(lc_full)):
        # still NaNs after trimming the lead -> too gappy to difference safely
        if not np.all(np.isfinite(lc_full)):
            return None, None

    tr_mask = _train_minute_mask(ts_arr)[first:]
    n_tr = int(np.count_nonzero(tr_mask))
    if n_tr < 200:
        return None, None

    chosen = None  # (d, w, t)
    for d in _FRACDIFF_D_GRID:
        w = _ffd_weights(d)
        if len(w) < 2 or len(w) >= (n_tr // 2):
            continue
        fd_tr = _ffd_series_from_lc(lc_full[tr_mask], w)
        t = _adf_tstat(fd_tr)
        if chosen is None or t < chosen[2]:
            chosen = (d, w, t)
        if t <= _FRACDIFF_ADF_CRIT:
            chosen = (d, w, t)  # smallest d that clears the bar -> stop
            break
    if chosen is None:
        return None, None
    d, w, _t = chosen

    # Threshold: TRAIN total-variation per minute, scaled to the full series.
    fd_tr = _ffd_series_from_lc(lc_full[tr_mask], w)
    fd_tr = fd_tr[np.isfinite(fd_tr)]
    if len(fd_tr) < 50:
        return None, None
    total_tv_tr = float(np.sum(np.abs(np.diff(fd_tr))))
    n_tr_valid = len(fd_tr)
    n_all_valid = max(1, len(lc_full) - (len(w) - 1))
    avg_tv = total_tv_tr / max(1, n_tr_valid - 1)
    expected_total = avg_tv * n_all_valid
    thresh = _safe_thresh(expected_total, target_bars)
    if not np.isfinite(thresh) or thresh <= 0:
        return None, None
    return thresh, w


def _make_builder(bar_type, close, vol, ts_arr, target_bars):
    """Instantiate the right Builder with an auto-calibrated threshold."""
    if bar_type == "dollar":
        total = float(np.sum(np.asarray(close, dtype=float) * np.asarray(vol, dtype=float)))
        return DollarBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "tick":
        return TickBarBuilder(max(1, len(close) // max(1, int(target_bars))))

    if bar_type == "range":
        # Equal-price-move bars: tighter band when we want more (smaller) bars.
        thresh = 0.003 if int(target_bars) < 10000 else 0.002
        return RangeBarBuilder(thresh)

    if bar_type == "logdollar":
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        dv = c * v
        valid = (c > 0) & (v > 0)
        total = float(np.sum(np.log1p(dv[valid]))) if np.any(valid) else 0.0
        return LogDollarBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "entropy":
        edges, probs, T = _fit_entropy_axis(close, vol, ts_arr, target_bars)
        if edges is None:
            return None
        return EntropyBarBuilder(T, edges=edges, probs=probs)

    if bar_type == "imbalance":
        # Threshold from TRAIN signed-dollar volatility, random-walk scaled to
        # ~target_bars: |theta| after k minutes ~ sigma*sqrt(k), so a threshold of
        # sigma*sqrt(minutes_per_bar) crosses roughly every target interval. Fit on
        # TRAIN minutes only (causal); applied forward unchanged.
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        sd = np.sign(ret) * (c * v)
        keep = tr & np.isfinite(sd) & (c > 0) & (v > 0) & (sd != 0)
        sdt = sd[keep]
        if len(sdt) < 100:
            return None
        sigma = float(np.std(sdt))
        m = max(1.0, len(c) / max(1, int(target_bars)))   # target minutes per bar
        thresh = sigma * math.sqrt(m)
        if not np.isfinite(thresh) or thresh <= 0:
            return None
        return DollarImbalanceBarBuilder(thresh)

    if bar_type == "tickimb":
        # ±1 random-walk scaling: |Σ b| ~ sqrt(k), so a count threshold sqrt(m)
        # crosses roughly every m minutes. Pure counts -> no fit, causal.
        m = max(1.0, len(close) / max(1, int(target_bars)))
        return TickImbalanceBarBuilder(max(2.0, math.sqrt(m)))

    if bar_type == "volumeimb":
        # Threshold from TRAIN signed-VOLUME volatility, random-walk scaled to
        # ~target_bars: |theta| after k minutes ~ sigma*sqrt(k), so a threshold of
        # sigma*sqrt(minutes_per_bar) crosses roughly every target interval. Same
        # recipe as the signed-dollar `imbalance` axis but the per-minute weight is
        # share VOLUME (sign(ret)*vol) rather than notional (sign(ret)*close*vol).
        # Fit on TRAIN minutes only (causal); applied forward unchanged.
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        sv = np.sign(ret) * v
        keep = tr & np.isfinite(sv) & (c > 0) & (v > 0) & (sv != 0)
        svt = sv[keep]
        if len(svt) < 100:
            return None
        sigma = float(np.std(svt))
        m = max(1.0, len(c) / max(1, int(target_bars)))   # target minutes per bar
        thresh = sigma * math.sqrt(m)
        if not np.isfinite(thresh) or thresh <= 0:
            return None
        return VolumeImbalanceBarBuilder(thresh)

    if bar_type == "fracdiff":
        # Memory-preserving clock: smallest d whose TRAIN FFD log-price passes the
        # ADF unit-root test, threshold = TRAIN total-variation scaled to
        # ~target_bars. Both fit on TRAIN minutes only (causal), applied forward.
        thresh, weights = _fit_fracdiff_axis(close, vol, ts_arr, target_bars)
        if thresh is None or weights is None:
            return None
        return FracDiffBarBuilder(thresh, weights=weights)

    if bar_type == "dc":
        # Reversal threshold delta = TRAIN minute-return std * sqrt(minutes/bar):
        # a delta-reversal recurs ~ every (delta/sigma)^2 minutes in a random walk,
        # so this targets ~target_bars reversals. Fit on TRAIN only (causal).
        c = np.asarray(close, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        trr = ret[tr & np.isfinite(ret)]
        if len(trr) < 100:
            return None
        sigma = float(np.std(trr))
        m = max(1.0, len(c) / max(1, int(target_bars)))
        delta = sigma * math.sqrt(m)
        if not np.isfinite(delta) or delta <= 0:
            return None
        return DirectionalChangeBarBuilder(delta)

    # Default / unknown -> volatility bar (Wang's workhorse axis).
    c = np.asarray(close, dtype=float)
    v = np.asarray(vol, dtype=float)
    total = 0.0
    last_lc = None
    for i in range(len(c)):
        if c[i] <= 0 or v[i] <= 0:
            last_lc = None
            continue
        lc_val = math.log(c[i])
        if last_lc is not None:
            total += (lc_val - last_lc) ** 2 * math.sqrt(v[i])
        last_lc = lc_val
    return VolBarBuilder(_safe_thresh(total, target_bars))


# ----------------------------------------------------------------------------
# Dispatch — build a full bar series from minute data.
# ----------------------------------------------------------------------------
def build_bars(close, vol, ts_arr, bar_type="vol", target_bars=15000):
    """Build bars from minute data on the requested axis.

    Args:
        close:   np.array of minute close prices.
        vol:     np.array of minute volumes.
        ts_arr:  np.array of minute timestamps (aligned with close/vol).
        bar_type: one of ["dollar","tick","vol","range","logdollar","entropy",
                  "imbalance","tickimb","volumeimb","fracdiff"].
                  Unknown values fall back to the volatility axis.
        target_bars: approximate number of bars desired; each axis auto-computes
                  its threshold to hit roughly this count.

    Returns:
        lc:     np.array of log-close per bar.
        lr:     np.array of log-returns per bar (lr[0] = 0).
        N:      int number of bars.
        bar_ts: np.array of bar emission timestamps.
    """
    close = np.asarray(close)
    vol = np.asarray(vol)
    ts_arr = np.asarray(ts_arr)
    n_min = len(close)

    builder = _make_builder(bar_type, close, vol, ts_arr, target_bars)
    if builder is None:
        # Axis could not be calibrated (e.g. entropy with too little TRAIN data).
        return np.array([]), np.array([]), 0, np.array([])

    bars = []
    bar_ts_raw = []
    for i in range(n_min):
        b = builder.update(ts_arr[i], close[i], vol[i])
        if b is not None:
            bars.append(b)
            bar_ts_raw.append(ts_arr[i])

    N = len(bars)
    if N == 0:
        return np.array([]), np.array([]), 0, np.array([])

    lc = np.array([x["log_close"] for x in bars], dtype=float)
    lr = np.zeros(N)
    lr[1:] = lc[1:] - lc[:-1]
    bar_ts = np.array(bar_ts_raw)
    return lc, lr, N, bar_ts
