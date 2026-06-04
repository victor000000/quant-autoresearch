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


class KyleImpactBarBuilder:
    """Price-impact / illiquidity bar (NEW, microstructure axis): accumulate
    |delta log-close| / sqrt(volume) — price MOVE per unit sqrt-liquidity (a Kyle-lambda /
    Amihud-illiquidity flavour). Emits FAST when price moves on THIN volume (high price
    impact == likely informed flow) and SLOWLY when heavy volume barely moves price.
    Orthogonal to the SIZE axes: vol bars weight ret^2*sqrt(vol) and dollar bars weight
    close*vol (both speed up on size); this speeds up on IMPACT-per-size, a distinct clock.
    O(1) incremental so the batch build and the online infer are byte-exact; the threshold
    is auto-calibrated from TRAIN minutes only (causal, see _make_builder)."""

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
            contrib = abs(lc - self.last_lc) / math.sqrt(vol)
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


class ZCusumBarBuilder:
    """Standardized-cumsum CUSUM bar (Wang's LogDollar event clock). Driver
    x = log(close*vol); rolling-standardize over a trailing window (causal):
    z = (x - mu_w)/sd_w; run a SYMMETRIC CUSUM filter
        S+ = max(0, S+ + z),  S- = min(0, S- + z),
    emit a bar when S+ >= T or -S- >= T, then reset both. Samples densely at
    directional RUNS in standardized log-dollar flow and sparsely in calm drift —
    an event clock, unlike the count/threshold bars. Rolling stats are kept
    incrementally (ring buffer, O(1)/update) so the SAME update() drives the batch
    build and the online infer -> byte-exact. Leak-free: the window is trailing
    (no future); the threshold T comes from the minute count only (no fit)."""

    def __init__(self, threshold, window=1950):
        self.thresh = float(threshold)          # T (uniform name for builder_threshold)
        self.w = int(window)
        self._ring = [0.0] * self.w
        self._cnt = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.sp = 0.0
        self.sn = 0.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        self.close_lc = math.log(close)
        x = math.log(close * vol)
        pos = self._cnt % self.w
        if self._cnt >= self.w:                 # evict the value leaving the window
            old = self._ring[pos]
            self._sum -= old
            self._sumsq -= old * old
        self._ring[pos] = x
        self._sum += x
        self._sumsq += x * x
        self._cnt += 1
        n = self.w if self._cnt >= self.w else self._cnt
        if n < 30:                              # warm-up: too few for a stable z
            return None
        mu = self._sum / n
        var = self._sumsq / n - mu * mu
        if var <= 1e-18:
            return None
        z = (x - mu) / math.sqrt(var)
        self.sp = max(0.0, self.sp + z)
        self.sn = min(0.0, self.sn + z)
        if self.sp >= self.thresh or -self.sn >= self.thresh:
            self.sp = 0.0
            self.sn = 0.0
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


class RunBarBuilder:
    """Custom axis: RUN-PERSISTENCE bars. Accumulate |log-return| WITHIN a directional
    run; RESET the accumulator whenever the move sign flips; emit a bar when the within-run
    accumulation >= threshold. This is the DUAL of `dc` (which samples at REVERSALS, dense
    in chop): the run clock fires on SUSTAINED one-direction runs — dense during persistent
    trends, silent during choppy two-sided action — so it isolates exactly the momentum
    structure the ker/trend_scan labelers exploit. A trend-momentum asset (GLD/SOXX) should
    get a cleaner trend signal off this clock than off a notional clock (logdollar) that
    samples chop and trend alike. Threshold fit on TRAIN by simulation (causal, see
    _fit_run_axis); the builder itself fits nothing — it only consumes the frozen threshold.
    """

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.theta = 0.0
        self.sign = 0.0
        self.last_lc = None
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            self.last_lc = None      # break the run chain across invalid prints
            return None
        lc = math.log(close)
        self.close_lc = lc
        if self.last_lc is None:
            self.last_lc = lc
            return None
        d = lc - self.last_lc
        self.last_lc = lc
        s = 1.0 if d > 0 else (-1.0 if d < 0 else self.sign)
        if s != self.sign and self.sign != 0.0:
            self.theta = abs(d)      # sign flip -> start a fresh run
        else:
            self.theta += abs(d)
        if s != 0.0:
            self.sign = s
        if self.theta >= self.thresh:
            self.theta = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


class SpectralCycleBarBuilder:
    """Custom axis: SPECTRAL / dominant-CYCLE clock. Maintain a causal band-pass oscillator
    bp = EMA_fast(logprice) - EMA_slow(logprice) (a MACD-type filter that isolates one frequency
    band of the price path); emit a bar at each ZERO-CROSSING of bp — i.e. each completed half-cycle
    of the dominant oscillation in that band. DENSE during oscillation / regime churn, SPARSE during
    clean trends: the spectral DUAL of the magnitude/flow clocks. Built for regime-OSCILLATING assets
    (UUP/dollar) whose edge lives in the cycle, not the drift. The fast EMA span is fit on TRAIN by
    simulation (causal, see _fit_spectral_axis); slow = 4*fast. The builder consumes only the frozen
    span (stored as .thresh so builder_threshold/verify-replay work uniformly).
    """

    _SLOW_RATIO = 4.0

    def __init__(self, threshold):
        self.thresh = float(threshold)                  # fast EMA span (uniform 'thresh' slot)
        slow = self.thresh * self._SLOW_RATIO
        self.af = 2.0 / (self.thresh + 1.0)
        self.as_ = 2.0 / (slow + 1.0)
        self.ef = None
        self.es = None
        self.last_bp = 0.0
        self.started = False
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        self.close_lc = lc
        if self.ef is None:
            self.ef = lc
            self.es = lc
            return None
        self.ef += self.af * (lc - self.ef)
        self.es += self.as_ * (lc - self.es)
        bp = self.ef - self.es
        crossed = (bp > 0 and self.last_bp <= 0) or (bp < 0 and self.last_bp >= 0)
        prev = self.last_bp
        self.last_bp = bp
        if bp != 0.0:
            self.started = True
        if crossed and self.started and prev != 0.0:
            return {"ts_close": ts, "log_close": lc}
        return None


class VpinBarBuilder:
    """Custom axis: VPIN / bulk-volume-classified order-flow TOXICITY clock (Easley, Lopez de
    Prado & O'Hara 2012 'Flow Toxicity and Liquidity'; AFML Ch.19.5.2 + Ch.2.3.2). Mined
    2026-06-03 as the AFML information-bar method we had MISSED.

    Unlike the imbalance axes (`imbalance`/`tickimb`/`volumeimb`) that classify each minute's
    volume 100%/0% buy-or-sell by the HARD tick rule (sign of the price change), Bulk Volume
    Classification splits it SOFTLY: buy-fraction f = Phi(z), z = standardized minute log-return,
    Phi = standard-normal CDF. A +0.3-sigma minute is ~62% buy / 38% sell, so the accumulator
    weights flow by HOW one-sided the move was, not by a binary sign. Accumulate the one-sided
    imbalance theta += vol*|2f-1| (the VPIN numerator) and emit a bar when theta >= threshold.

    sigma is an ONLINE TRAILING std of recent minute returns (causal ring buffer) so the builder
    needs ONLY the scalar threshold -> BUILDER_CLASSES-compatible (online-verifiable), no second
    fitted param. Phi is parameter-free (math.erf). The threshold is TRAIN-fit in _make_builder.
    A genuinely new functional class (soft probabilistic classification clock), not a reweighting
    of the existing hard-sign accumulators.
    """

    _SW = 240   # trailing minutes for the online sigma (BVC scale)

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.theta = 0.0
        self.last_lc = None
        self.close_lc = None
        self._rs = []          # trailing minute log-returns ring for online sigma
        self._ss = 0.0         # running SUM OF SQUARES of _rs (O(1) rolling variance — recomputing
        #                        sum(x*x) every minute over ~1M minutes timed QC out; this is the fix)

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            self.last_lc = None      # break the return chain across invalid prints
            return None
        lc = math.log(close)
        if self.last_lc is not None:
            r = lc - self.last_lc
            self._rs.append(r)
            self._ss += r * r
            if len(self._rs) > self._SW:
                old = self._rs.pop(0)
                self._ss -= old * old
            if len(self._rs) >= 20:
                sigma = math.sqrt(self._ss / len(self._rs)) if self._ss > 0.0 else 0.0
                if sigma > 1e-12:
                    z = r / sigma
                    f = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))   # Phi(z): BVC buy fraction
                    self.theta += vol * abs(2.0 * f - 1.0)
        self.last_lc = lc
        self.close_lc = lc
        if self.theta >= self.thresh:
            self.theta = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


class JumpBarBuilder:
    """Custom axis: Lee-Mykland JUMP-DETECTION clock (Lee & Mykland 2008, Rev. Financial Studies). Mined
    2026-06-04. Emit a bar ONLY when a statistically-significant JUMP occurs — when the current return,
    standardized by a LOCAL bipower-variation volatility, exceeds a threshold (|r|/sigma_bipower >= thresh).
    Bars therefore CLUSTER at the explosive / discontinuous moves (the dollar's 2014-15/2022 surges that
    sadf showed carry UUP's edge) and are silent during diffusive chop — a fundamentally different sampling
    than the volume/flow/info clocks. Bipower vol = (pi/2)*mean(|r_{j-1}||r_j|) over a trailing window
    (jump-robust). O(1) per minute (running product-sum ring — the vpin O(n) timeout lesson). The threshold
    is a TRAIN quantile of |L| (in _make_builder) targeting ~target_bars -> scalar -> BUILDER_CLASSES-compatible.
    """

    _K = 60   # trailing window for the bipower local-vol estimate

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.last_lc = None
        self.close_lc = None
        self._prev_ar = None     # previous |log-return|
        self._prods = []         # ring of consecutive products |r_{j-1}|*|r_j|
        self._psum = 0.0         # running sum of _prods (O(1) bipower)

    def update(self, ts, close, vol):
        if close <= 0:
            self.last_lc = None      # break the return chain across invalid prints
            return None
        lc = math.log(close)
        emit = None
        if self.last_lc is not None:
            ar = abs(lc - self.last_lc)
            if self._prev_ar is not None and len(self._prods) >= 20:
                bv = (math.pi / 2.0) * self._psum / len(self._prods)
                sigma = math.sqrt(bv) if bv > 0.0 else 0.0
                if sigma > 1e-12 and ar / sigma >= self.thresh:
                    emit = {"ts_close": ts, "log_close": lc}
            if self._prev_ar is not None:
                p = self._prev_ar * ar
                self._prods.append(p)
                self._psum += p
                if len(self._prods) > self._K:
                    self._psum -= self._prods.pop(0)
            self._prev_ar = ar
        self.last_lc = lc
        self.close_lc = lc
        return emit


class VolOfVolBarBuilder:
    """Custom axis: VOL-OF-VOL clock (second-order volatility / volatility-of-volatility). Mined
    2026-06-04. Every built axis is FIRST-order: vol/logdollar weight the variance LEVEL, jump
    standardizes a single return, kyle/imbalance/vpin clock impact/flow. NONE clocks the RATE-OF-
    CHANGE of volatility. This axis emits a bar when the cumulative absolute change in (log) spot
    variance crosses a threshold, so bars CLUSTER at vol REPRICINGS — the minutes where the
    volatility regime is turning — and stay silent while vol drifts (whether vol is high or low).

    Spot variance is estimated jump-robustly with the Lee-Mykland BIPOWER form sigma2 =
    (pi/2)*mean(|r_{j-1}||r_j|) over a trailing window (K=60); a single isolated jump inflates one
    product but is washed out by the window mean, so a lone jump does NOT masquerade as vol-of-vol
    (this is what separates it from the `jump` axis, which fires ON the jump itself). The
    accumulator sums |v - prev_v| where v = log(sigma2) (log so equal MULTIPLICATIVE vol changes
    weigh equally, scale-free across the dollar's calm/stress regimes).

    O(1) per minute: the bipower mean is kept with a running product-sum ring (the vpin/jump O(n)
    timeout lesson), and v / |dv| are single scalars. The builder needs ONLY the scalar threshold
    -> BUILDER_CLASSES-compatible (online byte-exact replay), no second fitted param. The threshold
    is a rate-then-scale TRAIN fit (mean per-minute |dv| over TRAIN minutes x full length /
    target_bars) in _make_builder — OOS-invariant. UUP regime turns are preceded by vol repricing,
    so this over-samples regime-onset minutes the size/flow clocks reach late ('sample where the
    edge RESOLVES'). A genuinely second-order signal, not a reweighting of the first-order axes.
    """

    _K = 60   # trailing window for the bipower spot-variance estimate (matches the jump axis)

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.last_lc = None
        self.close_lc = None
        self.cum = 0.0
        self._prev_ar = None     # previous |log-return|
        self._prods = []         # ring of consecutive products |r_{j-1}|*|r_j| (bipower)
        self._psum = 0.0         # running sum of _prods (O(1) bipower mean)
        self._prev_v = None      # previous v = log(bipower spot variance)

    def update(self, ts, close, vol):
        if close <= 0:
            self.last_lc = None      # break the return chain across invalid prints
            return None
        lc = math.log(close)
        emit = None
        if self.last_lc is not None:
            ar = abs(lc - self.last_lc)
            if self._prev_ar is not None and len(self._prods) >= 20:
                bv = (math.pi / 2.0) * self._psum / len(self._prods)   # bipower spot variance
                if bv > 1e-300:
                    v = math.log(bv)
                    if self._prev_v is not None:
                        self.cum += abs(v - self._prev_v)   # vol-of-vol increment (|dlog sigma2|)
                    self._prev_v = v
            if self._prev_ar is not None:
                p = self._prev_ar * ar
                self._prods.append(p)
                self._psum += p
                if len(self._prods) > self._K:
                    self._psum -= self._prods.pop(0)
            self._prev_ar = ar
        self.last_lc = lc
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            emit = {"ts_close": ts, "log_close": lc}
        return emit


class WaveletBarBuilder:
    """Custom axis: WAVELET multi-resolution ENERGY clock (a-trous / undecimated causal Haar
    cascade, J=3). Mined 2026-06-04. Every built spectral-family axis is SINGLE-band: `spectral`
    is a one-band EMA-difference ZERO-CROSSING timing clock and `vol`/`volofvol` clock a SINGLE
    realised-variance scale. NONE decomposes the path into MULTIPLE resolutions. This axis runs the
    a-trous (undecimated, shift-invariant) wavelet transform online and emits a bar when the
    cumulative COARSE-SCALE detail ENERGY crosses a threshold, so bars CLUSTER at moves on the
    trend/swing scale (period ~2^J minutes) and stay silent during fine high-frequency chop — the
    multi-resolution DUAL of the magnitude/flow clocks, an orthogonal-ADD inside GLD's own logdollar
    TREND mechanism ('sample where the trend/swing resolves, not the high-frequency chop').

    a-trous cascade on the log-price s[t] = log(close[t]):
        A_0[t] = s[t]
        A_j[t] = 0.5 * (A_{j-1}[t] + A_{j-1}[t - 2^(j-1)])     for j = 1 .. J
        D_J    = A_{J-1} - A_J            (coarsest detail band)
    Only CURRENT/PAST taps (lags 1, 2, 4 for J=3) enter, so it is STRICTLY causal — the decimated
    DWT would need future samples and would leak; the undecimated a-trous does NOT. The accumulator
    sums D_J^2 (coarse-band energy). Past taps are held in a fixed-depth A-history ring per level
    (max depth 2^(J-1) = 4 floats), so the transform is O(J) = O(3) per minute — a single scalar,
    no per-minute O(window) recompute (the vpin/jump O(n) timeout lesson). The builder needs ONLY
    the scalar threshold -> BUILDER_CLASSES-compatible (online byte-exact replay), no second fitted
    param; the threshold is a rate-then-scale TRAIN fit (mean per-minute D_J^2 over TRAIN minutes x
    full length / target_bars) in _make_builder — OOS-invariant. A genuinely multi-resolution signal,
    not a reweighting of the first-order or single-band axes.
    """

    _J = 3   # cascade depth; coarse band ~ 2^J-minute swing scale

    def __init__(self, threshold):
        self.thresh = float(threshold)
        self.cum = 0.0
        self.close_lc = None
        # _hist[j] = recent A_j values (the INPUT to cascade level j+1), held to lag 2^j.
        self._hist = [[] for _ in range(self._J)]
        self._lag = [1 << j for j in range(self._J)]   # taps at 1, 2, 4

    def _detail_sq(self, lc):
        """Coarse-band detail energy D_J^2 at this minute; advances the A-history rings.
        Pure scalar arithmetic on current/past taps only (strictly causal). The SAME method
        path is used in _make_builder calibration, so the fit matches the replay bit-for-bit."""
        a_in = lc                  # A_0[t]
        a_jm1 = lc                 # becomes A_{J-1}[t] (input to the last level)
        a_j = lc                   # becomes A_J[t] (output of the last level)
        for j in range(self._J):
            buf = self._hist[j]
            lag = self._lag[j]
            if len(buf) >= lag:
                a_lag = buf[-lag]
            elif buf:
                a_lag = buf[0]      # warm-up: clamp to earliest known tap (deterministic)
            else:
                a_lag = a_in
            a_out = 0.5 * (a_in + a_lag)    # A_{j+1}[t] = 0.5(A_j[t] + A_j[t-2^j])
            buf.append(a_in)
            if len(buf) > lag:
                buf.pop(0)
            a_jm1 = a_in
            a_j = a_out
            a_in = a_out
        d_j = a_jm1 - a_j           # D_J = A_{J-1} - A_J
        return d_j * d_j

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        self.close_lc = lc
        self.cum += self._detail_sq(lc)
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": lc}
        return None


# Registry — names must be EXACTLY these and in this order.
# ----------------------------------------------------------------------------
_AXES_ORDER = ["dollar", "tick", "vol", "range", "logdollar", "entropy", "imbalance", "tickimb", "volumeimb", "fracdiff", "dc", "zcusum", "kyle", "run", "spectral", "vpin", "jump", "volofvol", "wavelet"]
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
    "zcusum": ZCusumBarBuilder,
    "kyle": KyleImpactBarBuilder,
    "run": RunBarBuilder,
    "spectral": SpectralCycleBarBuilder,
    "vpin": VpinBarBuilder,
    "jump": JumpBarBuilder,
    "volofvol": VolOfVolBarBuilder,
    "wavelet": WaveletBarBuilder,
}


# ----------------------------------------------------------------------------
# Threshold auto-calibration helpers (run on the minute stream, pre-build).
# ----------------------------------------------------------------------------
def _safe_thresh(total, target_bars):
    """Threshold that yields ~target_bars bars from a positive accumulator total."""
    tb = max(1, int(target_bars))
    return (total / tb) if total > 0 else 1e-9


def _train_minute_mask(ts_arr):
    """Boolean mask of minutes strictly before TRAIN_END (the header global).

    FAILS LOUD if TRAIN_END is unavailable. The old silent "all minutes are TRAIN"
    fallback would fit bar thresholds on the FULL series (incl. OOS) — a look-ahead
    leak. The G3 invariant requires bar thresholds be TRAIN-only, so we refuse to
    proceed without TRAIN_END rather than silently leak.
    """
    try:
        cutoff = np.datetime64(TRAIN_END)  # noqa: F821 — global from header
    except Exception as e:
        raise RuntimeError("TRAIN_END unavailable in _train_minute_mask — refusing the "
                           "full-series fallback (it would leak OOS into bar thresholds)") from e
    ts64 = np.array([np.datetime64(str(t)) for t in ts_arr])
    return ts64 < cutoff


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


def _fit_run_axis(close, vol, ts_arr, target_bars):
    """TRAIN-fit the run-bar magnitude threshold by simulation (causal). Accumulate
    |log-return| WITHIN a directional run (reset on sign flip); emit when the within-run
    accumulation >= thresh. Reset-on-flip wastes sub-threshold run magnitude, so there is no
    clean closed form (unlike the imbalance axes) -> binary-search the threshold on TRAIN
    minutes so the TRAIN emission count matches the TRAIN share of target_bars. Fit on TRAIN
    only; applied forward unchanged."""
    c = np.asarray(close, dtype=float)
    ret = _minute_log_returns(c)
    tr = _train_minute_mask(ts_arr)
    r = ret[tr & np.isfinite(ret)]
    if len(r) < 200:
        return None
    sigma = float(np.std(r))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    target_train = max(50.0, float(target_bars) * (len(r) / max(1, len(c))))
    ar = np.abs(r)
    sr = np.sign(r)

    def emit_count(thresh):
        theta = 0.0
        sign = 0.0
        n = 0
        for i in range(len(r)):
            s = sr[i]
            if s == 0.0:
                s = sign
            if s != sign and sign != 0.0:
                theta = ar[i]
            else:
                theta += ar[i]
            if s != 0.0:
                sign = s
            if theta >= thresh:
                theta = 0.0
                n += 1
        return n

    lo, hi = sigma * 0.5, sigma * 300.0
    for _ in range(22):                       # binary search: higher thresh -> fewer bars
        mid = 0.5 * (lo + hi)
        if emit_count(mid) > target_train:
            lo = mid
        else:
            hi = mid
    thresh = 0.5 * (lo + hi)
    if not np.isfinite(thresh) or thresh <= 0:
        return None
    return thresh


def _fit_spectral_axis(close, vol, ts_arr, target_bars):
    """TRAIN-fit the spectral clock's fast EMA span by simulation (causal). Count zero-crossings of
    the band-pass oscillator bp = EMA_fast - EMA_4*fast over TRAIN log-prices; a larger span = slower
    oscillator = FEWER crossings, so binary-search the span so the TRAIN crossing count matches the
    TRAIN share of target_bars. The span (a frequency-band choice) is the only fitted param and is
    frozen forward — fit on TRAIN minutes only, applied unchanged (causal)."""
    c = np.asarray(close, dtype=float)
    tr = _train_minute_mask(ts_arr)
    lc = np.log(c[tr & (c > 0)])
    if len(lc) < 200:
        return None
    target_train = max(50.0, float(target_bars) * (len(lc) / max(1, len(c))))

    def cross_count(fast):
        slow = fast * SpectralCycleBarBuilder._SLOW_RATIO
        af = 2.0 / (fast + 1.0)
        a_s = 2.0 / (slow + 1.0)
        ef = lc[0]
        es = lc[0]
        last = 0.0
        started = False
        n = 0
        for i in range(1, len(lc)):
            ef += af * (lc[i] - ef)
            es += a_s * (lc[i] - es)
            bp = ef - es
            crossed = (bp > 0 and last <= 0) or (bp < 0 and last >= 0)
            if crossed and started and last != 0.0:
                n += 1
            if bp != 0.0:
                started = True
            last = bp
        return n

    lo, hi = 2.0, max(16.0, (len(lc) / max(1.0, target_train)) * 8.0)
    for _ in range(20):                       # binary search: larger span -> fewer bars
        mid = 0.5 * (lo + hi)
        if cross_count(mid) > target_train:
            lo = mid
        else:
            hi = mid
    fast = 0.5 * (lo + hi)
    if not np.isfinite(fast) or fast < 1.0:
        return None
    return fast


def _make_builder(bar_type, close, vol, ts_arr, target_bars):
    """Instantiate the right Builder with an auto-calibrated threshold."""
    if bar_type == "dollar":
        # Threshold = TRAIN average per-minute notional x full length / target_bars.
        # Fit the rate on TRAIN minutes only (causal) and scale by the (benign) full
        # minute count so we still target ~target_bars bars. Using the full-series
        # SUM here would leak OOS notional into the OOS bar boundaries.
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        tr = _train_minute_mask(ts_arr)
        dvm = c * v
        keep = tr & np.isfinite(dvm) & (dvm > 0)
        if not np.any(keep):
            return None
        total = float(np.mean(dvm[keep])) * len(c)        # TRAIN rate x full length
        return DollarBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "tick":
        return TickBarBuilder(max(1, len(close) // max(1, int(target_bars))))

    if bar_type == "range":
        # Equal-price-move bars: tighter band when we want more (smaller) bars.
        thresh = 0.003 if int(target_bars) < 10000 else 0.002
        return RangeBarBuilder(thresh)

    if bar_type == "logdollar":
        # TRAIN average per-minute log-notional x full valid count / target_bars.
        # Rate fit on TRAIN only (causal); a full-series SUM would leak OOS notional.
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        dv = c * v
        valid = (c > 0) & (v > 0)
        tr = _train_minute_mask(ts_arr)
        keep = tr & valid
        if not np.any(keep):
            return None
        # LEAK FIX (2026-06-03 adversarial audit): the count multiplier must be OOS-INVARIANT yet
        # preserve the intended bar coarseness. int(np.sum(valid)) (full-series valid count) LEAKED OOS
        # validity; plain len(c) (all minutes incl. invalid) is OOS-invariant but OVER-counts when many
        # minutes are invalid, changing coarseness. Correct fix: extrapolate the TRAIN valid DENSITY to
        # the full length — full_valid_est = (TRAIN valid / TRAIN minutes) * len(c). TRAIN-only (no OOS),
        # and == np.sum(valid) when the valid fraction is stable across the split.
        _trc = max(1, int(np.sum(tr)))
        total = float(np.mean(np.log1p(dv[keep]))) * (int(np.sum(keep)) * len(c) / _trc)
        return LogDollarBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "zcusum":
        # Standardized-cumsum CUSUM clock (Wang's LogDollar). z is unit-variance, so a
        # symmetric CUSUM crosses ~ every T^2 minutes -> T = sqrt(minutes/bar) targets
        # ~target_bars. Count-only threshold (causal, no fit); rolling window is trailing.
        m = max(1.0, len(close) / max(1, int(target_bars)))
        return ZCusumBarBuilder(math.sqrt(m))

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

    if bar_type == "kyle":
        # Price-impact clock. Per-minute impact = |log-return| / sqrt(volume). Threshold =
        # TRAIN average impact x full valid count / target_bars (same rate-on-TRAIN-then-scale
        # recipe as dollar/logdollar; a full-series SUM would leak OOS impact into bar bounds).
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = np.abs(ret) / np.sqrt(np.where(v > 0, v, np.nan))
        valid = (c > 0) & (v > 0) & np.isfinite(imp)
        keep = tr & valid
        if not np.any(keep):
            return None
        # LEAK FIX (2026-06-03 adversarial audit): OOS-invariant TRAIN-valid-density extrapolation
        # (not int(np.sum(valid)) which leaked OOS validity, nor plain len(c) which changes coarseness). See logdollar.
        _trc = max(1, int(np.sum(tr)))
        total = float(np.mean(imp[keep])) * (int(np.sum(keep)) * len(c) / _trc)
        return KyleImpactBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "vpin":
        # VPIN / bulk-volume-classification toxicity clock. Per-minute BVC imbalance =
        # vol*|2*Phi(z)-1|, z = return / online-trailing-sigma (the SAME causal rolling sigma the
        # builder uses at runtime, so calibration matches replay). Threshold = TRAIN mean of that
        # per-minute imbalance x full length / target_bars (rate-on-TRAIN-then-scale, OOS-invariant;
        # a full-series SUM would leak OOS flow into bar bounds). sigma is causal (trailing only).
        c = np.asarray(close, dtype=float)
        v = np.asarray(vol, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        terms = np.full(len(c), np.nan)
        _rs = []
        _ss = 0.0          # O(1) rolling sum-of-squares (matches the builder; avoids O(n*240) timeout)
        for i in range(len(c)):
            if not np.isfinite(ret[i]) or c[i] <= 0 or v[i] <= 0:
                continue
            ri = float(ret[i])
            _rs.append(ri)
            _ss += ri * ri
            if len(_rs) > 240:
                _old = _rs.pop(0)
                _ss -= _old * _old
            if len(_rs) >= 20:
                sg = math.sqrt(_ss / len(_rs)) if _ss > 0.0 else 0.0
                if sg > 1e-12:
                    z = ri / sg
                    f = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    terms[i] = float(v[i]) * abs(2.0 * f - 1.0)
        keep = tr & np.isfinite(terms)
        if not np.any(keep):
            return None
        total = float(np.mean(terms[keep])) * len(c)       # TRAIN rate x full length (OOS-invariant)
        return VpinBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "jump":
        # Lee-Mykland JUMP clock. Standardize each return by an O(1) trailing bipower vol; |L|=|r|/sigma.
        # The emission threshold is the TRAIN quantile of |L| at (1 - target_bars/len) so ~target_bars of
        # the most jump-like minutes emit. TRAIN-only distribution (leak-safe); a single scalar (the
        # builder recomputes sigma online, identical to replay). matches the JumpBarBuilder math exactly.
        c = np.asarray(close, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        Lstat = np.full(len(c), np.nan)
        _prev = None
        _prods = []
        _psum = 0.0
        for i in range(len(c)):
            if not np.isfinite(ret[i]) or c[i] <= 0:
                continue
            ar = abs(float(ret[i]))
            if _prev is not None and len(_prods) >= 20:
                bv = (math.pi / 2.0) * _psum / len(_prods)
                sg = math.sqrt(bv) if bv > 0.0 else 0.0
                if sg > 1e-12:
                    Lstat[i] = ar / sg
            if _prev is not None:
                p = _prev * ar
                _prods.append(p)
                _psum += p
                if len(_prods) > 60:
                    _psum -= _prods.pop(0)
            _prev = ar
        keep = tr & np.isfinite(Lstat)
        if int(np.sum(keep)) < 200:
            return None
        q = max(0.0, min(0.999, 1.0 - float(target_bars) / max(1, len(c))))
        thresh = float(np.quantile(Lstat[keep], q))         # TRAIN-quantile jump threshold (OOS-invariant)
        if not np.isfinite(thresh) or thresh <= 0:
            return None
        return JumpBarBuilder(thresh)

    if bar_type == "volofvol":
        # VOL-OF-VOL clock. Per-minute term = |v - prev_v|, v = log(bipower spot variance) over an
        # O(1) trailing ring (the SAME math the builder replays online). Threshold = TRAIN mean of that
        # per-minute |dv| x full length / target_bars (rate-on-TRAIN-then-scale, OOS-invariant; a
        # full-series SUM would leak OOS vol-of-vol into bar bounds). The bipower estimate is causal
        # (trailing products only). A single scalar -> BUILDER_CLASSES-compatible; the builder
        # recomputes v online identically, so calibration matches replay byte-for-byte.
        c = np.asarray(close, dtype=float)
        ret = _minute_log_returns(c)
        tr = _train_minute_mask(ts_arr)
        terms = np.full(len(c), np.nan)
        _prev = None
        _prods = []
        _psum = 0.0
        _prev_v = None
        for i in range(len(c)):
            if not np.isfinite(ret[i]) or c[i] <= 0:
                continue
            ar = abs(float(ret[i]))
            if _prev is not None and len(_prods) >= 20:
                bv = (math.pi / 2.0) * _psum / len(_prods)
                if bv > 1e-300:
                    v = math.log(bv)
                    if _prev_v is not None:
                        terms[i] = abs(v - _prev_v)
                    _prev_v = v
            if _prev is not None:
                p = _prev * ar
                _prods.append(p)
                _psum += p
                if len(_prods) > 60:
                    _psum -= _prods.pop(0)
            _prev = ar
        keep = tr & np.isfinite(terms)
        if int(np.sum(keep)) < 200:
            return None
        total = float(np.mean(terms[keep])) * len(c)       # TRAIN rate x full length (OOS-invariant)
        return VolOfVolBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "wavelet":
        # WAVELET multi-resolution energy clock (a-trous causal Haar, J=3). Per-minute term =
        # D_J^2, the coarse-band detail energy, computed by REPLAYING the builder's own _detail_sq
        # method over the minute stream (the SAME code path the builder runs online -> the fit
        # matches the replay bit-for-bit; no second numpy reimplementation that could drift).
        # Threshold = TRAIN mean of that per-minute D_J^2 x full length / target_bars
        # (rate-on-TRAIN-then-scale, OOS-invariant; a full-series SUM would leak OOS energy into bar
        # bounds). The a-trous taps are current/past only (lags 1,2,4) so the term is strictly
        # causal. A single scalar -> BUILDER_CLASSES-compatible (online byte-exact replay).
        c = np.asarray(close, dtype=float)
        tr = _train_minute_mask(ts_arr)
        _wb = WaveletBarBuilder(1e300)            # sentinel thresh: never resets cum; pure term reader
        terms = np.full(len(c), np.nan)
        for i in range(len(c)):
            ci = float(c[i])
            if ci <= 0:
                continue
            terms[i] = _wb._detail_sq(math.log(ci))   # identical method path to the online builder
        keep = tr & np.isfinite(terms)
        if int(np.sum(keep)) < 200:
            return None
        total = float(np.mean(terms[keep])) * len(c)       # TRAIN rate x full length (OOS-invariant)
        return WaveletBarBuilder(_safe_thresh(total, target_bars))

    if bar_type == "run":
        thresh = _fit_run_axis(close, vol, ts_arr, target_bars)
        if thresh is None:
            return None
        return RunBarBuilder(thresh)

    if bar_type == "spectral":
        fast = _fit_spectral_axis(close, vol, ts_arr, target_bars)
        if fast is None:
            return None
        return SpectralCycleBarBuilder(fast)

    # Default / unknown -> volatility bar (Wang's workhorse axis).
    # Threshold = TRAIN average per-minute vol term x full count / target_bars.
    # The per-minute realized-vol term is averaged over TRAIN minutes only (causal);
    # summing it over the full series would leak OOS volatility into bar boundaries.
    c = np.asarray(close, dtype=float)
    v = np.asarray(vol, dtype=float)
    tr = _train_minute_mask(ts_arr)
    terms = np.full(len(c), np.nan)
    last_lc = None
    for i in range(len(c)):
        if c[i] <= 0 or v[i] <= 0:
            last_lc = None
            continue
        lc_val = math.log(c[i])
        if last_lc is not None:
            terms[i] = (lc_val - last_lc) ** 2 * math.sqrt(v[i])
        last_lc = lc_val
    keep = tr & np.isfinite(terms)
    if not np.any(keep):
        return None
    total = float(np.mean(terms[keep])) * len(c)       # TRAIN rate x full length
    return VolBarBuilder(_safe_thresh(total, target_bars))


# Scalar-threshold builder classes by axis name — lets a verification run rebuild
# bars ONLINE from a SAVED frozen threshold (no _make_builder recompute needed).
# entropy/fracdiff omitted: they also need fitted params (edges/probs/T, weights/d).
BUILDER_CLASSES = {
    "dollar": DollarBarBuilder, "tick": TickBarBuilder, "vol": VolBarBuilder,
    "range": RangeBarBuilder, "logdollar": LogDollarBarBuilder,
    "imbalance": DollarImbalanceBarBuilder, "tickimb": TickImbalanceBarBuilder,
    "volumeimb": VolumeImbalanceBarBuilder, "dc": DirectionalChangeBarBuilder,
    "zcusum": ZCusumBarBuilder, "kyle": KyleImpactBarBuilder,
    "run": RunBarBuilder, "spectral": SpectralCycleBarBuilder,
    "vpin": VpinBarBuilder, "jump": JumpBarBuilder,
    "volofvol": VolOfVolBarBuilder, "wavelet": WaveletBarBuilder,
}


def builder_threshold(b):
    """The frozen scalar threshold of a built builder (uniform across classes).
    try/except attribute access (not getattr) so this file is lint-clean as a STANDALONE
    QC project module — QC's cloud compiler treats the getattr 'discouraged' warning as a
    build error for non-main files (it was a tolerated warning when concatenated)."""
    try:
        return b.thresh
    except AttributeError:
        pass
    try:
        return b.thresh_pct
    except AttributeError:
        pass
    try:
        return b.delta
    except AttributeError:
        return None


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
