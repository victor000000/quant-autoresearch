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
_AXES_ORDER = ["dollar", "tick", "vol", "range", "logdollar", "entropy"]
AXES = {
    "dollar": DollarBarBuilder,
    "tick": TickBarBuilder,
    "vol": VolBarBuilder,
    "range": RangeBarBuilder,
    "logdollar": LogDollarBarBuilder,
    "entropy": EntropyBarBuilder,
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
        bar_type: one of ["dollar","tick","vol","range","logdollar","entropy"].
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
