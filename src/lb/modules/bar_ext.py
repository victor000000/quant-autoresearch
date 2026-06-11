"""bar_ext — bar-builder EXTENSION module (separate QC project file).

bar_builder.py's minified render sits ~1k under QC's 64,000-byte/file budget, so new
axes land HERE (the "split further" the 2026-06-06 backlog note anticipated). Same
rules as bar_builder.py: top-level imports ONLY numpy/math; no sibling imports —
anything needed from bar_builder is passed IN as an argument; lint-clean (no getattr,
no nested-quote f-strings); every threshold fit TRAIN-only + extrapolated causally.

Contents:
  logdollar_rc — Wang's DE-SCALED notional clock (2026-06-10, R1868+). Plain logdollar
  freezes ONE TRAIN-fit threshold, so when notional levels drift over years the bar
  duration drifts with them (Wang: "3bn turnover today is not comparable to 3bn five
  years ago" — the axis itself drifts). Here the COARSENESS target (minutes-per-bar)
  is TRAIN-fit with the exact leak-fixed logdollar recipe, while the threshold LEVEL
  re-scales at each bar emission from a strictly-TRAILING window of per-minute
  accumulation — bars keep ~constant information density across notional regimes.
  Causal by construction: the threshold in force for a bar was set at the PREVIOUS
  emission from minutes strictly before it; appending OOS data can never change
  earlier bars (append-OOS-invariant, guarded by tests/test_bar_threshold_leak.py).
"""
import math

import numpy as np


class LogDollarRCBarBuilder:
    """logdollar accumulator with a rolling-causal (de-scaled) emission threshold."""

    def __init__(self, mins_per_bar, warm_thresh, win_minutes=23400):
        # 23400 minutes ~= 60 trading days of US RTH — long enough to be stable,
        # short enough to track multi-month notional regime shifts.
        self.mpb = float(mins_per_bar)        # TRAIN-fit coarseness target (dimensionless)
        self.win = max(100, int(win_minutes))
        self.ring = [0.0] * self.win          # trailing per-minute contributions (O(1) ring)
        self.head = 0
        self.count = 0
        self.rsum = 0.0
        self.min_fill = max(1000, self.win // 10)
        self.cum = 0.0
        self.thresh = max(1e-9, float(warm_thresh))   # TRAIN-fit warmup threshold
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        x = math.log1p(close * vol)
        self.close_lc = math.log(close)
        self.cum += x
        # The emission decision uses the threshold set at a PREVIOUS emission — the
        # current minute can never influence its own bar boundary.
        emit = self.cum >= self.thresh
        old = self.ring[self.head]
        self.ring[self.head] = x
        self.head = (self.head + 1) % self.win
        if self.count < self.win:
            self.count += 1
            self.rsum += x
        else:
            self.rsum += x - old
        if emit:
            self.cum = 0.0
            if self.count >= self.min_fill:
                # de-scaled threshold: trailing per-minute rate x fixed coarseness
                self.thresh = max(1e-9, (self.rsum / self.count) * self.mpb)
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


def make_logdollar_rc(close, vol, ts_arr, target_bars, train_minute_mask):
    """Calibrate + instantiate. Mirrors the leak-fixed logdollar recipe exactly:
    rate fit on TRAIN minutes only; full count via TRAIN valid-DENSITY extrapolation
    (never np.sum(valid) over the full series, which leaked OOS validity)."""
    c = np.asarray(close, dtype=float)
    v = np.asarray(vol, dtype=float)
    dv = c * v
    valid = (c > 0) & (v > 0)
    tr = train_minute_mask(ts_arr)
    keep = tr & valid
    if not np.any(keep):
        return None
    trc = max(1, int(np.sum(tr)))
    full_valid_est = int(np.sum(keep)) * len(c) / trc
    rate_tr = float(np.mean(np.log1p(dv[keep])))
    mins_per_bar = max(1.0, full_valid_est / max(1.0, float(target_bars)))
    warm = max(1e-9, rate_tr * mins_per_bar)
    return LogDollarRCBarBuilder(mins_per_bar, warm)


class Session2BarBuilder:
    """SESSION-ANCHORED clock (Wang frontier #4, SPY session momentum, 2026-06-10):
    exactly two bars per session — the 15:30 bar (NY minute-of-day 930) and the
    16:00 close bar (960). A 1-bar-forward label at the 15:30 bar is the LAST-HALF-
    HOUR move (the Gao-Han-Li-Zhou target); trailing features at 15:30 carry the
    morning return (the documented predictor). No fitted parameters at all —
    pure wall-clock anchors, append-OOS-invariant by construction."""

    def __init__(self):
        self.prev_mod = None
        self.prev_day = None
        self.fired_1530 = False
        self.fired_eod = False
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        t64 = np.datetime64(ts, "m")
        mod = int(t64.astype(np.int64) % 1440)          # minute-of-day (exchange-local stamps)
        day = t64.astype("datetime64[D]")
        if self.prev_day is None or day != self.prev_day:
            self.prev_day = day
            self.fired_1530 = False
            self.fired_eod = False
        self.close_lc = math.log(close)
        emit = False
        if not self.fired_1530 and mod >= 930 and mod < 960:
            self.fired_1530 = True
            emit = True
        elif not self.fired_eod and mod >= 960:
            self.fired_eod = True
            emit = True
        self.prev_mod = mod
        if emit:
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


def make_session2(close, vol, ts_arr, target_bars, train_minute_mask):
    """No calibration — the clock is pure wall-time anchors."""
    return Session2BarBuilder()


class GapFlowBarBuilder:
    """OVERNIGHT-GAP-WEIGHTED variance clock (gapflow, 2026-06-10 invention round).

    The only clock that treats the close-to-open gap as a DISTINCT information
    event: within a session the accumulator advances by (dlc)^2 like a variance
    clock; at a day rollover it advances by kappa*(gap)^2, where kappa re-weights
    overnight variance to its TRAIN-fit share of intraday variance. Asset physics:
    international-equity ETFs' NAV information arrives while the US is closed —
    their home markets trade in the gap (DXJ val_auc 0.71 on a gap-blind clock).
    kappa and T are both TRAIN-only-fit constants (append-OOS-invariant); the
    builder is O(1), causal, zero forward reach. close<=0 breaks the return chain
    (semivar convention). NOT in BUILDER_CLASSES (two fitted params, no online-
    verify scalar)."""

    def __init__(self, kappa, thresh):
        self.kappa = float(kappa)
        self.thresh = max(1e-12, float(thresh))
        self.cum = 0.0
        self.last_lc = None
        self.prev_day = None
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            self.last_lc = None          # break the return chain (semivar convention)
            return None
        day = np.datetime64(ts, "D")
        lc = math.log(close)
        self.close_lc = lc
        if self.last_lc is not None:
            d = lc - self.last_lc
            if self.prev_day is not None and day != self.prev_day:
                self.cum += self.kappa * d * d          # overnight gap event
            else:
                self.cum += d * d                       # within-session variance
        self.last_lc = lc
        self.prev_day = day
        if self.cum >= self.thresh:
            # CARRY the remainder (do not zero): a large gap is several bars' worth of
            # information — zeroing truncated ~60% of gap information in the sanity run
            # (240/600 bars). The TRAIN-replay calibration is rate-based, so carrying
            # keeps realized coarseness at target. One emission per minute max.
            self.cum -= self.thresh
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


def make_gapflow(close, vol, ts_arr, target_bars, train_minute_mask):
    """TRAIN-only fit of (kappa, T) by replaying the increment path on TRAIN minutes;
    T uses the sanctioned leak-fixed density recipe (mean TRAIN increment x TRAIN
    valid-density extrapolation / target_bars)."""
    c = np.asarray(close, dtype=float)
    tr = train_minute_mask(ts_arr)
    days = np.asarray(ts_arr).astype("datetime64[D]")
    n = len(c)
    intr = 0.0
    gap2 = 0.0
    n_inc = 0
    sum_intr_inc = 0.0
    last_lc = None
    prev_day = None
    for i in range(n):
        if not tr[i]:
            break                                        # TRAIN minutes are a prefix
        if c[i] <= 0:
            last_lc = None
            continue
        lc = math.log(c[i])
        if last_lc is not None:
            d = lc - last_lc
            if prev_day is not None and days[i] != prev_day:
                gap2 += d * d
            else:
                intr += d * d
                sum_intr_inc += d * d
            n_inc += 1
        last_lc = lc
        prev_day = days[i]
    if n_inc < 1000:
        return None
    kappa = min(500.0, max(1.0, intr / max(1e-18, gap2)))
    mean_u = (intr + kappa * gap2) / n_inc
    trc = max(1, int(np.sum(tr)))
    keep = tr & (c > 0)
    full_est = int(np.sum(keep)) * n / trc               # TRAIN valid-density extrapolation
    T = max(1e-12, mean_u * full_est / max(1.0, float(target_bars)))
    return GapFlowBarBuilder(kappa, T)


# Registry consumed by bar_builder's generic ext dispatch (one line per new axis,
# ZERO marginal bytes in bar_builder.py).
EXT_AXES = {"logdollar_rc": LogDollarRCBarBuilder, "sess2": Session2BarBuilder,
            "gapflow": GapFlowBarBuilder}
EXT_MAKERS = {"logdollar_rc": make_logdollar_rc, "sess2": make_session2,
              "gapflow": make_gapflow}


def make_ext(bar_type, close, vol, ts_arr, target_bars, train_minute_mask):
    fn = EXT_MAKERS.get(bar_type)
    return fn(close, vol, ts_arr, target_bars, train_minute_mask) if fn else None
