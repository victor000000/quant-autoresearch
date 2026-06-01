"""Custom bar builders. Edit freely — any axis type, any internal API.

Currently: Volatility Bars (Wang's 3rd axis type), Dollar Bars, Tick Bars.
"""
import math
import numpy as np


class VolBarBuilder:
    """Volatility bar: cumulative log_return² * sqrt(volume) → sample when threshold exceeded."""

    def __init__(self, threshold):
        self.thresh = threshold
        self.cum = 0.0
        self.last_close = None
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        lc = math.log(close)
        bar = None
        if self.last_close is not None:
            ret = lc - self.last_close
            contrib = (ret * ret) * math.sqrt(vol)
            if contrib > 0:
                self.cum += contrib
        self.last_close = lc
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            bar = {"ts_close": ts, "log_close": self.close_lc}
        return bar


class DollarBarBuilder:
    """Dollar bar: cumulative close * volume → sample when threshold exceeded."""

    def __init__(self, threshold):
        self.thresh = threshold
        self.cum = 0.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        lc = math.log(close)
        self.cum += close * vol
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class TickBarBuilder:
    """Tick bar: count trades → sample every N ticks."""

    def __init__(self, threshold):
        self.thresh = threshold
        self.count = 0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        self.count += 1
        self.close_lc = lc
        if self.count >= self.thresh:
            self.count = 0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class RangeBarBuilder:
    """Range bar: sample when price moves by threshold % from last sample.
    Normalizes price movement — each bar represents equal price change.
    Best for volatile/EM ETFs where time/dollar bars have irregular spacing."""

    def __init__(self, threshold_pct):
        self.thresh_pct = threshold_pct  # e.g., 0.005 = 0.5% price move
        self.last_close = None
        self.last_sample_close = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        bar = None
        if self.last_sample_close is not None and self.last_close is not None:
            pct_change = abs(close - self.last_sample_close) / self.last_sample_close
            if pct_change >= self.thresh_pct:
                lc = math.log(close)
                bar = {"ts_close": ts, "log_close": lc}
                self.last_sample_close = close
        elif self.last_sample_close is None:
            self.last_sample_close = close
        self.last_close = close
        return bar


def build_bars(close, vol, ts_arr, bar_type="vol", target_bars=15000):
    """Build bars from minute data.

    Args:
        close: np.array of closing prices
        vol: np.array of volumes
        ts_arr: np.array of timestamps
        bar_type: "vol", "dollar", or "tick"
        target_bars: approximate number of bars desired

    Returns:
        lc: np.array of log-close values per bar
        lr: np.array of log-returns per bar
        N: number of bars
        bar_ts: np.array of bar timestamps
    """
    # Compute threshold to hit ~target_bars
    if bar_type == "dollar":
        total_dollar = float(np.sum(close * vol))
        threshold = total_dollar / target_bars if total_dollar > 0 else 1e-9
        builder = DollarBarBuilder(threshold)
    elif bar_type == "tick":
        threshold = max(1, len(close) // target_bars)
        builder = TickBarBuilder(threshold)
    elif bar_type == "range":
        # Range bars: target ~0.3-0.5% per bar depending on asset volatility
        threshold = 0.003 if target_bars < 10000 else 0.002
        builder = RangeBarBuilder(threshold)
    else:  # vol (default)
        total_contrib = 0.0
        last_lc = None
        for i in range(len(close)):
            if close[i] <= 0 or vol[i] <= 0:
                continue
            lc_val = math.log(close[i])
            if last_lc is not None:
                total_contrib += (lc_val - last_lc) ** 2 * math.sqrt(vol[i])
            last_lc = lc_val
        threshold = total_contrib / target_bars if total_contrib > 0 else 1e-9
        builder = VolBarBuilder(threshold)

    # Build bars
    bars = []
    bar_ts_raw = []
    for i in range(len(close)):
        b = builder.update(ts_arr[i], close[i], vol[i])
        if b is not None:
            bars.append(b)
            bar_ts_raw.append(ts_arr[i])

    N = len(bars)
    lc = np.array([x["log_close"] for x in bars])
    lr = np.zeros(N)
    lr[1:] = lc[1:] - lc[:-1]
    bar_ts = np.array(bar_ts_raw)

    return lc, lr, N, bar_ts
