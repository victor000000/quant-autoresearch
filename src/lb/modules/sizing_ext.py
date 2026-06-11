"""sizing_ext — VAL sizing engine EXTENSION module (separate QC project file, 2026-06-10).

Moved VERBATIM out of templates/footer.py.tmpl: the GLD/UUP ensemble main.py renders sat
39 bytes under QC's 64,000-byte/file wall, blocking all future footer work. _size() is the
RENDER-TIME sizing rule shared semantically with templates/infer.py.tmpl (which keeps its
own self-contained copy — infer renders standalone, no extra files); _cstats_sized() is the
VAL scorer. CONFIG is injected by the footer at import time (sizing_ext.CONFIG = CONFIG)
because _cstats_sized reads CONFIG["rebal_band"]/CONFIG.get("aim_a"). Any change here MUST
mirror infer.py.tmpl's _size to keep the VAL == OOS rule identity.
"""
import math

import numpy as np

CONFIG = {}   # injected by the footer before any _cstats_sized call


_VOL_FAST = 10
_VOL_SLOW = 60


def _size(p, thresh, sizing, rbuf):
    """Return target weight in [0,1] for prob p under the named sizing mode.
      ramp        = min(1,(p-thresh)*200) if p>thresh else 0          (legacy)
      binary      = 1.0 if p>thresh else 0.0                          (label-responsive)
      cdf_plain   = clip(2*Phi((p-thresh)/sqrt(p(1-p)))-1,0,1) if p>thresh else 0
      cdf_overlay = cdf_plain * clip(std_slow/std_fast, 0.6, 1.0)     (vol-targeted)
      longshort   = +1 if p>thresh, -1 if p<1-thresh, else 0          (SHORT-capable)
      ls_cdf      = clip(2*Phi((p-.5)/sqrt(p(1-p)))-1, -1, 1)         (continuous long/short)
    """
    if sizing == "crashveto":          # tail-risk veto: full long unless a crash is predicted (p_crash > thresh) -> flat
        return 0.0 if p > thresh else 1.0
    if sizing == "longshort":
        return 1.0 if p > thresh else (-1.0 if p < 1.0 - thresh else 0.0)
    if sizing in ("ls_cdf", "ls_overlay"):
        pp = min(max(p, 1e-6), 1.0 - 1e-6)
        zc = (pp - 0.5) / math.sqrt(pp * (1.0 - pp))
        w = float(min(1.0, max(-1.0, 2.0 * 0.5 * (1.0 + math.erf(zc / math.sqrt(2.0))) - 1.0)))
        if sizing == "ls_overlay":              # vol-target the long/short magnitude, keep sign (causal)
            m = len(rbuf)
            if m >= _VOL_FAST + 2:
                fast = float(np.std(rbuf[-_VOL_FAST:]))
                slow = float(np.std(rbuf[-min(m, _VOL_SLOW):]))
                if fast > 1e-9:
                    w *= float(min(1.0, max(0.45, slow / fast)))
        return w
    if sizing == "binary":
        return 1.0 if p > thresh else 0.0
    if sizing == "ramp":
        return float(min(1.0, (p - thresh) * 200.0)) if p > thresh else 0.0
    if p <= thresh:
        return 0.0
    pp = min(max(p, 1e-6), 1.0 - 1e-6)
    z = (pp - thresh) / math.sqrt(pp * (1.0 - pp))
    b = 2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) - 1.0  # 2*Phi(z)-1
    b = float(min(1.0, max(0.0, b)))
    if sizing == "cdf_overlay":
        m = len(rbuf)
        if m >= _VOL_FAST + 2:
            fast = float(np.std(rbuf[-_VOL_FAST:]))
            slow = float(np.std(rbuf[-min(m, _VOL_SLOW):]))
            if fast > 1e-9:
                b *= float(min(1.0, max(0.6, slow / fast)))
    if sizing == "dd_overlay":          # drawdown-aware: throttle the bet when the asset is
        m = len(rbuf)                   # underwater vs its recent rolling peak (causal, rbuf only)
        if m >= _VOL_FAST + 2:
            win = np.asarray(rbuf[-min(m, _VOL_SLOW):], dtype=float)
            cum = np.cumsum(win)                       # recent log-equity of the asset
            dd = float(cum[-1] - float(np.max(cum)))   # <= 0: current log DD from recent peak
            ref = 3.0 * float(np.std(win)) * math.sqrt(len(win))   # vol-normalized DD scale (~3σ)
            if ref > 1e-9:
                b *= float(min(1.0, max(0.5, 1.0 + dd / ref)))     # 1.0 at peak -> 0.5 floor when deep
    if sizing == "ddbreaker":           # dd_overlay's smooth throttle + a HARD circuit-breaker: go FLAT
        m = len(rbuf)                   # on EXTREME drawdown (frac<0.4) to cut the TAIL (MaxDD), re-enter
        if m >= _VOL_FAST + 2:          # on recovery. Attacks the Calmar denominator harder than the
            win = np.asarray(rbuf[-min(m, _VOL_SLOW):], dtype=float)   # 0.5-floor smooth throttle. Causal (rbuf only).
            cum = np.cumsum(win)
            dd = float(cum[-1] - float(np.max(cum)))
            ref = 3.0 * float(np.std(win)) * math.sqrt(len(win))
            if ref > 1e-9:
                frac = 1.0 + dd / ref
                b = 0.0 if frac < 0.4 else b * float(min(1.0, max(0.4, frac)))
    return b


def _cstats_sized(probs, lc_arr, log_rets, thresh, sizing, tc=0.0005):
    """realistic_cstats variant that sizes via _size() (CONFIG['sizing']).
    Mirrors trainer.realistic_cstats accounting (decide-then-append causal rbuf,
    0.01 rebalance dead-band, tc on |delta|). Returns (calmar,trades,tot,mdd,ann,da)."""
    n = min(len(probs) - 1, len(log_rets) - 1, len(lc_arr) - 1)
    if n < 2:
        return 0, 0, 0, 0, 0, 0.0
    positions = np.zeros(n + 1)
    last_pos = 0.0
    trades = 0
    rbuf = []
    _aim_a = float(CONFIG.get("aim_a", 0.3))   # Garleanu-Pedersen partial-adjustment coefficient
    # 'aim' wraps a BASE sizing: partial-adjust toward that overlay's per-bar target. aim=plain CDF
    # bet; aim_dd=over dd_overlay (keep drawdown protection + cut turnover); aim_cdf=over cdf_overlay.
    _AIM_BASE = {"aim": "cdf_plain", "aim_dd": "dd_overlay", "aim_cdf": "cdf_overlay"}
    for i in range(n):
        _sz = _AIM_BASE.get(sizing, sizing)
        target = _size(float(probs[i]), thresh, _sz, rbuf)
        if sizing in _AIM_BASE:                # trade PARTIALLY toward the (overlaid) aim (Garleanu-Pedersen 2013):
            target = (1.0 - _aim_a) * last_pos + _aim_a * target   # x_t=(1-a)x_{t-1}+a*aim -> cuts turnover, keeps base overlay
        # Abs-delta dead-band — mirrors infer.py's rebalance band so val turnover/cost
        # matches OOS for long/short sizings too (the old long-only-flavoured condition
        # under-counted short flips).
        if abs(target - last_pos) > CONFIG["rebal_band"]:
            trades += 1
        positions[i] = target
        last_pos = target
        if i - 1 >= 0:
            rbuf.append(float(log_rets[i - 1]))
    if trades < 2:
        return 0, trades, 0, 0, 0, 0.0
    strat_rets = positions[:-1] * log_rets[1:n + 1]
    for i in range(1, n):
        if abs(positions[i] - positions[i - 1]) > CONFIG["rebal_band"]:
            strat_rets[i] -= tc * abs(positions[i] - positions[i - 1])
    cum = np.cumsum(strat_rets)                       # cumulative log-return == log-equity
    peak = np.maximum.accumulate(cum)
    dd = cum - peak                                   # log drawdown (<= 0)
    # COMPOUNDED metric, to match the REAL OOS estimator (QC 'Compounding Annual
    # Return' / 'Drawdown'): val must use the SAME definition as the keep gate, else
    # the synth selector reads systematically higher than OOS and is not a faithful
    # preview. (The residual val<->OOS gap after this is honest realism: OOS holds
    # continuously across minutes + overnight with real fees.)
    mdd = 1.0 - float(np.exp(float(np.min(dd))))      # max drawdown as a fraction
    if mdd < 1e-9:
        mdd = 1e-9
    cagr = float(np.exp(float(cum[-1]) * (880.0 / n))) - 1.0   # compounded annual return
    da = float(np.sum(1.0 - np.exp(dd)))              # compounded underwater area
    cal = cagr / mdd if mdd > 0.001 else 0
    return cal, trades, float(np.sum(strat_rets)), mdd, cagr, da


def _venn_abers(cal_s, cal_y, test_s):
    """Inductive Venn-Abers calibration (Vovk-Petej 2014; deep-v2 B4). For each test score s*, fit
    isotonic on the calibration set with (s*,0) appended -> p0 and with (s*,1) appended -> p1; the
    VA probability is p1/(1-p0+p1). Provably better-calibrated than a single isotonic fit (a
    multiprobability merge). Runs ONCE in the footer (post-training), reuses the embargoed-VAL
    calibration set; the resulting probs are saved + replayed by infer exactly like isotonic's."""
    from sklearn.isotonic import IsotonicRegression
    cs = np.asarray(cal_s, dtype=float)
    cy = np.asarray(cal_y, dtype=float)
    out = np.empty(len(test_s), dtype=float)
    base0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    for i, s in enumerate(test_s):
        s = float(s)
        g0 = base0.fit(np.append(cs, s), np.append(cy, 0.0))
        p0 = float(g0.predict([s])[0])
        g1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.append(cs, s), np.append(cy, 1.0))
        p1 = float(g1.predict([s])[0])
        den = 1.0 - p0 + p1
        out[i] = (p1 / den) if den > 1e-9 else p1
    return out
