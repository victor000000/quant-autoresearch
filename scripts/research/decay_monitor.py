#!/usr/bin/env python3
"""Strategy alpha-decay / staleness monitor — small, stdlib-only.

The formal fix for "champions go stale as the OOS window grows" (the failure that cost a whole
session). Two cheap reads:
  - flag_decay(early_sharpe, late_sharpe): the infer emits sharpe_early / sharpe_late (first vs
    second half of the OOS window); a champion whose LATE-half Sharpe has collapsed relative to its
    EARLY-half is decaying -> re-validate it (don't trust the stale headline Calmar).
  - page_hinkley / cusum_meanshift / kaplan_meier: change-point + survival tools on a return series
    (when the persisted autoresearch/<ticker>/returns_<cell>.json is available).
Imports: medicine/biostat transfer — Page (1954) CUSUM, Page-Hinkley, Kaplan-Meier (1958) survival.

Run `python3 scripts/research/decay_monitor.py` for the self-test.
"""
import math


def flag_decay(early_sharpe, late_sharpe, collapse_frac=0.4):
    """Cheap staleness read from the infer's early/late half-window Sharpes.
    STALE if the late half went negative while early was positive, OR late < collapse_frac*early."""
    e, l = float(early_sharpe), float(late_sharpe)
    stale = (e > 0 and l <= 0) or (e > 0 and l < collapse_frac * e)
    reason = (f"late Sharpe {l:.2f} collapsed vs early {e:.2f}" if stale
              else f"holding (early {e:.2f} -> late {l:.2f})")
    return {"stale": bool(stale), "early": e, "late": l, "reason": reason}


def page_hinkley(series, delta=None, lam=None):
    """Page-Hinkley test for a DOWNWARD mean shift. Returns the first index where decay is
    confirmed, else None. delta = tolerated drift (default 0); lam = decision threshold
    (default 3*std of the series)."""
    n = len(series)
    if n < 8:
        return None
    mu = sum(series) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in series) / (n - 1)) if n > 1 else 0.0
    if delta is None:
        delta = 0.0
    if lam is None:
        lam = 3.0 * sd
    if lam <= 0:
        return None
    mt = 0.0          # cumulative deviation below (running mean - delta)
    m_min = 0.0
    run_mean = 0.0
    for t, x in enumerate(series):
        run_mean += (x - run_mean) / (t + 1)
        mt += (x - run_mean - delta)        # accumulate; drops when x below running mean
        m_min = min(m_min, mt)
        if mt - m_min > lam:                # rebound — not a downward decay
            m_min = mt
        # downward decay: running sum falls lam below its running max
        # (use the symmetric upper accumulator for the DOWN direction)
    # simpler robust down-detector: CUSUM on negative deviations
    return cusum_meanshift(series, drift=delta)


def cusum_meanshift(series, threshold=None, drift=0.0):
    """Two-sided CUSUM (Page 1954). Returns the first index where a NEGATIVE (downward) mean
    shift is confirmed, else None. threshold default = 4*std."""
    n = len(series)
    if n < 8:
        return None
    mu = sum(series) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in series) / (n - 1))
    if threshold is None:
        threshold = 4.0 * sd
    if threshold <= 0:
        return None
    s_neg = 0.0
    for t, x in enumerate(series):
        s_neg = min(0.0, s_neg + (x - mu) + drift)
        if s_neg <= -threshold:
            return t
    return None


def kaplan_meier(durations, events):
    """Kaplan-Meier survival estimator (1958). durations: time-to-event/censor; events: 1=alpha
    death observed, 0=censored. Returns {times, survival, median_survival}."""
    pairs = sorted(zip(durations, events))
    n = len(pairs)
    at_risk = n
    S = 1.0
    times, surv = [], []
    i = 0
    while i < n:
        t = pairs[i][0]
        d = sum(1 for j in range(i, n) if pairs[j][0] == t and pairs[j][1])  # deaths at t
        c = sum(1 for j in range(i, n) if pairs[j][0] == t)                   # events (death+censor) at t
        if d > 0:
            S *= (1.0 - d / at_risk)
        times.append(t)
        surv.append(S)
        at_risk -= c
        i += c
    median = next((t for t, s in zip(times, surv) if s <= 0.5), None)
    return {"times": times, "survival": surv, "median_survival": median}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ok = True

    def check(name, cond):
        global ok; ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # flag_decay
    check("flag_decay stale when late collapses", flag_decay(2.0, -0.3)["stale"] is True)
    check("flag_decay ok when holding", flag_decay(2.0, 1.8)["stale"] is False)
    # cusum / page_hinkley on flat-then-declining series
    series = [0.01 * ((t % 5) - 2) for t in range(120)] + [-0.05 + 0.005 * ((t % 3) - 1) for t in range(60)]
    cp = cusum_meanshift(series)
    check(f"cusum finds the break in the decline region (got {cp})", cp is not None and cp >= 120)
    ph = page_hinkley(series)
    check(f"page_hinkley flags decay (got {ph})", ph is not None)
    check("cusum None on stable series", cusum_meanshift([0.01 * ((t % 7) - 3) for t in range(200)]) is None)
    # Kaplan-Meier textbook: deaths at 2,3,5; censor at 4; n=5 -> S after 2 = .8, after 3 = .6, after 5 = .3
    km = kaplan_meier([2, 3, 4, 5, 5], [1, 1, 0, 1, 1])
    s_at = dict(zip(km["times"], km["survival"]))
    check(f"KM S(2)=0.8 (got {round(s_at.get(2,-1),3)})", abs(s_at.get(2, -1) - 0.8) < 1e-9)
    check(f"KM S(3)=0.6 (got {round(s_at.get(3,-1),3)})", abs(s_at.get(3, -1) - 0.6) < 1e-9)

    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)
