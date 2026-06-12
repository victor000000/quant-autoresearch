#!/usr/bin/env python3
"""Regime-split validation (innovation backlog 2026-06-11 #6).

Require a champion's OOS edge to survive INDEPENDENTLY in both trailing-vol
halves. Regime is PAST-ONLY: day t is 'high-vol' iff the trailing 20-obs RMS of
the strategy's own daily returns exceeds the EXPANDING median of that RMS up to
t (no future information). Catches one-regime fragility that every aggregate
gate (Sharpe/Calmar/DSR on the full window) misses.

Proxy note: with no benchmark backtest on the project, regime is defined on the
STRATEGY curve, not the asset. For our long-only overlay champions the two track
closely when invested; flat stretches land in the low-vol half by construction.

Usage: python3 scripts/audit/regime_split.py TICKER BACKTEST_ID [TICKER BACKTEST_ID ...]
PASS = positive Sharpe in BOTH halves AND the weaker half >= 30% of the pooled Sharpe.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
import champion_series as cs  # noqa: E402


def split_stats(bid):
    eq = cs.equity_series(bid) or []
    closes = [c for _, c in eq]
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    n = len(rets)
    if n < 60:
        return None
    # trailing 20-obs RMS (past-only), expanding-median threshold (past-only)
    rms, med_hist, regime = [], [], []
    for t in range(n):
        lo = max(0, t - 20)
        w = rets[lo:t] or [0.0]
        r = math.sqrt(sum(x * x for x in w) / len(w))
        rms.append(r)
        hist = sorted(rms[: t + 1])
        med = hist[len(hist) // 2]
        med_hist.append(med)
        regime.append(1 if r > med else 0)

    def sharpe(xs):
        if len(xs) < 20:
            return float("nan")
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        return m / sd * math.sqrt(252) if sd > 1e-12 else float("nan")

    hi = [r for r, g in zip(rets, regime) if g == 1]
    lo_ = [r for r, g in zip(rets, regime) if g == 0]
    s_all, s_hi, s_lo = sharpe(rets), sharpe(hi), sharpe(lo_)
    weak = min(s_hi, s_lo)
    ok = s_hi > 0 and s_lo > 0 and (s_all <= 0 or weak >= 0.3 * s_all)
    return {"n": n, "n_hi": len(hi), "n_lo": len(lo_), "sharpe_all": s_all,
            "sharpe_hi": s_hi, "sharpe_lo": s_lo, "pass": ok}


def main(argv):
    pairs = list(zip(argv[::2], argv[1::2]))
    for tk, bid in pairs:
        st = split_stats(bid)
        if st is None:
            print(f"{tk}: insufficient curve")
            continue
        print(f"{tk}: Sharpe all {st['sharpe_all']:.2f} | high-vol {st['sharpe_hi']:.2f} "
              f"({st['n_hi']}d) | low-vol {st['sharpe_lo']:.2f} ({st['n_lo']}d) -> "
              f"{'PASS' if st['pass'] else 'FAIL (one-regime fragility)'}")


if __name__ == "__main__":
    main(sys.argv[1:])
