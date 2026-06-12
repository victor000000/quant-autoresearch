#!/usr/bin/env python3
"""Deflated CALMAR (innovation backlog 2026-06-11 #5).

We crown and weight by Calmar but historically deflated only Sharpe; maxDD is
the most selection-inflated order statistic of all. This audit asks: could the
champion's Calmar be the lucky MAX of N_trials no-edge strategies?

Method (Monte-Carlo max-of-N null):
  1. Pull the champion's OOS daily returns from its infer backtest curve.
  2. Null = stationary bootstrap (mean block 10) of the DEMEANED returns —
     same vol, fat tails and autocorrelation, zero edge.
  3. Simulate M sets of N_trials null paths; record the MAX Calmar of each set.
  4. PASS iff real Calmar > the 95th percentile of that max-of-N distribution.

Usage: python3 scripts/audit/deflated_calmar.py TICKER BACKTEST_ID N_TRIALS [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
import champion_series as cs  # noqa: E402

M_SETS = 400
PPY = 252


def calmar(rets):
    eq = np.cumprod(1.0 + rets, axis=-1)
    n = rets.shape[-1]
    cagr = eq[..., -1] ** (PPY / n) - 1.0
    peak = np.maximum.accumulate(eq, axis=-1)
    mdd = np.max(1.0 - eq / peak, axis=-1)
    return cagr / np.maximum(mdd, 1e-4)


def iaaft(rets, rng, iters=50):
    """IAAFT surrogate (backlog #12): exact amplitude distribution + amplitude
    spectrum (linear ACF) of the input, randomized phase. The orthogonal null to
    the stationary bootstrap — if a demeaned curve's max-of-N Calmar under IAAFT
    reaches the real value, the 'edge' is explainable by spectrum alone."""
    x = np.asarray(rets, float)
    sorted_x = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(iters):
        sp = np.fft.rfft(y)
        mag = np.abs(sp)
        mag[mag < 1e-20] = 1e-20
        y = np.fft.irfft(sp * amp / mag, n=len(x))
        y = sorted_x[np.argsort(np.argsort(y))]   # rank-remap to exact amplitudes
    return y


def stationary_bootstrap(rets, size, rng, mean_block=10):
    n = len(rets)
    p = 1.0 / mean_block
    idx = np.empty(size, dtype=np.int64)
    idx[0] = rng.integers(n)
    restart = rng.random(size) < p
    steps = rng.integers(n, size=size)
    for t in range(1, size):
        idx[t] = steps[t] if restart[t] else (idx[t - 1] + 1) % n
    return rets[idx]


def audit(tk, bid, n_trials, rng, null="bootstrap"):
    eq = cs.equity_series(bid) or []
    closes = np.array([c for _, c in eq], float)
    rets = closes[1:] / closes[:-1] - 1.0
    if len(rets) < 60:
        print(f"{tk}: insufficient curve")
        return
    real = float(calmar(rets))
    null_rets = rets - rets.mean()
    n = len(rets)
    gen = (lambda: iaaft(null_rets, rng)) if null == "iaaft" \
        else (lambda: stationary_bootstrap(null_rets, n, rng))
    maxes = np.empty(M_SETS)
    for s in range(M_SETS):
        paths = np.vstack([gen() for _ in range(n_trials)])
        maxes[s] = calmar(paths).max()
    p95 = float(np.quantile(maxes, 0.95))
    p50 = float(np.quantile(maxes, 0.50))
    pval = float(np.mean(maxes >= real))
    ok = real > p95
    print(f"{tk} [{null}]: real Calmar {real:.2f} | max-of-{n_trials} null: median {p50:.2f}, "
          f"95th {p95:.2f} | p(max>=real) {pval:.3f} -> "
          f"{'PASS (deflation-clearing)' if ok else 'FAIL (could be selection luck)'}")


def main(argv):
    null = "bootstrap"
    if argv and argv[0] == "--null":
        null = argv[1]
        argv = argv[2:]
    rng = np.random.default_rng(42)
    args = list(zip(argv[::3], argv[1::3], argv[2::3]))
    for tk, bid, nt in args:
        audit(tk, bid, int(nt), rng, null=null)


if __name__ == "__main__":
    main(sys.argv[1:])
