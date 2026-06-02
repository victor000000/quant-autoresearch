#!/usr/bin/env python3
"""Multiple-testing / backtest-overfitting rigor — small, dependency-free (stdlib only).

Implements the finance-grade selection-bias controls the efficiency review flagged as the #1
gap (RESEARCH_REVIEW_v2 Tier-1): Probabilistic & Deflated Sharpe (Bailey & Lopez de Prado 2014),
Probability of Backtest Overfitting via CSCV (Bailey-Borwein-LdP-Zhu 2014), Holm step-down (1979),
Benjamini-Hochberg FDR (1995). Pure stdlib so it runs anywhere; no mlfinlab/scipy/pandas.

Run `python3 scripts/stats_rigor.py` for the self-test.
"""
import math, itertools
from statistics import NormalDist

_ND = NormalDist()
_GAMMA = 0.5772156649015329  # Euler-Mascheroni
Phi = _ND.cdf
Phinv = _ND.inv_cdf


def probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=0.0):
    """PSR(sr>sr_benchmark): prob the true Sharpe exceeds the benchmark, correcting for skew,
    (non-excess) kurtosis and sample length n. sr is the PER-OBSERVATION Sharpe. Bailey-LdP 2012."""
    if n <= 1:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return Phi((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def expected_max_sharpe(var_trials_sr, n_trials):
    """E[max] of n_trials iid N(0, var) Sharpes — the 'best of N noise' benchmark. Bailey-LdP 2014."""
    if n_trials < 2 or var_trials_sr <= 0:
        return 0.0
    z1 = Phinv(1.0 - 1.0 / n_trials)
    z2 = Phinv(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_trials_sr) * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe_ratio(sr, n, skew, kurt, n_trials, var_trials_sr):
    """DSR = PSR against the best-of-N-trials benchmark. Prob the edge is real given the search size."""
    return probabilistic_sharpe_ratio(sr, n, skew, kurt,
                                      sr_benchmark=expected_max_sharpe(var_trials_sr, n_trials))


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm step-down (1979): FWER<=alpha, uniformly more powerful than Bonferroni. Returns
    list[bool] reject, in the ORIGINAL order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break  # step-down: once one fails, all larger fail
    return reject


def benjamini_hochberg(pvalues, q=0.10):
    """Benjamini-Hochberg (1995): control FDR<=q. Returns list[bool] reject, in ORIGINAL order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= rank / m * q:
            kmax = rank
    if kmax > 0:
        for rank, i in enumerate(order, start=1):
            if rank <= kmax:
                reject[i] = True
    return reject


def _sharpe(col):
    n = len(col)
    if n < 2:
        return 0.0
    mu = sum(col) / n
    var = sum((x - mu) ** 2 for x in col) / (n - 1)
    sd = math.sqrt(var)
    return mu / sd if sd > 1e-12 else 0.0


def pbo_cscv(returns_matrix, n_splits=16, max_combos=1000):
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.
    returns_matrix: list of T rows, each a list of N per-strategy returns. Split T into n_splits
    contiguous blocks; for each way to pick n_splits/2 blocks as IS: rank strategies by IS Sharpe,
    take the IS-best, find its OOS Sharpe rank -> relative rank w in (0,1) -> logit lambda; PBO =
    fraction of partitions with lambda<0 (IS-best below OOS median). Bailey-Borwein-LdP-Zhu 2014.
    Deterministic combo cap (stride) when C(n_splits, n_splits/2) is large. Returns dict."""
    T = len(returns_matrix)
    if T < n_splits or n_splits % 2 or not returns_matrix:
        return {"pbo": float("nan"), "n_combinations": 0, "lambdas": [], "note": "insufficient data"}
    N = len(returns_matrix[0])
    if N < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "lambdas": [], "note": "need >=2 strategies"}
    bsz = T // n_splits
    blocks = [list(range(b * bsz, (b + 1) * bsz)) for b in range(n_splits)]
    all_combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(all_combos) > max_combos:
        stride = max(1, len(all_combos) // max_combos)
        all_combos = all_combos[::stride][:max_combos]
    lambdas = []
    for combo in all_combos:
        is_blk = set(combo)
        is_rows = [r for b in is_blk for r in blocks[b]]
        oos_rows = [r for b in range(n_splits) if b not in is_blk for r in blocks[b]]
        is_sr = [_sharpe([returns_matrix[r][s] for r in is_rows]) for s in range(N)]
        oos_sr = [_sharpe([returns_matrix[r][s] for r in oos_rows]) for s in range(N)]
        best = max(range(N), key=lambda s: is_sr[s])
        # OOS rank of the IS-best (1=worst .. N=best); relative rank w in (0,1)
        rank = 1 + sum(1 for s in range(N) if oos_sr[s] < oos_sr[best])
        w = rank / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        lambdas.append(math.log(w / (1 - w)))
    pbo = sum(1 for l in lambdas if l <= 0) / len(lambdas) if lambdas else float("nan")
    return {"pbo": pbo, "n_combinations": len(lambdas), "lambdas": lambdas}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ok = True

    def check(name, cond):
        global ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # PSR monotonic in sr and in n
    check("PSR increases with sr", probabilistic_sharpe_ratio(0.10, 252, 0, 3) >
          probabilistic_sharpe_ratio(0.05, 252, 0, 3))
    check("PSR increases with n", probabilistic_sharpe_ratio(0.08, 500, 0, 3) >
          probabilistic_sharpe_ratio(0.08, 100, 0, 3))
    # expected_max_sharpe grows with n_trials
    check("E[maxSR] grows with n_trials", expected_max_sharpe(0.04, 50) > expected_max_sharpe(0.04, 5))
    # DSR < PSR(>0) when there were many trials (benchmark > 0)
    check("DSR <= PSR>0", deflated_sharpe_ratio(0.08, 252, 0, 3, 50, 0.04) <=
          probabilistic_sharpe_ratio(0.08, 252, 0, 3, 0.0) + 1e-9)
    # Holm: p=[0.01,0.04,0.03,0.005], alpha=0.05, m=4 -> sorted .005(<=.0125)T, .01(<=.0167)T,
    # .03(<=.025)F -> reject {.005,.01} = indices {3,0}
    h = holm_bonferroni([0.01, 0.04, 0.03, 0.005], 0.05)
    check("Holm rejects {0.005,0.01} only", h == [True, False, False, True])
    # BH classic: [0.001,0.008,0.039,0.041,0.042,0.06], q=0.05, m=6.
    # thresholds k/m*q: .0083,.0167,.025,.0333,.0417,.05 ; sorted p meet at k=5 (.042<=.0417? no)
    # k=4: .041<=.0333? no; k=2: .008<=.0167 yes -> kmax=2 -> reject 2 smallest {.001,.008}
    bh = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.06], 0.05)
    check("BH rejects 2 smallest", bh == [True, True, False, False, False, False])
    # PBO: genuine (strategy 0 persistently higher mean) -> low; pure noise (deterministic) -> ~0.5
    T = 256
    genuine = [[0.02 + 0.001 * ((r * 7 + 1) % 5 - 2)] + [0.001 * ((r * (s + 3)) % 7 - 3) for s in range(1, 8)] for r in range(T)]
    g = pbo_cscv(genuine, n_splits=8)
    check(f"PBO(genuine)<0.3 (got {round(g['pbo'],3)})", g["pbo"] < 0.3)
    # overfit matrix: IS-best is systematically OOS-worst -> PBO should be HIGH (detects overfitting)
    overfit = [[0.001 * (((r + 1) * (s + 2)) % 11 - 5) for s in range(8)] for r in range(T)]
    of = pbo_cscv(overfit, n_splits=8)
    check(f"PBO(overfit)>0.7 (got {round(of['pbo'],3)})", of["pbo"] > 0.7)

    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)
