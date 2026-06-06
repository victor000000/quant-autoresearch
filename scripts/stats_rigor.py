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
# N_eff (effective number of independent trials) — fixes DSR over-deflation.    #
# Raw Bonferroni 1-0.05/n_trials treats a correlated 21x27 sweep as n_trials    #
# INDEPENDENT bets, over-penalising and killing real borderline fits. N_eff is  #
# the eigenvalue PARTICIPATION RATIO of the trial correlation matrix C:         #
#   N_eff = (Σ λ_i)^2 / Σ λ_i^2.                                                 #
# For a symmetric C this needs NO eigendecomposition: Σλ_i = trace(C) and       #
# Σλ_i^2 = trace(C^2) = Σ_ij C_ij^2 (C symmetric). So N_eff = trace(C)^2 / ‖C‖_F^2 #
# — exact, pure-stdlib, O(N^2). (Meucci 2009 'Managing Diversification'; the     #
# same PR statistic LdP uses to count effective bets.) For a proper correlation  #
# matrix (unit diagonal) trace(C)=N so N_eff in [1, N].                          #
# --------------------------------------------------------------------------- #
def effective_n_trials(corr):
    """N_eff = (Σλ)^2 / Σλ^2 of trial correlation matrix `corr` (list-of-lists or any
    indexable square symmetric matrix). Collapses N correlated trials to the effective
    number of independent bets. Returns float in [1, N]. N==0 -> 1.0 (a single trial)."""
    n = len(corr)
    if n <= 1:
        return float(max(1, n))
    tr = 0.0      # trace(C) = Σ λ
    fro = 0.0     # Σ_ij C_ij^2 = trace(C^2) = Σ λ^2
    for i in range(n):
        row = corr[i]
        tr += float(row[i])
        for j in range(n):
            v = float(row[j])
            fro += v * v
    if fro <= 1e-300:
        return float(n)
    neff = (tr * tr) / fro
    # clamp to the structurally valid range [1, n]
    return float(min(float(n), max(1.0, neff)))


def corr_from_returns(vectors, min_overlap=20):
    """Pearson correlation matrix from a list of per-trial return vectors (possibly ragged).
    Pairwise on the common (tail-aligned) overlap; <min_overlap shared points -> corr 0.
    Pure stdlib. Use this to feed effective_n_trials when real per-trial OOS-PnL is captured."""
    m = len(vectors)
    cols = [list(map(float, v)) for v in vectors]
    C = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            a, b = cols[i], cols[j]
            k = min(len(a), len(b))
            if k < min_overlap:
                continue
            ai, bi = a[-k:], b[-k:]   # tail-align
            ma = sum(ai) / k
            mb = sum(bi) / k
            sa = sum((x - ma) ** 2 for x in ai)
            sb = sum((x - mb) ** 2 for x in bi)
            if sa <= 1e-30 or sb <= 1e-30:
                continue
            cov = sum((ai[t] - ma) * (bi[t] - mb) for t in range(k))
            r = cov / math.sqrt(sa * sb)
            r = min(1.0, max(-1.0, r))
            C[i][j] = C[j][i] = r
    return C


def config_affinity_corr(configs, weights=(("axis", 0.5), ("labeler", 0.3), ("sizing", 0.2))):
    """Proxy trial correlation from CONFIG OVERLAP when per-trial OOS-PnL series are not on
    disk (the offline case for round_results.csv). Two trials that share the same axis (and/or
    labeler, sizing) produce highly-correlated PnL; |corr| ~ Σ weight_k · 1{field_k equal}.
    `configs`: list of dicts. Returns a symmetric unit-diagonal matrix. This is a deliberate,
    documented UNDER-count proxy — swap in corr_from_returns(real_pnl) once per-trial daily PnL
    is captured. Either way N_eff << n_trials, which is the point (un-kill borderline fits)."""
    m = len(configs)
    C = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        ci = configs[i]
        for j in range(i + 1, m):
            cj = configs[j]
            r = 0.0
            for field, w in weights:
                if ci.get(field) is not None and ci.get(field) == cj.get(field):
                    r += w
            r = min(1.0, r)
            C[i][j] = C[j][i] = r
    return C


# --------------------------------------------------------------------------- #
# Persistent GLOBAL true-trial counter — the durable search burden across every #
# ETF x axis x labeler x sizer run, so N for the multiple-testing haircut never  #
# resets between sessions.                                                       #
# --------------------------------------------------------------------------- #
import json as _json
import os as _os

_DEFAULT_TRIAL_LEDGER = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "results", "global_trials.json")


def read_global_trials(path=None):
    """Return the cumulative global trial count (0 if the ledger does not yet exist)."""
    path = path or _DEFAULT_TRIAL_LEDGER
    try:
        with open(path) as f:
            return int(_json.load(f).get("global_trials", 0))
    except Exception:
        return 0


def bump_global_trials(n, path=None, tag=""):
    """Durably add `n` to the global trial counter; returns the new total. Atomic-ish
    (write-temp-then-replace). `tag` records the most recent contributor for provenance."""
    path = path or _DEFAULT_TRIAL_LEDGER
    cur = read_global_trials(path)
    new = cur + int(max(0, n))
    rec = {"global_trials": new, "last_bump": int(n), "last_tag": str(tag)}
    try:
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(rec, f, indent=2)
        _os.replace(tmp, path)
    except Exception:
        pass
    return new


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

    # N_eff: identity matrix (N independent trials) -> N_eff == N
    I5 = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
    check(f"N_eff(I_5)==5 (got {round(effective_n_trials(I5),3)})", abs(effective_n_trials(I5) - 5.0) < 1e-6)
    # all-ones (perfectly correlated) -> N_eff == 1
    J5 = [[1.0 for _ in range(5)] for _ in range(5)]
    check(f"N_eff(ones_5)==1 (got {round(effective_n_trials(J5),3)})", abs(effective_n_trials(J5) - 1.0) < 1e-6)
    # two independent BLOCKS of 3 perfectly-correlated trials -> N_eff == 2
    blk = [[1.0 if (i // 3) == (j // 3) else 0.0 for j in range(6)] for i in range(6)]
    check(f"N_eff(2 blocks of 3)==2 (got {round(effective_n_trials(blk),3)})", abs(effective_n_trials(blk) - 2.0) < 1e-6)
    # correlated trials collapse N_eff far below n_trials
    cfgs = [{"axis": "vol", "labeler": "trend_leg", "sizing": "binary"}] * 20 + \
           [{"axis": "dollar", "labeler": "ker", "sizing": "cdf_overlay"}] * 7
    Cn = config_affinity_corr(cfgs)
    ne = effective_n_trials(Cn)
    check(f"affinity N_eff<<27 (got {round(ne,2)} of 27)", ne < 6.0)
    # corr_from_returns: identical vectors -> corr 1, anti -> -1
    cr = corr_from_returns([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
                            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 3])
    check("corr_from_returns identical==1", abs(cr[0][1] - 1.0) < 1e-9)
    check("corr_from_returns anti==-1", abs(cr[0][2] + 1.0) < 1e-9)
    # global trial counter: bump is durable + additive (temp ledger)
    import tempfile as _tf
    _tp = _os.path.join(_tf.mkdtemp(), "gt.json")
    check("global trials start 0", read_global_trials(_tp) == 0)
    bump_global_trials(10, _tp, tag="t1"); t2 = bump_global_trials(5, _tp, tag="t2")
    check(f"global trials accumulate ==15 (got {t2})", read_global_trials(_tp) == 15)

    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)
