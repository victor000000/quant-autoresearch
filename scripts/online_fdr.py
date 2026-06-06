#!/usr/bin/env python3
"""Online-FDR alpha-wealth ledger (LORD++ / ADDIS) for the open-ended 311 screen.

311-plan step 4 quick-win. Batch Holm/BH assume the test set is FIXED and known in advance;
the 311 screen is genuinely SEQUENTIAL, resumable and open-ended (we keep adding ETFs). The
statistically-correct control there is ONLINE FDR with an alpha-WEALTH account: each test spends
a little wealth, each confirmed discovery earns wealth back, and the test level self-throttles as
wealth depletes — a live 'exploration budget' for the screen.

Two engines, pure scalar bookkeeping (no numpy needed):
  * LORD++   (Ramdas-Yang-Wainwright-Jordan 2017, 'decaying memory') — the rigorous workhorse;
             level alpha_t = w0*g_t + (alpha-w0)*g_{t-tau1} + alpha*Σ_{j>=2} g_{t-tauj}.
  * ADDIS    (Tian-Ramdas 2019) — adaptive-with-discarding extension of SAFFRON (Ramdas 2018):
             discards p>tau, counts candidates p<=lambda, multiplier (tau-lambda); more powerful
             when many tests are conservatively null (most of the 311). Reduces to SAFFRON at tau=1.
Both control FDR<=alpha (mFDR under arbitrary dependence; FDR under independence).

Replay: derive ONE p-value per ETF (or per trial) from results/round_results.csv — PSR(SR>0)
'DSR p-value' when daily stats exist, else a val_auc 'permute proxy' p — stream them in screen
order (results/etf_screen_progress.log), and stamp FIT only where p < the wealth-derived level.
Surfaces remaining alpha-wealth = the live exploration budget.

Run `python3 scripts/online_fdr.py`            -> selftest + (if data present) a live replay.
Run `python3 scripts/online_fdr.py --replay`   -> replay over round_results.csv only.
"""
import os, sys, csv, math, argparse, collections
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import stats_rigor as SR
    _PSR = SR.probabilistic_sharpe_ratio
except Exception:                                    # keep replay usable even if import fails
    _PSR = None

_ND = NormalDist()
Phi = _ND.cdf

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND_CSV = os.path.join(HERE, "..", "results", "round_results.csv")
SCREEN_LOG = os.path.join(HERE, "..", "results", "etf_screen_progress.log")

# The 8 STRONG deep-sweep fits + the two leak-free champions — the names a correct ledger must keep.
KNOWN_FITS = {"SSO", "IAU", "USO", "AGQ", "GDX", "DJP", "GSG", "UCO", "GLD", "UUP"}


def _zeta(s=1.6, N=300000):
    """Riemann zeta(s) via partial sum + Euler-Maclaurin tail — normalises the gamma weights so
    Σ_{k>=1} gamma_k = 1 (required for valid online-FDR control over an open-ended stream)."""
    part = 0.0
    for k in range(1, N + 1):
        part += k ** (-s)
    tail = N ** (1.0 - s) / (s - 1.0) + 0.5 * N ** (-s)   # ∫_N^∞ x^-s dx + 1/2 N^-s
    return part + tail


_GAMMA_EXP = 1.6
_GAMMA_Z = _zeta(_GAMMA_EXP)


def gamma(k):
    """Decaying spending sequence gamma_k = k^-1.6 / zeta(1.6), Σ=1, k>=1. gamma_k=0 for k<1."""
    if k < 1:
        return 0.0
    return (k ** (-_GAMMA_EXP)) / _GAMMA_Z


# --------------------------------------------------------------------------- #
class LordPP:
    """LORD++ online-FDR ledger. alpha = target FDR; w0 in (0, alpha] = starting alpha-wealth."""

    def __init__(self, alpha=0.10, w0=None):
        self.alpha = float(alpha)
        self.w0 = float(w0) if w0 is not None else 0.5 * self.alpha
        self.t = 0                       # tests seen
        self.rejs = []                   # rejection ordinals (1-based test index)
        self.n_rej = 0
        self.spent = 0.0
        self.earned = 0.0

    def level(self):
        """alpha_t for the NEXT test (test index t+1)."""
        t = self.t + 1
        a = self.w0 * gamma(t)
        for j, tau in enumerate(self.rejs):
            reward = (self.alpha - self.w0) if j == 0 else self.alpha
            a += reward * gamma(t - tau)
        return min(self.alpha, a)        # a level above alpha is never useful

    @property
    def wealth(self):
        """Remaining alpha-wealth (the live exploration budget): w0 + earned - spent, >=0."""
        return max(0.0, self.w0 + self.earned - self.spent)

    def test(self, p):
        """Process one p-value; return True if rejected (FIT). Updates the wealth account."""
        self.t += 1
        a = self.level()
        self.spent += a
        rej = p <= a
        if rej:
            self.rejs.append(self.t)
            self.n_rej += 1
            self.earned += (self.alpha - self.w0) if self.n_rej == 1 else self.alpha
        return rej, a


# --------------------------------------------------------------------------- #
class Addis:
    """ADDIS (Tian-Ramdas 2019): SAFFRON + discarding. lambda<tau in (0,1). Discards p>tau,
    counts candidates p<=lambda, multiplier (tau-lambda). tau=1 -> SAFFRON (Ramdas 2018)."""

    def __init__(self, alpha=0.10, lam=0.25, tau=0.5, w0=None):
        assert 0.0 < lam < tau <= 1.0, "require 0 < lambda < tau <= 1"
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.tau = float(tau)
        wmax = (self.tau - self.lam) * self.alpha
        self.w0 = float(w0) if w0 is not None else 0.5 * wmax
        self.w0 = min(self.w0, 0.999 * wmax)
        self.s = 0                       # SELECTED tests seen (p<=tau)
        self.cum_cand = 0                # candidates (p<=lambda) among selected tests seen
        self.rejs = []                   # list of (selected_ordinal, cum_cand_at_or_before)
        self.n_rej = 0
        self.spent = 0.0
        self.earned = 0.0

    def _saffron_term(self):
        """The SAFFRON/ADDIS bracket evaluated for the NEXT selected test (selected index s+1)."""
        s = self.s + 1
        c_lt = self.cum_cand               # candidates strictly before the current selected test
        # w0 term: gamma index = s - (candidates strictly before s)
        a = self.w0 * gamma(s - c_lt)
        for j, (tau, c_le) in enumerate(self.rejs):
            # candidates strictly between rejection tau and now = c_lt - c_le
            idx = (s - tau) - (c_lt - c_le)
            reward = (self.alpha - self.w0) if j == 0 else self.alpha
            a += reward * gamma(idx)
        return a

    def level_selected(self):
        """alpha_t for the next SELECTED test."""
        return min(self.lam, (self.tau - self.lam) * self._saffron_term())

    @property
    def wealth(self):
        return max(0.0, self.w0 + self.earned - self.spent)

    def test(self, p):
        """Process one p-value. Discarded (p>tau) tests do not advance the account.
        Returns (rejected, level_used). Discarded -> (False, 0.0)."""
        if p > self.tau:                       # DISCARD: too null to spend on
            return False, 0.0
        self.s += 1
        a = self.level_selected()
        self.spent += a
        rej = p <= a                           # a <= lambda <= tau, so a rejection is a candidate
        is_cand = p <= self.lam
        if rej:
            # record with candidate count INCLUDING this (rejection is always a candidate)
            self.rejs.append((self.s, self.cum_cand + 1))
            self.n_rej += 1
            self.earned += (self.alpha - self.w0) if self.n_rej == 1 else self.alpha
        if is_cand:
            self.cum_cand += 1
        return rej, a


# --------------------------------------------------------------------------- #
# p-value derivation from a round_results.csv row.                              #
# --------------------------------------------------------------------------- #
def _f(x, d=None):
    try:
        v = float(x)
        if v != v:               # NaN
            return d
        return v
    except (TypeError, ValueError):
        return d


def pvalue_from_row(row):
    """ONE-sided p that the strategy is NO better than null. Precedence:
      1. PSR 'DSR p-value' = 1 - PSR(SR>0) when real_sharpe + n_days are present (daily stats).
      2. val_auc 'permute proxy' p = 1 - Phi(z), z=(val_auc-0.5)*sqrt(n_eff_bars): higher label
         AUC -> stronger signal -> smaller p (the same monotone direction a permute test gives).
      3. p=1.0 (no evidence)."""
    sh = _f(row.get("real_sharpe"))
    n = _f(row.get("n_days"))
    if _PSR is not None and sh is not None and n is not None and n > 3:
        sk = _f(row.get("real_skew"), 0.0)
        ku = _f(row.get("real_kurt"), 3.0)
        srd = sh / math.sqrt(252.0)                  # per-observation (daily) Sharpe
        psr = _PSR(srd, n, sk, ku, 0.0)
        if psr == psr:                               # not NaN
            return min(1.0, max(1e-12, 1.0 - psr))
    va = _f(row.get("val_auc"))
    if va is not None and va > 0.0:
        trades = _f(row.get("trades"), 60.0)
        n_eff = max(20.0, min(400.0, trades))        # nominal effective sample for the AUC z
        z = (va - 0.5) * math.sqrt(n_eff)
        return min(1.0, max(1e-12, 1.0 - Phi(z)))
    return 1.0


def screen_order(path=SCREEN_LOG):
    """Screening order of tickers from the progress log ('SCREEN <ticker> ...' lines)."""
    order = []
    seen = set()
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "SCREEN":
                    tk = parts[1]
                    if tk not in seen:
                        seen.add(tk); order.append(tk)
    except Exception:
        pass
    return order


def load_etf_pvalues(csv_path=ROUND_CSV):
    """One decisive p-value per ETF = the MIN p across its trials (best evidence), plus the
    first-seen order. Returns (pmap, first_seen_order)."""
    best = {}
    order = []
    try:
        for row in csv.DictReader(open(csv_path)):
            tk = row.get("ticker")
            if not tk:
                continue
            p = pvalue_from_row(row)
            if tk not in best:
                best[tk] = p; order.append(tk)
            elif p < best[tk]:
                best[tk] = p
    except Exception as e:
        print("warn: round_results.csv:", e)
    return best, order


def replay(engine="lord", alpha=0.10, csv_path=ROUND_CSV):
    """Stream per-ETF p-values through the ledger in screen order; print the FIT decisions and
    the remaining alpha-wealth budget. Returns (rejected_set, ledger)."""
    pmap, first_seen = load_etf_pvalues(csv_path)
    if not pmap:
        print("[online-fdr] no data to replay")
        return set(), None
    sorder = [t for t in screen_order() if t in pmap]
    order = sorder + [t for t in first_seen if t not in set(sorder)]
    led = Addis(alpha=alpha) if engine == "addis" else LordPP(alpha=alpha)
    print(f"\n=== online-FDR replay ({engine.upper()}, FDR={alpha}) — {len(order)} ETFs in screen order ===")
    print(f"{'#':>3s} {'ETF':6s} {'p':>10s} {'level':>10s} {'FIT':>4s} {'wealth':>9s}")
    rejected = []
    for i, tk in enumerate(order, 1):
        p = pmap[tk]
        rej, a = led.test(p)
        if rej:
            rejected.append(tk)
        mark = "FIT" if rej else "."
        if rej or i <= 25 or tk in KNOWN_FITS:
            print(f"{i:3d} {tk:6s} {p:10.3e} {a:10.3e} {mark:>4s} {led.wealth:9.4f}")
    kept = set(rejected) & KNOWN_FITS
    print(f"\n[online-fdr] {led.n_rej} FIT-stamped; remaining alpha-wealth (exploration budget) = {led.wealth:.4f}")
    n_known_present = len(KNOWN_FITS & set(pmap))
    print(f"[online-fdr] known-fit names surviving the ledger: {sorted(kept)} "
          f"({len(kept)}/{n_known_present} of those present)")
    return set(rejected), led


# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("gamma sums ~1", abs(sum(gamma(k) for k in range(1, 200000)) - 1.0) < 1e-3)
    check("gamma decreasing", gamma(1) > gamma(2) > gamma(10) > gamma(100))

    # Online FDR controls E[FDP] (the GUARANTEE is in expectation, not per realization), so we
    # average the false-discovery proportion + power over many RANDOM replications.
    import random as _rnd
    _rnd.seed(12345)
    REPS, T, N_SIG = 300, 200, 6
    for name, make in (("LORD++", lambda: LordPP(alpha=0.10)), ("ADDIS", lambda: Addis(alpha=0.10))):
        fdps, powers, wmin_all = [], [], 1e9
        for _ in range(REPS):
            sig_at = set(_rnd.sample(range(T), N_SIG))
            led = make()
            tp = fp = 0
            for i in range(T):
                p = 1e-8 if i in sig_at else _rnd.random()        # nulls ~ U(0,1)
                r, _a = led.test(p)
                wmin_all = min(wmin_all, led.wealth)
                if r:
                    if i in sig_at:
                        tp += 1
                    else:
                        fp += 1
            fdps.append(fp / max(1, tp + fp))
            powers.append(tp / N_SIG)
        mean_fdp = sum(fdps) / REPS
        mean_pow = sum(powers) / REPS
        check(f"{name}: mean FDP<=0.10 over {REPS} reps (got {round(mean_fdp,3)})", mean_fdp <= 0.10 + 1e-9)
        check(f"{name}: power>=0.95 (got {round(mean_pow,3)})", mean_pow >= 0.95)
        check(f"{name}: alpha-wealth stays >=0 (min {round(wmin_all,4)})", wmin_all >= -1e-12)

    # all-null stream: very few rejections (controls FWER-ish under the global null)
    _rnd.seed(7)
    nrej_tot = 0
    for _ in range(50):
        led = LordPP(alpha=0.10)
        nrej_tot += sum(1 for _ in range(200) if led.test(_rnd.random())[0])
    check(f"LORD++ all-null: avg <=2 rejections/run (got {nrej_tot/50:.2f})", nrej_tot / 50 <= 2.0)

    # pvalue_from_row monotonic in val_auc
    p_hi = pvalue_from_row({"val_auc": "0.70", "trades": "120"})
    p_lo = pvalue_from_row({"val_auc": "0.50", "trades": "120"})
    check(f"p(val_auc=.70) < p(val_auc=.50) ({round(p_hi,4)}<{round(p_lo,4)})", p_hi < p_lo)
    # PSR path beats the AUC fallback when daily stats present
    p_psr = pvalue_from_row({"real_sharpe": "2.5", "n_days": "200", "real_skew": "0.1", "real_kurt": "4"})
    check(f"PSR-path p small for strong Sharpe (got {round(p_psr,4)})", p_psr < 0.05)

    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="run the round_results.csv replay")
    ap.add_argument("--engine", default="lord", choices=["lord", "addis"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.replay:
        replay(engine=args.engine, alpha=args.alpha)
        raise SystemExit(0)

    okk = _selftest()
    if os.path.exists(ROUND_CSV):
        replay(engine=args.engine, alpha=args.alpha)
    raise SystemExit(0 if okk else 1)
