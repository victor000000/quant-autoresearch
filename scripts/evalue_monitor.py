#!/usr/bin/env python3
"""ANYTIME-VALID e-value monitor for champion edges (efficiency-review v3 #1 upgrade).

We re-validate champions repeatedly as the OOS window grows ("records go stale"). That is optional
stopping / peeking, which INVALIDATES p-value- and DSR-based gates (Type-I error inflates under
continuous monitoring — the reproducibility-crisis cause). The fix: an e-process. E-values stay
Type-I-valid under continuous monitoring (Ville's inequality), so a champion's e-value can be checked
as often as we like; merging follow-up re-validations is just MULTIPLICATION.

Method: testing-by-betting (Waudby-Smith & Ramdas 2023). H0: mean return <= 0 (edge dead/noise).
Capital K_0=1; at each step bet a PREDICTABLE fraction lambda_t (from past returns only) of the next
return: K_t = K_{t-1} * (1 + lambda_t * r_t), lambda_t in [0, LAM_MAX]. Under H0, K_t is a non-negative
supermartingale with E[K_t]<=1 => an e-process. Ville: P(sup_t K_t >= 1/alpha) <= alpha. So
K (the e-value) >= 20 == anytime-valid significance at 0.05, VALID no matter how often we peek.

Returns floored at RET_FLOOR=-0.18 (a true lower bound on in-period weekly ETF returns, so no
anti-conservative clipping) so that with LAM_MAX=5, 1+lambda*r >= 0.1 > 0 always. Bet = adaptive plug-in lambda_t = clip(mean_{<t}/var_{<t}, 0, LAM_MAX)
(aGRAPA-style GROW bet).
"""
import json, os, math

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "results", "series_cache.json")
OUT = os.path.join(HERE, "HONEST_AUDIT.md")
LAM_MAX = 5.0          # bet cap; with return-floor -0.18 => 1+lambda*r >= 1+5*(-0.18)=0.1 > 0
RET_FLOOR = -0.18      # true lower bound on in-period weekly ETF returns (no -18% weeks) => no anti-conservative clipping; keeps 1+lambda*r>0


def betting_eprocess(returns, lam_max=LAM_MAX):
    """Anytime-valid e-value for H0: mean<=0, via a predictable GROW bet. Returns (e_value, path,
    decay_flag) where decay_flag is True if the e-process is STALLING in the recent half (evidence
    no longer accumulating -> possible decay)."""
    K = 1.0
    s = 0.0      # running sum of returns (for predictable mean)
    s2 = 0.0     # running sum of squares
    n = 0
    path = [1.0]
    for r in returns:
        r = max(RET_FLOOR, float(r))
        # predictable bet from PAST stats only (computed before seeing r)
        if n >= 5:
            mean = s / n
            var = max(1e-9, s2 / n - mean * mean)
            lam = mean / var
            lam = 0.0 if lam < 0 else (lam_max if lam > lam_max else lam)
        else:
            lam = 0.0
        K = K * (1.0 + lam * r)
        K = max(K, 1e-12)
        path.append(K)
        s += r; s2 += r * r; n += 1
    # decay heuristic: did the e-process grow in the FIRST half but stall/shrink in the SECOND?
    h = len(path) // 2
    # DECAY = genuinely alive by the midpoint (e-process reached real evidence, e>=5) BUT the recent
    # half stopped accumulating (added <10%). Requiring e>=5 (not mere 2x growth) avoids false-flagging
    # never-alive noise series whose early capital wandered up by luck.
    decay = (path[h] >= 5.0) and (path[-1] < path[h] * 1.1)
    return K, path, decay


def _selftest():
    import random
    rng = random.Random(0)
    # positive-drift edge: e-value should grow LARGE
    pos = [0.005 + 0.02 * rng.gauss(0, 1) for _ in range(300)]   # strong edge, Sharpe ~0.25/step
    ep, _, _ = betting_eprocess(pos)
    # zero-mean noise: e-value should rarely exceed 20 (false-positive control)
    maxe = 0.0
    for trial in range(200):
        rng2 = random.Random(1000 + trial)
        noise = [0.01 * rng2.gauss(0, 1) for _ in range(300)]
        e, _, _ = betting_eprocess(noise)
        maxe = max(maxe, e)
    fp_rate = 0.0
    cnt = 0
    for trial in range(200):
        rng2 = random.Random(5000 + trial)
        noise = [0.01 * rng2.gauss(0, 1) for _ in range(300)]
        e, _, _ = betting_eprocess(noise)
        cnt += (e >= 20)
    fp_rate = cnt / 200
    print(f"[selftest] positive-drift e-value = {ep:.1f} (want >>20)")
    print(f"[selftest] zero-mean false-positive rate at e>=20: {fp_rate:.3f} (want <= ~0.05, Ville bound)")
    return ep > 20 and fp_rate <= 0.05


def main():
    if not os.path.exists(CACHE):
        print("no series cache"); return
    cache = json.load(open(CACHE))
    champs = [("GLD", "band-0.03 champion 4.71"), ("SOXX", "3.02"), ("UUP", "1.30 fragile"),
              ("TIP", "diversifier"), ("DBC", "diversifier"), ("HYG", "diversifier")]
    lines = ["", "## Anytime-valid e-value monitor (peeking-robust; supersedes p-value/DSR re-checks)",
             "", "H0: mean return <= 0 (edge dead). E-value >= 20 = significant at 0.05, VALID under continuous",
             "monitoring (re-check anytime; merge re-validations by MULTIPLICATION). Testing-by-betting (WSR 2023).",
             "", "```", f"{'champ':5s} {'e-value':>10s} {'AV p=1/e':>9s} {'verdict':>14s}  decay?"]
    print("\n".join(lines[1:]))
    rows = []
    for tk, note in champs:
        if tk not in cache:
            continue
        ser = cache[tk]
        ts = sorted(ser, key=lambda x: int(x))
        rets = [ser[ts[i]] / ser[ts[i - 1]] - 1.0 for i in range(1, len(ts)) if ser[ts[i - 1]] > 0]
        e, _, decay = betting_eprocess(rets)
        avp = 1.0 / e if e > 0 else 1.0
        verdict = "ALIVE (e>=20)" if e >= 20 else ("weak (e>=1)" if e >= 1 else "no evidence")
        row = f"{tk:5s} {e:10.2f} {min(1.0,avp):9.4f} {verdict:>14s}  {'DECAY' if decay else 'holding'}"
        print(row); lines.append(row); rows.append((tk, e, decay))
    lines.append("```")
    lines.append("")
    lines.append("Anytime-valid: unlike DSR/p-values, these e-values stay honest no matter how many times we "
                 "re-check as the OOS window grows. Next re-validation just MULTIPLIES the new e-value in.")
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    m = "## Anytime-valid e-value monitor"
    if m in prev:
        prev = prev[:prev.index(m)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        ok = _selftest()
        print("SELFTEST", "PASS" if ok else "FAIL")
    else:
        main()
