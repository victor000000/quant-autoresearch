#!/usr/bin/env python3
"""TRAIN-only PREDICTABILITY pre-screen + mechanism router (311-plan step 1, top quick-win).

Replaces the hand-coded asset-class + AUM ordering in screen_etfs.py with a cheap, a-priori,
LEAK-SAFE triage that scores how EXPLOITABLE each series is and which MECHANISM it admits, so
driver-hours skip 0-EV / redundant names. Generalises beta_router.py / mechanism_router.py (which
emit a single beta_n / rho_k) to a composite of the SOTA forecastability statistics:

  * Lo-MacKinlay variance-ratio VR(q) with the HETEROSKEDASTICITY-ROBUST z* (Lo-MacKinlay 1988;
    Choi 1999) for q in {2,10} — the cleanest trend(VR>1)/revert(VR<1) discriminator.
  * Averaged HURST exponent = mean(R/S, DFA) — persistence (H>0.5 trend / H<0.5 revert).
  * Weighted PERMUTATION ENTROPY (Bandt-Pompe 2002; Fadlallah 2013) — a forecastability CEILING
    (low WPE => structured => exploitable).
  * Campbell-Thompson OOS-R2 (2008) of a cheap AR(1) predictor vs the prevailing mean.
  * Amihud (2002) illiquidity — implemented but DEFERRED in the footer probe (bars carry log_close
    only, no per-bar volume); supply dollar-volume to use it offline.

Routing (plan cut-points): VR>1 / H>0.55 -> trend (trend_leg); VR<1 / H<0.45 -> revert;
VR~1 / H~0.5 / R2<=0 -> buy-hold (SKIP).

LEAK-SAFETY: every statistic is computed on TRAIN bars ONLY. The estimators are written ONCE in
_PROBE_SRC and (a) exec'd into this module for offline/selftest use, (b) indent-injected into the
footer template — the SAME beta_router footer-injection probe technique — so the online TRAIN-only
signature is byte-identical to the offline library and can never depend on post-TRAIN bars.

Run `python3 scripts/predictability_screen.py`              -> selftest (offline, no QC).
Run `python3 scripts/predictability_screen.py --demo`       -> print full signatures of synthetics.
Run `python3 scripts/predictability_screen.py --run GLD USO -> footer-injection probe on QC (needs
                                                               the harness; computes TRAIN signatures).
"""
import sys, os, argparse, textwrap
import numpy as np   # exec(_PROBE_SRC) and the footer both reference the global name `np`


# --------------------------------------------------------------------------- #
# SINGLE SOURCE OF TRUTH for the estimators. Pure-numpy, no imports beyond the  #
# global `np`, no f-strings, no getattr (lint-clean for the QC minifier). This  #
# string is exec'd into the module (offline) AND indent-injected into the footer #
# (online) so the two paths are guaranteed identical.                           #
# --------------------------------------------------------------------------- #
_PROBE_SRC = r'''
def _ps_vr(r, q):
    """Lo-MacKinlay VR(q) and heteroskedasticity-robust z*. Returns (vr, z). VR>1 => positively
    autocorrelated returns (trend/momentum); VR<1 => mean reversion; z*~N(0,1) under the RW null."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < q + 2 or q < 2:
        return 1.0, 0.0
    mu = float(np.mean(r))
    dev = r - mu
    dev2 = dev * dev
    s_dev2 = float(np.sum(dev2))
    sig_a = s_dev2 / (n - 1)
    if sig_a <= 1e-300:
        return 1.0, 0.0
    csum = np.concatenate((np.array([0.0]), np.cumsum(r)))
    y = csum[q:] - csum[:-q]                      # overlapping q-period sums
    m = q * (n - q + 1) * (1.0 - float(q) / n)
    if m <= 0:
        return 1.0, 0.0
    sig_c = float(np.sum((y - q * mu) ** 2)) / m
    vr = sig_c / sig_a
    denom = s_dev2 * s_dev2
    theta = 0.0
    for j in range(1, q):
        num = float(np.sum(dev2[j:] * dev2[:-j]))
        delta = (num / denom) if denom > 1e-300 else 0.0
        w = (2.0 * (q - j) / q) ** 2
        theta += w * delta
    z = ((vr - 1.0) / np.sqrt(theta)) if theta > 1e-300 else 0.0
    return float(vr), float(z)


def _ps_hurst_rs(x):
    """Rescaled-range (R/S) Hurst exponent: slope of log(mean R/S) vs log(window)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 32:
        return 0.5
    logn = []
    logrs = []
    win = 8
    while win <= n // 2:
        k = n // win
        vals = []
        for c in range(k):
            seg = x[c * win:(c + 1) * win]
            z = seg - np.mean(seg)
            cz = np.cumsum(z)
            R = float(np.max(cz) - np.min(cz))
            S = float(np.std(seg))
            if S > 1e-300 and R > 0:
                vals.append(R / S)
        if vals:
            logn.append(np.log(win))
            logrs.append(np.log(np.mean(vals)))
        win *= 2
    if len(logn) < 3:
        return 0.5
    A = np.vstack([np.array(logn), np.ones(len(logn))]).T
    slope = float(np.linalg.lstsq(A, np.array(logrs), rcond=None)[0][0])
    return float(min(1.0, max(0.0, slope)))


def _ps_hurst_dfa(x):
    """Detrended fluctuation analysis exponent (~Hurst): integrate, linear-detrend per scale,
    slope of log F(s) vs log s."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 32:
        return 0.5
    y = np.cumsum(x - np.mean(x))
    logs = []
    logf = []
    s = 8
    while s <= n // 4:
        k = n // s
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        rms = []
        for c in range(k):
            seg = y[c * s:(c + 1) * s]
            coef = np.linalg.lstsq(A, seg, rcond=None)[0]
            fit = A @ coef
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        if rms:
            logs.append(np.log(s))
            logf.append(np.log(np.mean(rms)))
        s *= 2
    if len(logs) < 3:
        return 0.5
    A2 = np.vstack([np.array(logs), np.ones(len(logs))]).T
    slope = float(np.linalg.lstsq(A2, np.array(logf), rcond=None)[0][0])
    return float(min(1.0, max(0.0, slope)))


def _ps_wpe(x, m=3):
    """Weighted permutation entropy normalised to [0,1] (1 = max randomness). Amplitude-weighted
    ordinal patterns of length m (Fadlallah 2013)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < m + 2:
        return 1.0
    weights = {}
    total_w = 0.0
    for i in range(n - m + 1):
        w = x[i:i + m]
        pat = tuple(np.argsort(w, kind="mergesort").tolist())
        wt = float(np.var(w))
        weights[pat] = weights.get(pat, 0.0) + wt
        total_w += wt
    if total_w <= 1e-300:
        return 1.0
    H = 0.0
    for v in weights.values():
        p = v / total_w
        if p > 0:
            H -= p * np.log(p)
    fact = 1.0
    for kk in range(2, m + 1):
        fact *= kk
    norm = np.log(fact)
    return float(min(1.0, max(0.0, H / norm)))


def _ps_ct_r2(r, frac=0.7):
    """Campbell-Thompson OOS-R2 of an AR(1) one-step forecast vs the prevailing mean. ONE TRAIN
    split (fit on the first `frac`, evaluate on the rest) => leak-safe AND append-OOS-invariant."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 40:
        return 0.0
    h = int(n * frac)
    if h < 20 or n - h < 10:
        return 0.0
    x0 = r[:h - 1]
    y0 = r[1:h]
    mx = float(np.mean(x0))
    my = float(np.mean(y0))
    sxx = float(np.sum((x0 - mx) ** 2))
    if sxx <= 1e-300:
        return 0.0
    b = float(np.sum((x0 - mx) * (y0 - my))) / sxx
    a = my - b * mx
    bench = float(np.mean(r[:h]))
    pred = a + b * r[h - 1:n - 1]
    sse_m = float(np.sum((r[h:n] - pred) ** 2))
    sse_b = float(np.sum((r[h:n] - bench) ** 2))
    if sse_b <= 1e-300:
        return 0.0
    return float(1.0 - sse_m / sse_b)


def _ps_amihud(r, dvol):
    """Amihud 2002 illiquidity = 1e6 * mean(|r| / dollar_volume). DEFERRED in the footer probe
    (bars carry log_close only); supply a dollar-volume array to use it offline."""
    r = np.asarray(r, dtype=float)
    dvol = np.asarray(dvol, dtype=float)
    mask = np.isfinite(r) & np.isfinite(dvol) & (dvol > 0)
    if int(np.sum(mask)) < 5:
        return float("nan")
    return float(np.mean(np.abs(r[mask]) / dvol[mask]) * 1e6)


def _ps_signature(r, dvol=None):
    """Composite exploitability signature from a TRAIN return array (and optional dollar-volume)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    vr2, z2 = _ps_vr(r, 2)
    vr10, z10 = _ps_vr(r, 10)
    h_rs = _ps_hurst_rs(r)
    h_dfa = _ps_hurst_dfa(r)
    h = 0.5 * (h_rs + h_dfa)
    wpe = _ps_wpe(r, 3)
    ct = _ps_ct_r2(r)
    amh = _ps_amihud(r, dvol) if dvol is not None else float("nan")
    vr_sig = min(1.0, abs(z10) / 3.0)
    h_sig = min(1.0, abs(h - 0.5) / 0.25)
    wpe_sig = min(1.0, max(0.0, 1.0 - wpe))
    ct_sig = (min(1.0, max(0.0, ct / 0.05))) if ct > 0 else 0.0
    comp = 0.30 * vr_sig + 0.30 * h_sig + 0.20 * wpe_sig + 0.20 * ct_sig
    return {"vr2": vr2, "z2": z2, "vr10": vr10, "z10": z10, "hurst_rs": h_rs,
            "hurst_dfa": h_dfa, "hurst": h, "wpe": wpe, "ct_r2": ct, "amihud": amh,
            "composite": float(comp)}


def _ps_route(sig):
    """trend / revert / buy-hold from a signature dict (plan cut-points)."""
    vr = sig["vr10"]
    z = sig["z10"]
    h = sig["hurst"]
    ct = sig["ct_r2"]
    comp = sig["composite"]
    trend = (z > 1.64 and vr > 1.0) or (h > 0.55)
    revert = (z < -1.64 and vr < 1.0) or (h < 0.45)
    if trend and not revert:
        return "trend"
    if revert and not trend:
        return "revert"
    if comp < 0.15 and ct <= 0.0:
        return "buy-hold"
    return "trend" if (vr > 1.0 or h > 0.5) else "revert"
'''

# Make the estimators importable in THIS module (same code the footer will run).
exec(_PROBE_SRC, globals())

# Readable public aliases.
variance_ratio = _ps_vr                       # noqa: F821 (defined by exec above)
hurst_rs = _ps_hurst_rs                        # noqa: F821
hurst_dfa = _ps_hurst_dfa                      # noqa: F821
weighted_permutation_entropy = _ps_wpe         # noqa: F821
campbell_thompson_r2 = _ps_ct_r2               # noqa: F821
amihud_illiquidity = _ps_amihud                # noqa: F821
signature = _ps_signature                      # noqa: F821
route = _ps_route                              # noqa: F821


# --------------------------------------------------------------------------- #
# Footer-injection probe (the beta_router technique) — emits the TRAIN-only      #
# predictability signature + route per ticker via runtime statistics.            #
# --------------------------------------------------------------------------- #
FOOTER = "templates/footer.py.tmpl"
TARGET = "fwd_ret, fwd_vol = compute_forward_metrics(lc, lr)\n"


def _build_inject():
    """TARGET + a try-guarded block: nested estimator defs (from _PROBE_SRC) + the TRAIN-only
    emission. Defs are indented to 16 spaces (inside the try at 12); they reference the QC global
    `np`. All stats use tr_m & fv ONLY => leak-safe + append-OOS-invariant."""
    defs = textwrap.indent(_PROBE_SRC.strip("\n") + "\n", " " * 16)
    emit = (
        "            try:\n"
        + defs +
        "                _m = tr_m & fv & np.isfinite(lr)\n"
        "                _rtr = lr[_m]\n"
        "                if int(_rtr.size) > 64:\n"
        "                    _sig = _ps_signature(_rtr)\n"
        "                    self.set_runtime_statistic('ps_composite', str(round(float(_sig['composite']), 4)))\n"
        "                    self.set_runtime_statistic('ps_route', _ps_route(_sig))\n"
        "                    self.set_runtime_statistic('ps_vr2', str(round(float(_sig['vr2']), 4)))\n"
        "                    self.set_runtime_statistic('ps_z2', str(round(float(_sig['z2']), 4)))\n"
        "                    self.set_runtime_statistic('ps_vr10', str(round(float(_sig['vr10']), 4)))\n"
        "                    self.set_runtime_statistic('ps_z10', str(round(float(_sig['z10']), 4)))\n"
        "                    self.set_runtime_statistic('ps_hurst', str(round(float(_sig['hurst']), 4)))\n"
        "                    self.set_runtime_statistic('ps_wpe', str(round(float(_sig['wpe']), 4)))\n"
        "                    self.set_runtime_statistic('ps_ctr2', str(round(float(_sig['ct_r2']), 4)))\n"
        "            except Exception:\n"
        "                pass\n"
    )
    return TARGET + emit


def run_probe(tickers):
    """Inject the signature emission into the footer, run one TRAIN job per ticker, read+route."""
    from lb.harness.orchestrator import render_train_config
    from lb.harness.qc_client import submit_and_wait

    inject = _build_inject()
    orig = open(FOOTER).read()
    n = orig.count(TARGET)
    if n != 1:
        print(f"ABORT: anchor appears {n}x (need exactly 1)")
        return
    try:
        open(FOOTER, "w").write(orig.replace(TARGET, inject, 1))
        print(f"injected predictability-signature emission; running {tickers}", flush=True)
        print(f"{'ETF':6s} {'route':>9s} {'comp':>6s} {'vr10':>7s} {'z10':>7s} {'hurst':>6s} {'wpe':>6s} {'ctR2':>7s}", flush=True)
        for tk in tickers:
            cfg = {"ticker": tk, "axis": "logdollar", "labeler": "trend_leg", "thresh": 0.45, "sizing": "cdf_overlay"}
            tcode, extra = render_train_config(cfg)
            bt, st = submit_and_wait(tcode, f"pscreen_{tk}", timeout_s=300, extra_files=extra)
            rt = (bt.get("runtimeStatistics", {}) or {}) if isinstance(bt, dict) else {}
            print(f"{tk:6s} {str(rt.get('ps_route','?')):>9s} {str(rt.get('ps_composite','?')):>6s} "
                  f"{str(rt.get('ps_vr10','?')):>7s} {str(rt.get('ps_z10','?')):>7s} "
                  f"{str(rt.get('ps_hurst','?')):>6s} {str(rt.get('ps_wpe','?')):>6s} "
                  f"{str(rt.get('ps_ctr2','?')):>7s}  ({st})", flush=True)
    finally:
        open(FOOTER, "w").write(orig)
        print("restored original footer", flush=True)


# --------------------------------------------------------------------------- #
def _ar1(n, phi, seed, drift=0.0):
    """AR(1) return series r_t = drift + phi r_{t-1} + eps. phi>0 trend, phi<0 revert, phi=0 RW."""
    rng = np.random.RandomState(seed)
    eps = rng.normal(0.0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = drift + phi * r[t - 1] + eps[t]
    return r


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    N = 3000
    trend = _ar1(N, 0.30, 1)     # positive autocorr -> VR>1, H>0.5, route trend
    revert = _ar1(N, -0.30, 2)   # negative autocorr -> VR<1, H<0.5, route revert
    rw = _ar1(N, 0.0, 3)         # iid -> VR~1, H~0.5, low composite, route buy-hold

    st = _ps_signature(trend)
    sr = _ps_signature(revert)
    sw = _ps_signature(rw)
    print(f"  trend  : route={_ps_route(st):8s} vr10={st['vr10']:.3f} z10={st['z10']:+.2f} hurst={st['hurst']:.3f} wpe={st['wpe']:.3f} comp={st['composite']:.3f}")
    print(f"  revert : route={_ps_route(sr):8s} vr10={sr['vr10']:.3f} z10={sr['z10']:+.2f} hurst={sr['hurst']:.3f} wpe={sr['wpe']:.3f} comp={sr['composite']:.3f}")
    print(f"  rwalk  : route={_ps_route(sw):8s} vr10={sw['vr10']:.3f} z10={sw['z10']:+.2f} hurst={sw['hurst']:.3f} wpe={sw['wpe']:.3f} comp={sw['composite']:.3f}")

    check("VR>1 for trend (positive autocorr)", st["vr10"] > 1.0 and st["z10"] > 1.64)
    check("VR<1 for revert (negative autocorr)", sr["vr10"] < 1.0 and sr["z10"] < -1.64)
    check("VR~1 for random walk (|z|<2)", abs(sw["z10"]) < 2.0)
    check("Hurst trend > 0.5 > revert", st["hurst"] > 0.5 > sr["hurst"])
    check("route(trend)=='trend'", _ps_route(st) == "trend")
    check("route(revert)=='revert'", _ps_route(sr) == "revert")
    check("route(random walk)=='buy-hold'", _ps_route(sw) == "buy-hold")
    check("composite exploitable > RW", min(st["composite"], sr["composite"]) > sw["composite"])
    check("composite in [0,1]", all(0.0 <= s["composite"] <= 1.0 for s in (st, sr, sw)))
    # WPE: structured series < random walk (forecastability ceiling)
    check("WPE(trend)<WPE(rwalk) (more structure)", st["wpe"] < sw["wpe"])
    # Amihud offline path
    amh = _ps_amihud(np.abs(trend) + 1e-4, np.full(N, 1e7))
    check("amihud finite when dvol supplied", amh == amh and amh > 0)
    # injection builds + parses cleanly (the leak-safe probe the docstring promises).
    # Reconstruct the in-file context: TARGET sits at the method's 12-space body indent and the
    # emit block carries its own absolute indentation, so wrap as a 12-space function body.
    import ast as _ast
    inj = _build_inject()
    parses = True
    try:
        _ast.parse("def _f(self, fv, tr_m, lr, lc):\n            " + inj)
    except SyntaxError:
        parses = False
    check("footer injection block parses (leak-safe probe)", parses)

    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", metavar="TICKER", help="footer-injection probe on QC")
    ap.add_argument("--demo", action="store_true", help="print full signatures of synthetic series")
    args = ap.parse_args()

    if args.run:
        run_probe(args.run)
        raise SystemExit(0)
    if args.demo:
        for name, phi in (("trend(+0.3)", 0.3), ("revert(-0.3)", -0.3), ("randomwalk", 0.0)):
            s = _ps_signature(_ar1(3000, phi, 42))
            print(name, "->", _ps_route(s), {k: round(v, 4) for k, v in s.items()})
        raise SystemExit(0)
    raise SystemExit(0 if _selftest() else 1)
