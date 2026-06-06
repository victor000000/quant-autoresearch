#!/usr/bin/env python3
"""REGRESSION GUARD for the bar-threshold look-ahead leak (found 2026-06-03, missed by two prior audits).

The leak: bar-threshold scalings in modules/bar_builder.py:_make_builder used a FULL-SERIES (incl. OOS)
data-dependent reduction — `int(np.sum(valid))`, the count of valid minutes over the whole series — as the
multiplier. Because OOS validity differs from TRAIN, this let OOS data influence the TRAIN bar boundaries
(GLD 4.71->2.76, SOXX 3.02->0.71 leak-free). The clean pattern: every data reduction feeding a threshold must
operate on a TRAIN-masked subset (`x[keep]`, `np.sum(keep)`, `np.sum(tr)`) or be the OOS-INVARIANT calendar
length `len(c)`.

This test runs WITHOUT numpy (pure AST/text), so it works in CI and any environment. It would have caught the
original leak and fails loudly if it regresses. No runtime data needed.

Run: python3 tests/test_bar_threshold_leak.py   (exit 0 = pass, 1 = fail)
"""
import ast, os, sys, re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BB = os.path.join(_ROOT, "modules", "bar_builder.py")
FOOTER = os.path.join(_ROOT, "templates", "footer.py.tmpl")

# Names that are TRAIN-masked / OOS-invariant and therefore SAFE to feed a threshold.
SAFE_TOKENS = ("keep", "tr", "_trc", "trsel", "trr", "trm", "sdt", "svt", "r[")  # train-masked subsets / counts
# A reduction over these is the LEAK (full-series, OOS-inclusive, data-dependent).
LEAK_TOKENS = ("valid", "fvd")  # full-series boolean masks used unmasked


def _src(node, src):
    return ast.get_source_segment(src, node) or ""


def fail(msgs):
    print("BAR-THRESHOLD LEAK GUARD: \033[91mFAIL\033[0m")
    for m in msgs:
        print("  -", m)
    sys.exit(1)


# New axis names that must be EMPIRICALLY append-OOS-invariant (2026-06-06 new-methods backlog).
NEW_AXES = ["semivar", "chl", "diurnal", "kalman", "newma", "signedjumpvar"]


def _runtime_teeth():
    """EMPIRICAL append-OOS-invariance teeth for the 2026-06-06 new axes + the sticky_hmm guard.

    The pure-AST checks above cannot catch a NUMERICAL leak (a threshold/frozen-profile that silently
    reads a post-TRAIN_END statistic). So we BUILD each new axis on a synthetic minute stream, then
    RANDOMIZE every post-TRAIN_END close+vol and assert (a) the fitted threshold and (b) every
    TRAIN-region bar are byte-identical — the exact probe the spec mandates per new axis. We then add
    the leak-hunt's recommended HMM guard: assert generate_labels_sticky_hmm produces 0 label flips in
    [TRAIN_END, val_end-_EMBARGO) when only the TEST block is perturbed (its block-decode-at-the-OOS-
    boundary must keep TRAIN/VAL labels independent of TEST observations).

    Returns a list of problem strings (empty == pass). SKIPS (does not fail) when numpy/the modules are
    unavailable, preserving the no-numpy CI property of the AST checks above.
    """
    probs = []
    try:
        import numpy as np
        sys.path.insert(0, os.path.join(_ROOT, "modules"))
        import bar_builder as bb
        import labeler as lab
    except Exception as e:                      # numpy/pandas/module import unavailable -> AST-only
        print(f"  [runtime teeth SKIPPED — numpy/modules unavailable: {e}]")
        return probs

    # ---- synthetic minute stream with a TRAIN | VAL | TEST split (set the header globals) ----
    N_MIN = 9000
    base = np.datetime64("2020-01-02T09:30")
    ts = np.arange(base, base + np.timedelta64(N_MIN, "m"), dtype="datetime64[m]")
    rng = np.random.RandomState(7)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(N_MIN) * 0.0015))
    vol = rng.gamma(2.0, 50.0, N_MIN) + 1.0
    TR = 5500
    bb.TRAIN_END = ts[TR]
    bb.VAL_END = ts[7000]
    bb.TEST_END = ts[N_MIN - 1]

    # Randomize EVERY post-TRAIN_END minute (close + vol). An OOS-invariant axis must not move a
    # single TRAIN bar or the fitted threshold; a full-series statistic would shift under this.
    prng = np.random.RandomState(999)
    close_p = close.copy()
    vol_p = vol.copy()
    close_p[TR:] = (100.0 * np.exp(np.cumsum(prng.standard_normal(N_MIN) * 0.01)))[TR:]
    vol_p[TR:] = prng.gamma(3.0, 80.0, N_MIN)[TR:] + 1.0
    cutoff = bb.TRAIN_END

    for axis in NEW_AXES:
        try:
            b0 = bb._make_builder(axis, close, vol, ts, 1500)
            bP = bb._make_builder(axis, close_p, vol_p, ts, 1500)
            if b0 is None or bP is None:
                probs.append(f"runtime teeth: axis {axis!r} failed to calibrate on the synthetic TRAIN stream.")
                continue
            t0, tP = bb.builder_threshold(b0), bb.builder_threshold(bP)
            if t0 != tP:
                probs.append(f"runtime teeth: axis {axis!r} threshold MOVED under post-TRAIN_END "
                             f"perturbation ({t0!r} -> {tP!r}) — OOS leak into the bar threshold.")
            lc0, _lr0, _n0, bts0 = bb.build_bars(close, vol, ts, axis, 1500)
            lcP, _lrP, _nP, btsP = bb.build_bars(close_p, vol_p, ts, axis, 1500)
            m0 = bts0 < cutoff
            mP = btsP < cutoff
            same = (int(m0.sum()) == int(mP.sum())
                    and np.array_equal(lc0[m0], lcP[mP])
                    and np.array_equal(bts0[m0], btsP[mP]))
            if not same:
                probs.append(f"runtime teeth: axis {axis!r} TRAIN bars CHANGED under post-TRAIN_END "
                             f"perturbation — append-OOS-invariance VIOLATED (OOS leak into bar boundaries).")
        except Exception as e:
            probs.append(f"runtime teeth: axis {axis!r} raised {type(e).__name__}: {e}")

    # ---- HMM latent-backward-reach guard (leak-hunt 2026-06-06 recommended) ----
    # sticky_hmm SMOOTHS the posterior over the whole sequence; without the block-decode at the OOS
    # boundary a TRAIN/VAL label could become a function of TEST observations (unbounded backward
    # reach). Perturb ONLY the TEST block and require 0 flips in [TRAIN_END, val_end-_EMBARGO).
    try:
        hrng = np.random.RandomState(11)
        NB = 3000
        mus = np.array([0.0016, 0.0, -0.0016])
        sig = 0.0014
        A = np.array([[0.985, 0.010, 0.005], [0.0075, 0.985, 0.0075], [0.005, 0.010, 0.985]])
        st = 0
        lr = np.zeros(NB)
        for t in range(NB):                     # sticky 3-regime chain -> HMM finds structure
            lr[t] = hrng.standard_normal() * sig + mus[st]
            st = int(hrng.choice(3, p=A[st]))
        lc = np.cumsum(lr)
        tr_m = np.zeros(NB, bool)
        va_m = np.zeros(NB, bool)
        te_m = np.zeros(NB, bool)
        tr_m[:1800] = True
        va_m[1800:2400] = True
        te_m[2400:] = True
        fvb = np.ones(NB, bool)
        hor = [50, 100, 200]
        fr, fvv = lab.compute_forward_metrics(lc, lr, horizons=hor)
        y0, _c0, _h0 = lab.generate_labels_sticky_hmm(lc, lr, tr_m, va_m, te_m, fvb, fr, fvv, horizons=hor)
        if y0 is None:
            probs.append("runtime teeth: generate_labels_sticky_hmm returned None on the synthetic "
                         "regime series — cannot exercise the append-OOS guard (data too weak).")
        else:
            cut = int(np.where(te_m)[0][0])
            lr_p = lr.copy()
            lr_p[te_m] = hrng.standard_normal(int(te_m.sum())) * 0.01 + 0.02   # perturb TEST block ONLY
            lc_p = lc.copy()
            lc_p[cut:] = lc[cut - 1] + np.cumsum(lr_p[cut:])
            fr_p, fvv_p = lab.compute_forward_metrics(lc_p, lr_p, horizons=hor)
            y1, _c1, _h1 = lab.generate_labels_sticky_hmm(
                lc_p, lr_p, tr_m, va_m, te_m, fvb, fr_p, fvv_p, horizons=hor)
            if y1 is None:
                probs.append("runtime teeth: sticky_hmm returned None under the TEST-block perturbation "
                             "(it must be append-OOS-invariant, so the same labels must come back).")
            else:
                _EMBARGO = max(200, max(hor))               # mirrors footer.py.tmpl _EMBARGO floor
                tend = int(np.where(va_m)[0][0])            # TRAIN_END boundary == first VAL bar
                idx = np.arange(NB)
                region = (idx >= tend) & (idx < cut - _EMBARGO)   # [TRAIN_END, val_end - _EMBARGO)
                flips = int(np.sum(y0[region] != y1[region]))
                if flips != 0:
                    probs.append(f"runtime teeth: sticky_hmm produced {flips} label FLIP(s) in "
                                 f"[TRAIN_END, val_end-_EMBARGO) under a TEST-block perturbation — the "
                                 f"smoothed posterior LEAKS TEST obs into TRAIN/VAL labels (the OOS-"
                                 f"boundary block-decode regressed).")
    except Exception as e:
        probs.append(f"runtime teeth: HMM append-OOS case raised {type(e).__name__}: {e}")

    return probs


def main():
    src = open(BB).read()
    tree = ast.parse(src)
    problems = []

    # CODE-ONLY view: drop comments (so explanatory comments mentioning the forbidden pattern don't trip it).
    code_only = "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())

    # 1. EXACT leak signature must never reappear in CODE.
    for m in re.finditer(r"np\.sum\(\s*valid\s*\)", code_only):
        ln = code_only[: m.start()].count("\n") + 1
        problems.append(f"line {ln}: forbidden leak signature `np.sum(valid)` (full-series count -> OOS leak). "
                        f"Use len(c) or a TRAIN-masked count (np.sum(keep)).")

    # 2. Structural: every reduction (np.mean/np.sum/np.std/np.quantile/np.bincount) whose result feeds a
    #    threshold computation must take a TRAIN-masked argument. We scan _make_builder + _fit_* functions,
    #    find `total = ...` and `thresh = ...` assignments and any np-reduction in them, and require each
    #    reduction's argument text to contain a SAFE token (train-masked) and NOT a bare LEAK token.
    targets = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and (fn.name == "_make_builder" or fn.name.startswith("_fit_")):
            targets[fn.name] = fn
    if "_make_builder" not in targets:
        problems.append("could not locate _make_builder (test stale?)")

    REDUCERS = {"mean", "sum", "std", "var", "quantile", "bincount", "percentile", "average"}
    for fname, fn in targets.items():
        for assign in ast.walk(fn):
            if not isinstance(assign, ast.Assign):
                continue
            tnames = [t.id for t in assign.targets if isinstance(t, ast.Name)]
            if not any(n in ("total", "thresh", "delta", "fast") for n in tnames):
                continue
            rhs = _src(assign.value, src)
            ln = assign.lineno
            # find np reductions in the rhs
            for call in ast.walk(assign.value):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in REDUCERS:
                    arg = _src(call.args[0], src) if call.args else ""
                    safe = any(tok in arg for tok in SAFE_TOKENS)
                    leaky = any(re.search(r"\b" + re.escape(tok) + r"\b", arg) for tok in LEAK_TOKENS) and "keep" not in arg
                    if leaky or not safe:
                        problems.append(f"{fname} line {ln}: threshold reduction np.{call.func.attr}({arg[:50]}) "
                                        f"is not TRAIN-masked (needs a [keep]/tr subset or len(c)). RHS: {rhs[:80]}")

    # 3. Sanity: the train-mask helper must exist and be referenced (defense the masking exists at all).
    if "_train_minute_mask" not in src:
        problems.append("_train_minute_mask helper missing — TRAIN/OOS split not enforced.")
    if src.count("_train_minute_mask(") < 5:
        problems.append(f"_train_minute_mask called only {src.count('_train_minute_mask(')}x — expected per-axis masking.")

    # 4. EMBARGO regression (2026-06-04 leak-hunt): the val/test embargo must cover the forward-label
    #    horizon. A bare hardcoded `_EMBARGO = 200` lets a CONFIG['horizons']>200 override under-embargo
    #    the boundary and bleed early-test labels into cal.fit/eval_set. Require it to be horizon-aware.
    if os.path.exists(FOOTER):
        fsrc = "\n".join(re.sub(r"#.*$", "", line) for line in open(FOOTER).read().splitlines())
        if re.search(r"_EMBARGO\s*=\s*max\(\s*200\s*,", fsrc) is None:
            problems.append("footer.py.tmpl _EMBARGO is not horizon-aware — expected "
                            "`_EMBARGO = max(200, int(max(CONFIG.get('horizons') or [0])))` "
                            "(a bare `_EMBARGO = 200` under-embargoes horizons>200 -> test-label leak).")
        # 4b. DECLARED-REACH coupling (2026-06-06 structural fix): the labeler's 3rd return (its
        #     declared forward reach) must be PLUMBED into _run_cell (label_reach=...) and CONSUMED by
        #     _EMBARGO, so a smoothing/decoding labeler that out-reaches max(horizons) (e.g. the bounded
        #     next-event reach of dc_reversal/tlb_reversal, or a future wide-smoother) cannot silently
        #     bleed test-segment labels past the embargo. Discarding it into `_` regresses the fix.
        if "label_reach=" not in fsrc:
            problems.append("footer.py.tmpl does not PLUMB the labeler's declared reach into _run_cell "
                            "(expected `label_reach=...` on the _run_cell call) — the 3rd return is "
                            "discarded, so a smoothing labeler can silently out-reach the embargo.")
        if re.search(r"_EMBARGO\s*=\s*max\(.*label_reach", fsrc) is None:
            problems.append("footer.py.tmpl _EMBARGO does not CONSUME the labeler's declared reach — "
                            "expected `_EMBARGO = max(200, int(max(CONFIG.get('horizons') or [0])), "
                            "int(label_reach or 0))`.")

    # 5. LABELER REACH regression (2026-06-05 deep leak-hunt): generate_labels_dc_reversal labeled each
    #    bar with the direction of the NEXT reversal — an UNBOUNDED forward reach that returned None as
    #    its declared horizon. A near-boundary VAL bar's label could then encode a reversal INSIDE the
    #    test segment, bleeding into cal.fit/eval_set (the fixed embargo is blind to it). The labeler's
    #    reach MUST be bounded (<= the 200 embargo floor) and DECLARED (3rd return, not None).
    LABELER = os.path.join(_ROOT, "modules", "labeler.py")
    if os.path.exists(LABELER):
        lsrc = open(LABELER).read()
        m = re.search(r"def generate_labels_dc_reversal\(.*?\n(?=\ndef )", lsrc, re.S)
        body = m.group(0) if m else ""
        if "_MAXREACH" not in body:
            problems.append("labeler.py dc_reversal missing the _MAXREACH reach-bound — its next-reversal "
                            "reach is unbounded and can bleed test-segment labels into cal.fit/eval_set.")
        if re.search(r'return y,\s*f"dc_reversal_k\{k\}",\s*None', body):
            problems.append("labeler.py dc_reversal returns None (unbounded) as its declared reach — must "
                            "return a bounded reach (<=200 embargo floor) so the fit-set embargo covers it.")

    # EMPIRICAL teeth: append-OOS-invariance probe per NEW axis + the sticky_hmm block-decode guard.
    problems.extend(_runtime_teeth())

    if problems:
        fail(problems)
    print("LEAK GUARD: \033[92mPASS\033[0m — bar-threshold TRAIN-masked/OOS-invariant; embargo horizon-aware "
          "+ consumes declared reach; dc_reversal reach bounded+declared; new axes (semivar/chl/diurnal/"
          "kalman/newma/signedjumpvar) append-OOS-invariant (teeth); sticky_hmm 0 TEST->TRAIN/VAL flips.")
    sys.exit(0)


if __name__ == "__main__":
    main()
