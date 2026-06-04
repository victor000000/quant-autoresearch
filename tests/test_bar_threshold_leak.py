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
BB = os.path.join(_ROOT, "autoresearch", "modules", "bar_builder.py")
FOOTER = os.path.join(_ROOT, "autoresearch", "templates", "footer.py.tmpl")

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

    if problems:
        fail(problems)
    print("LEAK GUARD: \033[92mPASS\033[0m — bar-threshold scalings TRAIN-masked / OOS-invariant (no `np.sum(valid)`); "
          "embargo horizon-aware.")
    sys.exit(0)


if __name__ == "__main__":
    main()
