"""I5 guard: the QC header's _PSUF, the driver's _cell_key suffix, and the single
canonical lb.harness.psuf.cell_suffix must all produce the SAME ObjectStore cell-key
suffix for every config.

Why this matters: the cell key suffix is computed on the QC side (header _PSUF) and on
the local side (driver _cell_key). If they diverge by even one character, infer.py looks
up a NON-EXISTENT ObjectStore cell -> 0 trades -> Calmar 0.0 reported silently. The
inline copies diverged twice historically. Task 5 collapsed both to ONE definition:

    lb.harness.psuf.cell_suffix(cfg)

  - the driver's _cell_key delegates to it (suffix portion), and
  - the orchestrator INJECTS its source into the rendered QC header (replacing the
    __PSUF_FN__ placeholder), so the header computes `_PSUF = cell_suffix(CONFIG)`.

This test proves the full chain end-to-end:
  (a) the suffix extracted from the ACTUALLY-RENDERED header (post inspect.getsource
      injection AND minification) == cell_suffix(cfg)   [QC side is byte-identical], and
  (b) the driver's _cell_key suffix (prefix stripped)   == cell_suffix(cfg)   [local side].
(a) genuinely exercises the rendered artifact: cell_suffix is pulled back OUT of the
rendered main.py and executed — if injection failed, extraction or output would differ.
"""
import ast
import textwrap

from lb.harness import orchestrator
from lb.harness.psuf import cell_suffix
from lb.paths import ROOT

DRIVER = str(ROOT / "scripts" / "run_autoresearch_round.py")


# ---------------------------------------------------------------------------
# Extract _cell_key from the driver in isolation (safe; no network calls). The
# driver delegates the SUFFIX to cell_suffix, so we provide that name in the exec
# namespace; the PREFIX logic under test lives in the sliced function itself.
# ---------------------------------------------------------------------------

def _load_cell_key():
    """Parse and exec ONLY the _cell_key function from run_autoresearch_round.py."""
    src = open(DRIVER, encoding="utf-8").read()
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def _cell_key("))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l and not l[0].isspace() and not l.startswith("#"):
            end = i
            break
    fn_src = textwrap.dedent("\n".join(lines[start:end]))
    ns = {"cell_suffix": cell_suffix}   # driver delegates the suffix to the canonical fn
    exec(fn_src, ns)
    return ns["_cell_key"]

_cell_key = _load_cell_key()


# ---------------------------------------------------------------------------
# Extract _PSUF from the ACTUALLY-RENDERED header (post-injection + minify).
# render_train_config -> _load_header (injects inspect.getsource(cell_suffix) at the
# __PSUF_FN__ placeholder) -> _minify. We pull the injected cell_suffix def and the
# `_PSUF = cell_suffix(CONFIG)` assignment back out of the rendered main.py and exec
# them with CONFIG=cfg, obtaining the suffix exactly as QuantConnect would compute it.
# ---------------------------------------------------------------------------

def _rendered_psuf(cfg):
    """Evaluate _PSUF as defined in the rendered (injected) QC header, with CONFIG=cfg."""
    main, _extra = orchestrator.render_train_config(dict(cfg, ticker=cfg.get("ticker", "GLD")))
    tree = ast.parse(main)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "cell_suffix"), None)
    assert fn is not None, "rendered header is missing the injected cell_suffix def (__PSUF_FN__ not substituted?)"
    assign = next((n for n in tree.body
                   if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "_PSUF" for t in n.targets)), None)
    assert assign is not None, "rendered header is missing the _PSUF = cell_suffix(CONFIG) assignment"
    code = ast.unparse(ast.Module(body=[fn, assign], type_ignores=[]))
    ns = {"CONFIG": dict(cfg)}
    exec(code, ns)
    return ns["_PSUF"]


# ---------------------------------------------------------------------------
# Helper: strip the fixed axis/labeler/sizing/thresh prefix from _cell_key
# output to isolate the suffix — the part that must equal cell_suffix / _PSUF.
# ---------------------------------------------------------------------------

def _driver_suffix(cfg):
    """Return ONLY the suffix portion of _cell_key (strip the axis_label_siz_tNN prefix)."""
    prefix = (
        f"{cfg['axis']}_{cfg['labeler'].replace('+', '_x_')}"
        f"_{cfg['sizing']}_t{int(round(float(cfg['thresh']) * 100))}"
    )
    full = _cell_key(cfg)
    assert full.startswith(prefix), (
        f"_cell_key output {full!r} does not start with expected prefix {prefix!r}"
    )
    return full[len(prefix):]


# ---------------------------------------------------------------------------
# Test configs — cover every suffix flag combination.
# axis/labeler/sizing/thresh are anchor values (only affect the stripped prefix).
# ---------------------------------------------------------------------------

_ANCHOR = {"axis": "vol", "labeler": "tertile", "sizing": "binary", "thresh": 0.45}

CONFIGS = [
    # defaults: all flags at default → suffix is empty string
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "base", "calibration": "isotonic", "train_purge": False},
    # all non-default flags set → longest suffix
    {"permute_labels": True, "n_components": 15, "rebal_band": 0.05, "horizons": [5, 10],
     "reduce": "infogain", "features": "rich", "calibration": "venn_abers", "train_purge": True},
    # realyield feature set (the _ry suffix token)
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "realyield", "calibration": "isotonic", "train_purge": False},
    # termstruct feature set
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "termstruct", "calibration": "isotonic", "train_purge": False},
    # train_purge only
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "base", "calibration": "isotonic", "train_purge": True},
    # venn_abers calibration alone (isolates the _va suffix token)
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "base", "calibration": "venn_abers", "train_purge": False},
]


def test_canonical_cell_suffix_matches_driver_and_rendered_header():
    """cell_suffix == driver _cell_key suffix == rendered-header _PSUF, for every config."""
    for cfg in CONFIGS:
        full_cfg = dict(_ANCHOR, **cfg)
        canonical = cell_suffix(full_cfg)
        driver = _driver_suffix(full_cfg)
        rendered = _rendered_psuf(full_cfg)
        assert driver == canonical, (
            f"driver _cell_key suffix != canonical cell_suffix for {cfg}:\n"
            f"  driver    = {driver!r}\n  canonical = {canonical!r}"
        )
        assert rendered == canonical, (
            f"rendered-header _PSUF != canonical cell_suffix for {cfg}:\n"
            f"  rendered  = {rendered!r}\n  canonical = {canonical!r}"
        )
