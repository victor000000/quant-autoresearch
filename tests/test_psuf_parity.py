"""I5 guard: the header's _PSUF (QC side) must equal the driver's _cell_key suffix (local).

_cell_key in scripts/run_autoresearch_round.py produces the FULL cell key:
    {axis}_{labeler}_{sizing}_t{thresh*100}  +  SUFFIX

The header's _PSUF is ONLY the SUFFIX portion of that key.

This test strips the fixed prefix from _cell_key's output and verifies that the
remaining suffix equals _PSUF for a range of CONFIG combinations. A mismatch means
infer.py would look up a non-existent ObjectStore cell, returning 0 trades and
Calmar 0.0 silently.

_cell_key is extracted and exec'd in isolation (no network calls; the function
depends only on its 'cfg' argument and stdlib built-ins).
"""
import textwrap

from lb.paths import ROOT, TEMPLATES_DIR

HEADER = str(TEMPLATES_DIR / "header.py.tmpl")
DRIVER = str(ROOT / "scripts" / "run_autoresearch_round.py")

# ---------------------------------------------------------------------------
# Extract _cell_key from the driver in isolation (safe; no network calls).
# Read just the function lines and exec them so we don't trigger the module's
# harness/QC imports.
# ---------------------------------------------------------------------------

def _load_cell_key():
    """Parse and exec ONLY the _cell_key function from run_autoresearch_round.py."""
    src = open(DRIVER, encoding="utf-8").read()
    lines = src.splitlines()
    # Find start: first line of _cell_key definition
    start = next(i for i, l in enumerate(lines) if l.startswith("def _cell_key("))
    # Find end: next top-level definition or end of file after start
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l and not l[0].isspace() and not l.startswith("#"):
            end = i
            break
    fn_src = textwrap.dedent("\n".join(lines[start:end]))
    ns = {}
    exec(fn_src, ns)
    return ns["_cell_key"]

_cell_key = _load_cell_key()


# ---------------------------------------------------------------------------
# Extract _PSUF computation from the header template.
# _PSUF is defined on a single logical line in the template; exec it with a CONFIG.
# ---------------------------------------------------------------------------

def _header_psuf(cfg):
    """Evaluate the header's _PSUF expression with CONFIG=cfg."""
    src = open(HEADER, encoding="utf-8").read()
    line = next(l for l in src.splitlines() if l.strip().startswith("_PSUF"))
    ns = {"CONFIG": cfg}
    exec(line, ns)
    assert "_PSUF" in ns, f"exec of {line!r} did not bind _PSUF; update extractor"
    return ns["_PSUF"]


# ---------------------------------------------------------------------------
# Helper: strip the fixed axis/labeler/sizing/thresh prefix from _cell_key
# output to isolate the suffix — the part that must equal _PSUF.
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


def test_header_psuf_matches_driver_cell_key_suffix():
    """_PSUF (header/QC side) must equal the suffix of _cell_key (driver/local side)."""
    for cfg in CONFIGS:
        full_cfg = dict(_ANCHOR, **cfg)
        header_suffix = _header_psuf(full_cfg)
        driver_suffix = _driver_suffix(full_cfg)
        assert header_suffix == driver_suffix, (
            f"_PSUF mismatch for {cfg}:\n"
            f"  header _PSUF   = {header_suffix!r}\n"
            f"  driver suffix  = {driver_suffix!r}"
        )
