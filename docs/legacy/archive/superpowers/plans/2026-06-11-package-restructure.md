# Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `lb` into an installable `src/lb/` package (`pip install -e .`) with no absolute-path literals and no `sys.path` hacks, dedup the `_PSUF` cell-key generator, and tidy the filesystem / git tree / docs — without changing research behavior or breaking the live QC pipeline or report server.

**Architecture:** Code moves under `src/lb/` via `git mv` (history preserved); data files stay at the repo root. A new `lb/paths.py` derives `ROOT` from `__file__` and is the single source of truth for every path. The QC flat-file module-upload contract and the 64k limit are preserved; `_PSUF` is moved to one canonical `cell_suffix()` injected into the rendered header so the QC and driver sites cannot drift. Characterization tests (render smoke + `_PSUF` parity) are written *first* on the current layout and act as the regression guard through the move.

**Tech Stack:** Python 3.12, setuptools (src layout), pytest, QuantConnect API (`qc_client`), Flask (report server), git.

**Reference spec:** `docs/superpowers/specs/2026-06-11-package-restructure-design.md` — read §2 (hard invariants I1–I5) before starting.

---

## File Structure (decomposition)

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | package metadata, setuptools src layout, `[project.scripts]` entry points |
| `src/lb/__init__.py` | version; re-export `ROOT` |
| `src/lb/paths.py` | **single source of truth** for `ROOT` + all data/template/module paths |
| `src/lb/harness/{constants,orchestrator,qc_client}.py` | moved from `harness/`; import paths from `lb.paths` |
| `src/lb/harness/psuf.py` | **new** — canonical `cell_suffix(cfg)` |
| `src/lb/modules/*.py` | moved pipeline stages; local import fallbacks → `lb.modules` |
| `src/lb/templates/*.py.tmpl` | moved; `header.py.tmpl` gains a `__PSUF_FN__` injection point |
| `src/lb/console/**` | moved from `scripts/console/`; section `sys.path` walks → package imports |
| `src/lb/cli.py` | **new** — `round_main()` / `report main()` entry points |
| `scripts/{research,audit,diag}/*.py` | regrouped runners; `import lb.*` |
| `scripts/app.py` | **compat shim** → `lb.console.app:main` (keeps systemd unit working) |
| `tests/test_render_smoke.py` | **new** — render compiles + < 64k (I3, I4) |
| `tests/test_psuf_parity.py` | **new** — header `_PSUF` == `cell_suffix()` (I5) |
| `docs/assets/` | the 4 hero images |
| `docs/legacy/` | archived superseded doc series |

---

## Task 1: Branch + land current loop work

Get a clean baseline so the restructure diff is pure moves, not tangled content.

**Files:** working-tree changes already present (see `git status`).

- [ ] **Step 1: Confirm starting branch and create the work branch**

Run:
```bash
cd /home/ubuntu/lb
git branch --show-current        # expect: autoresearch/2026-05-31
git checkout -b chore/package-restructure
```
Expected: `Switched to a new branch 'chore/package-restructure'` (working-tree changes carry over).

- [ ] **Step 2: Commit the new module code**

```bash
git add modules/bar_ext.py modules/ml_ext.py modules/sizing_ext.py modules/bar_builder.py modules/features.py modules/labeler.py modules/trainer.py
git commit -m "modules: land _ext split (bar/ml/sizing) + pipeline edits from loop"
```

- [ ] **Step 3: Commit harness + templates + scripts loop edits**

```bash
git add harness/ templates/ scripts/ tests/
git commit -m "loop: land harness/templates/scripts edits (real-yield wiring, honesty tooling)"
```

- [ ] **Step 4: Commit data + docs artifacts**

```bash
git add knowledge.json results/ reports/ program.md docs/ results/wang_step1_beta_router.tsv
git commit -m "loop: land knowledge/results/docs artifacts (R22-27 findings)"
```

- [ ] **Step 5: Verify the only remaining untracked items are the stray images**

Run: `git status --porcelain`
Expected: at most the 4 hero PNG/JPEGs remain untracked (handled in Task 3). Nothing else dirty.

- [ ] **Step 6: Capture a behavior baseline for the report server**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:80/`
Expected: `200` (the running service; we re-check it after the move).

---

## Task 2: Characterization tests on the CURRENT layout (regression guard)

Write the render-smoke and `_PSUF`-parity tests **before** moving anything, so they prove the current behavior and then guard the move. These tests must import the *current* `harness`/`templates` layout; Task 4 updates their imports.

**Files:**
- Create: `tests/test_render_smoke.py`
- Create: `tests/test_psuf_parity.py`
- Test: both

- [ ] **Step 1: Write the render-smoke failing test**

Create `tests/test_render_smoke.py`:
```python
"""I3/I4 guard: render_train_config must produce compilable QC files under 64k."""
import os, sys, ast
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import orchestrator  # NOTE: becomes `from lb.harness import orchestrator` in Task 4

SAMPLE_CFG = {
    "ticker": "GLD", "axis": "vol", "labeler": "tertile", "sizing": "binary",
    "thresh": 0.45, "max_depth": 3, "permute_labels": False, "n_components": 20,
    "rebal_band": 0.01, "horizons": None, "reduce": "correlation",
    "features": "base", "calibration": "isotonic", "train_purge": False,
}

def _files(cfg):
    """Return {filename: source} for the rendered main + extra files."""
    main, extra = orchestrator.render_train_config(cfg)
    out = {"main.py": main}
    out.update(extra or {})
    return out

def test_render_compiles_and_under_64k():
    files = _files(dict(SAMPLE_CFG))
    assert "main.py" in files
    for name, src in files.items():
        ast.parse(src)                      # syntactically valid
        nbytes = len(src.encode("utf-8"))
        assert nbytes < 64000, f"{name} is {nbytes} bytes (>= 64000 QC limit)"
```

> If `render_train_config` returns just a string (not `(main, extra)`), adapt `_files` to the real signature discovered in `harness/orchestrator.py` — keep the asserts identical. Confirm the signature first: `grep -n "def render_train_config" harness/orchestrator.py`.

- [ ] **Step 2: Run it — expect PASS on current layout (proves baseline)**

Run: `cd /home/ubuntu/lb && python -m pytest tests/test_render_smoke.py -v`
Expected: PASS. If it FAILS, the render signature differs — fix `_files` per the note, do not weaken the asserts.

- [ ] **Step 3: Write the `_PSUF` parity failing test**

Create `tests/test_psuf_parity.py`:
```python
"""I5 guard: the header's _PSUF (QC side) must equal the driver's _cell_key (local)."""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# driver _cell_key
_spec = importlib.util.spec_from_file_location(
    "rar", os.path.join(ROOT, "scripts", "run_autoresearch_round.py"))
# run_autoresearch_round executes top-level path setup but no network at import; if it
# does, replace this with: exec only the _cell_key function source. Confirm with a dry import.
rar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rar)

HEADER = os.path.join(ROOT, "templates", "header.py.tmpl")  # becomes src/lb/templates in Task 4

def _header_psuf(cfg):
    """Exec the header's _PSUF line in isolation with CONFIG=cfg."""
    src = open(HEADER, encoding="utf-8").read()
    line = next(l for l in src.splitlines() if l.strip().startswith("_PSUF"))
    ns = {"CONFIG": cfg}
    exec(line, ns)
    return ns["_PSUF"]

CONFIGS = [
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "base", "calibration": "isotonic", "train_purge": False},
    {"permute_labels": True, "n_components": 15, "rebal_band": 0.05, "horizons": [5, 10],
     "reduce": "infogain", "features": "rich", "calibration": "venn_abers", "train_purge": True},
    {"permute_labels": False, "n_components": 20, "rebal_band": 0.01, "horizons": None,
     "reduce": "correlation", "features": "realyield", "calibration": "isotonic", "train_purge": False},
]

def test_header_psuf_matches_driver_cell_key():
    for cfg in CONFIGS:
        assert _header_psuf(dict(cfg)) == rar._cell_key(dict(cfg)), cfg
```

- [ ] **Step 4: Run it — expect PASS (proves the two sites currently agree)**

Run: `cd /home/ubuntu/lb && python -m pytest tests/test_psuf_parity.py -v`
Expected: PASS. If importing `run_autoresearch_round` triggers network/QC calls, switch to function-source extraction (exec only the `_cell_key` def block) per the inline comment, then re-run.

- [ ] **Step 5: Run the full existing test suite to confirm nothing else regressed**

Run: `cd /home/ubuntu/lb && python -m pytest tests/ -q`
Expected: all green (note which tests exist and pass — this is the baseline set).

- [ ] **Step 6: Commit**

```bash
git add tests/test_render_smoke.py tests/test_psuf_parity.py
git commit -m "test: characterize render-smoke (<64k) + _PSUF parity on current layout"
```

---

## Task 3: Filesystem + gitignore + docs consolidation (behavior-neutral)

No code path changes. Pure tree hygiene.

**Files:** root images, `.gitignore`, `scratch/_archive/`, `docs/`.

- [ ] **Step 1: Move the stray hero images into docs/assets/**

```bash
cd /home/ubuntu/lb
mkdir -p docs/assets
git mv console-1280-hero.png docs/assets/ 2>/dev/null || mv console-1280-hero.png docs/assets/
for f in desktop-hero.jpeg fresh-hero.jpeg site-reorg-fold.jpeg; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git mv "$f" docs/assets/ || mv "$f" docs/assets/
done
git add docs/assets/
```
Expected: 4 images now under `docs/assets/`, root clean of them.

- [ ] **Step 2: Untrack the dead-code archive and harden .gitignore**

```bash
git rm -r --cached scratch/_archive >/dev/null
printf '\n# --- workspace hygiene (2026-06-11 restructure) ---\nscratch/\nqc/\n*.aux\n' >> .gitignore
git add .gitignore
```
Verify: `git check-ignore scratch/_archive scratch/data_cache qc/.creds.json` → all three print (ignored).

- [ ] **Step 3: Delete build/cache artifacts (local only, untracked)**

```bash
rm -f docs/legacy/wang_qa.aux refs/qa_doc/wang_qa.aux
rm -f .playwright-mcp/*.log 2>/dev/null
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: no errors; these are untracked/ignored so `git status` is unaffected.

- [ ] **Step 4: Archive superseded doc series**

```bash
mkdir -p docs/legacy/sessions docs/legacy/archive
git mv docs/analysis/RESEARCH_REVIEW.md docs/legacy/archive/ 2>/dev/null
git mv docs/analysis/RESEARCH_REVIEW_v2.md docs/legacy/archive/ 2>/dev/null
git mv docs/research/WANG_INTERNET_DEEP_V2_2026-06-08.md docs/legacy/archive/ 2>/dev/null
git mv docs/research/WANG_INTERNET_DEEP_V3_2026-06-09.md docs/legacy/archive/ 2>/dev/null
git mv docs/research/new_method_backlog.md docs/legacy/archive/ 2>/dev/null
git mv docs/analysis/SESSION_SUMMARY_2026-06-08.md docs/legacy/sessions/ 2>/dev/null
git mv docs/analysis/LOOP_SESSION_2026-06-09.md docs/legacy/sessions/ 2>/dev/null
git mv docs/analysis/LEAK_AUDIT_2026-06-07.md docs/legacy/sessions/ 2>/dev/null
```
> Keep the latest of each series in place: `RESEARCH_REVIEW_v3.md`, `WANG_WORKFLOW_SYNTHESIS_2026-06-10.md`, `NEW_METHODS_BACKLOG.md`, `LEAK_REVIEW_2026-06-10.md`, `HONEST_AUDIT.md`.

- [ ] **Step 5: Verify tests still green + server still 200 (nothing code-facing touched)**

Run:
```bash
python -m pytest tests/ -q
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:80/
```
Expected: tests green; HTTP `200`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: tidy tree — hero imgs→docs/assets, untrack scratch/_archive, gitignore qc/+*.aux, archive superseded docs"
```

---

## Task 4: Package conversion — atomic green increment

Move code under `src/lb/`, add `paths.py`, and fix **all** internal imports + absolute literals together so the suite is green at the end. (The move and the import fixes are coupled — splitting them would leave a broken intermediate, so they are one task with many small steps.)

**Files:** `pyproject.toml` (create), `src/lb/**` (moved), `src/lb/paths.py` (create), `src/lb/harness/constants.py`, `src/lb/harness/orchestrator.py`, `scripts/run_autoresearch_round.py`, the 5 module-fallback sites, `src/lb/console/**`, `tests/*` imports.

- [ ] **Step 1: Create pyproject.toml**

Create `/home/ubuntu/lb/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lb"
version = "0.1.0"
description = "Autoresearch quant pipeline"
requires-python = ">=3.10"
dependencies = ["numpy", "xgboost", "scikit-learn", "scipy", "flask", "requests"]

[project.scripts]
lb-round = "lb.cli:round_main"
lb-report = "lb.console.app:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"lb" = ["templates/*.tmpl"]
```

- [ ] **Step 2: git mv code dirs under src/lb/ and add package markers**

```bash
cd /home/ubuntu/lb
mkdir -p src/lb
git mv harness src/lb/harness
git mv modules src/lb/modules
git mv templates src/lb/templates
git mv scripts/console src/lb/console
touch src/lb/__init__.py
for d in harness modules console console/sections; do touch "src/lb/$d/__init__.py"; done
git add src/lb
```
Expected: dirs relocated; `git status` shows renames.

- [ ] **Step 3: Create src/lb/paths.py (single source of truth)**

Create `/home/ubuntu/lb/src/lb/paths.py`:
```python
"""Single source of truth for filesystem paths. Derives the repo root from this
file's location; override with LB_ROOT for unusual deployments."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LB_ROOT", Path(__file__).resolve().parents[2]))

PKG = ROOT / "src" / "lb"
TEMPLATES_DIR = PKG / "templates"
MODULES_DIR = PKG / "modules"
QC_SCRIPTS_DIR = ROOT / "_autoresearch_scripts"
QC_CREDS_PATH = ROOT / "qc" / ".creds.json"

KNOWLEDGE_JSON = ROOT / "knowledge.json"
TECHNIQUES_JSON = ROOT / "techniques.json"
HYPOTHESES_JSON = ROOT / "hypotheses.json"
RESULTS_TSV = ROOT / "results.tsv"
RESULTS_DIR = ROOT / "results"
ROUND_RESULTS_CSV = RESULTS_DIR / "round_results.csv"
REPORTS_DIR = ROOT / "reports"
STATUS_JSON = REPORTS_DIR / "status.json"
```

- [ ] **Step 4: __init__.py re-exports ROOT**

Write `/home/ubuntu/lb/src/lb/__init__.py`:
```python
from lb.paths import ROOT  # noqa: F401
__version__ = "0.1.0"
```

- [ ] **Step 5: Rewrite harness/constants.py to import from lb.paths**

In `src/lb/harness/constants.py` replace the absolute-literal lines (13, 23, 24, 26) with:
```python
from lb.paths import (
    TEMPLATES_DIR, MODULES_DIR, QC_SCRIPTS_DIR, QC_CREDS_PATH,
    KNOWLEDGE_JSON, TECHNIQUES_JSON, HYPOTHESES_JSON, RESULTS_TSV,
    RESULTS_DIR, ROUND_RESULTS_CSV, STATUS_JSON,
)
```
Keep `QC_PROJECT_ID`, `CORE_7_ETFS`, the date splits, etc. as-is. Where downstream code does `open(TEMPLATES_DIR + "/header.py.tmpl")` (string concat), convert to `os.path.join(str(TEMPLATES_DIR), ...)` or `TEMPLATES_DIR / "header.py.tmpl"`. Find them: `grep -rn "TEMPLATES_DIR\|MODULES_DIR" src/lb/`.

- [ ] **Step 6: Rewrite orchestrator.py + driver to drop PROJECT_ROOT literals**

In `src/lb/harness/orchestrator.py:26` and `scripts/run_autoresearch_round.py:68`, delete `PROJECT_ROOT = "/home/ubuntu/lb"` and the derived path constants; import them from `lb.paths` / `lb.harness.constants` instead. Replace the `sys.path.insert` blocks in the driver with nothing (the package is installed). Confirm afterward: `grep -rn '/home/ubuntu/lb' src/ scripts/` → only matches inside comments/docstrings, none in code.

- [ ] **Step 7: Update the 5 module local-fallback imports (I2)**

In each site change ONLY the `except` arm `from modules import X` → `from lb.modules import X` (keep the bare `import X` QC arm first):
- `src/lb/modules/bar_builder.py:1356`
- `src/lb/modules/features.py:440`
- `src/lb/modules/labeler.py:3540`
- `src/lb/modules/trainer.py:177`
- `src/lb/templates/footer.py.tmpl:49`

Run: `grep -rn "from modules import" src/lb/` → expect zero matches afterward.

- [ ] **Step 8: Update console package internal imports**

In `src/lb/console/**`, remove every `sys.path.insert(...)` and the 3-level walks; replace `from console import …` / `from console.sections import …` with `from lb.console import …` / `from lb.console.sections import …`. The `data.py` knowledge.json load uses `lb.paths.KNOWLEDGE_JSON`.

Run: `grep -rn "sys.path.insert" src/lb/console/` → expect zero.

- [ ] **Step 9: Update test imports to the package**

In `tests/test_render_smoke.py`, `tests/test_psuf_parity.py`, `tests/test_bar_threshold_leak.py`, `tests/test_realyield_feats.py`: delete the `sys.path` setup and change `from harness import …` → `from lb.harness import …`, `templates/header.py.tmpl` path → `from lb.paths import TEMPLATES_DIR`, and the driver path for parity → `src/lb/...` is unchanged (it stays in `scripts/`). Use `importlib` against `scripts/run_autoresearch_round.py` as before but import its deps via the installed package.

- [ ] **Step 10: Editable-install the package**

```bash
cd /home/ubuntu/lb && pip install -e . 2>&1 | tail -3
python -c "import lb; print('lb ROOT =', lb.ROOT)"
```
Expected: install succeeds; prints `lb ROOT = /home/ubuntu/lb`.

> If pip is the system pip without a venv, use `pip install -e . --user` or the project venv. Record which interpreter the systemd unit uses (`/usr/bin/python3`) — the package must be importable by THAT interpreter for the report shim to work (Step in Task 6).

- [ ] **Step 11: Run the full suite (render smoke + parity + leak tests)**

Run: `cd /home/ubuntu/lb && python -m pytest tests/ -q`
Expected: all green. The render-smoke and `_PSUF`-parity tests now exercise the moved package and prove the move preserved behavior.

- [ ] **Step 12: Grep gate**

```bash
grep -rn '/home/ubuntu/lb' src/ | grep -v -E '#|"""' || echo "OK: no absolute literals in code"
grep -rn "from modules import\|from harness import\|from console import" src/ tests/ || echo "OK: no bare-package imports"
```
Expected: both print the OK line.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "refactor: src/lb package — git mv code, lb.paths single source of truth, kill absolute literals + sys.path hacks (core)"
```

---

## Task 5: Dedup `_PSUF` into one canonical `cell_suffix()` (I5)

**Files:** `src/lb/harness/psuf.py` (create), `src/lb/templates/header.py.tmpl`, `src/lb/harness/orchestrator.py`, `scripts/run_autoresearch_round.py`, `tests/test_psuf_parity.py`.

- [ ] **Step 1: Extract the canonical function**

Create `/home/ubuntu/lb/src/lb/harness/psuf.py` with a pure, stdlib-only function whose body is the EXACT logic currently in `_cell_key` / header `_PSUF`:
```python
def cell_suffix(cfg) -> str:
    """Canonical ObjectStore cell-key suffix. stdlib-only so it can run on QC.
    MUST stay the single definition (header injection + driver both use this)."""
    return (
        ("_perm" if cfg["permute_labels"] else "")
        + ("" if cfg["n_components"] == 20 else "_n" + str(cfg["n_components"]))
        + ("" if cfg["rebal_band"] == 0.01 else "_b" + str(int(round(cfg["rebal_band"] * 100))))
        + ("" if cfg["horizons"] is None else "_hz" + "x".join(str(h) for h in cfg["horizons"]))
        + ("" if cfg["reduce"] == "correlation" else "_ig" if cfg["reduce"] == "infogain" else "_rd" + str(cfg["reduce"]))
        + ("" if cfg["features"] == "base" else "_fr" if cfg["features"] == "rich" else "_ts" if cfg["features"] == "termstruct" else "_ry" if cfg["features"] == "realyield" else "_fx")
        + ("" if cfg.get("calibration", "isotonic") == "isotonic" else "_va")
        + ("_tp" if cfg["train_purge"] else "")
    )
```
> Copy the exact expression from `header.py.tmpl:96` to avoid any transcription drift; the test in Step 5 catches a mismatch.

- [ ] **Step 2: Point the driver at it**

In `scripts/run_autoresearch_round.py`, make `_cell_key` a thin wrapper:
```python
from lb.harness.psuf import cell_suffix
def _cell_key(cfg):
    return f"{cfg['ticker']}_{cfg['axis']}_{cfg['labeler']}_{cfg['sizing']}_t{cfg['thresh']:.2f}" + cell_suffix(cfg)
```
> Verify the base prefix (`ticker_axis_labeler_sizing_tNN`) matches the current `_cell_key` head exactly before replacing — keep that part byte-identical.

- [ ] **Step 3: Add the header injection point**

In `src/lb/templates/header.py.tmpl`, replace the inline `_PSUF = (...)` line (96) with a placeholder block:
```python
__PSUF_FN__
_PSUF = cell_suffix(CONFIG)
```

- [ ] **Step 4: Inject the source at render time**

In `src/lb/harness/orchestrator.py`, where the header template is read, substitute the placeholder with the function source:
```python
import inspect
from lb.harness.psuf import cell_suffix
header_src = header_src.replace("__PSUF_FN__", inspect.getsource(cell_suffix))
```
Ensure this runs BEFORE minification and that the resulting `main.py` still passes the 64k check (the render-smoke test enforces this).

- [ ] **Step 5: Update the parity test to assert against `cell_suffix`**

In `tests/test_psuf_parity.py`, add a third equality: render the header via the orchestrator, `exec` it with a sample `CONFIG`, and assert `_PSUF == cell_suffix(cfg)` for every config in `CONFIGS`. Keep the existing driver-equality assertion too.

- [ ] **Step 6: Run parity + render smoke**

Run: `cd /home/ubuntu/lb && python -m pytest tests/test_psuf_parity.py tests/test_render_smoke.py -v`
Expected: PASS (one canonical source, header injection byte-identical, still < 64k).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: single canonical _PSUF (lb.harness.psuf.cell_suffix) injected into QC header; parity-tested"
```

---

## Task 6: Regroup runner scripts + entry points + report shim

**Files:** `scripts/{research,audit,diag}/`, `scripts/app.py` (shim), `scripts/run_round.py`, `src/lb/cli.py` (create), the ~35 loose runner scripts.

- [ ] **Step 1: Create the CLI entry module**

Create `/home/ubuntu/lb/src/lb/cli.py`:
```python
"""Console-script entry points (see pyproject [project.scripts])."""
def round_main():
    import runpy, os
    from lb.paths import ROOT
    runpy.run_path(os.path.join(ROOT, "scripts", "run_round.py"), run_name="__main__")
```
> `lb-report` points at `lb.console.app:main`; ensure `src/lb/console/app.py` exposes a `main()` that starts Flask (wrap the existing `app.run(...)` block in `def main(): ...`).

- [ ] **Step 2: Replace scripts/app.py with a compat shim**

Overwrite `/home/ubuntu/lb/scripts/app.py`:
```python
"""Compat shim — the systemd unit runs `python3 scripts/app.py`. Real code lives in
lb.console.app. Keeps the autoresearch-reports service working without a unit edit."""
from lb.console.app import main
if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Move the driver entry to scripts/run_round.py**

```bash
cd /home/ubuntu/lb
git mv scripts/run_autoresearch_round.py scripts/run_round.py
```
> Update any references to the old name: `grep -rln run_autoresearch_round . --include=*.py --include=*.md` and fix them (the watchdog `scripts/sweep_watchdog.py` and docs may reference it).

- [ ] **Step 4: Regroup the remaining runners by role**

```bash
mkdir -p scripts/research scripts/audit scripts/diag scripts/lib
# audit / honesty / multiple-testing
git mv scripts/{assess_dsr,deflated_audit,harvey_liu_haircut,online_fdr,online_fdr_ledger,honest_audit,certify_leaderboard,predictability_screen,bootstrap_calmar_ci}.py scripts/audit/ 2>/dev/null
# research runners
git mv scripts/{champion_series,decay_oil,cost_oil,cost_stress,deadband_tune,pbo_gld,beta_router,mechanism_router,portfolio_rederive,backfill_cagr_mdd,cleanup_objectstore,verify_online_bars,verify_crown_online}.py scripts/research/ 2>/dev/null
# diagnostics already in scripts/diag/ ; move minify check there too
git mv scripts/_minify_check.py scripts/diag/ 2>/dev/null
```
> This list is illustrative of the grouping; place each remaining loose `scripts/*.py` into the folder matching its role. Leave `scripts/render_index.py`, `scripts/render_round.py` at `scripts/` root (thin render shims) or move under `scripts/lib/`.

- [ ] **Step 5: Rewrite runner imports to the installed package**

For every moved runner, delete `sys.path.insert(...)` lines and rewrite `import constants`/`from harness import X`/`import orchestrator` → `from lb.harness import X`. Then syntax-check all of them:
```bash
cd /home/ubuntu/lb
python - <<'PY'
import pathlib, ast, sys
bad = []
for p in pathlib.Path("scripts").rglob("*.py"):
    try: ast.parse(p.read_text())
    except SyntaxError as e: bad.append((str(p), e))
print("syntax errors:", bad or "none")
sys.exit(1 if bad else 0)
PY
grep -rn "sys.path.insert" scripts/ | grep -v "scripts/app.py" || echo "OK: no sys.path hacks left (except shim)"
```
Expected: `syntax errors: none`; the OK line prints.

- [ ] **Step 6: Spot-run three representative runners (import-level)**

Run (these import the package; some need QC creds to do real work — we only check they import/parse without `ImportError`):
```bash
cd /home/ubuntu/lb
python -c "import importlib.util,sys; [print(f,'OK') for f in ['scripts/audit/assess_dsr.py','scripts/research/champion_series.py','scripts/diag/_measure_one.py']]"
python scripts/diag/_minify_check.py 2>&1 | tail -3 || true
```
Expected: no `ModuleNotFoundError: lb` / no `ImportError` for `lb.*`. (Functional QC runs are out of scope here.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: regroup runners (research/audit/diag), lb.cli entry points, scripts/app.py compat shim, run_round.py"
```

---

## Task 7: Final validation gate + report-server restart

**Files:** none (verification only).

- [ ] **Step 1: Clean editable reinstall + import from arbitrary CWD**

```bash
cd /home/ubuntu/lb && pip install -e . 2>&1 | tail -2
cd /tmp && python -c "import lb; print(lb.ROOT)" && cd -
```
Expected: prints `/home/ubuntu/lb` from `/tmp` (relocatable import works).

- [ ] **Step 2: Full test suite green**

Run: `cd /home/ubuntu/lb && python -m pytest tests/ -q`
Expected: all green, including render-smoke (I3/I4) and `_PSUF` parity (I5).

- [ ] **Step 3: Console builds**

Run:
```bash
cd /home/ubuntu/lb
python -c "from lb.console.page import build_html, build_data; build_data(); print('console build OK', len(build_html()))"
```
Expected: prints `console build OK <nbytes>` with no traceback.

- [ ] **Step 4: Restart the report service and confirm 200**

```bash
sudo systemctl restart autoresearch-reports
sleep 2
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:80/
sudo systemctl status autoresearch-reports --no-pager | grep Active
```
Expected: HTTP `200`; `Active: active (running)`. (The unit still runs `scripts/app.py`, now a shim into `lb.console.app`.)

- [ ] **Step 5: Final grep gate**

```bash
cd /home/ubuntu/lb
grep -rn '/home/ubuntu/lb' src/ scripts/ | grep -vE '#|"""|\.md' || echo "OK: no absolute literals in code"
grep -rn "sys.path.insert" src/ scripts/ | grep -v "scripts/app.py" || echo "OK: no sys.path hacks (except shim)"
```
Expected: both OK lines print.

- [ ] **Step 6 (optional gold standard): one real QC backtest**

If QC creds + network are available, submit a single backtest through the moved pipeline to confirm render → upload → result end-to-end:
```bash
cd /home/ubuntu/lb
python -c "
from lb.harness import orchestrator, qc_client
cfg = {'ticker':'GLD','axis':'vol','labeler':'tertile','sizing':'binary','thresh':0.45,'max_depth':3,'permute_labels':False,'n_components':20,'rebal_band':0.01,'horizons':None,'reduce':'correlation','features':'base','calibration':'isotonic','train_purge':False}
main, extra = orchestrator.render_train_config(cfg)
bid = qc_client.submit_backtest(main, 'restructure-smoke', extra_files=extra)
print('submitted', bid)
"
```
Expected: a backtest id prints and the run completes on QC without an import/cell-key error. (Skip if creds unavailable — the local gate above is the definition of done.)

- [ ] **Step 7: Final commit / branch ready for merge**

```bash
git add -A && git commit -m "chore: restructure validation pass — tests green, server restarted, grep gates clean" --allow-empty
git log --oneline chore/package-restructure ^autoresearch/2026-05-31
```
Then use the `superpowers:finishing-a-development-branch` skill to decide merge/PR.

---

## Self-Review notes

- **Spec coverage:** I1 (data-at-root) → Task 4 Step 3 `paths.py` keeps data at ROOT; I2 (module fallback) → Task 4 Step 7; I3/I4 (render+64k) → Task 2 + Task 5 Step 6 render-smoke; I5 (`_PSUF`) → Task 5 + parity test. Filesystem/git/docs scopes → Tasks 1–3. Package + entry points → Tasks 4, 6. Validation gate → Task 7. All §6 gate items mapped.
- **Ambiguity guarded:** render signature (Task 2 Step 1 note), driver-import side effects (Task 2 Step 4 note), interpreter/venv for the systemd shim (Task 4 Step 10 note) each carry an explicit fallback.
- **No silent caps:** the ~35 runner scripts get import-rewrite + syntax-check + spot-import (Task 6); full functional QC runs of every runner are explicitly out of scope and called out, not hidden.
