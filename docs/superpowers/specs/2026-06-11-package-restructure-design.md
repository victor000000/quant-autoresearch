# Design — convert `lb` to an installable `src/lb/` package

**Date:** 2026-06-11
**Status:** approved (design); awaiting spec review
**Scope:** deep workspace reorganization — filesystem tidy-up, git-tree cleanup,
docs consolidation, **and** a full code-architecture restructure into an
installable package (`pip install -e .`) with zero `sys.path` hacks.

---

## 1. Motivation

The repo works but carries structural debt that makes it fragile and hard to move:

- **Hardcoded absolute paths.** `"/home/ubuntu/lb"` is baked into
  `harness/constants.py` (`TEMPLATES_DIR`, `MODULES_DIR`), `harness/orchestrator.py`,
  `scripts/run_autoresearch_round.py`, and `scripts/diag/_measure_one.py`. The repo
  cannot be cloned to another path without edits.
- **A `sys.path.insert` maze.** ~15 scripts manipulate `sys.path` (`insert(0, ".")`,
  `insert(0, "harness")`, absolute inserts). The 9 `scripts/console/sections/*.py`
  files each walk `dirname(dirname(dirname(__file__)))` to reach `scripts/`. All of
  this breaks the moment a file moves.
- **A duplicated cell-key generator.** `_PSUF` exists inline in
  `templates/header.py.tmpl:96` (executed on QuantConnect) and is mirrored by
  `_cell_key()` in `run_autoresearch_round.py:637`. The driver's docstring records that
  the two **diverged twice** — and divergence is a silent leak/zero-trade bug.
- **Root and tree clutter.** 4 unreferenced hero images at root, 34 tracked
  dead-code files under `scratch/_archive/`, stale `.aux`/`.playwright-mcp/` artifacts,
  ~25 uncommitted modified files tangled with new work, and several superseded doc
  series (`RESEARCH_REVIEW` v1/v2/v3, `WANG_*` v2/v3, dated session logs).

Goal: a relocatable, `pip install`-able package with one source of truth for paths
and the cell key, a clean working tree, and a tidy docs/filesystem — **without changing
any research behavior or breaking the live QC pipeline or report server.**

---

## 2. Hard invariants (must not break)

These are contracts the restructure must preserve exactly. They define the
validation gate in §6.

### I1 — Data files stay at the repo root
`knowledge.json`, `techniques.json`, `hypotheses.json`, `results.tsv`, and
`results/` are read/written by path by both the autoresearch loop and the live
Flask server (`scripts/app.py`, currently running as a systemd service). **Only code
moves; data stays put.** Paths to them become a single source of truth in `lb/paths.py`.

### I2 — The QC module-upload contract
`modules/{bar_builder,bar_ext,features,ml_ext,labeler,trainer}.py` and
`templates/footer.py.tmpl` are uploaded to QuantConnect as **flat project files** and
import each other with a two-arm fallback:

```python
try:
    import bar_ext as _bx_mod              # QC cloud: flat project files
except Exception:
    from modules import bar_ext as _bx_mod # local repo / tests
```

The **bare import (QC arm) stays first and unchanged.** Only the local fallback arm
changes `modules` → `lb.modules`. Sites (verified):
`modules/bar_builder.py:1353-1356`, `modules/features.py:438-440`,
`modules/labeler.py:3537-3540`, `modules/trainer.py:175-177`,
`templates/footer.py.tmpl:47-49` (footer is QC-only; its fallback is cosmetic but
updated for consistency).

### I3 — Templates are read from disk at render time
`render_train_config()` reads `*.py.tmpl` and the module `.py` files from
`constants.TEMPLATES_DIR` / `MODULES_DIR`, minifies, and uploads. Moving these dirs is
**only** a constants change — no import semantics involved.

### I4 — The 64,000-byte QC per-file limit
After minification, `main.py` and every extra file (`bar_builder`, `bar_ext`,
`ml_ext`, `sizing_ext`) must each stay `< 64000` bytes. The restructure must not push
any file over budget (the injected `_PSUF` source in §4.3 is the only new content
added to a rendered file — it must be measured).

### I5 — `_PSUF` byte-identical across both call sites
The ObjectStore cell key written by the footer (header `_PSUF`) and the key the driver
computes to read it back must be **byte-identical** for every config, or inference reads
a non-existent cell → 0 trades → Calmar 0.0. This is the leak/zero-trade landmine.

---

## 3. Target layout

```
lb/
├── pyproject.toml                # [project] lb; [project.scripts] entry points; setuptools src layout
├── README.md  program.md  .gitignore
├── src/lb/
│   ├── __init__.py               # version; re-export ROOT
│   ├── paths.py                  # ROOT = Path(__file__).resolve().parents[2]; ALL data paths
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── constants.py          # TEMPLATES_DIR/MODULES_DIR derived from paths.ROOT
│   │   ├── orchestrator.py
│   │   ├── qc_client.py
│   │   └── psuf.py               # NEW — single source of truth for the cell-key suffix
│   ├── modules/                  # bar_builder, bar_ext, features, ml_ext, labeler, trainer, sizing_ext
│   │   └── __init__.py
│   ├── templates/                # *.py.tmpl  (package data, still read from disk)
│   └── console/                  # was scripts/console/ — Flask site
│       ├── __init__.py  app.py  page.py  data.py  primitives.py  render_index.py
│       └── sections/*.py         # 3-level sys.path walks → package imports
├── scripts/                      # thin runners; `import lb.*`, no sys.path hacks
│   ├── research/                 # champion_series, decay_*, pbo_*, cost_stress, deadband_tune, …
│   ├── audit/                    # leak/honesty/dsr checks, honest_audit
│   ├── diag/                     # _measure_one, reports_to_html
│   ├── run_round.py              # entry shim → lb.cli round
│   └── app.py                    # COMPAT shim → lb.console.app:main (keeps systemd service alive)
├── tests/                        # imports updated to lb.*
├── docs/
│   ├── assets/                   # the 4 hero images land here (or deleted)
│   └── legacy/                   # superseded doc series archived here
└── knowledge.json techniques.json hypotheses.json results.tsv results/   # DATA — root, unchanged
```

**Entry points** (`pyproject.toml [project.scripts]`):
- `lb-round = lb.cli:round_main` — wraps the autoresearch tournament driver.
- `lb-report = lb.console.app:main` — starts the Flask report server.

---

## 4. Key design decisions

### 4.1 `lb/paths.py` — one source of truth
```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # src/lb/paths.py -> repo root
TEMPLATES_DIR = ROOT / "src" / "lb" / "templates"
MODULES_DIR   = ROOT / "src" / "lb" / "modules"
KNOWLEDGE_JSON = ROOT / "knowledge.json"
TECHNIQUES_JSON = ROOT / "techniques.json"
HYPOTHESES_JSON = ROOT / "hypotheses.json"
RESULTS_TSV = ROOT / "results.tsv"
RESULTS_DIR = ROOT / "results"
ROUND_RESULTS_CSV = RESULTS_DIR / "round_results.csv"
STATUS_JSON = ROOT / "reports" / "status.json"
```
`constants.py`, `orchestrator.py`, and the driver import from here. Every
`"/home/ubuntu/lb"` literal is deleted. An override hook
(`LB_ROOT` env var) is supported for unusual deployments but defaults to the derived path.

### 4.2 Modules remain importable both as package and as flat QC files
Local fallback arms change to `from lb.modules import …` (I2). Because the QC bare-import
arm is tried first and only succeeds on QC (no top-level `bar_ext` on the local path),
local execution deterministically reaches the `lb.modules` arm. Verified by the render
smoke test + the existing `tests/test_bar_threshold_leak.py` (its `sys.path` walk is
replaced by a real `import lb.modules.*`).

### 4.3 `_PSUF` dedup — inject the shared source into the header (I5)
`lb/harness/psuf.py` defines a pure, stdlib-only function:
```python
def cell_suffix(cfg) -> str:
    ...  # the single canonical implementation (moved verbatim from _cell_key)
```
- **Driver** imports it directly: `_cell_key` becomes a thin wrapper around `cell_suffix`.
- **QC header**: `header.py.tmpl` gets a `__PSUF_FN__` placeholder. At render time the
  orchestrator substitutes `inspect.getsource(cell_suffix)` followed by
  `_PSUF = cell_suffix(CONFIG)`. The same source executes on QC → byte-identical key.
- **Parity test** (`tests/test_psuf_parity.py`): for a battery of configs, render the
  header, `exec` it to obtain `_PSUF`, and assert equality with `cell_suffix(cfg)`; also
  assert the rendered header stays `< 64000` bytes (I4).
- **Fallback if injection proves awkward** (e.g. minify/byte issues): keep `_PSUF` as a
  single shared string template constant consumed by both sites. Decision recorded in the
  plan; injection is preferred because it cannot drift.

### 4.4 Console package imports
The 9 `sections/*.py` 3-level `sys.path` walks and the `render_index`/`app.py` inserts
are replaced by `from lb.console import primitives as P` etc. `scripts/app.py` remains as
a 3-line compat shim (`from lb.console.app import main; main()`) so the running systemd
unit (`python3 .../scripts/app.py`) keeps working without a unit-file edit; updating the
unit to `lb-report` is offered as an optional follow-up.

---

## 5. Migration phases (each an isolated commit on branch `chore/package-restructure`)

1. **Land current loop work.** Commit the ~25 modified files + new `modules/*_ext.py` +
   2 new docs + `results/wang_step1_beta_router.tsv` in legible commits so the
   restructure diff is pure moves, not tangled content. Then branch.
2. **Filesystem + gitignore + docs** (no code-behavior change):
   - move 4 hero images → `docs/assets/` (or delete — see open question);
   - `git rm --cached -r scratch/_archive` (34 files) and ensure `scratch/` is ignored;
   - harden `.gitignore` (`qc/`, `*.aux`); delete `*.aux` + stale `.playwright-mcp/` logs;
   - archive superseded docs (`RESEARCH_REVIEW` v1/v2, `WANG_INTERNET_DEEP` v2/v3, dated
     `SESSION_*`/`LOOP_SESSION_*`/older `LEAK_AUDIT_*`) → `docs/legacy/`.
3. **Package skeleton.** Add `pyproject.toml`; `git mv` `harness/ modules/ templates/
   scripts/console/` → `src/lb/…`; add `__init__.py`; `pip install -e .`.
4. **Kill path landmines.** Add `lb/paths.py`; rewrite `constants.py`, `orchestrator.py`,
   driver to import it; delete every absolute literal.
5. **Dedup `_PSUF`.** Add `lb/harness/psuf.py`; wire driver + header injection; add parity
   test.
6. **Fix imports.** Module local fallbacks → `lb.modules`; console walks → package
   imports; `scripts/*` runners → `import lb.*`; regroup runners into
   `research/ audit/ diag/`.
7. **Entry points.** `lb/cli.py` + `[project.scripts]`; keep `scripts/app.py` shim.

Phases 1–2 are independently valuable and low-risk; 3–7 are the package conversion.

---

## 6. Validation gate (definition of done)

- `pip install -e .` succeeds; `python -c "import lb"` works from any CWD.
- **`pytest` green** — including the existing leak/threshold tests with imports updated.
- **Render smoke test**: `render_train_config(sample_cfg)` produces a `main.py` that
  `compile()`s and is `< 64000` bytes; every extra file compiles and is under budget (I3, I4).
- **`_PSUF` parity test**: shared `cell_suffix()` == rendered-header `_PSUF` across a
  config battery (I5).
- **Console**: `build_html()` / `build_data()` run without error; restart the running
  `app.py` server and confirm the dashboard renders.
- **Grep gate**: zero remaining `"/home/ubuntu/lb"` literals; zero `sys.path.insert`
  outside the compat shim.
- **Optional gold standard** (offered, not assumed): one real QC backtest submission via
  `qc_client` to confirm the full render → upload → backtest path end-to-end (needs creds,
  ~minutes).

---

## 7. Risks & rollback

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| In-QC module import breaks | low | I2 keeps the bare-import arm first; render smoke + leak test exercise the local arm |
| `_PSUF` divergence (silent 0-trade) | medium | §4.3 single source + injection + parity test (I5) |
| Rendered file exceeds 64k after injection | low | I4 byte assertion in the parity/smoke test; string-template fallback (§4.3) |
| Running Flask service breaks | low | `scripts/app.py` compat shim; explicit restart + render check |
| History lost on moved files | low | `git mv` throughout; phases are separate commits |

**Rollback:** the whole conversion lives on `chore/package-restructure`; phases are
discrete commits, so any phase can be reverted independently, and the branch can be
abandoned without touching the working `autoresearch/2026-05-31` branch.

---

## 8. Open questions (defaults chosen; confirm or override at spec review)

1. **Hero images**: move to `docs/assets/` (keep) vs. delete outright.
   **Default: move to `docs/assets/`** (cheap, preserves the website screenshots).
2. **Real QC backtest in the done-gate**: **Default: NO** — local render smoke +
   parity test is the gate; a real backtest is offered as an optional confirmation.
3. **systemd unit**: **Default: keep the `scripts/app.py` shim**, leave the unit
   untouched (no sudo needed); updating it to `lb-report` is an optional follow-up.
