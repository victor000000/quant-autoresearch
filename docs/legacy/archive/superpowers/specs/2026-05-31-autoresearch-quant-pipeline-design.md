# Autoresearch for ML Quant Trading Pipeline — Design Spec

**Date**: 2026-05-31
**Status**: Approved
**Inspired by**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

## 1. Overview

An autonomous AI agent loop that runs quant trading strategy experiments overnight on QuantConnect Cloud. The agent reads research directives, freely modifies pipeline code, submits backtests to 2 parallel QC nodes (5-min timeout each), evaluates results against multi-gate criteria, keeps or discards changes via git, and iterates indefinitely.

**Human role**: Write `program.md` and seed `techniques.json`. Review results in the morning.
**Agent role**: Discover ideas from papers/transcripts/web, implement them, run experiments, learn from results.

## 2. Architecture

```
                         ┌──────────────────────────┐
                         │  HUMAN INTERFACE          │
                         │  program.md               │
                         │  techniques.json (seed)   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  IDEA AGENT               │
                         │  Searches: arXiv, SSRN,   │
                         │  Google, Bing, web,       │
                         │  pdfs/, uni/transcripts/  │
                         │  → enriches techniques.json│
                         └────────────┬─────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EXECUTOR AGENT                             │
│                                                                  │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │bar_builder.py│ labeler.py   │ features.py  │ trainer.py   │  │
│  │(any axis)    │(any labeling)│(any features)│(any model)   │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
│                    │ Agent freely modifies all 4                  │
│                    │ git add + commit                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOOP ORCHESTRATOR                             │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │    QC Node 1        │    │    QC Node 2        │             │
│  │  ETF A              │    │  ETF B              │             │
│  │  Submit → Poll /30s │    │  Submit → Poll /30s │             │
│  │  DELETE if >5 min   │    │  DELETE if >5 min   │             │
│  └──────────┬──────────┘    └──────────┬──────────┘             │
│             └─────────────┬────────────┘                         │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────┐             │
│  │         MULTI-GATE EVALUATOR                    │             │
│  │  G0: Completed (not timeout)                    │             │
│  │  G1: Calmar OOS > 3.0                          │             │
│  │  G2: Trades OOS > 80                           │             │
│  │  G3: No lookahead bias                         │             │
│  │  G4: Not overfit (|train_AUC - val_AUC| < 0.05)│             │
│  │  ALL pass → KEEP  |  ANY fail → DISCARD        │             │
│  └────────────────────────────────────────────────┘             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH MEMORY                               │
│  results.tsv         knowledge.json        techniques.json       │
│  (experiment log)    (structured findings) (ranked idea queue)  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. File Structure

```
lb/
├── autoresearch/                    ← NEW: autonomous research system
│   ├── program.md                   ← HUMAN: goals, constraints, criteria
│   ├── techniques.json              ← IDEA AGENT: ranked idea queue
│   ├── results.tsv                  ← EXECUTOR: experiment log (git-untracked)
│   ├── knowledge.json               ← AGENT: structured findings
│   │
│   ├── modules/                     ← EDITABLE: agent freely modifies
│   │   ├── bar_builder.py           ← Custom axis (dollar, tick, vol, range, ...)
│   │   ├── labeler.py               ← Labeling (KMeans, BGM, Carry, DBSCAN, ...)
│   │   ├── features.py              ← Feature engineering (entropy, FFT, AE, ...)
│   │   └── trainer.py               ← Model + train + sweep + evaluation
│   │
│   ├── harness/                     ← FIXED: agent does NOT modify
│   │   ├── orchestrator.py          ← 2-node submit, poll, timeout, cancel
│   │   ├── evaluator.py             ← Multi-gate evaluation logic
│   │   ├── qc_client.py             ← QC API (extends experiment_summary/tools/api_curl.py)
│   │   └── constants.py             ← ETF universe, splits, gate thresholds
│   │
│   └── templates/                   ← FIXED: concatenation templates
│       ├── header.py.tmpl           ← QC boilerplate, data collection
│       └── footer.py.tmpl           ← ObjectStore save, runtime stats export
│
├── experiment_summary/              ← EXISTING: historical results
├── pdfs/                            ← EXISTING: AFML.pdf, CFI.pdf, MLAM.pdf, wang slides
├── uni/transcripts/                 ← EXISTING: 37 Wang course transcripts
└── lean_workspace/                  ← EXISTING: generated QC scripts
```

## 4. Template Rendering

The 4 agent-editable modules are concatenated into a single QC-compatible script:

```
header.py.tmpl  ─┐
bar_builder.py  ─┤
labeler.py      ─┼── concat → replace __TICKER__ → QC script
features.py     ─┤
trainer.py      ─┘
footer.py.tmpl  ─┘
```

**Header (fixed)**: `AlgorithmImports`, constants (`TICKER`, `TRAIN_END`, `VAL_END`, `TEST_END`), `QCAlgorithm` class with `initialize()` and `on_data()` for minute data collection.

**Footer (fixed)**: `ObjectStore.save()` for predictions, `set_runtime_statistic()` for `best_cal`, `train_auc`, `val_auc`.

**Agent freedom**: No forced function signatures. No import whitelist. The agent can invent any axis type, any labeling scheme, any feature set, any model. The only requirements:
1. Uses `__TICKER__` placeholder (gets replaced per ETF)
2. Calls `self.set_runtime_statistic("best_cal", ...)`, `("train_auc", ...)`, `("val_auc", ...)`
3. Calls `self.object_store.save(...)` for predictions

**Pre-submit validation** (30 seconds, catches crash-causing errors only):
- Syntax check (`compile()`)
- Required runtime stat keys present
- `__TICKER__` placeholder present

## 5. Multi-Gate Evaluator

| Gate | Metric | Threshold | Source |
|------|--------|-----------|--------|
| G0 | Completion | Status == "Completed" | `bt["status"]` |
| G1 | Calmar OOS | > 3.0 on TEST period | `CAGR / MDD` from runtime stats |
| G2 | Trade Count | > 80 on TEST period | `Total Orders` from QC statistics |
| G3 | No Lookahead | Zero future-data leaks | Timestamp audit in evaluator |
| G4 | No Overfit | `|train_AUC - val_AUC| < 0.05` | AUC from runtime stats |

**Outcome per experiment (2 ETFs tested simultaneously)**:
- BOTH pass all gates → KEEP (git stays on commit)
- ONE passes all gates → KEEP (at least one ETF improved)
- NEITHER passes → DISCARD (`git reset --hard HEAD~1`)
- Any timeout → TIMEOUT (`git reset`, idea re-queued with lower priority)
- Any crash → CRASH (`git reset`, fix and retry ONCE)

## 6. NEVER STOP Loop Protocol

### Setup (once per session)
1. Read `program.md`, `techniques.json`, `knowledge.json`, `results.tsv`
2. Read all 4 modules + `harness/constants.py`
3. Verify QC credentials and data cache
4. Create git branch: `autoresearch/YYYY-MM-DD`
5. Run BASELINE on 2 ETFs → log to `results.tsv`
6. If technique queue < 5: run IDEA AGENT search pass

### Experiment Loop (indefinite)

```
LOOP:
  1. PICK: Pop highest-priority idea from techniques.json
     - Cross-check with knowledge.json to skip dead ends
     - If tried on some ETFs with promise → test remaining ETFs
     - If never tried → proceed

  2. IMPLEMENT: Edit modules to implement the idea
     - git add modules/ && git commit -m "<technique>: <change>"

  3. SELECT: Pick 2 ETFs from core-7
     - Prefer assets where similar techniques worked
     - Rotate through all 7 across experiments

  4. SUBMIT: Render + submit to both QC nodes simultaneously
     - Replace __TICKER__ → ETF_A, ETF_B
     - Submit via qc_client

  5. MONITOR: Poll both every 30 seconds
     - Completed → read full result
     - Elapsed > 300s → DELETE backtest, mark TIMEOUT
     - Error → read logs, mark CRASH

  6. EVALUATE: Run multi-gate evaluation per ETF
     - All gates pass → KEEP
     - Any gate fails → DISCARD (git reset)

  7. LOG: Append results.tsv, update knowledge.json, update techniques.json
     - If queue < 5: trigger IDEA AGENT search pass
     - If promising partial results: clone idea for remaining ETFs
     - If results suggest new idea: append to queue

  8. REPEAT
```

### Crash Recovery
- Compile/runtime error → read QC logs, fix ONCE if trivial, otherwise log CRASH and move on
- OOM → reduce model size or batch size, retry once
- Never retry more than once per crash

### Stuck Detection
- Last 5 experiments all DISCARD with same gate → log warning
- Queue empty + idea agent finds nothing → log, suggest human review

### NEVER STOP
- No iteration limit
- Agent must not ask "should I continue?"
- Only exit: human interrupt, credential expiry, hardware failure

## 7. Idea Discovery (Idea Agent)

The agent searches broadly for new quant strategy techniques:

| Source | Method | Example Queries |
|--------|--------|-----------------|
| arXiv | Search `q-fin.TR`, `q-fin.ST`, `cs.LG` | "unsupervised regime detection financial time series" |
| SSRN | Search recent papers | "machine learning trading strategy discovery" |
| Google Scholar | Academic search | "novel labeling method quant finance 2025" |
| Bing/Web | General search | "custom bar types information-driven ML" |
| pdfs/ (local) | Grep + semantic search | AFML.pdf, CFI.pdf, MLAM.pdf, wang slides |
| uni/transcripts/ (local) | Wang course transcripts | "大家可以试一下", "更好的方法", technique names |

**Output**: Enriched `techniques.json` entries with source attribution, hypothesis, confidence, and suggested target assets.

**Deduplication**: Cross-references with `knowledge.json` to avoid retrying dead ends. Filters out techniques already tried and failed on all target assets.

**Frequency**: Triggered when queue < 5 items, or every ~10 experiments.

## 8. Research Memory

### `techniques.json`
```json
{
  "queue": [{
    "id": "t042",
    "technique": "Hurst-adaptive labeling on vol bars",
    "description": "...",
    "source": {"type": "pdf", "ref": "AFML.pdf §3.4", "url": null},
    "hypothesis": "...",
    "priority": 4,
    "applicable_assets": ["GLD", "GDX", "XLE"],
    "status": "queued",
    "tried_count": 0
  }],
  "dead_ends": ["t001", "t015"],
  "last_idea_search": "2026-06-01T03:40:00Z"
}
```

### `knowledge.json`
```json
{
  "techniques": {
    "carry_labeling": {
      "description": "Always long when forward vol < median",
      "results": {
        "GLD": {"status": "kept", "best_calmar": 1.61, "trades": 45},
        "QQQ": {"status": "dead_end", "reason": "1-class on tech ETFs"}
      },
      "verdict": "Works on commodities. Fails on tech/equity."
    }
  },
  "axis_types": { "...": "..." },
  "dead_ends_global": [
    "HMM labeling (never selected over KMeans)",
    "CUSUM labeling (never selected)",
    "FracDiff features (hurt OOS)"
  ],
  "frontier": {
    "best_calmar_overall": 1.61,
    "current_bottleneck": "minute data + XGBoost ceiling",
    "unexplored": ["tick data", "deep learning features", "cross-asset features"]
  }
}
```

### `results.tsv`
```
commit	calmar	trades	status	description
a1b2c3d	0.85	45	keep	baseline: dollar+kmeans+xgb on GLD
b2c3d4e	0.92	52	keep	tick bars + GMM labeling on IWM
c3d4e5f	0.00	0	crash	OOM: doubled model depth
d4e5f6g	0.45	12	discard	Hurst-adaptive: too few trades
```

## 9. Implementation Plan

### Phase 1: Foundation (harness/)
1. **`qc_client.py`** — Extend `experiment_summary/tools/api_curl.py`:
   - Add `delete_backtest(pid, bid)` via `POST /backtests/delete`
   - Add `read_bt_with_timeout(pid, bid, timeout_s)` — poll + auto-delete
   - Add `submit_and_wait(pid, code, name, timeout_s)` — full lifecycle

2. **`constants.py`** — Define:
   - `CORE_7_ETFS = ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD"]`
   - `TRAIN_END`, `VAL_END`, `TEST_END` dates
   - `TIME_BUDGET = 300` (5 minutes)
   - `GATE_THRESHOLDS = {calmar: 3.0, trades: 80, auc_divergence: 0.05}`
   - `QC_PROJECT_ID = 31338454`

3. **`evaluator.py`** — Multi-gate evaluation:
   - `evaluate(bt_result)` → `{g0_pass, g1_pass, g2_pass, g3_pass, g4_pass, details}`
   - `lookahead_audit(script_text)` → `{pass, violations}`

4. **`orchestrator.py`** — Loop engine:
   - `render_script(modules, ticker)` → concatenated QC script
   - `run_experiment(commit_hash, ticker_a, ticker_b)` → `{result_a, result_b}`
   - `validate_script(script_text)` → `[errors]`
   - `run_loop(program_md, techniques_json)` → main entry point

### Phase 2: Modules (modules/)
5. Extract current best pipeline (v384 style) into the 4 module files as baseline
6. Ensure each module is self-contained and can be independently edited

### Phase 3: Templates (templates/)
7. Create `header.py.tmpl` and `footer.py.tmpl` from existing pipeline structure
8. Verify concatenation produces working QC scripts

### Phase 4: Knowledge System
9. Initialize `knowledge.json` from existing `experiment_summary/results/` data
10. Initialize `techniques.json` with promising unexplored ideas
11. Write `program.md` with research goals and constraints

### Phase 5: Integration & Test
12. End-to-end test: manual run of one experiment through full loop
13. Overnight dry run with strict monitoring
14. Tune based on first-night results

## 10. Key Design Decisions

1. **4 modules, not 1 file**: Splits the pipeline into independently editable units while keeping the search space wide open. The agent can modify one module without touching the others.

2. **Concatenation, not imports**: Matches QC's single-file requirement. Simpler than AST merging. Modules just need to not have name collisions.

3. **No forced signatures**: The agent can restructure how modules communicate. Only the runtime stat export keys are mandatory.

4. **Multi-gate evaluation**: Single-metric optimization (Calmar only) leads to overfit strategies with 3 trades. The 4-gate system enforces robustness.

5. **Knowledge deduplication**: Prevents the agent from retrying HMM labeling for the 15th time. `dead_ends_global` is seeded from the 150+ pipeline versions of historical failures.

6. **2-node parallelism**: Matches QC Cloud free tier limit. 2 ETFs tested per experiment, ~16 experiments per 8-hour night (at ~30 min per full cycle including agent think time).

7. **5-minute timeout with active cancellation**: QC backtests that run longer are killed via `POST /backtests/delete`. This bounds experiment duration and forces the agent to design efficient pipelines.

8. **Git-based state management**: Exact Karpathy pattern. Branch per session, commit on improvement, reset on regression. Clean audit trail.
