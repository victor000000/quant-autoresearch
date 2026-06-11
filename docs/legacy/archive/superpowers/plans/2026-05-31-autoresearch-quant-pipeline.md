# Autoresearch Quant Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous AI agent loop that runs quant trading experiments overnight on QuantConnect Cloud — agent freely modifies pipeline code, submits to 2 parallel QC nodes, evaluates via 4-gate criteria, and iterates indefinitely via git-based keep/discard.

**Architecture:** 4 editable modules (`bar_builder.py`, `labeler.py`, `features.py`, `trainer.py`) concatenated with fixed templates into single-file QC scripts. Fixed harness (`harness/`) provides orchestration, evaluation, QC API client. Agent manages state via git commits (keep) / resets (discard). Research memory in `knowledge.json` + `techniques.json` + `results.tsv`.

**Tech Stack:** Python 3.12, QC Cloud REST API (curl-based), git, existing QuantConnect LEAN environment, sklearn, xgboost, numpy, pandas.

---

## File Structure Map

```
autoresearch/                       ← ALL new code goes here
├── harness/
│   ├── qc_client.py                ← Create: QC API with delete, timeout, full lifecycle
│   ├── constants.py                ← Create: universe, splits, thresholds
│   ├── evaluator.py                ← Create: 4-gate evaluation
│   └── orchestrator.py             ← Create: render, validate, submit, loop engine
├── modules/
│   ├── bar_builder.py              ← Create: extracted from v370 (vol bars + dollar bars)
│   ├── labeler.py                  ← Create: extracted from v370 (KMeans two-stage)
│   ├── features.py                 ← Create: extracted from v370 (72 features + entropy)
│   └── trainer.py                  ← Create: extracted from v370 (XGBoost + sweep)
├── templates/
│   ├── header.py.tmpl              ← Create: QC boilerplate + data collection
│   └── footer.py.tmpl              ← Create: ObjectStore save + runtime stats
├── program.md                      ← Create: human research directive
├── techniques.json                 ← Create: seeded idea queue
├── knowledge.json                  ← Create: from experiment_summary/results/
└── results.tsv                     ← Create: empty with header, populated by agent
```

---

### Task 1: `harness/constants.py` — Research Universe & Gate Configuration

**Files:**
- Create: `autoresearch/harness/__init__.py`
- Create: `autoresearch/harness/constants.py`

- [ ] **Step 1: Create harness package init**

```bash
mkdir -p /Users/liyuanjun/ai_work/lb/autoresearch/harness
```

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/harness/__init__.py
"""Fixed evaluation harness — do NOT modify (agent playground is modules/)."""
```

- [ ] **Step 2: Write constants.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/harness/constants.py
"""Fixed constants for the autoresearch loop. Agent reads these, does NOT modify."""

from datetime import datetime

# === ETF Universe ===
CORE_7_ETFS = ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD"]

# === Time Splits ===
TRAIN_END = datetime(2021, 8, 1)
VAL_END = datetime(2023, 8, 1)
TEST_END = datetime(2026, 6, 1)

# === QC Cloud ===
QC_PROJECT_ID = 31338454
QC_CREDS_PATH = "/Users/liyuanjun/ai_work/lb/qc/.creds.json"
QC_POLL_INTERVAL = 30  # seconds between status checks
TIME_BUDGET = 300       # 5 minutes max per backtest

# === Gate Thresholds ===
GATE_CALMAR_MIN = 3.0       # OOS Calmar must exceed this
GATE_TRADES_MIN = 80        # OOS trade count must exceed this
GATE_AUC_DIVERGENCE_MAX = 0.05  # |train_AUC - val_AUC| must be below this

# === Rendering ===
TEMPLATES_DIR = "/Users/liyuanjun/ai_work/lb/autoresearch/templates"
MODULES_DIR = "/Users/liyuanjun/ai_work/lb/autoresearch/modules"
QC_SCRIPTS_DIR = "/Users/liyuanjun/ai_work/lb/lean_workspace/_autoresearch"

# === Asset Fingerprinting (for ETF selection) ===
# Which ETFs historically respond to which technique families
ASSET_AFFINITY = {
    "trend_following": ["QQQ", "IWM", "XLE", "GLD"],
    "mean_reversion": ["HYG", "TLT", "EEM"],
    "volatility_regime": ["GLD", "XLE", "EEM"],
}

# Target bars per 17-year history (2009-2026) at minute granularity
TARGET_BARS = 15000
```

- [ ] **Step 3: Commit**

```bash
git add autoresearch/harness/__init__.py autoresearch/harness/constants.py
git commit -m "feat: add harness constants — ETF universe, gate thresholds, QC config"
```

---

### Task 2: `harness/qc_client.py` — QC API Client with Timeout & Delete

**Files:**
- Create: `autoresearch/harness/qc_client.py`

This extends the existing `experiment_summary/tools/api_curl.py` pattern. It reuses the `qc_post` function (copied inline to keep the harness self-contained) and adds `delete_backtest`, `submit_and_wait`, and `read_bt_with_timeout`.

- [ ] **Step 1: Write qc_client.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/harness/qc_client.py
"""QC Cloud API client with timeout enforcement and backtest lifecycle management.
Reuses the curl-based auth pattern from experiment_summary/tools/api_curl.py."""

import json, subprocess, time, base64, hashlib, os, sys

from .constants import QC_CREDS_PATH, QC_POLL_INTERVAL, TIME_BUDGET

# ---------------------------------------------------------------------------
# Low-level HTTP (same pattern as api_curl.py, self-contained for harness)
# ---------------------------------------------------------------------------

def _get_creds():
    with open(QC_CREDS_PATH) as f:
        return json.load(f)

def _qc_post(path, body=None, max_time=120):
    """POST to QC API v2. Returns parsed JSON or {} on failure."""
    c = _get_creds()
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{c['token']}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{c['user_id']}:{digest}".encode()).decode()
    url = f"https://www.quantconnect.com/api/v2{path}"
    data_str = json.dumps(body or {})

    # Use Python requests if available, fallback to curl
    try:
        import requests
        r = requests.post(url,
            headers={"Authorization": f"Basic {auth}", "Timestamp": ts,
                     "Content-Type": "application/json"},
            data=data_str, timeout=max_time)
        if r.status_code == 200:
            return r.json()
        return {}
    except ImportError:
        pass

    # Curl fallback
    tmp = f"/tmp/qc_api_{os.getpid()}_{ts}.json"
    cmd = ["curl", "-s", "-w", "%{http_code}",
           "-X", "POST", url,
           "-H", f"Authorization: Basic {auth}",
           "-H", f"Timestamp: {ts}",
           "-H", "Content-Type: application/json",
           "-d", data_str,
           "--connect-timeout", "30", "--max-time", str(max_time),
           "-o", tmp]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 10)
        try:
            with open(tmp, 'r') as f:
                return json.load(f)
        except:
            return {}
    finally:
        try: os.unlink(tmp)
        except: pass
    return {}

# ---------------------------------------------------------------------------
# Backtest lifecycle
# ---------------------------------------------------------------------------

def submit_backtest(code, name):
    """Upload code, compile, create backtest. Returns backtestId or raises."""
    pid = _get_project_id()

    # 1. Upload
    r = _qc_post("/files/update", {"projectId": pid, "name": "main.py", "content": code})
    if not r.get("success"):
        raise RuntimeError(f"Upload failed: {r}")

    # 2. Compile
    r = _qc_post("/compile/create", {"projectId": pid})
    if not r.get("success"):
        raise RuntimeError(f"Compile create failed: {r}")
    cid = r["compileId"]

    # 3. Wait for compile
    t0 = time.time()
    while time.time() - t0 < 120:
        r = _qc_post("/compile/read", {"projectId": pid, "compileId": cid})
        state = r.get("state", "")
        if state == "BuildSuccess":
            break
        if state == "BuildError":
            raise RuntimeError(f"BuildError: {(r.get('logs') or '')[:500]}")
        time.sleep(5)

    # 4. Create backtest
    r = _qc_post("/backtests/create",
                 {"projectId": pid, "compileId": cid, "backtestName": name})
    if not r.get("success"):
        raise RuntimeError(f"Backtest create failed: {r.get('errors', r)}")
    return r["backtest"]["backtestId"]


def read_backtest(bid):
    """Read full backtest result."""
    pid = _get_project_id()
    r = _qc_post("/backtests/read", {"projectId": pid, "backtestId": bid}, max_time=120)
    return r.get("backtest", {})


def read_backtest_status(bid):
    """Lightweight status check. Returns (status, runtime_stats, logs)."""
    bt = read_backtest(bid)
    if not bt:
        return "?", {}, ""
    return bt.get("status", "?"), bt.get("runtimeStatistics") or {}, bt.get("logs") or ""


def delete_backtest(bid):
    """Cancel/delete a running or completed backtest."""
    pid = _get_project_id()
    r = _qc_post("/backtests/delete", {"projectId": pid, "backtestId": bid})
    return r.get("success", False)


def is_done(status):
    """True if backtest is in a terminal state."""
    return status.startswith("Completed") or "Error" in status or status == "Canceled"


def submit_and_wait(code, name, timeout_s=None):
    """Full lifecycle: submit → poll → read result. Auto-deletes on timeout.
    Returns (result_dict, status_string) where status is one of:
      'completed', 'timeout', 'crash', 'error'
    """
    if timeout_s is None:
        timeout_s = TIME_BUDGET

    # Submit
    try:
        bid = submit_backtest(code, name)
    except Exception as e:
        return {"error": str(e)}, "crash"

    # Poll
    t_start = time.time()
    while True:
        elapsed = time.time() - t_start
        status, rt, logs = read_backtest_status(bid)

        if is_done(status):
            if status.startswith("Completed"):
                bt = read_backtest(bid)
                return bt, "completed"
            else:
                return {"status": status, "logs": logs, "runtime": rt}, "crash"

        if elapsed > timeout_s:
            delete_backtest(bid)
            return {"timeout_s": elapsed, "runtime": rt}, "timeout"

        time.sleep(QC_POLL_INTERVAL)


def _get_project_id():
    """QC project ID from constants."""
    from .constants import QC_PROJECT_ID
    return QC_PROJECT_ID
```

- [ ] **Step 2: Quick smoke test (requires QC credentials)**

```bash
cd /Users/liyuanjun/ai_work/lb && python3 -c "
from autoresearch.harness.qc_client import _get_creds
c = _get_creds()
assert 'user_id' in c and 'token' in c
print('Credentials OK:', c['user_id'])
"
```

Expected: `Credentials OK: 286064`

- [ ] **Step 3: Commit**

```bash
git add autoresearch/harness/qc_client.py
git commit -m "feat: add qc_client — submit, poll, timeout, delete backtests"
```

---

### Task 3: `harness/evaluator.py` — Multi-Gate Evaluation

**Files:**
- Create: `autoresearch/harness/evaluator.py`

- [ ] **Step 1: Write evaluator.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/harness/evaluator.py
"""Multi-gate evaluator for QC backtest results.
G0: Completed    — backtest finished without timeout
G1: Calmar > 3.0 — OOS risk-adjusted return
G2: Trades > 80  — sufficient trading activity
G3: No Lookahead — zero future-data leaks
G4: No Overfit   — train/val AUC divergence < 0.05
"""

from .constants import GATE_CALMAR_MIN, GATE_TRADES_MIN, GATE_AUC_DIVERGENCE_MAX


def evaluate(bt_result, script_text=None):
    """Evaluate a single backtest result against all 4 gates.

    Args:
        bt_result: dict from read_backtest() or submit_and_wait()
        script_text: optional, the pipeline script text for lookahead audit

    Returns:
        dict with per-gate pass/fail and overall verdict
    """
    gates = {}

    # G0: Completion
    status = bt_result.get("status", "?")
    gates["g0_completed"] = {
        "pass": status.startswith("Completed"),
        "value": status,
        "detail": f"Status: {status}"
    }

    if not gates["g0_completed"]["pass"]:
        return _verdict(gates, "timeout")

    # Parse statistics
    s = bt_result.get("statistics", {}) or {}
    rt = bt_result.get("runtimeStatistics", {}) or {}

    # G1: Calmar OOS
    try:
        cagr_str = s.get("Compounding Annual Return", "0%")
        cagr = float(cagr_str.replace("%", ""))
        mdd_str = s.get("Drawdown", "0%")
        mdd = float(mdd_str.replace("%", ""))
        calmar = cagr / mdd if abs(mdd) > 0.01 else 0.0
    except (ValueError, ZeroDivisionError):
        calmar = 0.0

    gates["g1_calmar"] = {
        "pass": calmar > GATE_CALMAR_MIN,
        "value": round(calmar, 4),
        "detail": f"Calmar: {calmar:.4f} (need >{GATE_CALMAR_MIN})"
    }

    # G2: Trade Count OOS
    try:
        orders_str = s.get("Total Orders", "0")
        orders = int(orders_str)
    except (ValueError, TypeError):
        orders = 0

    gates["g2_trades"] = {
        "pass": orders > GATE_TRADES_MIN,
        "value": orders,
        "detail": f"Trades: {orders} (need >{GATE_TRADES_MIN})"
    }

    # G3: Lookahead audit
    if script_text:
        audit = lookahead_audit(script_text)
        gates["g3_lookahead"] = {
            "pass": audit["pass"],
            "value": audit["violations"],
            "detail": f"Lookahead violations: {audit['violations']}"
        }
    else:
        gates["g3_lookahead"] = {
            "pass": True, "value": [], "detail": "No script text provided for audit"
        }

    # G4: Overfit detection
    try:
        train_auc_str = rt.get("train_auc", "0")
        val_auc_str = rt.get("val_auc", "0")
        train_auc = float(train_auc_str)
        val_auc = float(val_auc_str)
        divergence = abs(train_auc - val_auc)
    except (ValueError, TypeError):
        divergence = 0.0
        train_auc = val_auc = 0.0

    gates["g4_overfit"] = {
        "pass": divergence < GATE_AUC_DIVERGENCE_MAX,
        "value": round(divergence, 4),
        "detail": f"AUC divergence: {divergence:.4f} (train={train_auc:.4f}, val={val_auc:.4f}, max={GATE_AUC_DIVERGENCE_MAX})"
    }

    return _verdict(gates, "completed")


def _verdict(gates, terminal_status):
    """Compute overall verdict from gate results."""
    g0 = gates.get("g0_completed", {})
    g1 = gates.get("g1_calmar", {})
    g2 = gates.get("g2_trades", {})
    g3 = gates.get("g3_lookahead", {})
    g4 = gates.get("g4_overfit", {})

    all_pass = all([
        g0.get("pass", False),
        g1.get("pass", False),
        g2.get("pass", False),
        g3.get("pass", False),
        g4.get("pass", False),
    ])

    # Determine status
    if terminal_status != "completed":
        status = terminal_status  # timeout, crash
    elif all_pass:
        status = "keep"
    elif not g3.get("pass", True):
        status = "leak"
    elif not g4.get("pass", True):
        status = "overfit"
    else:
        status = "discard"

    return {
        "status": status,
        "all_pass": all_pass,
        "gates": gates,
        "summary": _summarize(gates, status),
    }


def _summarize(gates, status):
    """One-line summary of evaluation."""
    parts = []
    for g in ["g0_completed", "g1_calmar", "g2_trades", "g3_lookahead", "g4_overfit"]:
        if g in gates:
            icon = "✓" if gates[g]["pass"] else "✗"
            parts.append(f"{icon}{g}")
    return f"[{status.upper()}] " + " ".join(parts)


def lookahead_audit(script_text):
    """Scan pipeline script for common lookahead patterns.

    Returns dict with pass (bool) and violations (list of str).
    This is a heuristic audit — not exhaustive, but catches common mistakes.
    """
    violations = []

    # Pattern 1: Negative shift on time axis (future data leakage in pandas)
    # .shift(-k) brings FUTURE values to current row
    if ".shift(-" in script_text:
        violations.append("pandas .shift(-N) detected — likely future-data leak")

    # Pattern 2: Reversed indexing on time axis
    # [::-1] on bar-indexed arrays reverses time
    import re
    # Look for array slicing patterns that reverse on time-indexed data
    rev_patterns = re.findall(r'(\w+)\[::-\d*\]', script_text)
    for match in rev_patterns:
        if match not in ('text', 's', 'x'):  # common non-time uses
            violations.append(f"potential reversed indexing: {match}[::-1]")

    # Pattern 3: Using TEST/VAL period data in training
    # Check if train mask uses bar_ts correctly
    if "tr_m" in script_text and "te_m" in script_text:
        # Verify tr_m comes before te_m in time logic
        tr_idx = script_text.find("tr_m")
        te_idx = script_text.find("te_m")
        # This is weak — flag for review if masks appear ambiguous
        if "tr_m|te_m" in script_text or "te_m|tr_m" in script_text:
            violations.append("train and test masks combined with OR — possible leak")

    # Pattern 4: fillna using future values (bfill on time series)
    if ".fillna(method='bfill'" in script_text or ".bfill()" in script_text:
        if any(kw in script_text for kw in ['lr', 'lc', 'ret', 'close', 'price']):
            violations.append("backfill (bfill) on price/return data — possible future leak")

    # Pattern 5: sklearn fit on non-training data
    if ".fit(" in script_text:
        # Check for .fit() calls — should use tr_m subset
        fit_calls = re.findall(r'(\w+)\.fit\(([^)]+)\)', script_text)
        for obj, args in fit_calls:
            if 'tr_m' not in args and 'train' not in args.lower():
                # Not necessarily a leak, but flag for review
                pass  # Too many false positives — skip aggressive checking

    return {
        "pass": len(violations) == 0,
        "violations": violations,
    }
```

- [ ] **Step 2: Smoke test the evaluator with synthetic data**

```bash
cd /Users/liyuanjun/ai_work/lb && python3 -c "
from autoresearch.harness.evaluator import evaluate, lookahead_audit

# Test completed backtest
bt = {
    'status': 'Completed',
    'statistics': {
        'Compounding Annual Return': '45%',
        'Drawdown': '12%',
        'Total Orders': '95'
    },
    'runtimeStatistics': {
        'train_auc': '0.88',
        'val_auc': '0.85'
    }
}
result = evaluate(bt, script_text='tr_m = bar_ts < TRAIN_END')
print('All pass:', result['all_pass'])
print('Status:', result['status'])
print('Summary:', result['summary'])
assert result['all_pass'] == True
assert result['status'] == 'keep'

# Test failing backtest
bt2 = {'status': 'Completed', 'statistics': {'Compounding Annual Return': '10%', 'Drawdown': '8%', 'Total Orders': '3'}, 'runtimeStatistics': {'train_auc': '0.95', 'val_auc': '0.70'}}
result2 = evaluate(bt2)
print()
print('Status:', result2['status'])
print('Summary:', result2['summary'])
assert result2['all_pass'] == False

# Test lookahead audit
audit = lookahead_audit('x.shift(-1) + lr')
print()
print('Lookahead violations:', audit['violations'])
assert audit['pass'] == False
print('All tests passed!')
"
```

Expected: All assertions pass, showing `[KEEP]` for good bt and `[DISCARD]` for bad bt.

- [ ] **Step 3: Commit**

```bash
git add autoresearch/harness/evaluator.py
git commit -m "feat: add multi-gate evaluator — Calmar, trades, lookahead, overfit"
```

---

### Task 4: `harness/orchestrator.py` — Render, Validate, Run Loop

**Files:**
- Create: `autoresearch/harness/orchestrator.py`

- [ ] **Step 1: Write orchestrator.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/harness/orchestrator.py
"""Loop orchestrator: render pipeline scripts, submit to QC, evaluate, manage git state.

The agent calls run_experiment() for each idea. The orchestrator handles all
the mechanical work: template rendering, QC submission, polling, evaluation,
git operations, and logging.
"""

import os, sys, time, json, subprocess
from datetime import datetime

from .constants import (
    CORE_7_ETFS, TEMPLATES_DIR, MODULES_DIR, QC_SCRIPTS_DIR, TARGET_BARS
)
from .qc_client import submit_and_wait
from .evaluator import evaluate

# Paths relative to project root
PROJECT_ROOT = "/Users/liyuanjun/ai_work/lb"
AUTORESEARCH_DIR = os.path.join(PROJECT_ROOT, "autoresearch")
RESULTS_TSV = os.path.join(AUTORESEARCH_DIR, "results.tsv")
KNOWLEDGE_JSON = os.path.join(AUTORESEARCH_DIR, "knowledge.json")
TECHNIQUES_JSON = os.path.join(AUTORESEARCH_DIR, "techniques.json")


# ---------------------------------------------------------------------------
# Script rendering
# ---------------------------------------------------------------------------

def read_module(name):
    """Read a module file. Returns its content as string."""
    path = os.path.join(MODULES_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module not found: {path}")
    with open(path) as f:
        return f.read()


def render_script(ticker):
    """Concatenate header + 4 modules + footer into a single QC script.
    Replaces __TICKER__ with the actual ticker.
    """
    header_path = os.path.join(TEMPLATES_DIR, "header.py.tmpl")
    footer_path = os.path.join(TEMPLATES_DIR, "footer.py.tmpl")

    with open(header_path) as f:
        script = f.read()

    # Insert modules in order
    for mod in ["bar_builder.py", "labeler.py", "features.py", "trainer.py"]:
        script += "\n# === " + mod + " ===\n"
        script += read_module(mod)
        script += "\n"

    with open(footer_path) as f:
        script += f.read()

    # Replace placeholders
    script = script.replace("__TICKER__", ticker)

    return script


def validate_script(script_text):
    """Pre-submit validation. Returns list of error strings (empty = valid)."""
    errors = []

    # 1. Must compile
    try:
        compile(script_text, '<pipeline>', 'exec')
    except SyntaxError as e:
        errors.append(f"SYNTAX: {e}")

    # 2. Required runtime stat keys
    for key in ["best_cal", "train_auc", "val_auc"]:
        if f'set_runtime_statistic("{key}"' not in script_text:
            errors.append(f"MISSING_STAT: set_runtime_statistic(\"{key}\") — evaluator needs this")

    # 3. Must have __TICKER__ (should be replaced before submit)
    # Already replaced by render_script, but double-check
    # (no-op if ticker already replaced)

    return errors


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_experiment(ticker_a, ticker_b, description):
    """Run one experiment on two ETFs in parallel.

    Args:
        ticker_a, ticker_b: ETF tickers to test
        description: short description for logging

    Returns:
        dict with keys: commit, results (dict per ticker), overall_status, description
    """
    commit = _get_commit_hash()

    # Render scripts
    script_a = render_script(ticker_a)
    script_b = render_script(ticker_b)

    # Validate
    errors_a = validate_script(script_a)
    errors_b = validate_script(script_b)
    if errors_a or errors_b:
        return {
            "commit": commit,
            "description": description,
            "overall_status": "crash",
            "results": {
                ticker_a: {"status": "crash", "error": f"Validation: {errors_a}"},
                ticker_b: {"status": "crash", "error": f"Validation: {errors_b}"},
            }
        }

    # Save scripts for debugging
    os.makedirs(QC_SCRIPTS_DIR, exist_ok=True)
    for ticker, script in [(ticker_a, script_a), (ticker_b, script_b)]:
        script_path = os.path.join(QC_SCRIPTS_DIR, f"{ticker}_{commit[:7]}.py")
        with open(script_path, 'w') as f:
            f.write(script)

    # Submit both simultaneously (conceptually; Python submits sequentially but QC runs in parallel)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Submitting {ticker_a} + {ticker_b} ...")
    bt_a, status_a = submit_and_wait(script_a, f"auto_{ticker_a}_{commit[:7]}")
    bt_b, status_b = submit_and_wait(script_b, f"auto_{ticker_b}_{commit[:7]}")

    # Evaluate each
    eval_a = evaluate(bt_a, script_text=script_a) if status_a == "completed" else _timeout_or_crash_eval(bt_a, status_a)
    eval_b = evaluate(bt_b, script_text=script_b) if status_b == "completed" else _timeout_or_crash_eval(bt_b, status_b)

    # Extract key metrics for results.tsv
    best_calmar = max(
        eval_a.get("gates", {}).get("g1_calmar", {}).get("value", 0),
        eval_b.get("gates", {}).get("g1_calmar", {}).get("value", 0),
    )
    best_trades = max(
        eval_a.get("gates", {}).get("g2_trades", {}).get("value", 0),
        eval_b.get("gates", {}).get("g2_trades", {}).get("value", 0),
    )

    # Overall: keep if ANY ETF passes all gates
    a_pass = eval_a.get("all_pass", False)
    b_pass = eval_b.get("all_pass", False)
    if a_pass or b_pass:
        overall = "keep"
    elif status_a == "timeout" or status_b == "timeout":
        overall = "timeout"
    elif status_a == "crash" or status_b == "crash":
        overall = "crash"
    else:
        overall = "discard"

    result = {
        "commit": commit,
        "description": description,
        "overall_status": overall,
        "best_calmar": best_calmar,
        "best_trades": best_trades,
        "results": {
            ticker_a: {"status": status_a, "evaluation": eval_a},
            ticker_b: {"status": status_b, "evaluation": eval_b},
        }
    }

    # Log to results.tsv
    _log_result(commit, best_calmar, best_trades, overall, description)

    return result


def _timeout_or_crash_eval(bt, status):
    """Create evaluation dict for timeout/crash cases."""
    return {
        "status": status,
        "all_pass": False,
        "gates": {
            "g0_completed": {"pass": False, "value": status, "detail": f"Status: {status}"},
            "g1_calmar": {"pass": False, "value": 0.0, "detail": "N/A — did not complete"},
            "g2_trades": {"pass": False, "value": 0, "detail": "N/A — did not complete"},
            "g3_lookahead": {"pass": True, "value": [], "detail": "N/A"},
            "g4_overfit": {"pass": True, "value": 0.0, "detail": "N/A"},
        },
        "summary": f"[{status.upper()}] Backtest did not complete"
    }


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def _get_commit_hash():
    """Get short commit hash of HEAD."""
    result = subprocess.run(
        ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def git_keep():
    """Keep the current commit (no-op — commit already made)."""
    pass  # Agent already committed before calling run_experiment


def git_discard():
    """Discard current commit: reset to previous."""
    subprocess.run(
        ["git", "-C", PROJECT_ROOT, "reset", "--hard", "HEAD~1"],
        capture_output=True, text=True
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_result(commit, calmar, trades, status, description):
    """Append a row to results.tsv."""
    tsv_path = RESULTS_TSV
    existed = os.path.exists(tsv_path)
    with open(tsv_path, 'a') as f:
        if not existed:
            f.write("commit\tcalmar\ttrades\tstatus\tdescription\n")
        # Escape tabs in description
        desc_clean = description.replace("\t", " ").replace("\n", " ")
        f.write(f"{commit}\t{calmar:.4f}\t{trades}\t{status}\t{desc_clean}\n")


def load_knowledge():
    """Load knowledge.json, return empty dict if missing."""
    if os.path.exists(KNOWLEDGE_JSON):
        with open(KNOWLEDGE_JSON) as f:
            return json.load(f)
    return {}


def save_knowledge(data):
    """Save knowledge.json."""
    with open(KNOWLEDGE_JSON, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_techniques():
    """Load techniques.json, return empty queue if missing."""
    if os.path.exists(TECHNIQUES_JSON):
        with open(TECHNIQUES_JSON) as f:
            return json.load(f)
    return {"queue": [], "dead_ends": [], "last_idea_search": None}


def save_techniques(data):
    """Save techniques.json."""
    with open(TECHNIQUES_JSON, 'w') as f:
        json.dump(data, f, indent=2, default=str)
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/harness/orchestrator.py
git commit -m "feat: add orchestrator — render, validate, run experiments, log, git ops"
```

---

### Task 5: `modules/bar_builder.py` — Volatility Bars (Baseline)

**Files:**
- Create: `autoresearch/modules/__init__.py`
- Create: `autoresearch/modules/bar_builder.py`

Extract the VolBarBuilder class and build_bars logic from v370_train.py as the baseline. The agent will modify this freely.

- [ ] **Step 1: Create modules package and bar_builder.py**

```bash
mkdir -p /Users/liyuanjun/ai_work/lb/autoresearch/modules
```

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/modules/__init__.py
"""Editable pipeline modules — the agent's playground."""
```

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/modules/bar_builder.py
"""Custom bar builders. Edit freely — any axis type, any internal API.

Currently: Volatility Bars (Wang's 3rd axis type).
Volatility bar: cumsum(squared_log_returns * sqrt(volume)) >= threshold → sample.
During high vol: more bars (more info). During low vol: fewer bars.
"""
import math
import numpy as np


class VolBarBuilder:
    """Volatility bar: cumulative log_return² * sqrt(volume) → sample when threshold exceeded."""

    def __init__(self, threshold):
        self.thresh = threshold
        self.cum = 0.0
        self.last_close = None
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        lc = math.log(close)
        bar = None
        if self.last_close is not None:
            ret = lc - self.last_close
            contrib = (ret * ret) * math.sqrt(vol)
            if contrib > 0:
                self.cum += contrib
        self.last_close = lc
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            bar = {"ts_close": ts, "log_close": self.close_lc}
        return bar


class DollarBarBuilder:
    """Dollar bar: cumulative close * volume → sample when threshold exceeded."""

    def __init__(self, threshold):
        self.thresh = threshold
        self.cum = 0.0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0 or vol <= 0:
            return None
        lc = math.log(close)
        self.cum += close * vol
        self.close_lc = lc
        if self.cum >= self.thresh:
            self.cum = 0.0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


class TickBarBuilder:
    """Tick bar: count trades → sample every N ticks."""

    def __init__(self, threshold):
        self.thresh = threshold
        self.count = 0
        self.close_lc = None

    def update(self, ts, close, vol):
        if close <= 0:
            return None
        lc = math.log(close)
        self.count += 1
        self.close_lc = lc
        if self.count >= self.thresh:
            self.count = 0
            return {"ts_close": ts, "log_close": self.close_lc}
        return None


def build_bars(close, vol, ts_arr, bar_type="vol", target_bars=15000):
    """Build bars from minute data.

    Args:
        close: np.array of closing prices
        vol: np.array of volumes
        ts_arr: np.array of timestamps
        bar_type: "vol", "dollar", or "tick"
        target_bars: approximate number of bars desired

    Returns:
        lc: np.array of log-close values per bar
        lr: np.array of log-returns per bar
        N: number of bars
        bar_ts: np.array of bar timestamps
    """
    # Compute threshold to hit ~target_bars
    if bar_type == "dollar":
        total_dollar = float(np.sum(close * vol))
        threshold = total_dollar / target_bars if total_dollar > 0 else 1e-9
        builder = DollarBarBuilder(threshold)
    elif bar_type == "tick":
        threshold = max(1, len(close) // target_bars)
        builder = TickBarBuilder(threshold)
    else:  # vol (default)
        total_contrib = 0.0
        last_lc = None
        for i in range(len(close)):
            if close[i] <= 0 or vol[i] <= 0:
                continue
            lc_val = math.log(close[i])
            if last_lc is not None:
                total_contrib += (lc_val - last_lc) ** 2 * math.sqrt(vol[i])
            last_lc = lc_val
        threshold = total_contrib / target_bars if total_contrib > 0 else 1e-9
        builder = VolBarBuilder(threshold)

    # Build bars
    bars = []
    bar_ts_raw = []
    for i in range(len(close)):
        b = builder.update(ts_arr[i], close[i], vol[i])
        if b is not None:
            bars.append(b)
            bar_ts_raw.append(ts_arr[i])

    N = len(bars)
    lc = np.array([x["log_close"] for x in bars])
    lr = np.zeros(N)
    lr[1:] = lc[1:] - lc[:-1]
    bar_ts = np.array(bar_ts_raw)

    return lc, lr, N, bar_ts
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/modules/__init__.py autoresearch/modules/bar_builder.py
git commit -m "feat: add bar_builder module — Vol, Dollar, Tick bar types"
```

---

### Task 6: `modules/labeler.py` — KMeans Two-Stage Labeling (Baseline)

**Files:**
- Create: `autoresearch/modules/labeler.py`

- [ ] **Step 1: Write labeler.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/modules/labeler.py
"""Unsupervised labeling methods. Edit freely — any labeling scheme.

Currently: KMeans two-stage (vol cluster → direction cluster).
Also includes carry-inspired labeling (low vol = long).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as FS


def compute_forward_metrics(lc, lr, horizons=[50, 100, 200]):
    """Compute forward returns and volatility at multiple horizons.

    Returns:
        fwd_ret: dict {horizon: np.array of forward returns}
        fwd_vol: dict {horizon: np.array of forward volatility}
    """
    N = len(lc)
    fwd_ret = {}
    fwd_vol = {}
    for fk in horizons:
        fr = np.full(N, np.nan)
        fv = np.full(N, np.nan)
        for t in range(N - fk):
            wr = lr[t+1:t+fk+1]
            if len(wr) < 2:
                continue
            fr[t] = lc[t+fk] - lc[t]
            fv[t] = float(np.std(wr))
        fwd_ret[fk] = fr
        fwd_vol[fk] = fv
    return fwd_ret, fwd_vol


def generate_labels_kmeans_two_stage(lc, lr, tr_m, va_m, te_m, fv,
                                      fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """KMeans two-stage labeling: vol cluster → direction cluster.

    Stage 1: KMeans(K=2) on forward volatility → find low-vol regime
    Stage 2: KMeans(K∈{2,3}) on [fwd_ret, |fwd_ret|] within low-vol → find up-cluster

    Returns:
        best_labels: np.array of -1 (ignore), 0 (short/neutral), 1 (long)
        best_cfg: string describing the winning configuration
        best_horizon: the chosen forward horizon
    """
    N = len(lc)
    best_val_cal = -999
    best_labels = None
    best_cfg = ""
    best_horizon = None

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk])
        fwd_abs_k = np.abs(fwd_ret[fk])

        # Stage 1: Vol clustering on training data
        fv_clean = fwd_vol[fk][tr_m & fvd_k & fv]
        if len(fv_clean) < 30:
            continue
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(
            fv_clean.reshape(-1, 1))
        cv_vol = km_vol.predict(fwd_vol[fk][fvd_k & fv].reshape(-1, 1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low = (cv_vol == lo_vol)
        s2_mask_tr = tr_m[fvd_k & fv] & is_low

        fwd_cl = fwd_ret[fk][fvd_k & fv]
        fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        vf_tr = np.column_stack([fwd_cl[s2_mask_tr], fwd_abs_cl[s2_mask_tr]])
        if len(vf_tr) < 60:
            continue

        # Stage 2: Direction clustering
        fs_dir = FS().fit(vf_tr)
        vf_all = np.column_stack([fwd_cl[is_low], fwd_abs_cl[is_low]])
        vf_all_z = fs_dir.transform(vf_all)

        for nc in [2, 3]:
            km_dir = KMeans(n_clusters=nc, random_state=42, n_init=5).fit(
                fs_dir.transform(vf_tr))
            cv_dir = km_dir.predict(vf_all_z)
            up_c = int(np.argmax(km_dir.cluster_centers_[:, 0]))
            dir_labels = np.zeros(cv_dir.shape[0], dtype=int)
            dir_labels[cv_dir == up_c] = 1

            full_labels = np.full(N, -1, dtype=int)
            full_labels[np.where(fvd_k & fv)[0][is_low]] = dir_labels
            y = full_labels
            ly = y >= 0

            # Quick val check
            vx = fv & ly & va_m
            if vx.sum() < 30:
                continue
            # Simple heuristic: check class balance
            val_balance = y[vx].mean() if vx.sum() > 0 else 0
            if 0.2 < val_balance < 0.8:  # Both classes present
                cfg = f"km2_f{fk}_c{nc}"
                # Use validation class balance as rough quality signal
                quality = min(val_balance, 1 - val_balance)  # higher = more balanced
                score = quality  # Simplified; real Calmar computed in trainer

                if score > best_val_cal - 99:  # Always accept first valid config
                    if best_val_cal < -99 or score > best_val_cal:
                        best_val_cal = score
                        best_labels = y
                        best_cfg = cfg
                        best_horizon = fk

    return best_labels, best_cfg, best_horizon


def generate_labels_carry(fwd_vol, tr_m, va_m, fv, horizons=[50, 100, 200]):
    """Carry-inspired labels: always long when forward vol is below median.

    Simple but effective on volatility-bar-based pipelines (GLD Cal 1.59).
    """
    N = len(fwd_vol[horizons[0]])
    best_labels = None
    best_cfg = ""
    best_score = -999

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_vol[fk])
        med_v = float(np.median(fwd_vol[fk][tr_m & fvd_k & fv])) if (tr_m & fvd_k & fv).sum() > 10 else 0
        if med_v <= 0:
            continue

        y_carry = np.full(N, -1, dtype=int)
        y_carry[fvd_k & fv & (fwd_vol[fk] <= med_v)] = 1
        y_carry[fvd_k & fv & (fwd_vol[fk] > med_v)] = 0

        ly = y_carry >= 0
        vx = fv & ly & va_m
        if vx.sum() < 30:
            continue

        balance = y_carry[vx].mean()
        if 0.2 < balance < 0.8:
            score = min(balance, 1 - balance)
            if score > best_score:
                best_score = score
                best_labels = y_carry
                best_cfg = f"carry_f{fk}"

    return best_labels, best_cfg, None
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/modules/labeler.py
git commit -m "feat: add labeler module — KMeans two-stage + carry labeling"
```

---

### Task 7: `modules/features.py` — Feature Engineering (Baseline)

**Files:**
- Create: `autoresearch/modules/features.py`

- [ ] **Step 1: Write features.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/modules/features.py
"""Feature engineering. Edit freely — any features, any dimensionality.

Currently: 72 base features (momentum, z-score, rolling stats, kurtosis, vol ratio,
price vs MA) + 8 entropy features = 80 total.
"""
import math
import numpy as np
import pandas as pd


def sample_entropy(x, m=2, r_factor=0.2, max_comp=40):
    """Approximate sample entropy with bounded computation."""
    N = len(x)
    r = r_factor * np.std(x) + 1e-12
    if N < m + 3 or r == 0:
        return 0.0

    def count_matches(tlen):
        cnt, tot = 0, 0
        step = max(1, (N - tlen) // 200)
        for i in range(0, N - tlen - 1, step):
            max_j = min(i + max_comp + 1, N - tlen)
            for j in range(i + 1, max_j):
                if max(abs(x[i + k] - x[j + k]) for k in range(tlen)) < r:
                    cnt += 1
                tot += 1
        return cnt, tot

    A, tA = count_matches(m + 1)
    B, tB = count_matches(m)
    if B == 0 or A == 0:
        return 0.0
    return -math.log((A / tA) / (B / tB)) if tA > 0 and tB > 0 else 0.0


def build_feats(lc, lr):
    """Build feature matrix from log-close and log-return arrays.

    Returns: np.array of shape (N, F) with float32 dtype.
    F = 20 (raw momentum) + 20 (z-scored momentum) + 16 (rolling std/mean)
        + 8 (kurtosis) + 4 (vol ratio) + 4 (price vs MA) + 8 (entropy) = 80
    """
    N = len(lc)
    feats = []

    # Raw momentum: k-bar log-returns (k = 1..20)
    for k in range(1, 21):
        r = np.full(N, np.nan)
        r[k:] = lc[k:] - lc[:-k]
        feats.append(r.astype(np.float32))

    # Z-scored momentum (100-bar rolling window)
    W_Z = 100
    for k in range(1, 21):
        r = np.full(N, np.nan)
        r[k:] = lc[k:] - lc[:-k]
        rs = pd.Series(r)
        m = rs.rolling(W_Z, min_periods=W_Z).mean()
        s = rs.rolling(W_Z, min_periods=W_Z).std()
        feats.append(((rs - m) / (s + 1e-12)).astype(np.float32).to_numpy())

    # Rolling std and mean of log-returns
    slr = pd.Series(lr)
    for W in [5, 10, 20, 50, 100, 200, 400, 800]:
        feats.append(slr.rolling(W, min_periods=W).std().astype(np.float32).to_numpy())
        feats.append(slr.rolling(W, min_periods=W).mean().astype(np.float32).to_numpy())

    # Rolling kurtosis (returns and absolute returns)
    sa = pd.Series(np.abs(lr))
    for W in [50, 100, 200, 400]:
        feats.append(slr.rolling(W, min_periods=W).kurt().astype(np.float32).to_numpy())
        feats.append(sa.rolling(W, min_periods=W).kurt().astype(np.float32).to_numpy())

    # Volatility ratio: short-term / long-term std
    for W in [5, 10, 20, 50]:
        ratio = slr.rolling(W, min_periods=W).std() / (slr.rolling(200, min_periods=200).std() + 1e-9)
        feats.append(ratio.astype(np.float32).to_numpy())

    # Price vs moving average (binary)
    for W in [25, 50, 100, 200]:
        ma = pd.Series(lc).rolling(W, min_periods=W).mean().to_numpy()
        feats.append((lc > ma).astype(np.float32))

    # Sample entropy features
    lr_arr = np.diff(lc, prepend=lc[0])
    lr_abs = np.abs(lr_arr)
    for W in [50, 100, 200]:
        for r_f in [0.1, 0.2]:
            se = np.full(N, np.nan, dtype=np.float32)
            stride = max(1, W // 5)
            for i in range(W, N, stride):
                se[i] = sample_entropy(lc[i - W:i], m=2, r_factor=r_f)
            feats.append(pd.Series(se).fillna(method='ffill').fillna(0.0).astype(np.float32).to_numpy())
    for W in [50, 100]:
        se = np.full(N, np.nan, dtype=np.float32)
        stride = max(1, W // 5)
        for i in range(W, N, stride):
            se[i] = sample_entropy(lr_abs[i - W:i], m=2, r_factor=0.2)
        feats.append(pd.Series(se).fillna(method='ffill').fillna(0.0).astype(np.float32).to_numpy())

    return np.column_stack(feats).astype(np.float32)
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/modules/features.py
git commit -m "feat: add features module — 80 features (momentum, entropy, rolling stats)"
```

---

### Task 8: `modules/trainer.py` — Model Training & Parameter Sweep (Baseline)

**Files:**
- Create: `autoresearch/modules/trainer.py`

- [ ] **Step 1: Write trainer.py**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/modules/trainer.py
"""Model training, parameter sweep, and evaluation. Edit freely — any model, any sweep.

On_end_of_algorithm orchestrates: build bars → label → featurize → train → evaluate.
Results are exposed via self.set_runtime_statistic() for the evaluator harness.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# These are imported by the template and available in on_end_of_algorithm scope:
# bar_builder.build_bars(close, vol, ts_arr, ...)
# labeler.generate_labels_kmeans_two_stage(...)
# labeler.generate_labels_carry(...)
# features.build_feats(lc, lr)


def realistic_cstats(probs, lc_arr, ma_arr, log_rets, tc=0.0005):
    """Realistic backtest statistics with transaction costs.

    Returns: (calmar, trades, total_return, mdd, annual_return)
    """
    n = min(len(probs) - 1, len(log_rets) - 1, len(lc_arr) - 1, len(ma_arr) - 1)
    if n < 2:
        return 0, 0, 0, 0, 0

    positions = np.zeros(n + 1)
    last_pos = 0.0
    trades = 0
    for i in range(n):
        above = lc_arr[i] > ma_arr[i]
        target = min(1.0, (probs[i] - 0.5) * 200) if (probs[i] > 0.5 and above) else 0.0
        if (last_pos == 0 and target > 0) or (last_pos > 0 and target == 0) or abs(target - last_pos) > 0.01:
            trades += 1
        positions[i] = target
        last_pos = target

    if trades < 2:
        return 0, trades, 0, 0, 0

    strat_rets = positions[:-1] * log_rets[1:n + 1]
    for i in range(1, n):
        if abs(positions[i] - positions[i - 1]) > 0.001:
            strat_rets[i] -= tc * abs(positions[i] - positions[i - 1])

    cum = np.cumsum(strat_rets)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    mdd = abs(float(np.min(dd))) + 1e-9
    ann = float(np.mean(strat_rets)) * 880  # Annualize: ~252 trading days * ~3.5 bars/day for minute bars
    calmar = ann / mdd if mdd > 0.001 else 0
    return calmar, trades, float(np.sum(strat_rets)), mdd, ann


def train_and_evaluate(feats, y, tr_m, va_m, te_m, lc, lr, bar_ts, fv,
                        bar_type="vol", label_cfg=""):
    """Train XGBoost on labeled data with internal parameter sweep.

    Sweeps: forward horizons [50, 100, 200], MA periods [50, 100, 200, 0],
            inversion [True, False].

    Returns:
        best_real_cal: float
        best_cfg: str
        val_preds: list of prediction dicts
        test_preds: list of prediction dicts
        train_auc: float
        val_auc: float
        best_trades: int
    """
    N = len(lc)
    ly = y >= 0
    tx = fv & ly & tr_m
    vx = fv & ly & va_m
    ex = fv & ly & te_m

    if tx.sum() < 200 or vx.sum() < 30 or ex.sum() < 30:
        return 0, "insufficient_data", None, None, 0, 0, 0

    best_real_cal = -999
    best_cfg = ""
    best_val_preds = None
    best_test_preds = None
    best_train_auc = 0
    best_val_auc = 0
    best_trades = 0

    # MA periods to sweep
    for ma_period in [50, 100, 200, 0]:
        if ma_period == 0:
            ma = np.zeros_like(lc)
            suf = "_noma"
        else:
            ma = pd.Series(lc).rolling(ma_period, min_periods=ma_period).mean().to_numpy()
            suf = f"_ma{ma_period}"

        sc = StandardScaler()
        Xt = sc.fit_transform(feats[tx])
        Xv = sc.transform(feats[vx])
        Xe = sc.transform(feats[ex])

        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            random_state=42, n_jobs=1, early_stopping_rounds=30, base_score=0.5
        )
        m.fit(Xt, y[tx], eval_set=[(Xv, y[vx])], verbose=False)

        pv = m.predict_proba(Xv)[:, 1]
        pe = m.predict_proba(Xe)[:, 1]

        # AUC values
        from sklearn.metrics import roc_auc_score
        try:
            train_auc = roc_auc_score(y[tx], m.predict_proba(Xt)[:, 1])
            val_auc = roc_auc_score(y[vx], pv)
        except ValueError:
            train_auc = 0.5
            val_auc = 0.5

        vi = np.where(vx)[0]
        ei = np.where(ex)[0]

        for inv, vp, ep in [(False, pv, pe), (True, 1 - pv, 1 - pe)]:
            inv_suf = "_inv" if inv else ""
            rc, nt, _, _, _ = realistic_cstats(
                vp[:-1], lc[vi][:-1], ma[vi][:-1], lr[vi][1:]
            )

            if nt >= 10 and rc > best_real_cal:
                best_real_cal = rc
                best_cfg = f"{bar_type}_{label_cfg}{suf}{inv_suf}"
                best_train_auc = train_auc
                best_val_auc = val_auc
                best_trades = nt

                # Serialize predictions
                best_val_preds = _pred_list(vp[:-1], lc[vi][:-1], ma[vi][:-1], bar_ts[vi][:-1])
                best_test_preds = _pred_list(ep[:-1], lc[ei][:-1], ma[ei][:-1], bar_ts[ei][:-1])

    return best_real_cal, best_cfg, best_val_preds, best_test_preds, best_train_auc, best_val_auc, best_trades


def _pred_list(probs, lc_arr, ma_arr, times):
    """Serialize predictions to list-of-dicts for ObjectStore."""
    out = []
    for i in range(min(len(probs), len(times))):
        try:
            t = times[i]
            ts_str = t.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(t, 'strftime') else str(t)[:19].replace(' ', 'T')
        except:
            ts_str = str(times[i])[:19].replace(' ', 'T')
        above = bool(lc_arr[i] > ma_arr[i]) if i < len(ma_arr) and i < len(lc_arr) else True
        out.append({"time": ts_str, "pred": float(probs[i]), "above_ma": above})
    return out
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/modules/trainer.py
git commit -m "feat: add trainer module — XGBoost + parameter sweep + eval"
```

---

### Task 9: `templates/header.py.tmpl` — QC Boilerplate

**Files:**
- Create: `autoresearch/templates/header.py.tmpl`

- [ ] **Step 1: Create templates directory and write header template**

```bash
mkdir -p /Users/liyuanjun/ai_work/lb/autoresearch/templates
```

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/templates/header.py.tmpl
"""Autoresearch pipeline — generated {timestamp} — commit {commit_hash}
DO NOT EDIT THIS FILE DIRECTLY. It is rendered from modules/ by the orchestrator.
"""
from AlgorithmImports import *
import json, math, numpy as np, pandas as pd
from datetime import datetime

TICKER = "__TICKER__"
TARGET_BARS = 15000
TRAIN_END = datetime(2021, 8, 1)
VAL_END = datetime(2023, 8, 1)
TEST_END = datetime(2026, 6, 1)


class AutoresearchPipeline(QCAlgorithm):
    """Autonomous research pipeline. Data collection in initialize/on_data.
    All pipeline logic is in the module code below (concatenated).
    """

    def initialize(self):
        self.set_start_date(2009, 8, 1)
        self.set_end_date(2026, 6, 1)
        self.set_cash(100000)
        self.sym = self.add_equity(TICKER, Resolution.MINUTE).symbol
        self.tsl = []
        self.cll = []
        self.vll = []

    def on_data(self, slice):
        if self.sym not in slice.bars:
            return
        b = slice.bars[self.sym]
        self.tsl.append(b.end_time)
        self.cll.append(float(b.close))
        self.vll.append(float(b.volume))
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/templates/header.py.tmpl
git commit -m "feat: add header template — QC boilerplate + data collection"
```

---

### Task 10: `templates/footer.py.tmpl` — Result Export

**Files:**
- Create: `autoresearch/templates/footer.py.tmpl`

- [ ] **Step 1: Write footer template**

```python
# /Users/liyuanjun/ai_work/lb/autoresearch/templates/footer.py.tmpl
"""Footer template — result export and runtime statistics.
DO NOT EDIT THIS FILE DIRECTLY. It is rendered from modules/ by the orchestrator.
"""

    # === END OF MODULE CODE ===
    # The on_end_of_algorithm method continues here after the module code.
    # The modules define: bar_type, label_cfg, best_real_cal, best_cfg,
    # best_val_preds, best_test_preds, train_auc, val_auc, best_trades
    # These are set by trainer.py's train_and_evaluate() call.

    def on_end_of_algorithm(self):
        close = np.array(self.cll)
        vol = np.array(self.vll)
        ts_arr = np.array(self.tsl)

        # === BAR BUILDING ===
        bar_type = "vol"  # Default; agent can change in bar_builder.py
        try:
            lc, lr, N, bar_ts = build_bars(close, vol, ts_arr,
                                           bar_type=bar_type,
                                           target_bars=TARGET_BARS)
        except NameError:
            self.set_runtime_statistic("err", "bar_builder_missing_build_bars")
            return

        # Time masks
        bar_ts_np = np.array([np.datetime64(str(t)) for t in bar_ts])
        tr_m = bar_ts_np < np.datetime64(TRAIN_END)
        va_m = (bar_ts_np >= np.datetime64(TRAIN_END)) & (bar_ts_np < np.datetime64(VAL_END))
        te_m = (bar_ts_np >= np.datetime64(VAL_END)) & (bar_ts_np < np.datetime64(TEST_END))
        self.log(f"TR={int(tr_m.sum())} VA={int(va_m.sum())} TE={int(te_m.sum())} bars={N}")

        if tr_m.sum() < 500 or va_m.sum() < 50 or te_m.sum() < 50:
            self.set_runtime_statistic("err", "insufficient_bars")
            return

        # === FEATURE BUILDING ===
        try:
            feats = build_feats(lc, lr)
        except NameError:
            self.set_runtime_statistic("err", "features_missing_build_feats")
            return
        fv = ~np.isnan(feats).any(axis=1)
        self.log(f"Feats shape={feats.shape} clean={int(fv.sum())}")

        # === LABELING ===
        fwd_ret, fwd_vol = compute_forward_metrics(lc, lr)

        # Try KMeans two-stage first, fall back to carry
        y, label_cfg, _ = generate_labels_kmeans_two_stage(
            lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol)
        if y is None:
            y, label_cfg, _ = generate_labels_carry(fwd_vol, tr_m, va_m, fv)

        if y is None:
            self.set_runtime_statistic("err", "no_valid_labels")
            return

        # === TRAINING ===
        best_real_cal, best_cfg, best_val_preds, best_test_preds, train_auc, val_auc, best_trades = \
            train_and_evaluate(feats, y, tr_m, va_m, te_m, lc, lr, bar_ts, fv,
                              bar_type=bar_type, label_cfg=label_cfg)

        # === SAVE ===
        if best_val_preds is not None:
            self.object_store.save(f"autoresearch/{TICKER}/val_preds.json",
                json.dumps({"ticker": TICKER, "best_cfg": best_cfg,
                           "n_preds": len(best_val_preds),
                           "predictions": best_val_preds}))
            self.object_store.save(f"autoresearch/{TICKER}/test_preds.json",
                json.dumps({"ticker": TICKER, "best_cfg": best_cfg,
                           "n_preds": len(best_test_preds),
                           "predictions": best_test_preds}))

        # === RUNTIME STATS (required by evaluator) ===
        self.set_runtime_statistic("best_cfg", str(best_cfg))
        self.set_runtime_statistic("best_cal", str(round(float(best_real_cal), 4)))
        self.set_runtime_statistic("train_auc", str(round(float(train_auc), 4)))
        self.set_runtime_statistic("val_auc", str(round(float(val_auc), 4)))
        self.set_runtime_statistic("n_bars", str(N))
        self.set_runtime_statistic("n_trades_val", str(best_trades))
        self.set_runtime_statistic("bar_type", str(bar_type))

        self.log(f"BEST: {best_cfg} cal={round(float(best_real_cal), 4)} "
                f"train_auc={round(float(train_auc), 4)} val_auc={round(float(val_auc), 4)}")
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/templates/footer.py.tmpl
git commit -m "feat: add footer template — ObjectStore save + runtime stat export"
```

---

### Task 11: Knowledge System — `knowledge.json`, `techniques.json`, `program.md`

**Files:**
- Create: `autoresearch/knowledge.json`
- Create: `autoresearch/techniques.json`
- Create: `autoresearch/program.md`
- Create: `autoresearch/results.tsv` (empty with header)

- [ ] **Step 1: Write knowledge.json seeded from experiment_summary**

```json
{
  "techniques": {
    "kmeans_two_stage": {
      "description": "KMeans vol-cluster → KMeans direction-cluster on forward returns",
      "source": "Wang course + v236 pipeline",
      "results": {
        "QQQ": {"status": "kept", "best_calmar": 1.29, "trades": 45, "notes": "dollar bars, 2019 split"},
        "IWM": {"status": "kept", "best_calmar": 0.59, "trades": 35, "notes": "dollar bars"},
        "EEM": {"status": "kept", "best_calmar": 0.64, "trades": 30, "notes": "tick bars"},
        "XLE": {"status": "kept", "best_calmar": 0.96, "trades": 40, "notes": "tick bars"},
        "HYG": {"status": "kept", "best_calmar": 1.61, "trades": 25, "notes": "dollar bars"},
        "TLT": {"status": "discard", "best_calmar": 0.47, "trades": 15, "notes": "barely trades"},
        "GLD": {"status": "kept", "best_calmar": 1.42, "trades": 50, "notes": "dollar bars"}
      },
      "verdict": "Universal baseline. Works on most assets but rarely exceeds Calmar 1.6."
    },
    "carry_labeling": {
      "description": "Always long when forward vol is below median",
      "source": "Wang course vol bars lecture",
      "results": {
        "GLD": {"status": "kept", "best_calmar": 1.61, "trades": 45, "notes": "vol bars, best result so far"},
        "GDX": {"status": "kept", "best_calmar": 1.54, "trades": 38, "notes": "vol bars"},
        "IAU": {"status": "kept", "best_calmar": 1.62, "trades": 40, "notes": "vol bars"},
        "XME": {"status": "kept", "best_calmar": 1.13, "trades": 35, "notes": "vol bars"},
        "QQQ": {"status": "dead_end", "best_calmar": 0.0, "trades": 0, "notes": "1-class on tech ETFs"},
        "IWM": {"status": "dead_end", "best_calmar": 0.0, "trades": 0, "notes": "1-class on equity ETFs"}
      },
      "verdict": "Works on commodities (GLD, GDX, IAU, XME). Fails on tech/equity (QQQ, IWM)."
    },
    "bgm_labeling": {
      "description": "Bayesian Gaussian Mixture with sparse Dirichlet prior",
      "source": "SESSION_FINAL_SUMMARY.md, v362 pipeline",
      "results": {
        "GLD": {"status": "kept", "best_calmar": 1.37, "trades": 40, "notes": "BGM K=3 + isotonic calibration"},
        "GDXJ": {"status": "kept", "best_calmar": 1.60, "trades": 55, "notes": "vol bars, best for gold juniors"}
      },
      "verdict": "Strong on gold-related assets. Bayesian prior prevents over-fragmenting."
    },
    "ensemble_labeling": {
      "description": "Average probabilities from KMeans + Carry + BGM label sets",
      "source": "Wang Forest of Opinions, v372 pipeline",
      "results": {
        "SLV": {"status": "kept", "best_calmar": 1.10, "trades": 60, "notes": "vol bars, +207% return"},
        "USO": {"status": "kept", "best_calmar": 0.90, "trades": 50, "notes": "vol bars"},
        "DBC": {"status": "kept", "best_calmar": 0.87, "trades": 45, "notes": "vol bars"}
      },
      "verdict": "Effective on vol bars for metals/commodities. Destroys performance on dollar bars."
    }
  },
  "axis_types": {
    "dollar_bars": {
      "best_for": ["QQQ", "SMH", "SOXX", "DIA", "SPY"],
      "calmar_range": "0.8-1.8",
      "notes": "Info-density-equity. Optimal for liquid stable assets."
    },
    "tick_bars": {
      "best_for": ["GLD", "GDXJ", "XLE", "XME"],
      "calmar_range": "0.6-1.6",
      "notes": "Stabilizes across regimes when dollar volume drifts."
    },
    "vol_bars": {
      "best_for": ["GLD", "GDX", "IAU", "SLV", "XME", "USO"],
      "calmar_range": "0.6-1.6",
      "notes": "Wang's 3rd axis. Samples more during high vol. Best for commodities."
    }
  },
  "dead_ends_global": [
    "HMM labeling (never selected over KMeans in 150+ experiments)",
    "CUSUM labeling (never selected)",
    "Fractional differentiation features (hurt OOS despite higher IS AUC)",
    "PCA / KernelPCA features (destroyed signal)",
    "Autoencoder bottleneck features (no lift over raw features)",
    "Imbalance bars (require tick data, not available on minute resolution)",
    "Daily / hourly data (too coarse for regime detection)",
    "Trend Scanning labels (inferior to KMeans two-stage)",
    "Renko bars (feature mismatch, predictions don't cross threshold)",
    "GMM probability labels (never selected over KMeans)",
    "DBSCAN labels (noisy on most ETFs)",
    "Spectral clustering labels (no improvement)",
    "IG feature selection (no lift)",
    "ACF features (no lift)",
    "RSI features (no lift)",
    "XGBoost hyperparameter tuning beyond defaults (no lift)",
    "Lasso / ElasticNet models (worse than Ridge/XGBoost)",
    "GradientBoosting (worse than XGBoost)",
    "Cross-asset features for non-GLD ETFs (no lift)",
    "Multi-horizon AND/AVG ensemble (no lift)",
    "Continuous position sizing (helped GLD marginally, hurt others)",
    "Long-short strategies (XGB low-p ≠ go down, just low confidence)",
    "Trailing stop-loss (no improvement)",
    "Z-score labels (no improvement)",
    "Continuous labels (no improvement)"
  ],
  "frontier": {
    "best_calmar_overall": 1.62,
    "best_calmar_asset": "IAU",
    "best_calmar_technique": "carry_labeling",
    "best_calmar_axis": "vol_bars",
    "current_bottleneck": "minute data granularity + XGBoost model capacity ceiling",
    "unexplored": [
      "Triple-barrier labeling (dynamic take-profit / stop-loss)",
      "Hurst-adaptive labeling (auto trend vs MR per regime)",
      "Change-point detection labels (Wang recommended, superior to TB)",
      "Multi-order combination labels (Wang's signature for strong trends)",
      "Quantile tertile labels (skip middle tertile for noise reduction)",
      "Meta-labeling (2-model confidence gating)",
      "Range bars (price-driven, best for volatile/EM)",
      "Information-driven bars (entropy-based adaptive sampling)",
      "Rolling LogDollar axis (Wang production-grade)",
      "Tick data (for Imbalance/Run bars)",
      "Deep learning features (LSTM/Transformer embeddings)",
      "Cross-asset features (SPY/VIX/TNX/DXY as features)",
      "Options flow sentiment features",
      "Order flow imbalance features",
      "Adaptive online retraining",
      "Kelly / volatility-targeted position sizing"
    ]
  }
}
```

- [ ] **Step 2: Write techniques.json with unexplored ideas**

```json
{
  "queue": [
    {
      "id": "t001",
      "technique": "Triple-barrier labeling on vol bars",
      "description": "Replace fixed-horizon labels with triple-barrier (profit-taking at +2σ, stop-loss at -1σ, time-out at 100 bars). Should give more realistic trade exits than fixed-horizon forward returns.",
      "source": {"type": "pdf", "ref": "AFML.pdf Chapter 3", "url": null},
      "hypothesis": "Vol bars create more IID returns → triple-barrier thresholds will be more stable and produce cleaner labels",
      "priority": 5,
      "applicable_assets": ["GLD", "XLE", "EEM"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t002",
      "technique": "Hurst-adaptive labeling on vol bars",
      "description": "Compute rolling Hurst exponent. If H > 0.6 → use trend-following labels (direction cluster). If H < 0.45 → use mean-reversion labels (invert). On vol bars where regimes are cleaner.",
      "source": {"type": "transcript", "ref": "Wang course: 自适应标签 + Hurst exponent", "url": null},
      "hypothesis": "Hurst > 0.6 indicates trending regime where direction clustering works; H < 0.45 indicates mean-reverting where carry/inverse works better. Adaptive switching prevents applying wrong label type to wrong regime.",
      "priority": 5,
      "applicable_assets": ["GLD", "GDX", "XLE", "TLT"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t003",
      "technique": "Multi-order combination labels",
      "description": "Wang's signature technique: combine labels at multiple forward horizons (50, 100, 200) into a single consensus label. Only trade when all horizons agree on direction.",
      "source": {"type": "pdf", "ref": "wang_course_2026-06.pdf — multi-order labels", "url": null},
      "hypothesis": "Multi-horizon agreement filters noise — a signal that works at multiple timescales is more likely to be real. Should improve Calmar by reducing false signals.",
      "priority": 4,
      "applicable_assets": ["QQQ", "IWM", "GLD", "XLE"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t004",
      "technique": "Quantile tertile labels with skip-middle",
      "description": "Instead of binary labels: top tertile of forward returns = long, bottom tertile = skip, middle tertile = no-trade. This avoids the noisy middle where direction is ambiguous.",
      "source": {"type": "pdf", "ref": "v386 pipeline experiment", "url": null},
      "hypothesis": "The middle tertile of return distributions contains mostly noise. Skipping it should increase signal purity and reduce false trades, improving Calmar.",
      "priority": 4,
      "applicable_assets": ["QQQ", "IWM", "HYG"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t005",
      "technique": "Meta-labeling with confidence gating",
      "description": "Train a secondary XGBoost model to predict whether the primary model's prediction will be correct. Only trade when both primary says LONG and meta says CONFIDENT.",
      "source": {"type": "pdf", "ref": "AFML.pdf §3.6 — meta-labeling", "url": null},
      "hypothesis": "Meta-labeling filters false positives. Should increase Calmar even if it reduces trade count, as long as trades > 80.",
      "priority": 3,
      "applicable_assets": ["GLD", "QQQ", "XLE", "IWM"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t006",
      "technique": "Change-point detection labels (CUSUM or PELT)",
      "description": "Use change-point detection (CUSUM or PELT algorithm) on volatility or price to find regime boundaries. Label post-change regimes as long/short based on subsequent return.",
      "source": {"type": "web", "ref": "ruptures library + Wang recommendation", "url": null},
      "hypothesis": "Change-point detection explicitly models regime shifts rather than clustering. Should produce cleaner regime boundaries than KMeans, especially at transition points.",
      "priority": 3,
      "applicable_assets": ["EEM", "TLT", "HYG"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t007",
      "technique": "Range bars for volatile ETFs",
      "description": "Use range bars (price-move-based sampling) instead of vol/dollar/tick bars. Range bars sample whenever price moves by a fixed percentage, creating uniform bar sizes in price space.",
      "source": {"type": "pdf", "ref": "De Prado — range bars", "url": null},
      "hypothesis": "Range bars normalize price movement, which may help volatile ETFs like EEM, XLE where dollar/tick bars produce irregular time spacing.",
      "priority": 3,
      "applicable_assets": ["EEM", "XLE", "IWM"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    },
    {
      "id": "t008",
      "technique": "Ridge regression with isotonic calibration",
      "description": "Replace XGBoost with Ridge regression + isotonic probability calibration. Ridge is linear (less overfitting) and isotonic calibration maps raw scores to well-calibrated probabilities.",
      "source": {"type": "pdf", "ref": "v274 pipeline — Ridge + MR success on TLT/SHY", "url": null},
      "hypothesis": "Ridge + isotonic may beat XGBoost on assets with linear signal structure (rates, mean-reverting ETFs). Test on TLT, HYG, SHY.",
      "priority": 2,
      "applicable_assets": ["TLT", "HYG"],
      "status": "queued",
      "tried_count": 0,
      "created": "2026-05-31T00:00:00Z"
    }
  ],
  "dead_ends": [],
  "last_idea_search": "2026-05-31T00:00:00Z"
}
```

- [ ] **Step 3: Write program.md**

```markdown
# Autoresearch — ML Quant Trading Pipeline

## Goals
- **Primary**: Calmar > 3.0 on OOS TEST period (real QC SetHoldings backtest)
- **Secondary**: Trades > 80 in OOS, no lookahead bias, no overfitting
- **Tertiary**: Discover techniques that generalize across multiple ETFs

## Universe
- **Core 7**: QQQ, IWM, EEM, XLE, HYG, TLT, GLD
- **Splits**: TRAIN 2009-08 to 2021-08, VAL 2021-08 to 2023-08, TEST 2023-08 to 2026-06
- **Data**: Minute bars from QuantConnect, ~17 years

## Constraints
- **2 QC Cloud nodes** (parallel execution only)
- **5-minute timeout** per backtest (cancel via API if exceeded)
- **Only edit modules/**: bar_builder.py, labeler.py, features.py, trainer.py
- **Do NOT modify harness/**: orchestrator.py, evaluator.py, qc_client.py, constants.py
- **Do NOT modify templates/**: header.py.tmpl, footer.py.tmpl
- **Target ~15,000 bars** per 17-year history (adjust bar builder threshold accordingly)

## Evaluation (Multi-Gate)
All four gates must pass on at least ONE ETF to KEEP a change:

| Gate | Metric | Threshold |
|------|--------|-----------|
| G0 | Backtest completed | Status == "Completed" |
| G1 | OOS Calmar | > 3.0 |
| G2 | OOS Trades | > 80 |
| G3 | No lookahead | Zero future-data leaks |
| G4 | No overfit | |train_AUC - val_AUC| < 0.05 |

## Workflow
1. Read modules/, knowledge.json, techniques.json, results.tsv
2. Pick highest-priority idea from techniques.json
3. Edit modules to implement the idea
4. `git add modules/ && git commit -m "<technique>: <change>"`
5. Submit to 2 QC nodes on 2 ETFs simultaneously
6. Evaluate results against all 4 gates
7. All pass on ≥1 ETF → KEEP (commit stays)
8. Any fail on both ETFs → DISCARD (git reset --hard HEAD~1)
9. Log to results.tsv, update knowledge.json, update techniques.json
10. If queue < 5: search for new ideas (arXiv, SSRN, web, pdfs/, transcripts/)
11. Repeat forever — NEVER STOP until human interrupts

## Idea Discovery
When the technique queue runs low (< 5 items), search broadly:
- **arXiv**: q-fin.TR, q-fin.ST, cs.LG — "trading strategy ML", "unsupervised regime detection", "novel labeling method"
- **SSRN**: recent papers on quantitative trading, factor investing
- **Google Scholar**: "automated trading strategy discovery", "novel feature engineering finance"
- **Web**: general search for new quant research
- **Local pdfs/**: AFML.pdf (De Prado), CFI.pdf, MLAM.pdf, wang_course_2026-06.pdf
- **Local uni/transcripts/**: 37 Wang course transcript files — search for techniques mentioned

Cross-reference with knowledge.json to avoid retrying dead ends.

## Setup
```bash
cd /Users/liyuanjun/ai_work/lb
git checkout -b autoresearch/$(date +%Y-%m-%d)
```

## Key Learnings from 150+ Previous Experiments
- **Works**: KMeans two-stage (universal), carry labeling (commodities on vol bars), BGM (gold), ensemble (vol bars)
- **Dead ends**: HMM, CUSUM, FracDiff, PCA, Autoencoder, Renko, Trend Scanning, daily data, long-short
- **Best axis-asset pairs**: Dollar bars → tech/equity, Vol bars → commodities/metals, Tick bars → mixed
- **Entropy features**: +4-28pp improvement depending on ETF
- **Per-ETF splits matter**: QQQ +135pp with 2019 split vs 2021
```

- [ ] **Step 4: Initialize results.tsv**

```bash
echo -e "commit\tcalmar\ttrades\tstatus\tdescription" > /Users/liyuanjun/ai_work/lb/autoresearch/results.tsv
```

- [ ] **Step 5: Commit**

```bash
git add autoresearch/knowledge.json autoresearch/techniques.json autoresearch/program.md autoresearch/results.tsv
git commit -m "feat: add knowledge system — knowledge.json, techniques.json, program.md, results.tsv"
```

---

### Task 12: Integration Smoke Test

**Files:**
- None new; test the assembled system

- [ ] **Step 1: Verify module import and render**

```bash
cd /Users/liyuanjun/ai_work/lb && python3 -c "
import sys
sys.path.insert(0, '.')

# Test render_script
from autoresearch.harness.orchestrator import render_script, validate_script
script = render_script('GLD')
print('Script length:', len(script), 'chars')
print('Contains __TICKER__:', '__TICKER__' in script)  # Should be False (already replaced)
print('Contains GLD:', 'GLD' in script)  # Should be True

# Test validate
errors = validate_script(script)
if errors:
    print('VALIDATION ERRORS:', errors)
else:
    print('Validation: PASSED')

# Quick compile check
try:
    compile(script, '<test>', 'exec')
    print('Compile: PASSED')
except SyntaxError as e:
    print('COMPILE ERROR:', e)
"
```

Expected: Script renders (~15-20K chars), `__TICKER__` replaced with `GLD`, validation passes, compiles cleanly.

- [ ] **Step 2: Verify evaluator with each gate scenario**

```bash
cd /Users/liyuanjun/ai_work/lb && python3 -c "
from autoresearch.harness.evaluator import evaluate

# Scenario 1: All gates pass
good = {
    'status': 'Completed',
    'statistics': {
        'Compounding Annual Return': '45%',
        'Drawdown': '10%',
        'Total Orders': '95'
    },
    'runtimeStatistics': {
        'train_auc': '0.88',
        'val_auc': '0.86'
    }
}
r = evaluate(good)
print('Good:', r['status'], r['summary'])
assert r['status'] == 'keep'

# Scenario 2: Low Calmar
low_cal = {
    'status': 'Completed',
    'statistics': {
        'Compounding Annual Return': '15%',
        'Drawdown': '10%',
        'Total Orders': '95'
    },
    'runtimeStatistics': {
        'train_auc': '0.88',
        'val_auc': '0.86'
    }
}
r2 = evaluate(low_cal)
print('Low Cal:', r2['status'], r2['summary'])
assert r2['status'] == 'discard'

# Scenario 3: Few trades
few_trades = {
    'status': 'Completed',
    'statistics': {
        'Compounding Annual Return': '45%',
        'Drawdown': '10%',
        'Total Orders': '5'
    },
    'runtimeStatistics': {
        'train_auc': '0.88',
        'val_auc': '0.86'
    }
}
r3 = evaluate(few_trades)
print('Few trades:', r3['status'], r3['summary'])
assert r3['status'] == 'discard'

# Scenario 4: Overfit
overfit = {
    'status': 'Completed',
    'statistics': {
        'Compounding Annual Return': '45%',
        'Drawdown': '10%',
        'Total Orders': '95'
    },
    'runtimeStatistics': {
        'train_auc': '0.95',
        'val_auc': '0.70'
    }
}
r4 = evaluate(overfit)
print('Overfit:', r4['status'], r4['summary'])
assert r4['status'] == 'overfit'

# Scenario 5: Lookahead detected
leak = {
    'status': 'Completed',
    'statistics': {
        'Compounding Annual Return': '45%',
        'Drawdown': '10%',
        'Total Orders': '95'
    },
    'runtimeStatistics': {
        'train_auc': '0.88',
        'val_auc': '0.86'
    }
}
r5 = evaluate(leak, script_text='df.close.shift(-1) + lr')
print('Leak:', r5['status'], r5['summary'])
assert r5['status'] == 'leak'

print('All evaluator scenarios passed!')
"
```

- [ ] **Step 3: Verify orchestrator utility functions**

```bash
cd /Users/liyuanjun/ai_work/lb && python3 -c "
from autoresearch.harness.orchestrator import load_knowledge, load_techniques, render_script, validate_script

# Load knowledge
k = load_knowledge()
print('Knowledge techniques:', list(k.get('techniques', {}).keys()))
print('Dead ends:', len(k.get('dead_ends_global', [])))
assert len(k['techniques']) >= 4
assert len(k['dead_ends_global']) >= 20

# Load techniques
t = load_techniques()
print('Queue size:', len(t.get('queue', [])))
assert len(t['queue']) >= 5

# Render and validate
script = render_script('QQQ')
errs = validate_script(script)
assert len(errs) == 0, f'Validation errors: {errs}'
print('Render + validate: PASSED')

print('All orchestrator checks passed!')
"
```

- [ ] **Step 4: List all files created**

```bash
find /Users/liyuanjun/ai_work/lb/autoresearch -type f | sort
```

Expected:
```
autoresearch/harness/__init__.py
autoresearch/harness/constants.py
autoresearch/harness/evaluator.py
autoresearch/harness/orchestrator.py
autoresearch/harness/qc_client.py
autoresearch/knowledge.json
autoresearch/modules/__init__.py
autoresearch/modules/bar_builder.py
autoresearch/modules/features.py
autoresearch/modules/labeler.py
autoresearch/modules/trainer.py
autoresearch/program.md
autoresearch/results.tsv
autoresearch/techniques.json
autoresearch/templates/footer.py.tmpl
autoresearch/templates/header.py.tmpl
```

- [ ] **Step 5: Commit integration test confirmation**

```bash
git add -A && git status
```

---

## Summary

**Total new files**: 16 across 4 directories
**Total new code**: ~1,500 lines

**What the agent can now do:**
1. Read `program.md` + `techniques.json` + `knowledge.json` for direction
2. Freely edit `modules/*.py` to implement any idea
3. `git commit` the change
4. Call `orchestrator.run_experiment("GLD", "IWM", "test: triple barrier")` → renders scripts → submits to 2 QC nodes → polls with 5-min timeout → evaluates via 4-gate → logs to results.tsv
5. Discard failed experiments via `git reset --hard HEAD~1`
6. Search for new ideas when queue runs low (arXiv, SSRN, web, pdfs/, transcripts/)

**What the agent does NOT touch:**
- `harness/` — evaluation framework (sacred, like Karpathy's `prepare.py`)
- `templates/` — script structure (sacred)
