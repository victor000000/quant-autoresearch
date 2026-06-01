"""Loop orchestrator: two-phase train→infer with REAL Calmar from QC backtest."""

import os, sys, time, json, subprocess, ast
from datetime import datetime


def _minify(src):
    """Strip comments + docstrings so the rendered script fits QC's 64,000-char
    main.py limit. Semantics-preserving (AST round-trip); validated by compile."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)
                    and isinstance(b[0].value.value, str)):
                node.body = b[1:] or [ast.Pass()]
    out = ast.unparse(tree)
    compile(out, "<minified>", "exec")
    return out

from .constants import (CORE_7_ETFS, TEMPLATES_DIR, MODULES_DIR, QC_SCRIPTS_DIR, TARGET_BARS)
from .qc_client import submit_and_wait
from .evaluator import evaluate

PROJECT_ROOT = "/home/ubuntu/lb"
AUTORESEARCH_DIR = os.path.join(PROJECT_ROOT, "autoresearch")
RESULTS_TSV = os.path.join(AUTORESEARCH_DIR, "results.tsv")
KNOWLEDGE_JSON = os.path.join(AUTORESEARCH_DIR, "knowledge.json")
TECHNIQUES_JSON = os.path.join(AUTORESEARCH_DIR, "techniques.json")


def read_module(name):
    path = os.path.join(MODULES_DIR, name)
    if not os.path.exists(path): raise FileNotFoundError(f"Module not found: {path}")
    with open(path) as f: return f.read()


def render_script(ticker, axis=None):
    """Render ONE QC train script = header + modules (in order) + footer.

    The rendered modules are bar_builder.py, labeler.py, features.py, trainer.py
    (calibrator/consensus/ensembler stay out — the footer's FIXED downstream only
    needs reduce_dims/realistic_cstats from trainer.py). All share ONE global
    namespace, so modules must not import siblings.

    axis: if given, substitute the AXIS placeholder so the train run sweeps just
    that one axis; if None, the literal '__AXIS__' remains and the footer sweeps
    ALL axes.
    """
    header_path = os.path.join(TEMPLATES_DIR, "header.py.tmpl")
    footer_path = os.path.join(TEMPLATES_DIR, "footer.py.tmpl")
    with open(header_path) as f: script = f.read()
    for mod in ["bar_builder.py", "labeler.py", "features.py", "trainer.py"]:
        script += f"\n# === {mod} ===\n" + read_module(mod) + "\n"
    with open(footer_path) as f: script += f.read()
    script = script.replace("__TICKER__", ticker)
    if axis is not None:
        script = script.replace("__AXIS__", axis)
    return _minify(script)


def render_train_config(config):
    """Render ONE single-config TRAIN script for a hypothesis CONFIG.

    CONFIG = {"ticker","axis","labeler","thresh","sizing"} is injected at RENDER
    TIME by substituting the header's five placeholders (__TICKER__, __AXIS__,
    __LABELER__, __THRESH__, __SIZING__) — NO code edits between hypotheses. With
    axis+labeler both substituted the footer enters HYPOTHESIS MODE: it builds ONLY
    CONFIG['axis'] and runs ONLY CONFIG['labeler'] as ONE cell, sizing on VAL via
    _size(CONFIG['sizing'], CONFIG['thresh']) and SAVING that sizing+thresh into the
    cell payload so infer (OOS) replays the identical rule.

    The cell written is autoresearch/{ticker}/cell_{axis}_{labeler}.json; pass that
    same '{axis}_{labeler}' to render_infer_cell as the CELL key.
    """
    header_path = os.path.join(TEMPLATES_DIR, "header.py.tmpl")
    footer_path = os.path.join(TEMPLATES_DIR, "footer.py.tmpl")
    with open(header_path) as f: script = f.read()
    for mod in ["bar_builder.py", "labeler.py", "features.py", "trainer.py"]:
        script += f"\n# === {mod} ===\n" + read_module(mod) + "\n"
    with open(footer_path) as f: script += f.read()
    script = (script
              .replace("__TICKER__", str(config["ticker"]))
              .replace("__AXIS__", str(config["axis"]))
              .replace("__LABELER__", str(config["labeler"]))
              .replace("__THRESH__", repr(float(config["thresh"])))
              .replace("__SIZING__", str(config["sizing"])))
    return _minify(script)


def render_infer_cell(ticker, cell):
    """Render the INFER script bound to one cell key '{axis}_{labeler}'.

    Substitutes __TICKER__/__CELL__ only; the infer template reads sizing+thresh
    from the saved cell payload, so OOS execution matches the train/VAL leg.
    NOT minified (infer template is already compact and well under 64k)."""
    infer_path = os.path.join(TEMPLATES_DIR, "infer.py.tmpl")
    with open(infer_path) as f:
        code = f.read()
    return code.replace("__TICKER__", str(ticker)).replace("__CELL__", str(cell))


def validate_script(script_text):
    errors = []
    try: compile(script_text, '<pipeline>', 'exec')
    except SyntaxError as e: errors.append(f"SYNTAX: {e}")
    for key in ["best_cal", "train_auc", "val_auc"]:
        if f'set_runtime_statistic("{key}"' not in script_text:
            errors.append(f'MISSING_STAT: {key}')
    return errors


def run_experiment(ticker_a, ticker_b, description):
    """Two-phase: train → infer → REAL Calmar from infer's QC statistics."""
    commit = _get_commit_hash()
    scripts = {t: render_script(t) for t in [ticker_a, ticker_b]}
    errors = {t: validate_script(scripts[t]) for t in scripts}
    if any(errors.values()):
        return {"commit": commit, "description": description, "overall_status": "crash",
                "best_calmar": 0, "best_trades": 0,
                "results": {t: {"status": "crash", "error": str(e)} for t, e in errors.items()}}

    results = {}; best_real_calmar = 0; best_real_trades = 0
    for ticker in [ticker_a, ticker_b]:
        train_script = scripts[ticker]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TRAIN {ticker} ...")
        bt_train, status_train = submit_and_wait(train_script, f"train_{ticker}_{commit[:7]}", timeout_s=480)
        rt_train = bt_train.get("runtimeStatistics", {}) or {}
        synth_cal = float(rt_train.get("best_cal", 0))
        train_auc = float(rt_train.get("train_auc", 0))
        val_auc = float(rt_train.get("val_auc", 0))
        print(f"  synth_cal={synth_cal:.4f} train_auc={train_auc:.4f} val_auc={val_auc:.4f}")

        if status_train != "completed":
            results[ticker] = {"status": status_train, "real_calmar": 0, "real_trades": 0,
                               "synth_calmar": synth_cal, "train_auc": train_auc, "val_auc": val_auc}
            continue

        # Phase 2: Infer loads predictions from ObjectStore, executes REAL SetHoldings
        print(f"[{datetime.now().strftime('%H:%M:%S')}] INFER {ticker} ...")
        infer_path = os.path.join(TEMPLATES_DIR, "infer.py.tmpl")
        with open(infer_path) as f:
            infer_code = f.read().replace("__TICKER__", ticker)

        bt_infer, status_infer = submit_and_wait(infer_code, f"infer_{ticker}_{commit[:7]}", timeout_s=180)
        st = bt_infer.get("statistics", {}) or {}

        try:
            cagr = float(st.get("Compounding Annual Return", "0%").replace("%", ""))
            mdd = float(st.get("Drawdown", "0%").replace("%", ""))
            real_calmar = cagr / mdd if abs(mdd) > 0.01 else 0
            real_trades = int(st.get("Total Orders", "0"))
        except (ValueError, TypeError):
            real_calmar = 0; real_trades = 0

        print(f"  >>> REAL Calmar: {real_calmar:.4f} | REAL Trades: {real_trades} | Status: {status_infer}")
        results[ticker] = {"status": status_infer, "real_calmar": real_calmar, "real_trades": real_trades,
                           "synth_calmar": synth_cal, "train_auc": train_auc, "val_auc": val_auc}
        best_real_calmar = max(best_real_calmar, real_calmar)
        best_real_trades = max(best_real_trades, real_trades)

    overall = "keep" if any(r["real_calmar"] > 3.0 and r["real_trades"] > 80 for r in results.values()) else "discard"
    if any(r["status"] == "crash" for r in results.values()): overall = "crash"

    result = {"commit": commit, "description": description, "overall_status": overall,
              "best_calmar": best_real_calmar, "best_trades": best_real_trades, "results": results}
    _log_result(commit, best_real_calmar, best_real_trades, overall, description)
    return result


def _get_commit_hash():
    result = subprocess.run(["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
    return result.stdout.strip()


def git_keep(): pass


def git_discard():
    subprocess.run(["git", "-C", PROJECT_ROOT, "reset", "--hard", "HEAD~1"],
                   capture_output=True, text=True)


def _log_result(commit, calmar, trades, status, description):
    tsv_path = RESULTS_TSV
    with open(tsv_path, 'a') as f:
        if os.path.getsize(tsv_path) == 0:
            f.write("commit\tcalmar\ttrades\tstatus\tdescription\n")
        desc_clean = description.replace("\t", " ").replace("\n", " ")
        f.write(f"{commit}\t{calmar:.4f}\t{trades}\t{status}\t{desc_clean}\n")


def load_knowledge():
    if os.path.exists(KNOWLEDGE_JSON):
        with open(KNOWLEDGE_JSON) as f: return json.load(f)
    return {}


def save_knowledge(data):
    with open(KNOWLEDGE_JSON, 'w') as f: json.dump(data, f, indent=2, default=str)


def load_techniques():
    if os.path.exists(TECHNIQUES_JSON):
        with open(TECHNIQUES_JSON) as f: return json.load(f)
    return {"queue": [], "dead_ends": [], "last_idea_search": None}


def save_techniques(data):
    with open(TECHNIQUES_JSON, 'w') as f: json.dump(data, f, indent=2, default=str)
