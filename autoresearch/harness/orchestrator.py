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


def _prune_labelers(src, keep):
    """Keep only the ONE labeler a hypothesis uses (+ compute_forward_metrics + its
    transitive top-level helpers), shrinking labeler.py from ~10 labelers to 1 so the
    rendered script stays under QC's 64,000-char limit. Falls back to the full source
    on any error. Hypothesis mode uses LABELERS[keep] only, so this is semantics-safe."""
    try:
        tree = ast.parse(src)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        keeps = keep.split("+") if "+" in keep else [keep]    # "+"-joined = ENSEMBLE (⑦)
        keepfns = [f"generate_labels_{k}" for k in keeps if f"generate_labels_{k}" in funcs]
        if not keepfns:
            return src
        # transitive closure of top-level funcs reachable from the kept labelers + compute_forward_metrics
        need, stack = set(), list(keepfns) + ["compute_forward_metrics"]
        while stack:
            fn = stack.pop()
            if fn in need or fn not in funcs:
                continue
            need.add(fn)
            for sub in ast.walk(funcs[fn]):
                if isinstance(sub, ast.Name) and sub.id in funcs and sub.id not in need:
                    stack.append(sub.id)
        minimal_labelers = "{" + ", ".join(f'"{k}": generate_labels_{k}' for k in keeps
                                           if f"generate_labels_{k}" in funcs) + "}"
        new_body = []
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                new_body.append(n)
            elif isinstance(n, ast.FunctionDef):
                if n.name in need:
                    new_body.append(n)
            elif isinstance(n, ast.Assign):
                tgts = [t.id for t in n.targets if isinstance(t, ast.Name)]
                if "LABELERS" in tgts:
                    new_body.append(ast.parse(f'LABELERS = {minimal_labelers}').body[0])
                elif "FEATURED_LABELERS" in tgts or "BASELINE_LABELERS" in tgts:
                    continue
                else:
                    new_body.append(n)
            else:
                new_body.append(n)
        tree.body = new_body
        out = ast.unparse(tree)
        compile(out, "<pruned-labelers>", "exec")
        return out
    except Exception:
        return src


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

    MULTI-FILE (2026-06-03): bar_builder.py is NOT concatenated into main.py — it is a
    SEPARATE QC project file that main.py imports. QC's 64,000-char limit is PER FILE, so
    splitting the largest module off keeps main.py well under the cap and unblocks big
    labeler ENSEMBLES (bgm+ker etc.) that previously overflowed. bar_builder's only external
    dependency is TRAIN_END, which main injects after importing it (bar_builder.TRAIN_END=...).
    Bars are byte-identical to the old concatenated build (same source, just imported), and
    infer/verify renders still concatenate bar_builder (they're small) so the leak-safe online
    replay path is untouched. Returns (main_code, {"bar_builder.py": bar_builder_code}).
    """
    header_path = os.path.join(TEMPLATES_DIR, "header.py.tmpl")
    footer_path = os.path.join(TEMPLATES_DIR, "footer.py.tmpl")
    with open(header_path) as f: script = f.read()
    # bar_builder is a SEPARATE file: import it + inject TRAIN_END (defined above in the
    # header) into its module namespace, then pull the names the footer uses into scope.
    script += ("\nimport bar_builder as _bbmod\n_bbmod.TRAIN_END = TRAIN_END\n"
               "from bar_builder import AXES, BUILDER_CLASSES, build_bars, "
               "builder_threshold, _make_builder\n")
    for mod in ["labeler.py", "features.py", "trainer.py"]:
        body = read_module(mod)
        if mod == "labeler.py" and config.get("labeler"):
            body = _prune_labelers(body, str(config["labeler"]))   # keep only the 1 used labeler (size)
        script += f"\n# === {mod} ===\n" + body + "\n"
    with open(footer_path) as f: script += f.read()
    script = (script
              .replace("__TICKER__", str(config["ticker"]))
              .replace("__AXIS__", str(config["axis"]))
              .replace("__LABELER__", str(config["labeler"]))
              .replace("__THRESH__", repr(float(config["thresh"])))
              .replace("__SIZING__", str(config["sizing"]))
              .replace("__MAXDEPTH__", str(int(config.get("max_depth", 3))))
              .replace("__PERMUTE__", "1" if config.get("permute_labels") else "0"))
    # Separate bar_builder.py file: standalone (TRAIN_END default None -> injected by main).
    bb = "TRAIN_END = None\n" + read_module("bar_builder.py")
    return _minify(script), {"bar_builder.py": _minify(bb)}


def render_verify(ticker, axis):
    """Render the ONLINE-BAR VERIFY backtest for one (ticker, axis): bar_builder
    module + verify template, replaying the feed from origin and rebuilding the OOS
    bars online from the footer-saved frozen threshold. Reads oosbars_{axis}.json
    (saved by a prior train run on that ticker/axis)."""
    bb = read_module("bar_builder.py")
    with open(os.path.join(TEMPLATES_DIR, "verify.py.tmpl")) as f:
        vt = f.read()
    code = ("from AlgorithmImports import *\nimport json\nimport math\nimport numpy as np\nimport pandas as pd\n\n"
            "# === bar_builder.py ===\n" + bb + "\n\n# === verify ===\n" + vt)
    return code.replace("__TICKER__", str(ticker)).replace("__AXIS__", str(axis))


def render_infer_online(ticker, cell):
    """Render the FROZEN-MODEL ONLINE infer + cross-check for one cell: bar_builder +
    features modules + the infer_online template. Loads the model bundle, rebuilds
    bars+features+model ONLINE from origin, asserts p_live == p_saved."""
    bb = read_module("bar_builder.py")
    ff = read_module("features.py")
    with open(os.path.join(TEMPLATES_DIR, "infer_online.py.tmpl")) as f:
        it = f.read()
    code = ("from AlgorithmImports import *\nimport json\nimport math\nimport numpy as np\n"
            "import pandas as pd\nimport xgboost as xgb\n\n"
            "# === bar_builder.py ===\n" + bb + "\n\n# === features.py ===\n" + ff
            + "\n\n# === infer_online ===\n" + it)
    return code.replace("__TICKER__", str(ticker)).replace("__CELL__", str(cell))


def render_portfolio(champions, leverage=1.0):
    """Render the 7-champion PORTFOLIO replay (Wang's endpoint ⑨⑩). champions =
    [[ticker, objectstore_cell_key, weight], ...]; each is replayed via its saved
    predictions at its own thresh+sizing, weight normalised then scaled by `leverage`
    (gross exposure = leverage). Self-contained template."""
    with open(os.path.join(TEMPLATES_DIR, "portfolio.py.tmpl")) as f:
        code = f.read()
    return code.replace("__CHAMPIONS__", json.dumps(champions)).replace("__LEVERAGE__", repr(float(leverage)))


def render_benchmark(tickers):
    """Render the PASSIVE buy-and-hold equal-weight benchmark for the given ETFs (the beta
    the strategy portfolio must beat). Self-contained template."""
    with open(os.path.join(TEMPLATES_DIR, "benchmark.py.tmpl")) as f:
        code = f.read()
    return code.replace("__TICKERS__", json.dumps(list(tickers)))


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
