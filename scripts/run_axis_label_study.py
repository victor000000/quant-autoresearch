#!/usr/bin/env python3
"""Driver: SCIENTIFIC axis x labeling study on QuantConnect.

For one ticker and lists of axes + labelers, this:
  1. Submits ONE TRAIN backtest per axis (header AXIS substituted with the axis
     name). The train footer sweeps EVERY labeler in the registry for that axis
     and saves each (axis,label) cell's TEST predictions to the ObjectStore.
  2. Submits ONE INFER backtest per (axis,label) cell (infer CELL substituted
     with "{axis}_{label}"). Infer replays that cell's saved predictions with
     real SetHoldings and the threshold persisted in the cell payload.
  3. Reads the REAL Calmar (= CAGR / Drawdown) and Total Orders from each infer
     backtest's statistics, plus the train run's synthetic stats (best_cal /
     train_auc / val_auc), and writes a tidy CSV + appends to results.tsv.

QuantConnect allows only 2 backtest nodes, so this runs SERIALLY (one submit at a
time) — simple and safe. Lots of prints so progress is obvious.

USAGE (NOT executed by codegen — run manually):
    python3 scripts/run_axis_label_study.py [TICKER] [AXES_CSV] [LABELERS_CSV]
Examples:
    python3 scripts/run_axis_label_study.py
    python3 scripts/run_axis_label_study.py GLD vol,dollar,tick kmeans2stage,carry
    python3 scripts/run_axis_label_study.py QQQ vol bgm,hmm
"""

import os
import sys
import csv
from datetime import datetime

# --- make the harness package importable (autoresearch/ has no __init__.py) ---
PROJECT_ROOT = "/home/ubuntu/lb"
AUTORESEARCH_DIR = os.path.join(PROJECT_ROOT, "autoresearch")
if AUTORESEARCH_DIR not in sys.path:
    sys.path.insert(0, AUTORESEARCH_DIR)

from harness.qc_client import submit_and_wait          # noqa: E402
from harness.orchestrator import render_script          # noqa: E402

# Where infer.py.tmpl lives (rendered manually per cell — the orchestrator's
# render_script only renders the TRAIN script).
INFER_TMPL = os.path.join(AUTORESEARCH_DIR, "templates", "infer.py.tmpl")

RESULTS_CSV = os.path.join(AUTORESEARCH_DIR, "axis_label_results.csv")
RESULTS_TSV = os.path.join(AUTORESEARCH_DIR, "results.tsv")

CSV_COLS = ["ticker", "axis", "label", "real_calmar", "trades",
            "synth_cal", "train_auc", "val_auc", "status"]

# Small pilot defaults (overridable via argv).
DEFAULT_TICKER = "GLD"
DEFAULT_AXES = ["vol", "dollar", "tick"]
DEFAULT_LABELERS = ["kmeans2stage", "carry", "hmm", "bgm"]


def _now():
    return datetime.now().strftime("%H:%M:%S")


def render_infer(ticker, cell):
    """Render the infer script for one cell ("{axis}_{label}")."""
    with open(INFER_TMPL) as f:
        code = f.read()
    return code.replace("__TICKER__", ticker).replace("__CELL__", cell)


def _f(x, default=0.0):
    try:
        return float(str(x).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def parse_train_stats(bt_train):
    """Extract synthetic stats from a TRAIN backtest's runtimeStatistics."""
    rt = bt_train.get("runtimeStatistics", {}) or {}
    return {
        "synth_cal": _f(rt.get("best_cal", 0)),
        "train_auc": _f(rt.get("train_auc", 0)),
        "val_auc": _f(rt.get("val_auc", 0)),
        "n_cells": rt.get("n_cells", "?"),
        "bar_type": rt.get("bar_type", "?"),
    }


def parse_infer_stats(bt_infer):
    """Extract REAL Calmar (= CAGR / Drawdown) and Total Orders from infer."""
    st = bt_infer.get("statistics", {}) or {}
    cagr = _f(st.get("Compounding Annual Return", "0%"))
    mdd = _f(st.get("Drawdown", "0%"))
    real_calmar = (cagr / mdd) if abs(mdd) > 0.01 else 0.0
    try:
        trades = int(st.get("Total Orders", "0"))
    except (ValueError, TypeError):
        trades = 0
    return real_calmar, trades


def write_csv(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    print(f"[{_now()}] wrote {len(rows)} rows -> {RESULTS_CSV}")


def append_tsv(rows):
    new = not os.path.exists(RESULTS_TSV) or os.path.getsize(RESULTS_TSV) == 0
    with open(RESULTS_TSV, "a") as f:
        if new:
            f.write("commit\tcalmar\ttrades\tstatus\tdescription\n")
        for r in rows:
            desc = f"axislabel {r['ticker']} {r['axis']}/{r['label']} " \
                   f"synth_cal={r['synth_cal']:.4f} val_auc={r['val_auc']:.4f}"
            desc = desc.replace("\t", " ").replace("\n", " ")
            f.write(f"axis_label\t{r['real_calmar']:.4f}\t{r['trades']}\t"
                    f"{r['status']}\t{desc}\n")
    print(f"[{_now()}] appended {len(rows)} rows -> {RESULTS_TSV}")


def run_study(ticker, axes, labelers):
    print(f"=== axis x labeling study: ticker={ticker} ===")
    print(f"    axes     = {axes}")
    print(f"    labelers = {labelers}")
    print(f"    {len(axes)} train runs + {len(axes)*len(labelers)} infer runs "
          f"(serial; QC has 2 nodes)\n")

    rows = []
    for ai, axis in enumerate(axes, 1):
        # --- Phase 1: TRAIN this axis (footer sweeps all labelers, saves cells).
        print(f"[{_now()}] ({ai}/{len(axes)}) TRAIN {ticker} axis={axis} ...")
        train_code = render_script(ticker, axis=axis)
        bt_train, status_train = submit_and_wait(
            train_code, f"train_{ticker}_{axis}", timeout_s=300)  # hard 5-min cap
        tstats = parse_train_stats(bt_train)
        print(f"    TRAIN status={status_train} synth_cal={tstats['synth_cal']:.4f} "
              f"train_auc={tstats['train_auc']:.4f} val_auc={tstats['val_auc']:.4f} "
              f"n_cells={tstats['n_cells']} bar_type={tstats['bar_type']}")

        if status_train != "completed":
            # Train failed: record every cell for this axis as failed; skip infer.
            for label in labelers:
                rows.append({
                    "ticker": ticker, "axis": axis, "label": label,
                    "real_calmar": 0.0, "trades": 0,
                    "synth_cal": tstats["synth_cal"],
                    "train_auc": tstats["train_auc"],
                    "val_auc": tstats["val_auc"],
                    "status": f"train_{status_train}",
                })
                print(f"      SKIP infer {axis}/{label} (train {status_train})")
            write_csv(rows)  # incremental save
            continue

        # --- Phase 2: INFER each (axis,label) cell, read REAL Calmar.
        for li, label in enumerate(labelers, 1):
            cell = f"{axis}_{label}"
            print(f"[{_now()}]   ({li}/{len(labelers)}) INFER {ticker} cell={cell} ...")
            infer_code = render_infer(ticker, cell)
            bt_infer, status_infer = submit_and_wait(
                infer_code, f"infer_{ticker}_{cell}", timeout_s=300)
            real_calmar, trades = parse_infer_stats(bt_infer)
            print(f"      >>> REAL Calmar={real_calmar:.4f} Trades={trades} "
                  f"status={status_infer}")
            rows.append({
                "ticker": ticker, "axis": axis, "label": label,
                "real_calmar": real_calmar, "trades": trades,
                "synth_cal": tstats["synth_cal"],
                "train_auc": tstats["train_auc"],
                "val_auc": tstats["val_auc"],
                "status": status_infer,
            })
            write_csv(rows)  # incremental save after every cell

    write_csv(rows)
    append_tsv(rows)
    print(f"\n[{_now()}] DONE: {len(rows)} cells. CSV={RESULTS_CSV}")
    return rows


def _parse_argv(argv):
    ticker = argv[1] if len(argv) > 1 else DEFAULT_TICKER
    axes = argv[2].split(",") if len(argv) > 2 and argv[2] else DEFAULT_AXES
    labelers = argv[3].split(",") if len(argv) > 3 and argv[3] else DEFAULT_LABELERS
    axes = [a.strip() for a in axes if a.strip()]
    labelers = [l.strip() for l in labelers if l.strip()]
    return ticker, axes, labelers


if __name__ == "__main__":
    tk, ax, lb = _parse_argv(sys.argv)
    run_study(tk, ax, lb)
