#!/usr/bin/env python3
"""Parallel axis×labeling driver — uses BOTH QC nodes.

QC has 2 backtest nodes but ONE shared project, and every submit overwrites
main.py. So we must serialize the short upload→compile→CREATE phase (each
backtest is locked to its own compiled code the moment it is created) and only
then run/poll up to 2 concurrently. submit_backtest() blocks through create,
which gives us that serialization for free; a 2-wide pool then keeps both nodes
busy during the long run/poll phase.

DO NOT run this concurrently with the serial driver (run_axis_label_study.py) —
two coordinators on one project WOULD race on main.py.

USAGE:
    python3 scripts/run_axis_label_parallel.py [TICKERS_CSV] [AXIS] [LABELERS_CSV]
    python3 scripts/run_axis_label_parallel.py QQQ,IWM,EEM,XLE,HYG,TLT,GLD vol carry,always_long
"""
import os, sys, csv, time
from datetime import datetime

PROJECT_ROOT = "/home/ubuntu/lb"
AR = os.path.join(PROJECT_ROOT)
if AR not in sys.path:
    sys.path.insert(0, AR)

from harness.qc_client import (submit_backtest, read_backtest, read_backtest_status,  # noqa: E402
                               delete_backtest, is_done)
from harness.orchestrator import render_script  # noqa: E402

INFER_TMPL = os.path.join(AR, "templates", "infer.py.tmpl")
RESULTS_DIR = os.path.join(AR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "axis_label_parallel_results.csv")
TIMEOUT = 300          # hard 5-min cap per backtest
POLL = 15              # seconds between status sweeps
MAX_INFLIGHT = 2       # QC node count


def _now():
    return datetime.now().strftime("%H:%M:%S")


def render_infer(ticker, cell):
    with open(INFER_TMPL) as f:
        code = f.read()
    return code.replace("__TICKER__", ticker).replace("__CELL__", cell)


def _f(x, d=0.0):
    try:
        return float(str(x).replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def run_pool(jobs):
    """jobs: list of (label, code). Keeps <=MAX_INFLIGHT backtests RUNNING at once.
    submit_backtest blocks through create (serializes main.py safely); polling then
    overlaps the run phase across both nodes. Returns {label: backtest_result_dict}."""
    results, inflight, pending = {}, {}, list(jobs)
    while pending or inflight:
        while pending and len(inflight) < MAX_INFLIGHT:
            label, code = pending.pop(0)
            print(f"[{_now()}] SUBMIT {label} (inflight={len(inflight)+1}/{MAX_INFLIGHT}, pending={len(pending)})")
            try:
                bid = submit_backtest(code, label)          # upload→compile→create (serial, safe)
                inflight[label] = (bid, time.time())
            except Exception as e:
                msg = str(e)
                # "no spare nodes" is TRANSIENT (a node is still busy) — re-queue and
                # wait for one to free instead of permanently failing the job.
                if "spare node" in msg.lower():
                    pending.insert(0, (label, code))
                    print(f"[{_now()}]   {label} no free node yet — re-queue + wait")
                    break
                results[label] = {"status": "crash", "error": msg}
                print(f"[{_now()}]   {label} submit CRASH: {msg[:120]}")
        done = []
        for label, (bid, t0) in inflight.items():
            status, _, _ = read_backtest_status(bid)
            if is_done(status):
                results[label] = read_backtest(bid) if status.startswith("Completed") else {"status": status}
                print(f"[{_now()}]   {label} -> {status}")
                done.append(label)
            elif time.time() - t0 > TIMEOUT:
                delete_backtest(bid)
                results[label] = {"status": "timeout"}
                print(f"[{_now()}]   {label} -> TIMEOUT (deleted)")
                done.append(label)
        for label in done:
            del inflight[label]
        if inflight or pending:   # sleep while work remains (incl. jobs waiting for a node)
            time.sleep(POLL)
    return results


def run_study(tickers, axis, labelers):
    print(f"=== PARALLEL axis×labeling: tickers={tickers} axis={axis} labelers={labelers} ===")
    print(f"    {len(tickers)} trains + {len(tickers)*len(labelers)} infers, {MAX_INFLIGHT} nodes\n")

    # Phase A — TRAIN every ticker (footer sweeps all labelers, saves cells), 2 nodes.
    train_jobs = [(f"train_{t}_{axis}", render_script(t, axis=axis)) for t in tickers]
    print(f"[{_now()}] PHASE A: {len(train_jobs)} trains")
    train_res = run_pool(train_jobs)
    tstats = {}
    for t in tickers:
        bt = train_res.get(f"train_{t}_{axis}", {})
        rt = bt.get("runtimeStatistics", {}) or {}
        tstats[t] = {"synth_cal": _f(rt.get("best_cal", 0)), "train_auc": _f(rt.get("train_auc", 0)),
                     "val_auc": _f(rt.get("val_auc", 0)),
                     "status": "completed" if str(bt.get("status", "")).startswith("Completed") else bt.get("status", "?")}

    # Phase B — INFER every (ticker,label) cell, 2 nodes.
    infer_jobs = []
    for t in tickers:
        if tstats[t]["status"] != "completed":
            continue
        for lab in labelers:
            infer_jobs.append((f"infer_{t}_{axis}_{lab}", render_infer(t, f"{axis}_{lab}")))
    print(f"\n[{_now()}] PHASE B: {len(infer_jobs)} infers")
    infer_res = run_pool(infer_jobs)

    # Collate.
    rows = []
    for t in tickers:
        for lab in labelers:
            bt = infer_res.get(f"infer_{t}_{axis}_{lab}", {})
            st = bt.get("statistics", {}) or {}
            rt = bt.get("runtimeStatistics", {}) or {}
            cagr, mdd = _f(st.get("Compounding Annual Return", "0%")), _f(st.get("Drawdown", "0%"))
            rows.append({
                "ticker": t, "axis": axis, "label": lab,
                "real_calmar": round(cagr / mdd, 4) if abs(mdd) > 0.01 else 0.0,
                "real_da": _f(rt.get("da_oos", 0.0)),
                "trades": int(st.get("Total Orders", "0") or 0) if str(st.get("Total Orders", "0")).isdigit() else 0,
                "synth_cal": tstats[t]["synth_cal"], "train_auc": tstats[t]["train_auc"],
                "val_auc": tstats[t]["val_auc"],
                "status": "completed" if str(bt.get("status", "")).startswith("Completed") else bt.get("status", "?"),
            })
    cols = ["ticker", "axis", "label", "real_calmar", "real_da", "trades",
            "synth_cal", "train_auc", "val_auc", "status"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[{_now()}] DONE: {len(rows)} cells -> {RESULTS_CSV}")
    for r in rows:
        print(f"  {r['ticker']:5s} {r['axis']:4s} {r['label']:12s} "
              f"Cal={r['real_calmar']:.4f} DA={r['real_da']:.2f} trades={r['trades']} {r['status']}")
    return rows


if __name__ == "__main__":
    tk = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD"])
    ax = sys.argv[2] if len(sys.argv) > 2 else "vol"
    lb = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["carry", "always_long"])
    run_study([t.strip() for t in tk if t.strip()], ax, [x.strip() for x in lb if x.strip()])
