#!/usr/bin/env python3
"""Online-determinism proof for a crowned cell (the manual's deployment gate:
"Don't trust a champion until its infer_online reports preds_match=1").

Two phases on the QC nodes:
  1. render_train_config(cfg) -> submit: trains the cell, saves model + predictions to ObjectStore.
  2. render_infer_online(ticker, cell) -> submit: rebuilds bars+features+model ONLINE from origin and
     asserts p_live == p_saved over matched OOS bars; emits runtime stat preds_match (1 iff max|d|<=1e-6).

Default target = the 2026-06-06 USO infogain crown (reduce=infogain lifted USO 2.18->3.42, +57%).
Run: python3 scripts/verify_crown_online.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.orchestrator import render_train_config, render_infer_online
from harness.qc_client import submit_and_wait

# (cfg for train-render, ObjectStore cell key for the online infer)
TARGET_CFG = dict(ticker="USO", axis="logdollar", labeler="revert", thresh=0.45,
                  sizing="cdf_overlay", reduce="infogain")
TARGET_CELL = "logdollar_revert_cdf_overlay_t45_ig"


def main():
    tk = TARGET_CFG["ticker"]
    print(f"[{tk}-online] PHASE 1 train (populate ObjectStore: model + saved preds) ...", flush=True)
    tcode, extra = render_train_config(TARGET_CFG)
    bt, st = submit_and_wait(tcode, f"{tk}_crown_online_train", timeout_s=540, extra_files=extra)
    print(f"  train status: {st}", flush=True)
    if st != "completed":
        print("TRAIN FAILED — aborting"); return

    print(f"[{tk}-online] PHASE 2 infer_online (rebuild ONLINE, assert p_live==p_saved) ...", flush=True)
    mcode, mextra = render_infer_online(tk, TARGET_CELL)
    bt2, st2 = submit_and_wait(mcode, f"{tk}_crown_infer_online", timeout_s=300, extra_files=mextra)
    rt = (bt2.get("runtimeStatistics", {}) or {}) if isinstance(bt2, dict) else {}
    pm = str(rt.get("preds_match"))
    print(f"  infer_online status: {st2}")
    print(f"  preds_match = {pm} | n_matched = {rt.get('n_matched')} | featlen_mismatch = {rt.get('featlen_mismatch')}")
    if pm == "1":
        print(f"VERDICT: {tk} crown is ONLINE-DEPLOYABLE — p_live==p_saved (<=1e-6), leak-free online replay. CONFIRMED.")
    else:
        print(f"VERDICT: NOT CONFIRMED (preds_match={pm}) — investigate before deploying.")


if __name__ == "__main__":
    main()
