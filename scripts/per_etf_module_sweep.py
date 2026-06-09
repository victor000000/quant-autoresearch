#!/usr/bin/env python3
"""PER-ETF ALL-MODULE SWEEP (user 2026-06-08: "every single ETF should have a best methodology through all modules").

deep_sweep_etfs.py covered axis x labeler with FIXED sizer/reduce. This Stage-2 sweep covers the UNSWEPT
modules the component-selection guide flagged as the 'next grid' — SIZER x REDUCE — per ETF, on each
champion's own axis + labeler + thresh (read from knowledge.json per_etf_best['config']). The driver logs
each leg's real OOS Calmar to round_results.csv and AUTO-UPDATES per_etf_best when a config beats the
champion — so every ETF ends with its best methodology across {axis,labeler}x{sizer,reduce}. Resumable.

Features/calibration are NOT swept (this session: feature-adds + venn_abers add noise / are neutral; base+
isotonic win). always_long champions are skipped (buy-hold ignores sizer/reduce).

Run: nohup python3 scripts/per_etf_module_sweep.py > /tmp/permodule.log 2>&1 &
"""
import os
import json
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(HERE, "scripts", "run_autoresearch_round.py")
KNOW = os.path.join(HERE, "knowledge.json")
PROG = os.path.join(HERE, "results", "etf_modulesweep_progress.log")

SIZERS = ["cdf_overlay", "dd_overlay", "cdf_plain"]
REDUCES = ["correlation", "infogain"]


def done_set():
    d = set()
    if os.path.exists(PROG):
        for l in open(PROG):
            if l.startswith("DONE "):
                d.add(l.split()[1])
    return d


def main():
    k = json.load(open(KNOW))
    pe = k.get("per_etf_best", {})
    done = done_set()
    todo = [t for t, v in pe.items() if v.get("config") and t not in done]
    todo.sort()
    log = open(PROG, "a")
    log.write(f"START module-sweep: {len(todo)} ETFs x {len(SIZERS)} sizers x {len(REDUCES)} reduces\n")
    log.flush()
    for tk in todo:
        cfg = pe[tk]["config"]
        lab = str(cfg.get("labeler", ""))
        if lab == "always_long" or not lab:
            log.write(f"SKIP {tk} (always_long/none)\n")
            log.write(f"DONE {tk}\n")
            log.flush()
            continue
        base = {"ticker": tk, "axis": cfg.get("axis", "logdollar"), "labeler": lab,
                "thresh": float(cfg.get("thresh", 0.45)),
                "n_components": int(cfg.get("n_components", 20))}
        cfgs = []
        for sz in SIZERS:
            for rd in REDUCES:
                cfgs.append(json.dumps({**base, "sizing": sz, "reduce": rd}))
        log.write(f"SWEEP {tk} ({len(cfgs)} configs; champ {cfg.get('sizing')}/{cfg.get('reduce')} @ {cfg.get('axis')}/{lab})\n")
        log.flush()
        for i in range(0, len(cfgs), 2):
            pair = cfgs[i:i + 2]
            if len(pair) == 1:
                pair = pair + [pair[0]]
            try:
                subprocess.run(["python3", DRIVER] + pair, cwd=HERE, timeout=1100,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log.write(f"ERR {tk} pair{i}: {e}\n")
                log.flush()
        log.write(f"DONE {tk}\n")
        log.flush()
    log.write("MODULE-SWEEP COMPLETE\n")
    log.flush()


if __name__ == "__main__":
    main()
