#!/usr/bin/env python3
"""DEEP SWEEP (user 2026-06-05: "did you try ALL custom axes + unsupervised labelers per ETF?").

The 5-method screen panel tests only 2 axes x ~6 labelers. This deep-sweep covers the FULL axis + labeler
space on the FIT-RELEVANT classes (commodity / currency / leveraged + the marginal names) — where edges
concentrate — WITHOUT the infeasible 21x27=567-config cross. For each ETF it tests:
  * EVERY axis (21) paired with the standard trend labeler trend_leg  -> tests each AXIS
  * EVERY distinct unsupervised labeler (~25) paired with the standard logdollar axis -> tests each LABELER
  * always_long (buy-hold baseline)
~47 configs/ETF. The driver logs each leg's real OOS Calmar + val_auc to round_results.csv; scripts/
screen_report.py then takes the BEST method per ETF (it already maxes over all non-always_long legs), so
the deep-sweep's extra methods automatically improve each ETF's verdict. Single coordinator. Resumable.

Run: nohup python3 scripts/deep_sweep_etfs.py > /tmp/arlogs/deepsweep.log 2>&1 &
"""
import sys, os, csv, json, subprocess

from lb.paths import ROOT as _LBROOT
HERE = str(_LBROOT)
CSV = os.path.join(HERE, "results", "etf_qc_confirmed_pre2009.csv")
DRIVER = os.path.join(HERE, "scripts", "run_round.py")
PROG = os.path.join(HERE, "results", "etf_deepsweep_progress.log")

# every bar axis (each tested with trend_leg)
AXES = ["dollar", "tick", "vol", "range", "logdollar", "entropy", "imbalance", "tickimb", "volumeimb",
        "fracdiff", "dc", "zcusum", "kyle", "run", "spectral", "vpin", "jump", "volofvol", "wavelet",
        "amihud", "ddonset"]
# every DISTINCT unsupervised labeler (each tested on logdollar); skip H-variants + triple_barrier variants
LABELERS = ["accel", "ker", "trend_leg", "sharpe_scan", "ofsc", "bde_cusum", "changepoint", "calmar_scan",
            "sadf_explosive", "hurst_persist", "sliced_wasserstein", "sortino_scan", "transfer_entropy_dir",
            "visgraph", "mfe_mae", "revert", "turn_scan", "perment", "tertile", "bgm", "agglomerative",
            "jump_model", "triple_barrier_meta", "trend_scan", "regime_gmm", "cusum_regime", "kmeans2stage"]
# "Leveraged/Inverse" REMOVED 2026-06-07 (leak-audit RAW re-validation): leveraged/inverse fits are
# ARTIFACTS — UCO(2x oil) permute-path, SSO(2x S&P)/AGQ(2x silver) RAW-collapse, SDS(-2x) negative.
# Mechanism: leverage decay + path amplification. They only burn QC (all DISCARD). Real energy edges
# (USO/DBC) live in the Commodity class, kept below. Re-add if a new leveraged hypothesis emerges.
FIT_CLASSES = {"Commodity", "Currency"}
# marginal/promising names from other classes that showed an edge in the 5-method screen
EXTRA = ["BWX", "SPTL", "BLV", "BIV", "IEO", "REM", "VOX", "MOO", "FXF", "XES"]
# already heavily researched (core book) — skip to avoid redundant compute
SKIP = {"GLD", "UUP", "SLV", "DBC", "TIP", "HYG", "TLT", "IWM", "EEM", "XLE", "QQQ", "SPY"}


def done_set():
    d = set()
    if os.path.exists(PROG):
        for l in open(PROG):
            if l.startswith("DONE "):
                d.add(l.split()[1])
    return d


def main():
    rows = list(csv.DictReader(open(CSV)))
    done = done_set()
    todo = [r["Ticker"].strip() for r in rows
            if (r.get("Asset_Class") in FIT_CLASSES or r["Ticker"].strip() in EXTRA)
            and r["Ticker"].strip() not in SKIP and r["Ticker"].strip() not in done]
    # de-dup preserve order
    seen = set(); todo = [t for t in todo if not (t in seen or seen.add(t))]
    log = open(PROG, "a")
    log.write(f"START deep-sweep: {len(todo)} ETFs x {len(AXES)} axes + {len(LABELERS)} labelers\n")
    log.flush()
    for tk in todo:
        # build the full config list for this ETF
        cfgs = []
        for ax in AXES:
            cfgs.append(json.dumps({"ticker": tk, "axis": ax, "labeler": "trend_leg",
                                    "thresh": 0.45, "sizing": "dd_overlay", "reduce": "infogain"}))
        for lab in LABELERS:
            cfgs.append(json.dumps({"ticker": tk, "axis": "logdollar", "labeler": lab,
                                    "thresh": 0.45, "sizing": "cdf_overlay"}))
        cfgs.append(json.dumps({"ticker": tk, "axis": "logdollar", "labeler": "always_long",
                                "thresh": 0.50, "sizing": "cdf_overlay"}))
        log.write(f"SWEEP {tk} ({len(cfgs)} configs)\n")
        log.flush()
        # run the driver two-at-a-time (2 QC nodes); pad odd tail with a repeat
        for i in range(0, len(cfgs), 2):
            pair = cfgs[i:i + 2]
            if len(pair) == 1:
                pair = pair + [pair[0]]
            try:
                subprocess.run(["python3", DRIVER] + pair, cwd=HERE, timeout=1000,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log.write(f"ERR {tk} pair{i}: {e}\n"); log.flush()
        log.write(f"DONE {tk}\n")
        log.flush()
    log.write("DEEP-SWEEP COMPLETE\n")
    log.flush()


if __name__ == "__main__":
    main()
