#!/usr/bin/env python3
"""Classify lean_workspace dirs into Wang pipeline modules.

Output: /tmp/reorg_mapping.csv with columns: dirname,module,reason
Modules:
  01_axis        - logdollar, realvar, znorm, dollar bars, custom-axis bake
  02_labels      - HMM (any), tertile, label-template, sign-label, triple-barrier label
  03_features    - PCA, VAE, raw features, dimred
  04_models      - XGB depth sweep, logreg, deep, model selection
  05_calibration - isotonic, Platt, conf filter, confidence sweep
  06_portfolio   - multi_, wstack, ensemble, all19, combo, portfolio
  07_walkfwd     - walk_fwd, wfwd, oos sweep
  99_archive     - dead-end / diagnostic / unused (diag_, check_, _cleanup)
  _templates     - _*_template, _zn_*_template, _walk_templ
  _uncertain     - everything that doesn't match
"""
import csv
import os
import re

WORKSPACE = "/home/txy/lb/lean_workspace"

ETF_PREFIXES = (
    "agg", "dbc", "eem", "efa", "ewj", "gld", "hyg", "iwm", "qqq", "shy",
    "slx", "spy", "tip", "tlt", "uup", "vnq", "vxx", "xlb", "xle", "xlf",
    "xli", "xlk", "xlp", "xlu", "xlv", "xly", "vea", "vti", "vwo", "bnd",
)


def has(name: str, *needles: str) -> bool:
    return any(n in name for n in needles)


def classify(name: str) -> tuple[str, str]:
    low = name.lower()

    if low.endswith("_template") or low.endswith("_templ"):
        return "_templates", "name ends with _template"

    if has(low, "_cleanup", "diag_", "_diag", "check_models", "_bench", "ffd_dsweep",
           "dv_analysis", "data_cache"):
        return "99_archive", "diagnostic/benchmark/cleanup"

    if has(low, "portfolio", "ensemble", "wstack", "all19", "_combo", "stack_",
           "_stack", "perf", "perasset", "hybrid_axis", "labels_simple") \
       or low.startswith("multi_") or low.startswith("zn_"):
        return "06_portfolio", "portfolio/multi/ensemble/zn output"

    if has(low, "walk_fwd", "wfwd", "walk_nox", "walk_sticky", "walk_templ"):
        return "07_walkfwd", "walk-forward keyword"

    if has(low, "calibrate", "iso_cal", "_platt", "platt", "iso_valdelta",
           "conf_filter", "confidence", "hiconf", "iso_hi",
           "iso_hmm", "lothresh", "bconf", "lower_thresh", "tip_hi",
           "_2dconf"):
        return "05_calibration", "calibration/conf-filter keyword"

    if has(low, "xgb", "logreg", "_pa_deep", "deep_long", "pa_deep",
           "universal_model", "_lreg", "_logreg", "_meta", "deepxgb",
           "_d5xgb", "_d3xgb", "d6_n5"):
        return "04_models", "model keyword (xgb/logreg/deep/meta)"

    if has(low, "pca", "vae", "_feat", "raw_features", "dimred", "p1c", "p1d",
           "_d6vae", "d5vae"):
        return "03_features", "feature/dimred keyword"

    if has(low, "hmm", "_lbl", "lbl_", "label", "tertile", "signlbl", "_tb_",
           "tb_label", "triple_barrier", "p1b", "_viterbi", "iso2d", "iso_scaled",
           "_iso_2d", "ballbl", "_ms_tb", "lblsrch", "_dbl_ddrg"):
        return "02_labels", "label/HMM/tertile keyword"

    if has(low, "logdollar", "realvar", "_axis", "dollar_v", "dollar_bar",
           "_dollar", "znorm", "_zn_", "_pa_", "p1a", "_xa_", "_detrend",
           "_volnorm", "_cumret", "_scaled", "atr_ksweep", "fine_ksweep",
           "multibar", "_pa7", "pa_full", "pa_r", "pa_z", "pa_v"):
        return "01_axis", "axis/dollar/znorm keyword"

    if has(low, "train_v1", "train_v2", "train_v3", "_pipeline", "_train_",
           "_oos_cpd", "p1_train", "_oos_", "pipeline_v"):
        return "08_pipelines", "end-to-end training pipeline (cross-module)"

    if has(low, "_400ft", "_hbal", "_spw", "feat_missing", "_pa7"):
        return "01_axis", "axis/feature-bake variant"

    if low in ("data", "data_check"):
        return "99_archive", "data scratch"

    if low.startswith("_"):
        return "_templates", "underscore-prefixed scratch"

    for pre in ETF_PREFIXES:
        if low == pre or low.startswith(pre + "_"):
            return "_uncertain", f"etf-prefixed {pre}_ but no module keyword"

    return "_uncertain", "no rule matched"


def main():
    dirs = sorted(d for d in os.listdir(WORKSPACE)
                  if os.path.isdir(os.path.join(WORKSPACE, d)))
    rows = []
    counts: dict[str, int] = {}
    for d in dirs:
        module, reason = classify(d)
        rows.append((d, module, reason))
        counts[module] = counts.get(module, 0) + 1

    out_csv = "/tmp/reorg_mapping.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dirname", "module", "reason"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print("\nCounts by module:")
    for m in sorted(counts):
        print(f"  {m:15s}  {counts[m]:5d}")


if __name__ == "__main__":
    main()
