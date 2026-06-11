#!/usr/bin/env python3
"""Backfill real_cagr / real_mdd into per_etf_best for the ETFs whose entries predate
the driver capturing them (IWM, TLT, QQQ). Re-runs each best config (train+infer) and
reads QC's 'Compounding Annual Return' + 'Drawdown'. Creates NO new rounds. Then
regenerates the dashboard so the leaderboard's CAGR/MDD columns are fully populated."""
import os, sys, json, subprocess
import importlib.util as _ilu
from lb.paths import ROOT as _ROOT
_spec = _ilu.spec_from_file_location("run_round", str(_ROOT / "scripts" / "run_round.py"))
R = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)  # driver-internal helpers (run_pool, _f, _cagr_from_stats, ...)

CONFIGS = {
    "IWM": {"ticker": "IWM", "axis": "imbalance", "labeler": "triple_barrier+bgm", "thresh": 0.45, "sizing": "cdf_overlay"},
    "TLT": {"ticker": "TLT", "axis": "range", "labeler": "triple_barrier", "thresh": 0.50, "sizing": "ls_overlay"},
    "QQQ": {"ticker": "QQQ", "axis": "logdollar", "labeler": "always_long", "thresh": 0.15, "sizing": "cdf_overlay"},
}


def cell_of(cfg):
    return (f"{cfg['axis']}_{cfg['labeler'].replace('+', '_x_')}_{cfg['sizing']}"
            f"_t{int(round(float(cfg['thresh']) * 100))}")


train_jobs = [(f"bf_train_{tk}", R.render_train_config(cfg)) for tk, cfg in CONFIGS.items()]
print(f"[backfill] training {len(train_jobs)} cells ...")
tr = R.run_pool(train_jobs)

infer_jobs = []
for tk, cfg in CONFIGS.items():
    bt = tr.get(f"bf_train_{tk}", {})
    if str(bt.get("status", "")).startswith("Completed"):
        infer_jobs.append((f"bf_infer_{tk}", R.render_infer_cell(tk, cell_of(cfg))))
    else:
        print(f"[backfill] {tk} train not completed ({bt.get('status','?')}) — skip")
print(f"[backfill] inferring {len(infer_jobs)} cells ...")
ir = R.run_pool(infer_jobs) if infer_jobs else {}

k = json.load(open(R.KNOWLEDGE_JSON))
pe = k["per_etf_best"]
for tk in CONFIGS:
    bt = ir.get(f"bf_infer_{tk}", {})
    st = (bt.get("statistics", {}) or {}) if isinstance(bt, dict) else {}
    if not st:
        print(f"[backfill] {tk}: NO stats (infer failed)")
        continue
    cagr, mdd, cal = R._cagr_from_stats(st), R._mdd_from_stats(st), R._calmar_from_stats(st)
    if tk in pe:
        pe[tk]["real_cagr"] = cagr
        pe[tk]["real_mdd"] = mdd
        print(f"[backfill] {tk}: cagr={cagr} mdd={mdd}  (calmar {cal} vs stored {pe[tk].get('real_calmar')})")
json.dump(k, open(R.KNOWLEDGE_JSON, "w"), indent=2)
subprocess.run(["python3", os.path.join(os.path.dirname(__file__), "render_index.py")])
print("[backfill] done — per_etf_best updated + dashboard regenerated")
