#!/usr/bin/env python3
"""Clean up QC Cloud ObjectStore: delete STALE per-cell prediction blobs under
'autoresearch/' while KEEPING the current champion cells (per_etf_best), all
non-cell keys (latest_key pointers, oosbars, metadata), and anything outside our
prefix. Runs as a tiny QC backtest (the QC-native way to manage the store).

  python3 scripts/cleanup_objectstore.py            # DRY-RUN (counts only, no delete)
  python3 scripts/cleanup_objectstore.py --delete    # actually delete the stale cells
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_autoresearch_round as R

DELETE = "--delete" in sys.argv
K = json.load(open(R.KNOWLEDGE_JSON))
pe = K.get("per_etf_best", {})

# Champion cell keys to KEEP (reconstructed from per_etf_best.config, matching the
# footer's save key: cell_{axis}_{labeler '+'->'_x_'}_{sizing}_t{int(thresh*100)}.json)
keep = set()
for tk, v in pe.items():
    c = v.get("config", {})
    if not c:
        continue
    lab = str(c["labeler"]).replace("+", "_x_")
    keep.add(f"autoresearch/{tk}/cell_{c['axis']}_{lab}_{c['sizing']}_t{int(round(float(c['thresh'])*100))}.json")
print(f"[cleanup] champions to KEEP ({len(keep)}):")
for k in sorted(keep):
    print("   ", k)

KEEP_JSON = json.dumps(sorted(keep))
DRY = "False" if DELETE else "True"

CODE = '''from AlgorithmImports import *
import json
KEEP = set(json.loads({keep!r}))
DRYRUN = {dry}
class StoreCleanup(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2026, 5, 1); self.set_end_date(2026, 5, 2); self.set_cash(100000)
        keys = []
        try:
            for kvp in self.object_store:
                keys.append(str(kvp.key))
        except Exception:
            try:
                keys = [str(k) for k in self.object_store.keys]
            except Exception:
                keys = []
        scanned = total_cells = champ_kept = noncell_kept = would_del = deleted = 0
        for ks in keys:
            if not ks.startswith("autoresearch/"):
                continue
            scanned += 1
            is_cell = ("/cell_" in ks) and ks.endswith(".json")
            if not is_cell:
                noncell_kept += 1; continue
            total_cells += 1
            if ks in KEEP:
                champ_kept += 1; continue
            would_del += 1
            if not DRYRUN:
                try:
                    self.object_store.delete(ks); deleted += 1
                except Exception:
                    pass
        self.set_runtime_statistic("scanned", str(scanned))
        self.set_runtime_statistic("total_cells", str(total_cells))
        self.set_runtime_statistic("champ_kept", str(champ_kept))
        self.set_runtime_statistic("noncell_kept", str(noncell_kept))
        self.set_runtime_statistic("would_delete", str(would_del))
        self.set_runtime_statistic("deleted", str(deleted))
        self.set_runtime_statistic("dryrun", str(DRYRUN))
        self.quit("objectstore cleanup done")
'''.format(keep=KEEP_JSON, dry=DRY)

name = "objectstore_cleanup_" + ("delete" if DELETE else "dryrun")
print(f"\n[cleanup] {'DELETING' if DELETE else 'DRY-RUN'} — submitting {name} ...")
res = R.run_pool([(name, CODE)])
bt = res.get(name, {})
rt = (bt.get("runtimeStatistics", {}) or {}) if isinstance(bt, dict) else {}
print(f"[cleanup] status={bt.get('status','?')}")
for k in ("scanned", "total_cells", "champ_kept", "noncell_kept", "would_delete", "deleted", "dryrun"):
    print(f"   {k:14s} = {rt.get(k, '—')}")
