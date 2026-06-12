#!/usr/bin/env python3
"""Clean up QC Cloud ObjectStore: delete STALE per-cell prediction blobs under
'autoresearch/' while KEEPING the current champion cells (per_etf_best), all
non-cell keys (latest_key pointers, oosbars, metadata), and anything outside our
prefix. Runs as a tiny QC backtest (the QC-native way to manage the store).

  python3 scripts/research/cleanup_objectstore.py            # DRY-RUN (counts only, no delete)
  python3 scripts/research/cleanup_objectstore.py --delete    # actually delete the stale cells
"""
import os, sys, json
import importlib.util as _ilu
from lb.paths import ROOT as _ROOT
_spec = _ilu.spec_from_file_location("run_round", str(_ROOT / "scripts" / "run_round.py"))
R = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(R)  # driver-internal helpers (run_pool, _f, _cagr_from_stats, ...)

DELETE = "--delete" in sys.argv
K = json.load(open(R.KNOWLEDGE_JSON))
pe = K.get("per_etf_best", {})

# Champion cell keys to KEEP (reconstructed from per_etf_best.config, matching the
# footer's FULL save key incl. _PSUF: cell_{axis}_{labeler '+'->'_x_'}_{sizing}_
# t{int(thresh*100)}{_PSUF}.json — the _PSUF mirror is REQUIRED (2026-06-10 fix:
# without it GLD `_n15_b3_ig` / IWM `_ig` champions reconstruct to nonexistent
# base keys and the real champion blobs would be DELETED as stale).
from lb.harness.psuf import cell_suffix as _psuf_canonical


def _psuf(c):
    """SINGLE SOURCE: lb.harness.psuf.cell_suffix (the inline mirror here went stale
    2026-06-12 — it lacked the _m/_cb/_fx{name} tokens and would have deleted the
    GLD lgbm_bag crown's blobs). Never re-inline."""
    return _psuf_canonical(c)


keep = set()
for tk, v in pe.items():
    c = v.get("config", {})
    if not c:
        continue
    lab = str(c["labeler"]).replace("+", "_x_")
    keep.add(f"autoresearch/{tk}/cell_{c['axis']}_{lab}_{c['sizing']}_"
             f"t{int(round(float(c['thresh'])*100))}{_psuf(c)}.json")
print(f"[cleanup] champions to KEEP ({len(keep)}):")
for k in sorted(keep):
    print("   ", k)

# DEPLOYED-BOOK tickers: keep EVERY cell (belt-and-suspenders — per_etf_best can lag
# the deployed config, e.g. UUP's book cell is bgm+sadf_explosive+ker 1.85 while
# per_etf_best stores the stale bgm+ker 0.45; an exact-match-only keep would delete
# the deployed member's blob). Cruft lives overwhelmingly in the ~300 screen tickers.
BOOK_TICKERS = ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO", "DIA", "FEZ", "PRFZ"]

KEEP_JSON = json.dumps(sorted(keep))
BOOK_JSON = json.dumps(BOOK_TICKERS)
DRY = "False" if DELETE else "True"

CODE = '''from AlgorithmImports import *
import json
KEEP = set(json.loads({keep!r}))
BOOK = set(json.loads({book!r}))
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
            parts = ks.split("/")
            if ks in KEEP or (len(parts) > 1 and parts[1] in BOOK):
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
'''.format(keep=KEEP_JSON, book=BOOK_JSON, dry=DRY)

name = "objectstore_cleanup_" + ("delete" if DELETE else "dryrun")
print(f"\n[cleanup] {'DELETING' if DELETE else 'DRY-RUN'} — submitting {name} ...")
res = R.run_pool([(name, CODE)])
bt = res.get(name, {})
rt = (bt.get("runtimeStatistics", {}) or {}) if isinstance(bt, dict) else {}
print(f"[cleanup] status={bt.get('status','?')}")
for k in ("scanned", "total_cells", "champ_kept", "noncell_kept", "would_delete", "deleted", "dryrun"):
    print(f"   {k:14s} = {rt.get(k, '—')}")
