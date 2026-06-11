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
# footer's FULL save key incl. _PSUF: cell_{axis}_{labeler '+'->'_x_'}_{sizing}_
# t{int(thresh*100)}{_PSUF}.json — the _PSUF mirror is REQUIRED (2026-06-10 fix:
# without it GLD `_n15_b3_ig` / IWM `_ig` champions reconstruct to nonexistent
# base keys and the real champion blobs would be DELETED as stale).
def _psuf(c):
    """Mirror templates/header.py.tmpl _PSUF exactly."""
    s = ""
    if c.get("permute_labels"):
        s += "_perm"
    if int(c.get("n_components", 20)) != 20:
        s += "_n" + str(int(c["n_components"]))
    if float(c.get("rebal_band", 0.01)) != 0.01:
        s += "_b" + str(int(round(float(c["rebal_band"]) * 100)))
    if c.get("horizons"):
        s += "_hz" + "x".join(str(int(h)) for h in c["horizons"])
    r = c.get("reduce", "correlation")
    if r != "correlation":
        s += "_ig" if r == "infogain" else "_rd" + str(r)
    f = c.get("features", "base")
    if f != "base":
        s += {"rich": "_fr", "termstruct": "_ts", "realyield": "_ry"}.get(f, "_fx")
    if c.get("calibration", "isotonic") != "isotonic":
        s += "_va"
    if c.get("train_purge"):
        s += "_tp"
    return s

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
BOOK_TICKERS = ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO"]

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
