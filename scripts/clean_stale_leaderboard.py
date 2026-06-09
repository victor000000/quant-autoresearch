#!/usr/bin/env python3
"""Clean the 10 STALE/FAKE per_etf_best entries (2026-06-09 module-sweep re-validation).

The module sweep re-ran every champion across sizer x reduce; 10 don't reproduce (artifact/leveraged/
leak-dead/decayed) but the driver's keep-HIGHER logic preserved the inflated stored value. This overwrites
each with its honest re-validated current best (max DEPLOYABLE Calmar from today's fresh sweep rows),
recording the old value as `stale_calmar_was` + a `revalidated_2026_06_09` flag (transparent, reversible).
Sanctioned LEADERBOARD maintenance (re-running overwrites stale cells — leaderboard-goes-stale memory);
the deployed book is separate + unaffected (none of these are book members). The human commits.
"""
import json
import csv
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALE = ["BIL", "QLD", "SPXL", "SSO", "AGQ", "ACWX", "XME", "XOP", "UUP", "SOXX"]
DATE = "2026-06-09"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    kpath = os.path.join(HERE, "knowledge.json")
    k = json.load(open(kpath))
    pe = k.get("per_etf_best", {})
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "round_results.csv"))))
    changed = []
    for tk in STALE:
        if tk not in pe:
            continue
        cfg = pe[tk].get("config") or {}
        lab = cfg.get("labeler")
        fresh = [r for r in rows if r.get("ticker") == tk and r.get("labeler") == lab
                 and str(r.get("timestamp", "")).startswith(DATE)]
        cand = [(f(r.get("real_calmar")), int(f(r.get("trades")) or 0), r.get("sizing"))
                for r in fresh if f(r.get("real_calmar")) is not None]
        if not cand:
            continue
        dep = [c for c in cand if c[1] >= 80]
        best = max(dep) if dep else max(cand)
        cal, tr, sz = best
        old = pe[tk].get("real_calmar")
        if old is None or cal >= 0.9 * old:        # only clean genuinely-stale entries
            continue
        pe[tk]["stale_calmar_was"] = old
        pe[tk]["real_calmar"] = round(cal, 4)
        pe[tk]["trades"] = tr
        pe[tk]["sizing_revalidated"] = sz
        pe[tk]["g2_pass"] = bool(tr >= 80 and cal > 0)
        pe[tk]["revalidated_2026_06_09"] = "STALE-cleaned: stored was inflated/artifact; this is the honest re-run"
        changed.append((tk, old, cal))
    with open(kpath, "w") as fh:
        json.dump(k, fh, indent=2, default=str)
    for tk, old, cal in changed:
        print(f"{tk}: {old} -> {cal} (cleaned)")
    print(f"\ncleaned {len(changed)} stale leaderboard entries (deployed book unaffected; human commits)")


if __name__ == "__main__":
    main()
