#!/usr/bin/env python3
"""Online-FDR (LOND) ledger over the round SEQUENCE — the anytime-valid multiple-testing honesty
lens (Javanmard-Montanari 2018), complementing the one-shot DSR/Holm/BH in honest_audit.py and the
sufficiency check in MinBTL. Walks results/round_results.csv in chronological order, assigns each
round a p-value = 1 - PSR(>0) of its winning leg, and runs LOND so the FDR among 'discoveries' is
<= alpha at ANY stopping round. Answers: across the whole search, which rounds were REAL discoveries
(not search-luck)?  Built 2026-06-08 (new-methods report tooling rec; LOND validated by simulation
in stats_rigor.py).

  python3 scripts/online_fdr_ledger.py            # print the ledger summary
"""
import csv
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats_rigor import online_fdr_lond, probabilistic_sharpe_ratio  # noqa: E402

CSV = os.path.join(HERE, "..", "results", "round_results.csv")
PPY = 252.0


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_sequence(rows):
    """One entry per round (chronological): the winning leg (higher real_sharpe), excluding
    always_long baselines (they carry no selection bias) and degenerate (<2 day) cells."""
    by_round = {}
    for r in rows:
        ts = r.get("timestamp") or ""
        sr = _fnum(r.get("real_sharpe"))
        if sr is None or (r.get("labeler") or "") == "always_long":
            continue
        n = int(_fnum(r.get("n_days")) or 0)
        if n < 2:
            continue
        e = {"sr": sr, "n": n, "sk": _fnum(r.get("real_skew")) or 0.0,
             "ku": _fnum(r.get("real_kurt")) or 3.0, "tk": r.get("ticker")}
        cur = by_round.get(ts)
        if cur is None or sr > cur["sr"]:
            by_round[ts] = e
    return [by_round[k] for k in sorted(by_round)]


def main(alpha=0.05):
    rows = list(csv.DictReader(open(CSV)))
    seq = build_sequence(rows)
    pvals = [max(0.0, 1.0 - probabilistic_sharpe_ratio(e["sr"] / math.sqrt(PPY), e["n"],
             e["sk"], e["ku"], 0.0)) for e in seq]
    rej, _alphas = online_fdr_lond(pvals, alpha=alpha)
    disc = Counter(seq[i]["tk"] for i in range(len(seq)) if rej[i])
    print(f"LOND online-FDR ledger (alpha={alpha}, anytime-valid):")
    print(f"  rounds in sequence : {len(seq)}")
    print(f"  discoveries survive: {sum(rej)}")
    print(f"  by ticker          : {dict(disc.most_common())}")
    print("  reading: a 'discovery' is a round whose edge survives sequential FDR control at the "
          "moment it was made — search-luck rounds do NOT survive. Concentration in one name = "
          "that name is the durable edge; cash proxies (BIL) are high-Sharpe artifacts, not alpha.")
    return sum(rej), dict(disc)


if __name__ == "__main__":
    main()
