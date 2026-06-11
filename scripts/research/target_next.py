#!/usr/bin/env python3
"""Targeting brief (LeGIT-style) — the FIRST thing to read before proposing a round.

Reads knowledge.json and prints, for the weakest ETF: its current best, every cell
already tried on it (so we don't repeat), the recent FINDING hubs, and the open
question. The agent then ENUMERATES candidate interventions, RANKS them by
(expected metric-gain x confidence) and edge-disambiguation value, and runs the top.

Usage:  python3 scripts/target_next.py            # weakest ETF
        python3 scripts/target_next.py TLT         # a specific ETF
"""
import json, os, sys

KJ = os.path.join(os.path.dirname(__file__), "..", "knowledge.json")
d = json.load(open(KJ))
pe = d.get("per_etf_best", {})
ranked = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))
want = sys.argv[1] if len(sys.argv) > 1 else (ranked[0][0] if ranked else None)

print("=" * 78)
print("LEADERBOARD (real OOS Calmar):")
for i, (k, v) in enumerate(ranked):
    tag = "  <== WEAKEST (target)" if i == 0 else ("  <- 2nd-weakest (bar to beat)" if i == 1 else "")
    print(f"  {k:5s} {v.get('real_calmar', 0):+.4f}  trades={v.get('trades')}  {v.get('cell','')}{tag}")
to_beat = ranked[1][1].get("real_calmar") if len(ranked) > 1 else None

print("=" * 78)
print(f"TARGET ETF: {want}   (persist until it beats 2nd-weakest = {to_beat})")
best = pe.get(want, {})
print(f"  current best: {best.get('real_calmar')}  cell={best.get('cell')}")

print("\nCELLS ALREADY TRIED ON THIS ETF (don't repeat; mind axis x labeler interaction):")
for k, v in sorted(d.get("cells", {}).items(),
                   key=lambda kv: kv[1].get("real_calmar", 0) if isinstance(kv[1], dict) else 0,
                   reverse=True):
    if want and want in k and isinstance(v, dict):
        leak = "  [LEAK-flagged]" if v.get("LEAK") else ""
        print(f"  {v.get('real_calmar'):+.4f}  tr={v.get('trades')}  {k}{leak}")

print("\nFINDING HUBS (reason each candidate intervention FROM one of these):")
for n in d.get("causal_graph", {}).get("nodes", []):
    if n.get("type") == "finding":
        print(f"  [{n['id']}] {n['label']}")

print("\nRECENT FINDINGS (chronological tail):")
for c in d.get("confirmed", [])[-6:]:
    print(f"  - {c[:200]}")

print("\nOPEN QUESTION / last plan:")
print(f"  {d.get('next_idea','')}")
print("=" * 78)
print("NOW: enumerate 3-5 candidate interventions, rank by (expected gain x confidence)")
print("AND edge-disambiguation (does it resolve an open finding?), run the top 1-2.")
print("Then VERIFY: confirm preds cover every causal bar (no look-ahead); if a KEEP")
print("shows a leak signature (Calmar >> buy-hold, or DA << buy-hold, or trades ~G2),")
print("run a label-independent control (always_long, same axis+sizing) before trusting.")
