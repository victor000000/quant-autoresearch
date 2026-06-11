#!/usr/bin/env python3
"""Seed knowledge.json['causal_graph'] with the full experiment causal DAG (rounds 1-22).

Single source of truth. Each round APPENDS nodes/edges here (see add_round helper),
then scripts/render_causal_graph.py regenerates the visual. Node types:
  finding   = a mechanism/learning HUB (causes downstream design choices)  [yellow]
  round     = an experiment                                                [grey]
  milestone = a KEEP / new per-ETF best                                    [green]
  decision  = a reframe / unlock / user policy                             [purple]
Optional node flag running:true -> dashed (in-flight).
"""
import json, os

KJ = os.path.join(os.path.dirname(__file__), "..", "knowledge.json")

# (id, type, phase, label)
NODES = [
    # --- Phase A: Landscape / the 6-round null ---
    ("r1", "round", "Landscape", "R1: every labeler = buy-hold (1 trade)"),
    ("f_sizing", "finding", "Landscape", "FINDING: bottleneck is module-8 SIZING - infer saturates 100% long (w=min(1,(p-t)*200))"),
    ("r2", "round", "Landscape", "R2: CDF x inverse-vol overlay -> 431 trades, DA -69%, but Calmar HALVED (over-de-levers)"),
    ("f_overlay", "finding", "Landscape", "FINDING: inverse-vol overlay is LABEL-INDEPENDENT - it washes out the labeler (carry = always_long)"),
    ("r3", "round", "Landscape", "R3: DD circuit-breaker -> both metrics WORSE"),
    ("f_stops", "finding", "Landscape", "FINDING: drawdown STOPS are pro-cyclical on drift-up assets (sell the dip, re-enter at highs)"),
    ("r4", "round", "Landscape", "R4: detuned sizing on VOL axis x7 ETF -> active but sub-buy-hold"),
    ("r5", "round", "Landscape", "R5: DOLLAR x labelers x7 ETF (2-node parallel) -> labelers identical; SEEDS per-ETF best"),
    ("r6", "round", "Landscape", "R6: overlay OFF -> every cell = buy-hold (model is always-long)"),
    ("f_null", "finding", "Landscape", "FINDING (6-round NULL): single-asset long-only ML cannot beat buy-hold - axis/labeler/sizing all washed out"),
    ("reframe", "decision", "Landscape", "REFRAME -> v2 tournament: need a label that CHANGES exposure; G2>80 trades; race 2 hyp/round, keep-if-better"),
    # --- Phase B: TLT (declining / two-sided bond) ---
    ("r7", "round", "TLT", "R7: TLT binary -> under-trade / 0 trades"),
    ("r8", "round", "TLT", "R8: TLT cdf_plain partial-long -> -0.29 (worse)"),
    ("f_longonly", "finding", "TLT", "FINDING: long-only cannot beat a DECLINING asset"),
    ("unlock_short", "decision", "TLT", "UNLOCK (user): shorting allowed -> ls_cdf / longshort"),
    ("r9", "round", "TLT", "R9: TLT ls_cdf/longshort -> still worse (-0.29 / 0)"),
    ("f_signal", "finding", "TLT", "FINDING: shorting alone doesn't help - the SIGNAL must predict direction -> need a DIRECTIONAL label"),
    ("policy", "decision", "TLT", "USER POLICY: persist on the weakest ETF until it beats the 2nd-weakest (never mark exhausted)"),
    ("r12", "round", "TLT", "R12: TLT tertile/logdollar/ls_cdf -> -0.05 (best yet), 71tr"),
    ("f_direction", "finding", "TLT", "FINDING: DIRECTIONAL label (tertile/triple_barrier) + SHORT (ls_cdf) lifts two-sided assets; remaining gap = ACTIVITY"),
    ("r13", "milestone", "TLT", "R13 WIN: TLT -0.15 -> +0.04 (dense DOLLAR, 151tr)"),
    ("f_density", "finding", "TLT", "FINDING: edge needs a DENSE axis to clear G2 (>80tr) - tick has edge but under-trades"),
    ("r15", "milestone", "TLT", "R15 WIN: TLT +0.04 -> +0.31 (RANGE axis = dense AND concentrated at rate-shocks, 234tr)"),
    ("r16", "round", "TLT", "R16: range tertile/longshort under-trade (~50tr)"),
    ("r17", "round", "TLT", "R17: on range, triple_barrier UNIQUELY high-edge (bgm/agglom low)"),
    ("f_logdollar", "finding", "TLT", "FINDING: logdollar compresses the flight-to-quality notional tail into dense, uniform regime coverage"),
    ("r18", "milestone", "TLT", "R18 WIN: TLT +0.31 -> +0.75 (logdollar) - overtakes IWM"),
    # --- Phase C: IWM (trending-up small-cap) ---
    ("r10", "round", "IWM", "R10: IWM dollar/carry -> +0.25/564tr (first clean active) but <0.65"),
    ("f_kmeans", "finding", "IWM", "FINDING: kmeans2stage is degenerate (0 trades) in single-config mode"),
    ("r11", "round", "IWM", "R11: IWM vol/bgm -> Calmar 1.13 but 24tr (G2 FAIL)"),
    ("f_bgm", "finding", "IWM", "FINDING: bgm = high-edge / low-DA but UNDER-TRADES on sparse axes -> deploy on a DENSE axis"),
    ("r19", "round", "IWM", "R19: TLT recipe ports -> +0.50/308tr but <0.65; SHORT legs HURT trending-up IWM"),
    ("f_character", "finding", "IWM", "FINDING: ASSET CHARACTER sets sizing - trending-up->long-bias (cdf_overlay/plain); two-sided/declining->short (ls_cdf). Diagnose: does shorting help or hurt?"),
    ("r20", "milestone", "IWM", "R20 WIN: IWM 0.65 -> +1.10 (bgm on DENSE logdollar + long-biased cdf_overlay, 394tr)"),
    # --- Phase D: XLE (trending-up energy) ---
    ("r21", "round", "XLE", "R21: XLE bgm cut DA 39->9 but <0.72; short -0.06 hurts (trending-up)"),
    ("r22", "round", "XLE", "R22: XLE bgm/triple_barrier cdf_plain still <0.72"),
    ("f_trenddom", "finding", "XLE", "FINDING: timed labels UNDER-RETURN strongly-trending XLE (cut CAGR more than MaxDD) - R6 null resurfacing"),
    ("r23", "round", "XLE", "R23: XLE multi_horizon (untested featured labeler), long-biased"),
]

# (src, dst, label)
EDGES = [
    ("r1", "f_sizing", "labeler can't express; w pins 100% long"),
    ("f_sizing", "r2", "de-saturate the sizing"),
    ("r2", "f_overlay", "carry = always_long"),
    ("r2", "r3", "over-de-levers -> try a DD breaker"),
    ("r3", "f_stops", "both metrics worse"),
    ("f_stops", "r4", "stops fail -> back to sizing"),
    ("f_overlay", "r4", "detune on vol axis"),
    ("r4", "r5", "3 sizing rounds never beat buy-hold -> pivot to axis x labeler"),
    ("r5", "f_overlay", "labelers identical (reinforces)"),
    ("r5", "r6", "turn overlay OFF"),
    ("r6", "f_null", "always-long -> buy-hold"),
    ("f_null", "reframe", ""),
    ("f_overlay", "reframe", ""),
    ("reframe", "r7", "TLT weakest (-0.15)"),
    ("r7", "r8", "binary under-trades"),
    ("r8", "f_longonly", ""),
    ("f_longonly", "unlock_short", ""),
    ("unlock_short", "r9", ""),
    ("r9", "f_signal", "shorting alone fails"),
    ("r9", "r10", "(pre-policy) moved to IWM"),
    ("policy", "r12", "return to TLT; don't exhaust"),
    ("f_signal", "r12", "need a directional label"),
    ("r10", "f_kmeans", ""),
    ("r11", "f_bgm", ""),
    ("r12", "f_direction", ""),
    ("f_direction", "r13", ""),
    ("r13", "f_density", "tick edge but under-trades"),
    ("f_density", "r15", "range = dense + at shocks"),
    ("r15", "r16", ""),
    ("r15", "r17", ""),
    ("r17", "f_logdollar", "range plateaus +0.31"),
    ("f_density", "r18", ""),
    ("f_direction", "r18", ""),
    ("f_logdollar", "r18", ""),
    ("r18", "r19", "TLT 0.75 beats IWM -> loop moves to IWM"),
    ("r19", "f_character", "short hurts trending-up IWM"),
    ("f_character", "r20", ""),
    ("f_bgm", "r20", "make bgm deployable on a dense axis"),
    ("f_density", "r20", "dense axis for G2"),
    ("r20", "r21", "IWM 1.10 -> loop moves to XLE"),
    ("f_character", "r21", "XLE trending-up -> long-bias"),
    ("f_bgm", "r21", "apply bgm"),
    ("r21", "r22", "cut DA but under-return -> drop vol-throttle"),
    ("r21", "f_trenddom", ""),
    ("r22", "f_trenddom", ""),
    ("f_null", "f_trenddom", "null resurfaces for a strong trend"),
    ("f_trenddom", "r23", "labels under-return -> try multi_horizon, stay long"),
]

d = json.load(open(KJ))
d["causal_graph"] = {
    "phases": ["Landscape", "TLT", "IWM", "XLE"],
    "nodes": [{"id": i, "type": t, "phase": p, "label": l} for (i, t, p, l) in NODES],
    "edges": [{"src": s, "dst": dd, "label": lb} for (s, dd, lb) in EDGES],
    "running": ["r23"],
}
json.dump(d, open(KJ, "w"), indent=2)
print(f"seeded causal_graph: {len(NODES)} nodes, {len(EDGES)} edges")
