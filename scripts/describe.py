#!/usr/bin/env python3
"""Translate a render-time CONFIG into a plain-English hypothesis description.
Shared by the driver (status panel), render_round, and render_index."""

AXES = {
    "dollar": "dollar-volume bars", "logdollar": "log-dollar bars",
    "imbalance": "signed-dollar imbalance (directional) bars",
    "tickimb": "tick-imbalance bars", "volumeimb": "volume-imbalance bars",
    "range": "equal-price-range bars", "vol": "volatility bars",
    "entropy": "entropy/surprise bars", "fracdiff": "fractional-difference (memory) bars",
    "dc": "directional-change (reversal) bars", "tick": "tick bars",
}
LAB = {
    "triple_barrier": "triple-barrier (forward σ-barrier) labels",
    "triple_barrier_tight": "tight triple-barrier labels",
    "bgm": "Bayesian-GMM regime labels", "multi_horizon": "multi-horizon trend labels",
    "tertile": "forward-tertile labels", "kmeans2stage": "k-means regime labels",
    "agglomerative": "agglomerative regime labels", "carry": "carry/low-vol labels",
    "dc_trend": "directional-change trend-state labels",
    "dc_reversal": "directional-change reversal labels",
    "regime_gmm": "causal-feature GMM regime labels",
    "cusum_regime": "CUSUM change-point regime labels",
    "hmm": "HMM regime labels (baseline)", "always_long": "always-long (buy & hold)",
}
SIZ = {
    "cdf_overlay": "long-only, vol-targeted", "cdf_plain": "long-only",
    "ls_overlay": "long/short, vol-targeted", "ls_cdf": "long/short (continuous)",
    "longshort": "long/short (binary ±1)", "binary": "binary long/flat",
    "ramp": "ramped long",
}


def labeler_phrase(labeler):
    parts = str(labeler).split("+")
    phrases = [LAB.get(p, p) for p in parts]
    return ("ensemble of " + " + ".join(phrases)) if len(parts) > 1 else phrases[0]


def describe(axis, labeler, thresh, sizing):
    a = AXES.get(axis, axis)
    s = SIZ.get(sizing, sizing)
    try:
        t = f"entry {float(thresh):.2f}"
    except (TypeError, ValueError):
        t = f"entry {thresh}"
    return f"Sample on {a}; {labeler_phrase(labeler)}; size {s}, {t}."


def describe_cfg(cfg):
    return describe(cfg.get("axis"), cfg.get("labeler"), cfg.get("thresh"), cfg.get("sizing"))
