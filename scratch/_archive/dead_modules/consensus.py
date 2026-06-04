"""Module ⑧: Consensus Filter (Wang workflow).
Only trade when the ensemble agrees sufficiently.
Proven: lifts GLD Cal 3.02 → 4.37 by filtering uncertain predictions.

Filter: trade long if min(p) > 0.5 AND avg(p) > 0.55
"""
import numpy as np


def apply_consensus(probs_ensemble, probs_per_seed, min_thresh=0.5, avg_thresh=0.55):
    """Apply consensus filter to ensemble predictions.

    For each prediction:
    - min(p) > min_thresh: ALL seeds agree it's a long signal
    - avg(p) > avg_thresh: average confidence exceeds threshold
    - Both must be true to trade

    Args:
        probs_ensemble: (N,) averaged probabilities across seeds
        probs_per_seed: (S, N) per-seed probabilities
        min_thresh: minimum probability for any individual seed
        avg_thresh: minimum average probability

    Returns:
        filtered_probs: (N,) consensus-filtered probabilities (0 if no consensus)
        consensus_mask: (N,) boolean mask of which bars pass consensus
    """
    probs = np.array(probs_per_seed)
    min_probs = np.min(probs, axis=0)
    avg_probs = np.mean(probs, axis=0)

    consensus_mask = (min_probs > min_thresh) & (avg_probs > avg_thresh)

    filtered = np.zeros_like(avg_probs)
    filtered[consensus_mask] = avg_probs[consensus_mask]

    return filtered, consensus_mask


def consensus_stats(probs_per_seed):
    """Compute consensus quality metrics."""
    probs = np.array(probs_per_seed)
    min_probs = np.min(probs, axis=0)
    avg_probs = np.mean(probs, axis=0)
    std_probs = np.std(probs, axis=0)

    return {
        "n_bars": len(avg_probs),
        "n_pass_min05": int(np.sum(min_probs > 0.5)),
        "n_pass_avg055": int(np.sum(avg_probs > 0.55)),
        "n_pass_both": int(np.sum((min_probs > 0.5) & (avg_probs > 0.55))),
        "mean_std": float(np.mean(std_probs)),
        "agreement_rate": float(np.mean(min_probs > 0.5)),
    }
