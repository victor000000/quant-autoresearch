#!/usr/bin/env python3
"""Unit tests for modules.features.realyield_feats — the exogenous real-yield
feature transform (2026-06-09 direction: gold <- 10y TIPS real yield + 2s10s slope).

The load-bearing property is CAUSALITY / APPEND-OOS-INVARIANCE: every feature at
bar i must depend only on ry[:i+1], so appending future bars never changes a past
feature value. This is the same invariant the leak contract demands of every axis
and the cross-asset feature path. Pure numpy/pandas — runs on the host, no QC.

    python3 tests/test_realyield_feats.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules"))
from features import realyield_feats  # noqa: E402


def _ry(n, seed=0):
    """A plausible real-yield path: random walk in basis points around ~1.0%."""
    rng = np.random.RandomState(seed)
    return 1.0 + np.cumsum(rng.randn(n)) * 0.01


def test_shapes_and_count():
    n = 400
    ry, sl = _ry(n), _ry(n, seed=1) - 0.5
    cols_no_sl = realyield_feats(ry, None, n)
    cols_sl = realyield_feats(ry, sl, n)
    assert len(cols_no_sl) == 6, f"expected 6 ry-only features, got {len(cols_no_sl)}"
    assert len(cols_sl) == 9, f"expected 9 ry+slope features, got {len(cols_sl)}"
    for c in cols_sl:
        assert c.shape == (n,), f"feature has wrong shape {c.shape}"
        assert c.dtype == np.float32
    print("ok — shapes (6 ry-only / 9 with slope) + float32")


def test_guards():
    assert realyield_feats([], None) == [], "empty ry must return []"
    # N mismatch -> [] (footer passes N to assert 1:1 alignment with the bar grid)
    assert realyield_feats(_ry(100), None, N=99) == [], "length!=N must return []"
    # mismatched slope length is dropped, ry features still returned
    cols = realyield_feats(_ry(200), _ry(150), 200)
    assert len(cols) == 6, "mismatched-length slope should be ignored, ry features kept"
    print("ok — guards (empty / N-mismatch / bad-slope-length)")


def test_causality_append_invariance():
    """THE leak test: features computed on a prefix must equal the same indices
    when more future bars are appended (no look-ahead anywhere in the transform)."""
    n_full = 600
    ry_full, sl_full = _ry(n_full, seed=7), _ry(n_full, seed=8) - 0.5
    k = 450  # prefix length
    full = realyield_feats(ry_full, sl_full, n_full)
    pref = realyield_feats(ry_full[:k], sl_full[:k], k)
    assert len(full) == len(pref) == 9
    for j, (cf, cp) in enumerate(zip(full, pref)):
        a, b = cf[:k], cp  # compare the overlapping [0, k) region
        # both NaN (warmup) or both equal within float tolerance
        mask = ~(np.isnan(a) & np.isnan(b))
        assert np.allclose(a[mask], b[mask], rtol=1e-5, atol=1e-6, equal_nan=True), \
            f"feature {j} is NOT causal — appending future bars changed a past value"
    print(f"ok — append-invariance: all 9 features causal over [0,{k}) under +{n_full - k} future bars")


def test_zscore_warmup_and_value():
    """Level z-score: NaN for the first W-1 bars, then a finite standardized value;
    on a pure ramp the rolling z-score is positive (level above its trailing mean)."""
    n = 300
    ry = 1.0 + np.arange(n) * 0.001  # strict ramp
    cols = realyield_feats(ry, None, n)
    z60 = cols[1]  # first z-score uses W=60
    assert np.all(np.isnan(z60[:59])), "z-score must be NaN during the W=60 warmup"
    assert np.isfinite(z60[59:]).all(), "z-score must be finite after warmup"
    assert (z60[100:] > 0).all(), "on a rising ramp the level sits above its trailing mean -> z>0"
    # the LEVEL feature is the raw yield, unchanged
    assert np.allclose(cols[0], ry.astype(np.float32)), "feature[0] must be the raw real-yield level"
    print("ok — z-score warmup is NaN, finite + positive on a ramp; level is raw")


if __name__ == "__main__":
    test_shapes_and_count()
    test_guards()
    test_causality_append_invariance()
    test_zscore_warmup_and_value()
    print("\nALL PASS — realyield_feats is causal (append-OOS-invariant), shaped, guarded.")
