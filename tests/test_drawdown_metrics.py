"""Drawdown-shape metrics (lb.metrics): Ulcer / Pain / time-under-water / Martin.

These capture drawdown depth AND duration — the underwater experience Calmar/MaxDD
(single deepest trough) miss on a long shallow bleed. Pure post-hoc on the realised
equity curve, so no leak; these tests pin the maths against hand-computed values.
"""
import math

from lb.metrics import (
    underwater, max_drawdown, pain_index, ulcer_index, max_dd_duration, martin_ratio,
)


def test_monotone_up_is_never_underwater():
    up = [100, 101, 103, 106, 110]
    assert underwater(up) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert ulcer_index(up) == 0.0
    assert pain_index(up) == 0.0
    assert max_drawdown(up) == 0.0
    assert max_dd_duration(up) == 0
    assert martin_ratio(up) == float("inf")


def test_v_shape_hand_computed():
    # [100,90,100] -> underwater [0,-0.1,0]
    v = [100, 90, 100]
    assert math.isclose(max_drawdown(v), 0.1, abs_tol=1e-12)
    assert math.isclose(pain_index(v), 0.1 / 3, abs_tol=1e-12)          # area/N
    assert math.isclose(ulcer_index(v), (0.01 / 3) ** 0.5, abs_tol=1e-12)  # RMS
    assert max_dd_duration(v) == 1


def test_same_maxdd_longer_underwater_scores_worse():
    """The whole point: identical MaxDD, but more time under water => higher Ulcer,
    higher Pain, longer duration. Calmar/MaxDD alone cannot tell these apart."""
    short = [100, 90, 100, 100, 100]
    long = [100, 90, 90, 90, 100]
    assert math.isclose(max_drawdown(short), max_drawdown(long), abs_tol=1e-12)
    assert ulcer_index(long) > ulcer_index(short)
    assert pain_index(long) > pain_index(short)
    assert max_dd_duration(long) > max_dd_duration(short)


def test_martin_ratio_sign_and_definition():
    # gentle uptrend with a dip: positive CAGR, finite ulcer -> positive, finite Martin
    eq = [100, 95, 100, 105, 110, 108, 115]
    ui = ulcer_index(eq)
    yrs = (len(eq) - 1) / 252.0
    cagr = (eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0
    assert math.isclose(martin_ratio(eq), cagr / ui, rel_tol=1e-12)
    assert martin_ratio(eq) > 0


def test_degenerate_inputs():
    assert ulcer_index([]) == 0.0
    assert max_drawdown([100]) == 0.0
    assert martin_ratio([100]) == 0.0           # <2 points -> undefined
    assert martin_ratio([0, 100]) == 0.0        # non-positive start -> undefined
