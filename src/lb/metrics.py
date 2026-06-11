"""Drawdown-SHAPE metrics — depth AND duration of the underwater experience.

Calmar/MaxDD see only the single deepest trough; they are blind to a long shallow
bleed (a decaying edge that drifts down without one big crash — e.g. UUP). Wang
Yiming's course (BV1VZLi6PE2B / BV1PQEQ6NETz) names the "drawdown AREA" (回撤面积)
as the trader's recovery-pain proxy: smaller area => faster recovery. This module
exposes the standard estimators of that idea, all operating on an EQUITY list (any
positive units) with pure stdlib:

    underwater(eq)      d_t = eq_t/peak_t - 1  (<=0), the underwater curve
    max_drawdown(eq)    -min d_t, worst trough as a POSITIVE fraction
    pain_index(eq)      mean |d_t| = drawdown AREA / N  (Becker Pain Index)
    ulcer_index(eq)     sqrt(mean d_t^2)  (Martin 1989; RMS drawdown)
    max_dd_duration(eq) longest run (points) spent below a prior peak
    martin_ratio(eq)    annualised CAGR / Ulcer  (Calmar's depth+duration kin)

All are post-hoc on the REALISED equity curve => no look-ahead, no leak. They are
length-normalised so they rank a book of unequal-horizon strategies fairly.
"""


def underwater(equity):
    """Per-point drawdown d_t = equity_t / running_peak_t - 1 (<= 0)."""
    out, peak = [], None
    for v in equity:
        peak = v if peak is None or v > peak else peak
        out.append(v / peak - 1.0 if peak else 0.0)
    return out


def max_drawdown(equity):
    """Worst peak-to-trough drop as a POSITIVE fraction (0.18 == -18%)."""
    dd = underwater(equity)
    return -min(dd) if dd else 0.0


def pain_index(equity):
    """Mean depth of the underwater curve = drawdown AREA / N (Becker Pain Index) —
    Wang's 回撤面积 normalised by length: the average ongoing pain of holding."""
    dd = underwater(equity)
    return -sum(dd) / len(dd) if dd else 0.0


def ulcer_index(equity):
    """Martin (1989) Ulcer Index = sqrt(mean(d_t^2)): RMS drawdown. Penalises deep
    AND long drawdowns (quadratic in depth, summed over time); ignores upside vol."""
    dd = underwater(equity)
    return (sum(d * d for d in dd) / len(dd)) ** 0.5 if dd else 0.0


def max_dd_duration(equity):
    """Longest stretch (in points) the curve spends STRICTLY below a prior peak —
    the 'time under water' that Calmar/MaxDD cannot see."""
    peak = None
    run = longest = 0
    for v in equity:
        if peak is None or v >= peak:
            peak, run = v, 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def martin_ratio(equity, ppy=252):
    """Annualised CAGR / Ulcer Index — the Calmar analogue that rewards spending
    LESS time and depth under water, not just a shallow single trough. ppy = points
    per year. Returns inf for a never-underwater curve, 0.0 if undefined."""
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    ui = ulcer_index(equity)
    if ui <= 0:
        return float("inf")
    yrs = (len(equity) - 1) / ppy
    if yrs <= 0:
        return 0.0
    cagr = (equity[-1] / equity[0]) ** (1.0 / yrs) - 1.0
    return cagr / ui
