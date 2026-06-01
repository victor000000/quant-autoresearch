"""Multi-gate evaluator for QC backtest results.
G0: Completed    — backtest finished without timeout
G1: Calmar > 3.0 — OOS risk-adjusted return
G2: Trades > 80  — sufficient trading activity
G3: No Lookahead — zero future-data leaks
G4: No Overfit   — train/val AUC divergence < 0.05
"""

from .constants import GATE_CALMAR_MIN, GATE_TRADES_MIN, GATE_AUC_DIVERGENCE_MAX


def evaluate(bt_result, script_text=None):
    """Evaluate a single backtest result against all 4 gates.

    Args:
        bt_result: dict from read_backtest() or submit_and_wait()
        script_text: optional, the pipeline script text for lookahead audit

    Returns:
        dict with per-gate pass/fail and overall verdict
    """
    gates = {}

    # G0: Completion
    status = bt_result.get("status", "?")
    gates["g0_completed"] = {
        "pass": status.startswith("Completed"),
        "value": status,
        "detail": f"Status: {status}"
    }

    if not gates["g0_completed"]["pass"]:
        return _verdict(gates, "timeout")

    # Parse statistics
    s = bt_result.get("statistics", {}) or {}
    rt = bt_result.get("runtimeStatistics", {}) or {}

    # G1: Calmar OOS
    try:
        cagr_str = s.get("Compounding Annual Return", "0%")
        cagr = float(cagr_str.replace("%", ""))
        mdd_str = s.get("Drawdown", "0%")
        mdd = float(mdd_str.replace("%", ""))
        calmar = cagr / mdd if abs(mdd) > 0.01 else 0.0
    except (ValueError, ZeroDivisionError):
        calmar = 0.0

    gates["g1_calmar"] = {
        "pass": calmar > GATE_CALMAR_MIN,
        "value": round(calmar, 4),
        "detail": f"Calmar: {calmar:.4f} (need >{GATE_CALMAR_MIN})"
    }

    # G2: Trade Count OOS
    try:
        orders_str = s.get("Total Orders", "0")
        orders = int(orders_str)
    except (ValueError, TypeError):
        orders = 0

    gates["g2_trades"] = {
        "pass": orders > GATE_TRADES_MIN,
        "value": orders,
        "detail": f"Trades: {orders} (need >{GATE_TRADES_MIN})"
    }

    # G3: Lookahead audit
    if script_text:
        audit = lookahead_audit(script_text)
        gates["g3_lookahead"] = {
            "pass": audit["pass"],
            "value": audit["violations"],
            "detail": f"Lookahead violations: {audit['violations']}"
        }
    else:
        gates["g3_lookahead"] = {
            "pass": True, "value": [], "detail": "No script text provided for audit"
        }

    # G4: Overfit detection
    try:
        train_auc_str = rt.get("train_auc", "0")
        val_auc_str = rt.get("val_auc", "0")
        train_auc = float(train_auc_str)
        val_auc = float(val_auc_str)
        divergence = abs(train_auc - val_auc)
    except (ValueError, TypeError):
        divergence = 0.0
        train_auc = val_auc = 0.0

    gates["g4_overfit"] = {
        "pass": divergence < GATE_AUC_DIVERGENCE_MAX,
        "value": round(divergence, 4),
        "detail": f"AUC divergence: {divergence:.4f} (train={train_auc:.4f}, val={val_auc:.4f}, max={GATE_AUC_DIVERGENCE_MAX})"
    }

    return _verdict(gates, "completed")


def _verdict(gates, terminal_status):
    """Compute overall verdict from gate results."""
    g0 = gates.get("g0_completed", {})
    g1 = gates.get("g1_calmar", {})
    g2 = gates.get("g2_trades", {})
    g3 = gates.get("g3_lookahead", {})
    g4 = gates.get("g4_overfit", {})

    all_pass = all([
        g0.get("pass", False),
        g1.get("pass", False),
        g2.get("pass", False),
        g3.get("pass", False),
        g4.get("pass", False),
    ])

    # Determine status
    if terminal_status != "completed":
        status = terminal_status  # timeout, crash
    elif all_pass:
        status = "keep"
    elif not g3.get("pass", True):
        status = "leak"
    elif not g4.get("pass", True):
        status = "overfit"
    else:
        status = "discard"

    return {
        "status": status,
        "all_pass": all_pass,
        "gates": gates,
        "summary": _summarize(gates, status),
    }


def _summarize(gates, status):
    """One-line summary of evaluation."""
    parts = []
    for g in ["g0_completed", "g1_calmar", "g2_trades", "g3_lookahead", "g4_overfit"]:
        if g in gates:
            icon = "✓" if gates[g]["pass"] else "✗"
            parts.append(f"{icon}{g}")
    return f"[{status.upper()}] " + " ".join(parts)


def lookahead_audit(script_text):
    """Scan pipeline script for common lookahead patterns.
    Returns dict with pass (bool) and violations (list of str).
    """
    import re
    violations = []

    # Pattern 1: Negative shift on time axis
    if ".shift(-" in script_text:
        violations.append("pandas .shift(-N) detected — likely future-data leak")

    # Pattern 2: Reversed indexing on time-indexed arrays
    rev_patterns = re.findall(r'(\w+)\[::-\d*\]', script_text)
    for match in rev_patterns:
        if match not in ('text', 's', 'x'):
            violations.append(f"potential reversed indexing: {match}[::-1]")

    # Pattern 3: Train and test masks combined with OR
    if "tr_m|te_m" in script_text or "te_m|tr_m" in script_text:
        violations.append("train and test masks combined with OR — possible leak")

    # Pattern 4: Backfill on financial data
    if ".fillna(method='bfill'" in script_text or ".bfill()" in script_text:
        if any(kw in script_text for kw in ['lr', 'lc', 'ret', 'close', 'price']):
            violations.append("backfill (bfill) on price/return data — possible future leak")

    return {
        "pass": len(violations) == 0,
        "violations": violations,
    }
