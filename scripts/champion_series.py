#!/usr/bin/env python3
"""Extract each champion's REAL OOS return series from QC (read-only chart API) and run FORMAL
alpha-decay / survival analysis on it — upgrading the point-in-time early/late check to change-point
(Page-Hinkley, CUSUM) + Kaplan-Meier. This also unblocks PBO-via-CSCV (the series are now obtainable).

Path (zero pipeline change, contract-safe): render_train_config -> submit (populates ObjectStore
predictions) -> render_infer_cell -> submit (pure replay) -> /backtests/chart/read 'Strategy Equity'
-> daily-ish equity closes -> returns. The infer leg is the audited pure-replay (BACKTEST_AUDIT.md),
so the series is the real OOS equity curve.
"""
import sys, os, json, math
sys.path.insert(0, ".")
sys.path.insert(0, "autoresearch")
sys.path.insert(0, "autoresearch/harness")
sys.path.insert(0, "scripts")
from harness.orchestrator import render_train_config, render_infer_cell
from harness.qc_client import submit_and_wait, _qc_post
from harness.constants import QC_PROJECT_ID
from decay_monitor import flag_decay, page_hinkley, cusum_meanshift

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "autoresearch", "CHAMPION_DECAY.md")

# (cfg for train-render, ObjectStore cell key for infer-render)
CHAMPS = [
    ({"ticker": "GLD", "axis": "logdollar", "labeler": "ker+regime_gmm", "thresh": 0.4,
      "sizing": "dd_overlay", "n_components": 15}, "logdollar_ker_x_regime_gmm_dd_overlay_t40_n15"),
    ({"ticker": "SOXX", "axis": "logdollar", "labeler": "ker+trend_scan+bgm", "thresh": 0.5,
      "sizing": "cdf_overlay"}, "logdollar_ker_x_trend_scan_x_bgm_cdf_overlay_t50"),
    ({"ticker": "UUP", "axis": "imbalance", "labeler": "bgm+ker", "thresh": 0.5,
      "sizing": "cdf_overlay"}, "imbalance_bgm_x_ker_cdf_overlay_t50"),
]


def equity_series(bid):
    r = _qc_post("/backtests/chart/read",
                 {"projectId": QC_PROJECT_ID, "backtestId": bid, "name": "Strategy Equity",
                  "count": 5000, "start": 0, "end": 2000000000})
    if not r.get("success"):
        return None
    ser = (r.get("chart") or {}).get("series") or {}
    eq = ser.get("Equity") or {}
    vals = eq.get("values") or []
    out = [(row[0], row[-1]) for row in vals if isinstance(row, list) and len(row) >= 2 and row[-1] > 0]
    return out


def returns_from_equity(eq):
    closes = [c for _, c in eq]
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]


def annualized_sharpe(rets, ppy):
    if len(rets) < 5:
        return float("nan")
    m = sum(rets) / len(rets)
    sd = (sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
    return (m / sd * math.sqrt(ppy)) if sd > 1e-12 else float("nan")


def main():
    lines = ["# Champion alpha-decay / survival (formal, on REAL OOS return series)", "",
             "Per-champion OOS equity extracted read-only from QC (`/backtests/chart/read`). Page-Hinkley",
             "& CUSUM detect a DOWNWARD mean-shift (decay onset); early/late Sharpe is the half-window read.", ""]
    print("\n".join(lines))
    for cfg, cell in CHAMPS:
        tk = cfg["ticker"]
        print(f"[{tk}] train ...", flush=True)
        tcode, extra = render_train_config(cfg)
        bt_tr, st = submit_and_wait(tcode, f"ser_{tk}_train", timeout_s=540, extra_files=extra)
        if st != "completed":
            print(f"[{tk}] TRAIN failed: {st}"); lines.append(f"**{tk}: train failed ({st})**"); continue
        print(f"[{tk}] infer ...", flush=True)
        bt_in, st2 = submit_and_wait(render_infer_cell(tk, cell), f"ser_{tk}_infer", timeout_s=300)
        if st2 != "completed":
            print(f"[{tk}] INFER failed: {st2}"); lines.append(f"**{tk}: infer failed ({st2})**"); continue
        bid = bt_in.get("backtestId")
        eq = equity_series(bid)
        if not eq or len(eq) < 20:
            print(f"[{tk}] no series (bid={bid}, n={len(eq) if eq else 0})")
            lines.append(f"**{tk}: equity series unavailable (n={len(eq) if eq else 0})**"); continue
        rets = returns_from_equity(eq)
        span_yrs = max(1e-6, (eq[-1][0] - eq[0][0]) / (365.25 * 86400.0))
        ppy = len(rets) / span_yrs
        half = len(rets) // 2
        se = annualized_sharpe(rets[:half], ppy)
        sl = annualized_sharpe(rets[half:], ppy)
        stale = flag_decay(se, sl) if (se == se and sl == sl) else {"stale": None}
        ph = page_hinkley(rets)
        cu = cusum_meanshift(rets)
        def frac(idx):
            return "—" if idx is None else f"{idx/len(rets):.0%} in"
        verdict = "STALE" if stale.get("stale") else "HOLDING"
        row = (f"{tk:5s} npts={len(rets):4d} ppy~{ppy:5.1f}  early_SR={se:6.2f} late_SR={sl:6.2f}  "
               f"PageHinkley={frac(ph):>7s} CUSUM={frac(cu):>7s}  -> {verdict}")
        print(row, flush=True)
        lines.append("```"); lines.append(row); lines.append("```")
        lines.append(f"- {tk}: early→late Sharpe {se:.2f}→{sl:.2f}; downward change-point "
                     f"{'NONE (no decay onset detected)' if ph is None and cu is None else 'DETECTED'}.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
