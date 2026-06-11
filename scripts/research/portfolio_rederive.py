#!/usr/bin/env python3
"""HONEST BOOK RE-DERIVATION. The deployed champion book (decorr_calmarsq, 2026-06-02) predates BOTH
the SOXX discovery (06-03, the most DSR-robust crown 0.959) and the honesty audits (UUP fragile).
This extracts each candidate's REAL OOS return series read-only and recomputes portfolio metrics under
several compositions/weights to answer: should SOXX be IN, and should UUP stay?

Series cached to autoresearch/results/series_cache.json (ts->equity) so re-runs are cheap.
"""
import sys, os, json, math
from lb.harness.orchestrator import render_train_config, render_infer_cell
from lb.harness.qc_client import submit_and_wait, _qc_post
from lb.harness.constants import QC_PROJECT_ID

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "results", "series_cache.json")
OUT = os.path.join(HERE, "docs", "analysis", "HONEST_AUDIT.md")

# candidate members: (cfg for train, ObjectStore cell, champion Calmar for weighting)
# Updated 2026-06-04 to CURRENT leak-free crowns: GLD trend_leg+regime_gmm+IG 4.02 (was old pre-leak
# ker+regime_gmm 4.545); SOXX REMOVED (leak-dead ~0.81, ~buy-hold); IWM trend_leg+IG 0.665 ADDED
# (provisional, decay-healthy, decorrelation candidate). UUP = gated bgm+ker 1.30 (sadf 1.85 is STALE/front-loaded).
MEMBERS = {
    "GLD":  (dict(ticker="GLD", axis="logdollar", labeler="trend_leg+regime_gmm", thresh=0.40, sizing="dd_overlay", n_components=15, rebal_band=0.03, reduce="infogain"), "logdollar_trend_leg_x_regime_gmm_dd_overlay_t40_n15_b3_ig", 4.022),
    "UUP":  (dict(ticker="UUP", axis="imbalance", labeler="bgm+ker", thresh=0.50, sizing="cdf_overlay"), "imbalance_bgm_x_ker_cdf_overlay_t50", 1.296),
    "IWM":  (dict(ticker="IWM", axis="logdollar", labeler="trend_leg", thresh=0.45, sizing="cdf_overlay", reduce="infogain"), "logdollar_trend_leg_cdf_overlay_t45_ig", 0.665),
    "HYG":  (dict(ticker="HYG", axis="logdollar", labeler="always_long", thresh=0.55, sizing="cdf_overlay"), "logdollar_always_long_cdf_overlay_t55", 1.828),
    "TIP":  (dict(ticker="TIP", axis="logdollar", labeler="always_long", thresh=0.45, sizing="cdf_overlay"), "logdollar_always_long_cdf_overlay_t45", 1.146),
    "DBC":  (dict(ticker="DBC", axis="logdollar", labeler="always_long", thresh=0.45, sizing="cdf_overlay"), "logdollar_always_long_cdf_overlay_t45", 0.912),
    # 2026-06-06: oil mean-reversion (3rd mechanism), fully validated (permute/decay/cost/DSR 0.915). __CELL__ -> latest_key (just-trained revert cell).
    "UCO":  (dict(ticker="UCO", axis="logdollar", labeler="revert", thresh=0.45, sizing="cdf_overlay"), "__CELL__", 3.506),
    "USO":  (dict(ticker="USO", axis="logdollar", labeler="revert", thresh=0.45, sizing="cdf_overlay"), "__CELL__", 2.175),
    # 2026-06-08 book-additivity test: IXP (global telecom trend) — screen's cleanest real (permute), decay-healthy,
    # non-redundant find, BUT idiosyncratic + Bonferroni-marginal. Does decorrelation outweigh fragility in the book?
    "IXP":  (dict(ticker="IXP", axis="logdollar", labeler="trend_leg+regime_gmm", thresh=0.40, sizing="dd_overlay", n_components=15, rebal_band=0.03, reduce="infogain"), "logdollar_trend_leg_x_regime_gmm_dd_overlay_t40_n15_b3_ig", 1.975),
    # 2026-06-08 equity-sleeve redundancy test: AAXJ (Asia sadf, event-concentrated) + EWL (Swiss ker, low-DA),
    # the other 2 permute-real screen finds. Do they add on top of IXP or are they redundant intl-equity?
    "AAXJ": (dict(ticker="AAXJ", axis="logdollar", labeler="sadf_explosive", thresh=0.50, sizing="cdf_overlay"), "__CELL__", 2.433),
    "EWL":  (dict(ticker="EWL", axis="logdollar", labeler="ker", thresh=0.45, sizing="dd_overlay"), "__CELL__", 1.980),
    # 2026-06-08 redundancy test: VV (US large-cap, sadf_explosive 2.05) — same event-concentration as AAXJ?
    "VV":   (dict(ticker="VV", axis="logdollar", labeler="sadf_explosive", thresh=0.50, sizing="cdf_overlay"), "__CELL__", 2.052),
    # 2026-06-08 DJP commodity-TIMING (ker 2.01, permute-real) — upgrade DBC's commodity buy-hold?
    "DJP":  (dict(ticker="DJP", axis="logdollar", labeler="ker", thresh=0.45, sizing="dd_overlay"), "__CELL__", 2.011),
}
CAL = {k: v[2] for k, v in MEMBERS.items()}


def equity_map(bid):
    r = _qc_post("/backtests/chart/read", {"projectId": QC_PROJECT_ID, "backtestId": bid,
                 "name": "Strategy Equity", "count": 5000, "start": 0, "end": 2000000000})
    if not r.get("success"):
        return {}
    vals = ((r.get("chart") or {}).get("series") or {}).get("Equity", {}).get("values") or []
    return {str(int(row[0])): float(row[-1]) for row in vals if isinstance(row, list) and len(row) >= 2 and row[-1] > 0}


def get_series(name):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if name in cache and len(cache[name]) > 50:
        return cache[name]
    cfg, cell, _ = MEMBERS[name]
    print(f"[{name}] train ...", flush=True)
    tcode, extra = render_train_config(cfg)
    _, st = submit_and_wait(tcode, f"book_{name}_train", timeout_s=540, extra_files=extra)
    if st != "completed":
        print(f"[{name}] train failed: {st}"); return None
    print(f"[{name}] infer ...", flush=True)
    bt, st2 = submit_and_wait(render_infer_cell(name, cell), f"book_{name}_infer", timeout_s=300)
    if st2 != "completed":
        print(f"[{name}] infer failed: {st2}"); return None
    em = equity_map(bt.get("backtestId"))
    if len(em) < 50:
        print(f"[{name}] series too short ({len(em)})"); return None
    cache[name] = em
    json.dump(cache, open(CACHE, "w"))
    print(f"[{name}] cached {len(em)} pts", flush=True)
    return em


def metrics(port_rets, ppy):
    eq = [1.0]
    for r in port_rets:
        eq.append(eq[-1] * (1.0 + r))
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    yrs = len(port_rets) / ppy
    cagr = eq[-1] ** (1.0 / yrs) - 1.0 if yrs > 0 and eq[-1] > 0 else 0.0
    m = sum(port_rets) / len(port_rets)
    sd = (sum((x - m) ** 2 for x in port_rets) / (len(port_rets) - 1)) ** 0.5
    sharpe = m / sd * math.sqrt(ppy) if sd > 1e-12 else 0.0
    calmar = cagr / mdd if mdd > 1e-6 else 0.0
    return dict(calmar=calmar, cagr=cagr * 100, mdd=mdd * 100, sharpe=sharpe)


def book(series, ts, names, scheme="cal2"):
    w = {n: (CAL[n] ** 2 if scheme == "cal2" else 1.0) for n in names}
    s = sum(w.values())
    w = {n: w[n] / s for n in names}
    port = []
    for i in range(1, len(ts)):
        r = 0.0
        for n in names:
            c1, c0 = series[n][ts[i]], series[n][ts[i - 1]]
            r += w[n] * (c1 / c0 - 1.0 if c0 > 0 else 0.0)
        port.append(r)
    return port, w


def main():
    series = {}
    for n in MEMBERS:
        s = get_series(n)
        if s:
            series[n] = s
    if len(series) < 4:
        print("too few series"); return
    common = sorted(set.intersection(*[set(s.keys()) for s in series.values()]), key=lambda x: int(x))
    span = (int(common[-1]) - int(common[0])) / (365.25 * 86400.0)
    ppy = len(common) / span
    print(f"\n{len(common)} common timestamps, ppy~{ppy:.1f}, members={list(series.keys())}\n", flush=True)
    avail = set(series.keys())
    COMPS = [
        ("CURRENT book (GLD/UUP/IWM/TIP/DBC/HYG, 6)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG"]),
        ("+ UCO oil-reversion 2x (7)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "UCO"]),
        ("+ USO oil-reversion 1x (7)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO"]),
        ("alpha core (GLD/UUP/IWM)", ["GLD", "UUP", "IWM"]),
        ("alpha core + USO 1x", ["GLD", "UUP", "IWM", "USO"]),
        ("alpha core + UCO 2x", ["GLD", "UUP", "IWM", "UCO"]),
        # 2026-06-08 IXP (telecom-trend) book-additivity test:
        ("+ IXP telecom-trend (7)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "IXP"]),
        ("+ USO + IXP (8)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO", "IXP"]),
        ("IXP replaces decayed IWM (6)", ["GLD", "UUP", "IXP", "TIP", "DBC", "HYG"]),
        ("+ USO + IXP + AAXJ + EWL equity sleeve (10)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO", "IXP", "AAXJ", "EWL"]),
        ("equity sleeve only (GLD/USO/IXP/AAXJ/EWL)", ["GLD", "USO", "IXP", "AAXJ", "EWL"]),
        ("sleeve(10) + VV (11) — redundant w/ AAXJ?", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO", "IXP", "AAXJ", "EWL", "VV"]),
        ("sleeve(10) + DJP (11)", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO", "IXP", "AAXJ", "EWL", "DJP"]),
        ("sleeve, DJP replaces DBC buy-hold (10)", ["GLD", "UUP", "IWM", "TIP", "DJP", "HYG", "USO", "IXP", "AAXJ", "EWL"]),
    ]
    lines = ["", "## Honest book re-derivation (real OOS series; decorr champion predates SOXX + audits)", "",
             f"Weights ∝ Calmar² (the deployed scheme), gross=1, on {len(common)}-pt common OOS grid.", "", "```",
             f"{'composition':44s} {'Calmar':>7s} {'CAGR%':>6s} {'MaxDD%':>7s} {'Sharpe':>7s}"]
    print(f"{'composition':44s} {'Calmar':>7s} {'CAGR%':>6s} {'MaxDD%':>7s} {'Sharpe':>7s}")
    for label, names in COMPS:
        names = [n for n in names if n in avail]
        if len(names) < 2:
            continue
        port, w = book(series, common, names, "cal2")
        m = metrics(port, ppy)
        row = f"{label:44s} {m['calmar']:7.3f} {m['cagr']:6.2f} {m['mdd']:7.2f} {m['sharpe']:7.3f}"
        print(row, flush=True)
        lines.append(row)
    lines.append("```")
    # 2026-06-08 SPLIT-HALF OOS stability: does the equity-sleeve lift hold in BOTH halves (robust) or one (window-fit)?
    h = len(common) // 2
    early, late = common[:h], common[h:]
    lines.append("\nSplit-half OOS stability (Calmar/Sharpe early | late):")
    lines.append("```")
    for label, names in [("6-name CURRENT", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG"]),
                         ("10-name +USO+IXP+AAXJ+EWL", ["GLD", "UUP", "IWM", "TIP", "DBC", "HYG", "USO", "IXP", "AAXJ", "EWL"]),
                         ("10-name DJP-replaces-DBC", ["GLD", "UUP", "IWM", "TIP", "DJP", "HYG", "USO", "IXP", "AAXJ", "EWL"]),
                         ("robust core GLD/USO/IXP", ["GLD", "USO", "IXP"])]:
        ns = [n for n in names if n in avail]
        pe, _ = book(series, early, ns, "cal2")
        pl, _ = book(series, late, ns, "cal2")
        me, ml = metrics(pe, ppy), metrics(pl, ppy)
        row = f"  {label:30s} early Cal={me['calmar']:6.2f} Sh={me['sharpe']:5.2f} | late Cal={ml['calmar']:6.2f} Sh={ml['sharpe']:5.2f}"
        print(row, flush=True); lines.append(row)
    lines.append("```")
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    marker = "## Honest book re-derivation"
    if marker in prev:
        prev = prev[:prev.index(marker)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
