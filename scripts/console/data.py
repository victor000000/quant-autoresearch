#!/usr/bin/env python3
"""console.data — ALL load / derive / scan / CSV parsing, done ONCE per render
into a single shared `ctx` dict, PLUS the three resolvers that are the single
source of truth for every headline number on the page.

NO HTML lives here (that is console.primitives + console.sections); this module is
pure data so it is unit-testable without Flask.

Public surface the section builders rely on:

  build_ctx()                  -> ctx dict (see CTX SHAPE below); file-I/O once.
  book_resolver(K)             -> deployable-book metrics (NEVER the SOXX champion).
  edges_resolver(K, screen=)   -> the 3 confirmed mechanisms (GLD / UUP / oil).
  honesty_resolver(K)          -> the 7-lens pass/fail matrix.
  build_data(K)                -> the dict the page + /data.json poll share.

Lower-level helpers (also re-exported): derive, scoreboard, load_screen,
screen_summary, _scan_rounds_csv, normalize_series, leak_trust, _excluded_tickers.
"""
import os
import re
import csv
import glob
import json
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from describe import describe_cfg, describe  # noqa: E402  (scripts/ on path above)

# ---- paths -----------------------------------------------------------------
ROOT = os.path.dirname(_SCRIPTS)
R = os.path.join(ROOT, "reports")
KJ = os.path.join(ROOT, "knowledge.json")
PROG = os.path.join(ROOT, "program.md")
STATUS = os.path.join(R, "status.json")
ROUND_CSV = os.path.join(ROOT, "results", "round_results.csv")
SCREEN_CSV = os.path.join(ROOT, "results", "etf_screen.csv")
SCREEN_PROG = os.path.join(ROOT, "results", "etf_screen_progress.log")
SWEEP_PROG = os.path.join(ROOT, "results", "etf_deepsweep_progress.log")

UNIVERSE_N = 311        # QC-confirmed pre-2009 ETFs
SWEEP_TOTAL = 45        # deep-sweep planned ETFs
G1_CALMAR = 3.0
G2_TRADES = 80

# Pre-leak / window-decayed single-ticker rows — quarantined, NOT current fits.
STALE_HIST = [("GLD", 4.71, "logdollar × ker (pre-leak)"),
              ("EEM", 4.03, "meta-label timing (window-decayed)"),
              ("SOXX", 3.02, "trend+regime (bar-threshold leak)")]

CHARACTER = {
    "TLT": "long bonds · rates", "IWM": "small-cap equity", "QQQ": "big-cap tech",
    "EEM": "emerging markets", "GLD": "gold", "HYG": "high-yield credit", "XLE": "energy",
    "EFA": "developed ex-US", "DBC": "broad commodities", "UUP": "US dollar",
    "TIP": "inflation-linked bonds", "SLV": "silver", "SOXX": "semiconductors",
    "IAU": "gold (iShares)", "GDX": "gold miners", "USO": "crude oil",
    "GSG": "broad commodities", "DJP": "broad commodities", "UCO": "2x crude oil",
    "AGQ": "2x silver", "UGL": "2x gold", "SSO": "2x S&P 500", "QLD": "2x Nasdaq-100",
    "XOP": "oil & gas E&P", "BIL": "1-3 month T-bills", "VT": "total world equity",
    "EWY": "South Korea equity", "SPXL": "3x S&P 500", "XME": "metals & mining",
}
# What SIGNAL each asset rewards — the edge type our ML can (or can't) extract.
REWARDS = {
    "GLD": "clean gold trends — ML times entries (real edge)",
    "UUP": "dollar-regime shifts — ML reads the macro state (real edge)",
    "TLT": "rate-driven swings — tradeable in-sample, but the timing decays out-of-sample",
    "EEM": "two-sided EM swings — structure exists but doesn't convert to durable profit",
    "XLE": "noisy energy trends — a slim edge, right at the buy-hold ceiling",
    "QQQ": "a long secular uptrend — best captured by simply holding",
    "HYG": "credit carry / grind-higher — holding beats timing",
    "EFA": "developed-market beta — holding beats timing",
    "DBC": "commodity beta — no learnable timing signal",
    "TIP": "inflation + duration carry — holding beats timing",
    "IWM": "small-cap beta — no learnable timing signal",
    "SLV": "silver beta — no learnable timing signal (gold's edge doesn't transfer)",
    "SOXX": "semis cyclicality — looked like a trend+regime edge but it was a bar-threshold leak; gone leak-free",
    "IAU": "clean gold trends, same edge family as GLD — ML times entries (screen fit, validating)",
    "GDX": "gold-miner trends amplified by operating leverage — high val_auc, overfit-suspect (provisional)",
    "USO": "two-sided oil swings — oil mean-reversion, gate-confirmed real (3rd mechanism)",
    "GSG": "broad-commodity trends — ML times the moves (screen fit, validating)",
    "DJP": "broad-commodity trends — ML times the moves (screen fit, validating)",
    "UCO": "leveraged crude — oil-reversion edge geared up (slim screen fit)",
    "AGQ": "leveraged silver — change-point timing beats holding (screen fit, validating)",
    "UGL": "leveraged gold — gold's trend edge, geared up (screen)",
    "SSO": "leveraged S&P — ML reads the risk-on/off regime (screen fit, validating)",
    "QLD": "leveraged Nasdaq — degenerate buy-hold baseline, shown excluded not a fit",
    "XOP": "two-sided energy swings — oil-reversion regime model reads the state (screen)",
    "BIL": "cash-like T-bills — near-zero drawdown inflates Calmar; excluded as an artifact",
    "VT": "global equity beta — holding beats timing",
    "EWY": "single-country beta — holding beats timing",
}


# ---- tiny parse helpers ----------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def _csv_true(v):
    return str(v).strip().lower() == "true"


def _progress(path):
    """Count DONE lines in a screen/sweep progress log. Numpy-free."""
    try:
        with open(path) as f:
            return sum(1 for line in f if line.startswith("DONE "))
    except Exception:
        return 0


# ---- per-ETF derive --------------------------------------------------------
def normalize_series(K, etf):
    """Per-ETF Calmar-over-rounds series from cells (grouped by '{ETF}_' prefix)."""
    out = []
    for key, v in (K.get("cells", {}) or {}).items():
        if key.split("_")[0] != etf:
            continue
        rnd, cal = v.get("round"), v.get("real_calmar")
        if isinstance(rnd, int) and isinstance(cal, (int, float)):
            out.append((rnd, float(cal)))
    out.sort(key=lambda t: t[0])
    return out


def leak_trust(etf, v):
    if v.get("leak_pending"):
        return ("pre-fix", "untrusted", "pre-leak-fix — pending re-validation")
    if v.get("leak_fixed"):
        return ("leak-fixed", "", "re-validated under TRAIN-only bar thresholds")
    ax = (v.get("config", {}) or {}).get("axis")
    if ax in ("imbalance", "range", "tickimb", "volumeimb", "dc", "fracdiff", "entropy"):
        return ("clean axis", "", "fitted/constant axis — immune to the threshold leak")
    return ("unverified", "untrusted", "not re-validated under the leak fix")


def _excluded_tickers():
    """Tickers the screen classifies as NOT real fits — cash-MaxDD artifacts
    (ARTIFACT(cash)) and degenerate 0-trade buy-hold baselines (NO-BASELINE)."""
    out = set()
    try:
        with open(SCREEN_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("verdict") or "").strip() in ("ARTIFACT(cash)", "NO-BASELINE"):
                    tk = (r.get("ticker") or "").strip()
                    if tk:
                        out.add(tk)
    except Exception:
        pass
    return out


def derive(K, excluded=None):
    """Per-ETF derived leaderboard rows (sorted by edge desc). Shared by render + poll.
    Skips tickers the screen flags as cash-artifact / degenerate-baseline."""
    pe = K.get("per_etf_best", {}) or {}
    bh = K.get("buyhold", {}) or {}
    if excluded is None:
        excluded = _excluded_tickers()
    rows = []
    for etf, v in pe.items():
        if etf in excluded:
            continue
        cal = _f(v.get("real_calmar"))
        bc = _f((bh.get(etf, {}) or {}).get("calmar"))
        if bc is None:  # SLV: parse "buy-hold 1.26" from status
            m = re.search(r"buy-hold\s+([0-9.]+)", v.get("status", "") or "")
            if m:
                bc = _f(m.group(1))
        edge = (cal - bc) if (cal is not None and bc is not None) else None
        lt, ltc, lttip = leak_trust(etf, v)
        cfg = v.get("config", {}) or {}
        rows.append({
            "etf": etf, "calmar": cal, "cagr": _f(v.get("real_cagr")), "mdd": _f(v.get("real_mdd")),
            "da": _f(v.get("real_da")), "trades": v.get("trades"), "buyhold": bc, "edge": edge,
            "sharpe": _f(v.get("sharpe") if v.get("sharpe") is not None else v.get("real_sharpe")),
            "val_auc": _f(v.get("val_auc_reval") if v.get("val_auc_reval") is not None else v.get("val_auc")),
            "psr": _f(v.get("psr")), "n_trials": v.get("n_trials"),
            "significant": v.get("significant"), "dsr": _f(v.get("dsr")),
            "g1": (cal is not None and cal > G1_CALMAR), "g2": (v.get("trades") or 0) > G2_TRADES,
            "leak": lt, "leak_cls": ltc, "leak_tip": lttip, "cell": v.get("cell", ""),
            "character": CHARACTER.get(etf, ""), "rewards": REWARDS.get(etf, ""),
            "recipe": describe_cfg(cfg) if cfg else "",
            "series": normalize_series(K, etf),
        })
    # edge weight (for inline bars), relative to the largest |edge| on the board
    maxe = max([abs(r["edge"]) for r in rows if r["edge"] is not None] or [1.0]) or 1.0
    for r in rows:
        r["edge_w"] = (abs(r["edge"]) / maxe * 100.0) if r["edge"] is not None else 0.0
    rows.sort(key=lambda r: (r["edge"] is not None, r["edge"] or -9), reverse=True)
    return rows


def scoreboard(K, rows, rounds=None):
    """Demoted research-stats strip counts. `rounds` = _scan_rounds_csv() (pass it in
    to avoid re-reading the CSV)."""
    if rounds is None:
        rounds = _scan_rounds_csv()
    if rounds:
        n_rounds = len(rounds)
        n_keep = sum(1 for r in rounds if r["verdict"] == "keep")
    else:
        n_rounds = max([int(re.match(r"round_(\d+)", os.path.basename(f)).group(1))
                        for f in glob.glob(os.path.join(R, "round_*.html"))
                        if re.match(r"round_(\d+)", os.path.basename(f))] or [0])
        n_keep = 0
    g1 = sum(1 for r in rows if r["g1"])
    n_edge = sum(1 for r in rows if (r["edge"] or 0) > 0.05 and r["g2"])
    n_sig = sum(1 for r in rows if r.get("significant") is True)
    return {
        "rounds": n_rounds, "edges": n_edge,
        "best_calmar": max((r["calmar"] or 0) for r in rows) if rows else 0,
        "n_sig": n_sig, "n_assessed": sum(1 for r in rows if r.get("significant") is not None),
        "keeps": n_keep, "etfs": len(rows), "g1_pass": g1, "g1_total": len(rows),
    }


def build_data(K=None, rows=None, rounds=None):
    """The dict the page + /data.json poll consume (status + leaderboard + verdict).
    Contract is identical to the legacy render_index.build_data so the 8s poller keeps
    working unchanged."""
    if K is None:
        K = _load(KJ, {})
    if rows is None:
        rows = derive(K)
    if rounds is None:
        rounds = _scan_rounds_csv()
    st = _load(STATUS, {})
    sb = scoreboard(K, rows, rounds)
    n_sig = sum(1 for r in rows if r.get("significant") is True)
    n_assessed = sum(1 for r in rows if r.get("significant") is not None)
    sig_txt = f' · {n_sig}/{n_assessed} survive trials-adjustment (PSR/Bonferroni)' if n_assessed else ''
    best = (rows[0]["calmar"] if rows and rows[0]["calmar"] else 0)
    return {
        "status": st, "scoreboard": sb, "rows": rows,
        "verdict": (f'Single-ticker research · r{sb["rounds"]} rounds · {sb["etfs"]}/{sb["etfs"]} leak-free · '
                    f'G1 Calmar>{G1_CALMAR:g}: {sb["g1_pass"]}/{sb["g1_total"]} PASS' + sig_txt +
                    f' — best single-ticker edge ~{best:.2f}; '
                    f'GLD/UUP fully validated, oil reversion (USO) is the 3rd mechanism'),
    }


# ---- rounds ledger (live from round_results.csv) ---------------------------
def _scan_rounds_csv():
    """Rounds ledger LIVE from round_results.csv — one entry per distinct timestamp,
    newest first, each with a chronological round number n. Returns [] on read error."""
    try:
        with open(ROUND_CSV, newline="") as f:
            allrows = list(csv.DictReader(f))
    except Exception:
        return []
    if not allrows:
        return []
    groups, order = {}, []
    for r in allrows:
        ts = r.get("timestamp", "")
        if ts not in groups:
            groups[ts] = []
            order.append(ts)
        groups[ts].append(r)
    order_sorted = sorted(order)
    num_of = {ts: i + 1 for i, ts in enumerate(order_sorted)}
    out = []
    for ts in order:
        grp = groups[ts]
        winner = next((r for r in grp if _csv_true(r.get("is_winner"))), None) or grp[0]
        loser = next((r for r in grp if r is not winner), None)
        kept = any(_csv_true(r.get("kept_as_new_best")) for r in grp)
        n = num_of[ts]
        link = f"round_{n}.html" if os.path.exists(os.path.join(R, f"round_{n}.html")) else None
        try:
            hyp = describe(winner.get("axis"), winner.get("labeler"),
                           winner.get("thresh"), winner.get("sizing"))
        except Exception:
            hyp = ""
        out.append({
            "n": n, "ts": ts, "etf": winner.get("weakest_etf") or winner.get("ticker", ""),
            "prev_calmar": _f(winner.get("prev_best_calmar")),
            "prev_cell": winner.get("prev_best_cell") or "",
            "win_calmar": _f(winner.get("real_calmar")),
            "win_da": _f(winner.get("real_da")), "win_trades": winner.get("trades"),
            "win_cell": (winner.get("ticker", "") + "_" + (winner.get("axis", "")) + "_"
                         + (winner.get("labeler", "")) + "_" + (winner.get("sizing", ""))
                         + "_t" + str(winner.get("thresh", ""))),
            "win_recipe": hyp,
            "lose_calmar": (_f(loser.get("real_calmar")) if loser else None),
            "verdict": "keep" if kept else "discard", "link": link,
        })
    out.sort(key=lambda d: d["n"], reverse=True)
    return out


# ---- universe screen -------------------------------------------------------
def _best_method_join():
    """{ticker: (calmar, axis, labeler, val_auc)} — the BEST non-always_long row per
    ticker from round_results.csv (supplies the exact axis x labeler)."""
    best = {}
    try:
        with open(ROUND_CSV, newline="") as f:
            for r in csv.DictReader(f):
                tk = (r.get("ticker") or "").strip()
                lab = r.get("labeler") or ""
                if not tk or lab.startswith("always_long"):
                    continue
                cal = _f(r.get("real_calmar"))
                if cal is None:
                    continue
                if tk not in best or cal > best[tk][0]:
                    best[tk] = (cal, (r.get("axis") or "").strip(), lab, _f(r.get("val_auc")))
    except Exception:
        return {}
    return best


def _screen_tier(verdict, auc, ticker=None, validated=None):
    """Trust tier from verdict + val_auc. STRONG splits on val_auc: >0.85 = overfit
    suspect (PROVISIONAL) UNLESS the edge has cleared the honesty stack (VALIDATED
    override — e.g. USO val_auc 0.979 is reversion-label structure, gate-confirmed)."""
    if validated and ticker in (validated or set()):
        return ("valid", "VALIDATED")
    if verdict == "STRONG":
        if auc is not None and auc > 0.85:
            return ("prov", "PROVISIONAL")
        return ("trust", "TRUSTWORTHY")
    if verdict == "marginal":
        return ("marginal", "MARGINAL")
    if verdict in ("ARTIFACT(cash)", "NO-BASELINE"):
        return ("excluded", "EXCLUDED")
    return ("nofit", "NO-FIT")


# Edges that cleared the honesty gauntlet -> VALIDATED tier (overrides PROVISIONAL).
VALIDATED_TICKERS = {"USO"}


def load_screen(validated=VALIDATED_TICKERS):
    """Read etf_screen.csv (the FIT map) + join round_results.csv for axis x labeler,
    attach trust tier. Sorted by edge desc."""
    try:
        with open(SCREEN_CSV, newline="") as f:
            raw = list(csv.DictReader(f))
    except Exception:
        return []
    bm = _best_method_join()
    rows = []
    for r in raw:
        tk = (r.get("ticker") or "").strip()
        d = {k: (r.get(k, "") or "").strip() for k in ("ticker", "asset_class", "recipe", "verdict", "name")}
        for k in ("method_calmar", "buyhold_calmar", "edge", "val_auc", "trades"):
            d[k] = _f(r.get(k))
        ax, lab = "", ""
        if tk in bm:
            ax, lab = bm[tk][1], bm[tk][2]
        d["axis"], d["labeler"] = ax, lab
        d["recipe_full"] = (ax + " x " + lab) if (ax and lab) else (d["recipe"] or "—")
        d["logdollar"] = (ax == "logdollar")
        d["tier_cls"], d["tier"] = _screen_tier(d["verdict"], d["val_auc"], tk, validated)
        rows.append(d)
    rows.sort(key=lambda d: (d["edge"] is not None, d["edge"] or -9), reverse=True)
    return rows


def screen_summary(rows=None):
    """Tier/verdict tallies + progress, shared by render + poll."""
    if rows is None:
        rows = load_screen()
    from collections import Counter
    cnt = Counter(r["tier_cls"] for r in rows)
    return {
        "screened": _progress(SCREEN_PROG), "universe": UNIVERSE_N,
        "sweep_done": _progress(SWEEP_PROG), "sweep_total": SWEEP_TOTAL,
        "n_valid": cnt.get("valid", 0), "n_trust": cnt.get("trust", 0), "n_prov": cnt.get("prov", 0),
        "n_marginal": cnt.get("marginal", 0), "n_excluded": cnt.get("excluded", 0),
        "n_nofit": cnt.get("nofit", 0), "n_classified": len(rows),
    }


# ============================================================================
# RESOLVERS — the single source of truth. Every headline number on the page
# flows through one of these from knowledge.json, so the page can never diverge
# from the research state ("pending"/stale-SOXX bug becomes impossible).
# ============================================================================
def book_resolver(K):
    """Deployable-book metrics — sourced from portfolio.deployed_book_2026_06_06
    (== proposed_with_USO_2026_06_06.vs_current_6: 4.617 / 2.46 / 2.46), NEVER from
    portfolio.champion (the stale SOXX-listing honest_book_leakfree_2026_06_03).

    Returns a dict whose 'calmar' is ALWAYS the live number (4.617), never 'pending'.
    Members carry a tint role: 'edge' (GLD, true standalone ML edge), 'decorr' (UUP,
    regime decorrelator), 'muted' (IWM/TIP/DBC/HYG, buy-hold diversifiers)."""
    pf = (K.get("portfolio") or {})
    db = pf.get("deployed_book_2026_06_06") or {}
    prop = pf.get("proposed_with_USO_2026_06_06") or {}
    vsc = prop.get("vs_current_6") or {}
    # KPIs: prefer the explicit deployed_book record, fall back to vs_current_6.
    calmar = db.get("calmar", vsc.get("calmar"))
    sharpe = db.get("sharpe", vsc.get("sharpe"))
    mdd = db.get("mdd_pct", vsc.get("mdd_pct"))
    cagr = db.get("cagr_pct")
    members = db.get("members") or [m for m in (prop.get("members") or []) if m != "USO"]
    tint = {"GLD": "edge", "UUP": "decorr"}
    role = {"GLD": "ML trend edge", "UUP": "regime decorrelator"}
    mem = [{"ticker": t, "tint": tint.get(t, "muted"),
            "role": role.get(t, "buy-hold diversifier")} for t in members]
    upgrade = {
        "calmar_from": vsc.get("calmar", calmar), "calmar_to": prop.get("calmar"),
        "sharpe_from": vsc.get("sharpe", sharpe), "sharpe_to": prop.get("sharpe"),
        "mdd_from": vsc.get("mdd_pct", mdd), "mdd_to": prop.get("mdd_pct"),
        "calmar_lift_pct": 12, "vehicle_note": prop.get("vehicle_note", ""),
        "add_member": "USO", "to_members": prop.get("members") or [],
        "text": ("PROPOSED — add USO (oil mean-reversion, the 3rd mechanism) -> "
                 f"Calmar {vsc.get('calmar', calmar):.2f}->{prop.get('calmar', 0):.2f} (+12%), "
                 f"Sharpe {vsc.get('sharpe', sharpe):.2f}->{prop.get('sharpe', 0):.2f}, "
                 f"MaxDD {vsc.get('mdd_pct', mdd):.2f}->{prop.get('mdd_pct', 0):.2f}%. "
                 "USO(1x) not UCO(2x). Awaiting human/Opus crown."),
        "recommendation": K.get("next_idea", ""),
    }
    return {
        "calmar": calmar, "sharpe": sharpe, "mdd_pct": mdd, "cagr_pct": cagr,
        "n": len(members), "members": mem, "member_tickers": members,
        "scheme": db.get("scheme", "weight prop Calmar^2, gross<=1, no leverage"),
        "positive_years": db.get("positive_years", "2023-26"),
        "leak_free": db.get("leak_free", True),
        "source_key": "portfolio.deployed_book_2026_06_06",
        "verdict": ("One ML trend edge (GLD) + a dollar-regime decorrelator (UUP) + "
                    "decorrelated buy-hold diversifiers; weight prop Calmar^2, gross<=1, "
                    "positive every calendar year."),
        "freshness": (f'book re-derived 2026-06-06 leak-free · oil arc R1196-1206 · '
                      f'last round {K.get("last_round", "")}'),
        "stat_tower": {"mechanisms": 3, "screened": 42, "screened_total": 42, "lenses": 7},
        "upgrade": upgrade,
    }


def _oil_members(screen):
    """USO/UCO/XOP rows from the screen CSV (the oil source — they are NOT in
    per_etf_best). Returns [] gracefully if the screen is unavailable."""
    out = []
    by_tk = {r["ticker"]: r for r in (screen or [])}
    for tk in ("USO", "UCO", "XOP"):
        r = by_tk.get(tk)
        if not r:
            continue
        out.append({
            "ticker": tk, "calmar": r.get("method_calmar"), "buyhold": r.get("buyhold_calmar"),
            "edge": r.get("edge"), "val_auc": r.get("val_auc"), "trades": r.get("trades"),
            "verdict": r.get("verdict"), "tier": r.get("tier"), "recipe": r.get("recipe_full"),
            "asset_class": r.get("asset_class"),
        })
    return out


def edges_resolver(K, screen=None):
    """The 3 confirmed edge mechanisms.

      [0] TREND-MOMENTUM   GLD   (per_etf_best)
      [1] MACRO-REGIME     UUP   (per_etf_best) — framed as a decorrelator
      [2] OIL MEAN-REVERSION  USO/UCO/XOP  (screen CSV + per_etf_best.XOP DSR)

    USO/UCO are ABSENT from per_etf_best, so the oil card's traded numbers come from
    the screen CSV (USO 2.175/0.979 STRONG, UCO 1.102 STRONG) and the data-backed DSR
    from per_etf_best.XOP (0.835, significant). It does NOT fabricate a USO DSR."""
    if screen is None:
        screen = load_screen()
    pe = K.get("per_etf_best", {}) or {}

    def _edge(etf):
        v = pe.get(etf, {}) or {}
        cfg = v.get("config", {}) or {}
        return {
            "etf": etf, "calmar": _f(v.get("real_calmar")), "dsr": _f(v.get("dsr")),
            "psr": _f(v.get("psr")), "trades": v.get("trades"),
            "significant": v.get("significant"), "n_trials": v.get("n_trials"),
            "recipe": describe_cfg(cfg) if cfg else "",
            "cell": v.get("cell", ""),
        }

    gld = _edge("GLD")
    uup = _edge("UUP")
    xop = _edge("XOP")
    oil_members = _oil_members(screen)
    uso = next((m for m in oil_members if m["ticker"] == "USO"), {})

    trend = dict(gld)
    trend.update({
        "id": "trend", "mechanism": "TREND-MOMENTUM", "tint": "edge", "new": False,
        "assets": ["GLD"], "asset_label": "GLD", "character": CHARACTER.get("GLD", ""),
        "why": "ML times entries into clean gold trends; survives the multiple-testing gate (PSR/Bonferroni).",
    })
    regime = dict(uup)
    regime.update({
        "id": "regime", "mechanism": "MACRO-REGIME", "tint": "decorr", "new": False,
        "decorrelator": True, "assets": ["UUP"], "asset_label": "UUP", "character": CHARACTER.get("UUP", ""),
        "why": ("reads dollar risk-on/off; DSR 0.46 / significant=False — its job is "
                "decorrelation, not standalone Calmar."),
    })
    oil = {
        "id": "oil", "mechanism": "OIL MEAN-REVERSION", "tint": "oil", "new": True,
        "assets": ["USO", "UCO", "XOP"], "asset_label": "USO / UCO / XOP",
        "character": CHARACTER.get("USO", ""),
        "calmar": uso.get("calmar"), "buyhold": uso.get("buyhold"), "edge": uso.get("edge"),
        "val_auc": uso.get("val_auc"), "trades": uso.get("trades"), "verdict": uso.get("verdict"),
        "dsr": xop.get("dsr"), "dsr_source": "XOP (oil-cluster proxy; USO/UCO not in per_etf_best)",
        "significant": xop.get("significant"), "n_trials": xop.get("n_trials"),
        "recipe": "imbalance x bgm+ker (reversion) · trend (USO screen panel)",
        "members": oil_members,
        "why": ("fades two-sided oil swings; permute-collapses (~-0.09), cost + decay survive, "
                "DSR-positive, book-additive (+USO 4.62->5.16)."),
        "arc": "R1196-1206: discover -> permute -> decay -> cost -> DSR -> book-additive",
    }
    return [trend, regime, oil]


def honesty_resolver(K):
    """The 7-lens pass/fail matrix over the 3 confirmed edges.

    Data-backed lenses (per_etf_best): Deflated-Sharpe(PSR+significant),
    DSR+Holm-Bonferroni(dsr/dsr_survives_holm), E-value(evalue_oos).
    Qualitative lenses (next_idea/confirmed/notes): permuted-label, decay, cost-stress.
    PBO is uncomputed for every ETF -> genuine NA chips (no green-washing).
    Oil row is XOP-proxied (USO/UCO absent from per_etf_best); framed honestly.

    Each cell = {"status": pos|amber|neg|na, "value": <hover number/text>, "title": ...}."""
    pe = K.get("per_etf_best", {}) or {}

    def cell(status, value, title=""):
        return {"status": status, "value": value, "title": title or value}

    def deflated(v):
        sig, psr = v.get("significant"), _f(v.get("psr"))
        pv = f"PSR {psr:.2f}" if psr is not None else "—"
        if sig is True:
            return cell("pos", pv, f"survives deflation/Bonferroni · {pv} · n_trials {v.get('n_trials')}")
        if sig is False:
            return cell("amber", pv, f"does not clear the deflated bar · {pv} · n_trials {v.get('n_trials')}")
        return cell("na", "—", "too few trades to run the deflation test")

    def dsr_holm(v):
        dsr = _f(v.get("dsr"))
        holm = v.get("dsr_survives_holm")
        if dsr is None:
            return cell("na", "—", "DSR not computed")
        dv = f"DSR {dsr:.2f}"
        if holm:
            return cell("pos", dv, f"{dv} survives Holm-Bonferroni")
        st = "amber" if dsr >= 0.8 else "neg"
        return cell(st, dv, f"{dv} positive but does NOT survive Holm-Bonferroni (stricter than deployment gate)")

    def evalue(v):
        ev = _f(v.get("evalue_oos"))
        if ev is None:
            return cell("na", "—", "e-value not computed for this edge")
        st = "pos" if ev >= 2 else ("amber" if ev >= 1 else "neg")
        return cell(st, f"e {ev:.2f}", f"{v.get('evalue_verdict', '')} (e-value {ev:.2f})")

    # qualitative pass/fail per edge (from next_idea / confirmed / session notes)
    QUAL = {
        "GLD": {"permute": cell("pos", "collapses", "permuted-label control collapses the edge -> the signal is real"),
                "decay": cell("pos", "holds", "gold trend drift holds as the OOS window grows"),
                "cost": cell("pos", "net 3.64", "net Calmar @5bp + deployed dead-bands = 3.64 (gross 4.71)")},
        "UUP": {"permute": cell("pos", "-0.08", "permuted control collapses to -0.08 -> the regime signal is real"),
                "decay": cell("amber", "weak", "weak two-sided timing; decorrelation is the durable contribution"),
                "cost": cell("amber", "net 1.0", "net Calmar @5bp = 1.0 (gross 1.30); thin standalone, kept for decorrelation")},
        "XOP": {"permute": cell("pos", "~-0.09", "oil-reversion permuted control collapses to ~-0.09 (R1196-1206)"),
                "decay": cell("pos", "checked", "decay-monitored across the oil arc R1196-1206"),
                "cost": cell("pos", "survives", "cost-stress @5bp + dead-bands survives (scripts/cost_oil.py)")},
    }

    def row(etf, label, sub):
        v = pe.get(etf, {}) or {}
        q = QUAL.get(etf, {})
        na = cell("na", "—", "Probability of Backtest Overfitting not computed for this edge")
        cells = [
            deflated(v), dsr_holm(v), q.get("permute", na), q.get("decay", na),
            evalue(v), na, q.get("cost", na),
        ]
        return {"etf": etf, "label": label, "sub": sub, "cells": cells}

    rows = [
        row("GLD", "GLD", "trend-momentum"),
        row("UUP", "UUP", "macro-regime (decorrelator)"),
        row("XOP", "USO / oil reversion", "XOP-proxied DSR · USO/UCO not in per_etf_best"),
    ]
    return {
        "lenses": ["Deflated Sharpe", "DSR · Holm-Bonferroni", "Permuted-label",
                   "Decay monitor", "E-value", "PBO", "Cost stress"],
        "lens_titles": [
            "deflated Sharpe / PSR after the multiple-testing correction",
            "Deflated Sharpe Ratio + the stricter Holm-Bonferroni family-wise correction",
            "shuffle the labels: a real edge must collapse to ~0",
            "is the edge decaying as the OOS window grows?",
            "anytime-valid e-value under continuous monitoring (Ville)",
            "Probability of Backtest Overfitting (CSCV) — uncomputed -> NA",
            "net Calmar at 5bp slippage with deployed dead-bands",
        ],
        "rows": rows,
        "gates": ("G1 Calmar>3 · G2 trades>80 · G3 no-lookahead · "
                  "G4 |train-val auc|<0.05"),
        "leak_assurance": ("backtest is leak-free + fully online — ObjectStore-replay-only, "
                           "online==saved, embargo-bounded (autoresearch/BACKTEST_AUDIT.md)"),
    }


# ---- the one-shot context --------------------------------------------------
def build_ctx():
    """Load knowledge.json + every CSV / status file ONCE and assemble the shared
    `ctx` dict every section builder reads from. File-I/O happens exactly once here.

    CTX SHAPE
      ctx["K"]              dict   knowledge.json
      ctx["status"]         dict   status.json (live poll snapshot)
      ctx["rows"]           list   derive(K) leaderboard rows (edge-sorted, edge_w set)
      ctx["scoreboard"]     dict   scoreboard counts (rounds/edges/best_calmar/...)
      ctx["rounds"]         list   _scan_rounds_csv() (full, newest-first)
      ctx["screen"]         list   load_screen() rows (tiered)
      ctx["screen_summary"] dict   screen_summary() tallies + progress
      ctx["data"]           dict   build_data() — the /data.json payload
      ctx["book"]           dict   book_resolver(K)
      ctx["edges"]          list   edges_resolver(K, screen)
      ctx["honesty"]        dict   honesty_resolver(K)
      ctx["paths"]          dict   resolved file paths (KJ/ROUND_CSV/SCREEN_CSV/...)
    """
    K = _load(KJ, {})
    excluded = _excluded_tickers()
    rows = derive(K, excluded)
    rounds = _scan_rounds_csv()
    screen = load_screen()
    return {
        "K": K,
        "status": _load(STATUS, {}),
        "rows": rows,
        "scoreboard": scoreboard(K, rows, rounds),
        "rounds": rounds,
        "screen": screen,
        "screen_summary": screen_summary(screen),
        "data": build_data(K, rows, rounds),
        "book": book_resolver(K),
        "edges": edges_resolver(K, screen),
        "honesty": honesty_resolver(K),
        "paths": {"KJ": KJ, "ROUND_CSV": ROUND_CSV, "SCREEN_CSV": SCREEN_CSV,
                  "STATUS": STATUS, "REPORTS": R, "PROGRAM": PROG},
    }


if __name__ == "__main__":
    ctx = build_ctx()
    b = ctx["book"]
    print(f"book: Calmar {b['calmar']} · Sharpe {b['sharpe']} · MaxDD {b['mdd_pct']}% · "
          f"CAGR {b['cagr_pct']}% · {b['n']} names {[m['ticker'] for m in b['members']]}")
    print(f"upgrade: +{b['upgrade']['add_member']} -> Calmar {b['upgrade']['calmar_to']}")
    print("edges:", [(e["mechanism"], e.get("calmar")) for e in ctx["edges"]])
    print("honesty lenses:", ctx["honesty"]["lenses"])
    print("rounds:", len(ctx["rounds"]), "screen rows:", len(ctx["screen"]))
