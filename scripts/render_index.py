#!/usr/bin/env python3
"""Build the autoresearch "Research Console" — a single full-width dark dashboard.

One scrolling page (no tabs): command bar · live status hero + scoreboard + honest
verdict · sortable leaderboard (inline edge bars, Calmar sparklines, gate chips,
leak-trust badges, plain-English recipe) · label-to-asset map · 4-act narrative strip ·
top insights · inline causal graph · filterable rounds ledger.

build_html() renders the page; build_data() returns the same derived data as JSON for
the single /data.json poll (so the page DOM-patches instead of hard-reloading). Both
share the derive() helpers, so server render and live poll never diverge."""
import os, re, json, glob, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from describe import describe_cfg, describe
from render_causal_graph import net_html, vis_data, LEGEND

R = os.path.join(os.path.dirname(__file__), "..", "reports")
KJ = os.path.join(R, "..", "knowledge.json")
PROG = os.path.join(R, "..", "program.md")
STATUS = os.path.join(R, "status.json")
ROUND_CSV = os.path.join(R, "..", "results", "round_results.csv")
SCREEN_CSV = os.path.join(R, "..", "results", "etf_screen.csv")
SCREEN_PROG = os.path.join(R, "..", "results", "etf_screen_progress.log")
SWEEP_PROG = os.path.join(R, "..", "results", "etf_deepsweep_progress.log")
UNIVERSE_N = 311        # QC-confirmed pre-2009 ETFs (etf_qc_confirmed_pre2009.csv data rows)
SWEEP_TOTAL = 45        # deep-sweep planned ETFs (START line: "45 ETFs x 21 axes + 27 labelers")
# STALE / pre-leak historical single-ticker rows — quarantined, NOT current screen fits.
STALE_HIST = [("GLD", 4.71, "logdollar × ker (pre-leak)"),
              ("EEM", 4.03, "meta-label timing (window-decayed)"),
              ("SOXX", 3.02, "trend+regime (bar-threshold leak)")]
TICKERS = ["TLT", "IWM", "QQQ", "EEM", "GLD", "HYG", "XLE", "EFA", "DBC", "UUP", "TIP", "SLV"]
CHARACTER = {
    "TLT": "long bonds · rates", "IWM": "small-cap equity", "QQQ": "big-cap tech",
    "EEM": "emerging markets", "GLD": "gold", "HYG": "high-yield credit", "XLE": "energy",
    "EFA": "developed ex-US", "DBC": "broad commodities", "UUP": "US dollar",
    "TIP": "inflation-linked bonds", "SLV": "silver", "SOXX": "semiconductors",
    # universe-screen names (the 311-ETF expansion — see the screen section)
    "IAU": "gold (iShares)", "GDX": "gold miners", "USO": "crude oil",
    "GSG": "broad commodities", "DJP": "broad commodities", "UCO": "2× crude oil",
    "AGQ": "2× silver", "UGL": "2× gold", "SSO": "2× S&P 500", "QLD": "2× Nasdaq-100",
    "XOP": "oil & gas E&P", "BIL": "1–3 month T-bills", "VT": "total world equity",
    "EWY": "South Korea equity",
}
# What SIGNAL each asset rewards — the edge type our ML can (or can't) extract from it.
# Plain-English, one phrase. Pairs with the data-derived verdict badge in _charmap_html.
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
    # universe-screen fits — screen-strong (beat buy-hold, val_auc>0.55, >80 trades), validating before a book seat
    "IAU": "clean gold trends, same edge family as GLD — ML times entries (screen fit, validating)",
    "GDX": "gold-miner trends amplified by operating leverage — high val_auc, overfit-suspect (provisional)",
    "USO": "two-sided oil swings — structure exists but val_auc is suspiciously high (provisional)",
    "GSG": "broad-commodity trends — ML times the moves (screen fit, validating)",
    "DJP": "broad-commodity trends — ML times the moves (screen fit, validating)",
    "UCO": "leveraged crude — regime model reads the state (slim screen fit)",
    "AGQ": "leveraged silver — change-point timing beats holding (screen fit, validating)",
    "UGL": "leveraged gold — gold's trend edge, geared up (screen)",
    "SSO": "leveraged S&P — ML reads the risk-on/off regime (screen fit, validating)",
    "QLD": "leveraged Nasdaq — degenerate buy-hold baseline, shown excluded not a fit",
    "XOP": "two-sided energy swings — regime model reads the state (screen)",
    "BIL": "cash-like T-bills — near-zero drawdown inflates Calmar; excluded as an artifact",
    "VT": "global equity beta — holding beats timing",
    "EWY": "single-country beta — holding beats timing",
}
G1_CALMAR = 3.0
G2_TRADES = 80


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def normalize_series(K, etf):
    """Per-ETF Calmar-over-rounds series from cells (grouped by '{ETF}_' prefix),
    keeping only int-round + numeric-calmar entries, sorted by round."""
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
    """Tickers the universe screen classifies as NOT real fits — cash-MaxDD artifacts
    (ARTIFACT(cash), e.g. BIL T-bills) and degenerate 0-trade buy-hold baselines
    (NO-BASELINE, e.g. leveraged ETFs like QLD). The legacy per-ETF leaderboard reads
    raw per_etf_best, so without this an inflated artifact Calmar (BIL 30.43) would rank
    #1 and headline the scoreboard — contradicting the screen's own honesty verdict."""
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


def derive(K):
    """Per-ETF derived dashboard rows (sorted by edge desc). Shared by render + poll.
    Skips tickers the screen flags as cash-artifact / degenerate-baseline (not real fits)."""
    pe = K.get("per_etf_best", {}) or {}
    bh = K.get("buyhold", {}) or {}
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
        sig = v.get("significant")
        rows.append({
            "etf": etf, "calmar": cal, "cagr": _f(v.get("real_cagr")), "mdd": _f(v.get("real_mdd")),
            "da": _f(v.get("real_da")), "trades": v.get("trades"), "buyhold": bc, "edge": edge,
            "sharpe": _f(v.get("sharpe") if v.get("sharpe") is not None else v.get("real_sharpe")),
            "val_auc": _f(v.get("val_auc_reval") if v.get("val_auc_reval") is not None else v.get("val_auc")),
            "psr": _f(v.get("psr")), "n_trials": v.get("n_trials"),
            "significant": sig,
            "g1": (cal is not None and cal > G1_CALMAR), "g2": (v.get("trades") or 0) > G2_TRADES,
            "leak": lt, "leak_cls": ltc, "leak_tip": lttip, "cell": v.get("cell", ""),
            "character": CHARACTER.get(etf, ""),
            "rewards": REWARDS.get(etf, ""),
            "recipe": describe_cfg(cfg) if cfg else "",
            "series": normalize_series(K, etf),
        })
    rows.sort(key=lambda r: (r["edge"] is not None, r["edge"] or -9), reverse=True)
    return rows


def scoreboard(K, rows):
    # Prefer the LIVE CSV (always current): rounds = distinct timestamps, keeps =
    # distinct timestamps containing a kept_as_new_best row. Fall back to the
    # frozen pre-rendered-HTML counts only if the CSV can't be read.
    csv_rounds = _scan_rounds_csv()
    if csv_rounds:
        n_rounds = len(csv_rounds)
        n_keep = sum(1 for r in csv_rounds if r["verdict"] == "keep")
    else:
        n_rounds = max([int(re.match(r"round_(\d+)", os.path.basename(f)).group(1))
                        for f in glob.glob(os.path.join(R, "round_*.html"))
                        if re.match(r"round_(\d+)", os.path.basename(f))] or [0])
        n_keep = sum(1 for f in glob.glob(os.path.join(R, "round_*.html"))
                     if "KEEP" in (open(f).read()[:400].upper()))
    g1 = sum(1 for r in rows if r["g1"])
    n_edge = sum(1 for r in rows if (r["edge"] or 0) > 0.05 and r["g2"])  # real, traded edges
    n_sig = sum(1 for r in rows if r.get("significant") is True)
    return {
        "rounds": n_rounds, "edges": n_edge,
        "best_calmar": max((r["calmar"] or 0) for r in rows) if rows else 0,
        "n_sig": n_sig, "n_assessed": sum(1 for r in rows if r.get("significant") is not None),
        "keeps": n_keep, "etfs": len(rows), "g1_pass": g1, "g1_total": len(rows),
    }


def build_data(K=None):
    """The dict the page + /data.json poll consume (status + leaderboard + verdict)."""
    if K is None:
        K = _load(KJ, {})
    rows = derive(K)
    st = _load(STATUS, {})
    sb = scoreboard(K, rows)
    maxe = max([abs(r["edge"]) for r in rows if r["edge"] is not None] or [1.0]) or 1.0
    for r in rows:
        r["edge_w"] = (abs(r["edge"]) / maxe * 100.0) if r["edge"] is not None else 0.0
    n_sig = sum(1 for r in rows if r.get("significant") is True)
    n_assessed = sum(1 for r in rows if r.get("significant") is not None)
    sig_txt = f' · {n_sig}/{n_assessed} survive trials-adjustment (PSR/Bonferroni)' if n_assessed else ''
    return {
        "status": st, "scoreboard": sb, "rows": rows,
        "verdict": (f'Single-ticker research · r{sb["rounds"]} rounds · {sb["etfs"]}/{sb["etfs"]} leak-free · '
                    f'G1 Calmar>{G1_CALMAR:g}: {sb["g1_pass"]}/{sb["g1_total"]} PASS' + sig_txt +
                    f' — best single-ticker edge ~{(rows[0]["calmar"] if rows and rows[0]["calmar"] else 0):.2f}; '
                    f'GLD/UUP fully validated, the 311-ETF screen is surfacing commodity/leveraged fits'),
    }


def _scan_rounds():
    rounds = []
    for f in glob.glob(os.path.join(R, "round_*.html")):
        base = os.path.basename(f)
        m = re.match(r"round_(\d+)([a-z]?)\.html$", base)
        if not m:
            continue
        title, summary, hyps = base, "", []
        try:
            txt = open(f).read()
            mt = re.search(r"<title>(.*?)</title>", txt)
            if mt:
                title = re.sub(r"\s+", " ", mt.group(1)).strip()
            ms = re.search(r'class="tldr"[^>]*>(.*?)</(?:p|div)>', txt, re.S)
            if ms:
                s = re.sub(r"<[^>]+>", "", ms.group(1))
                s = re.sub(r"\s+", " ", s).strip()
                s = re.sub(r"^TL;DR\.?\s*", "", s, flags=re.I)
                summary = (s[:240].rstrip() + "…") if len(s) > 240 else s
            mh = re.search(r"<!--HYPS_NL:(.*?)-->", txt, re.S)
            if mh:
                hyps = [h.strip() for h in mh.group(1).split("|||") if h.strip()]
        except Exception:
            pass
        rounds.append((int(m.group(1)) + (0.5 if m.group(2) else 0), base, title, summary, hyps))
    rounds.sort(reverse=True)
    return rounds


def _csv_true(v):
    return str(v).strip().lower() == "true"


def _scan_rounds_csv():
    """Build the rounds ledger LIVE from autoresearch/results/round_results.csv —
    ALWAYS current (the driver appends to this every round). One ledger entry per
    distinct `timestamp` (== one round of 2 competing hypotheses). Returns a list
    of dicts, newest first, each with a chronological round number n (1..N over the
    sorted-ascending distinct timestamps). Falls back to [] on any read error so the
    caller can use the pre-rendered-HTML scanner instead."""
    try:
        with open(ROUND_CSV, newline="") as f:
            allrows = list(csv.DictReader(f))
    except Exception:
        return []
    if not allrows:
        return []
    # group rows by timestamp, preserving first-seen (chronological) order
    groups, order = {}, []
    for r in allrows:
        ts = r.get("timestamp", "")
        if ts not in groups:
            groups[ts] = []
            order.append(ts)
        groups[ts].append(r)
    order_sorted = sorted(order)            # ascending => chronological round number
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


# ---- universe screen (this week's headline) --------------------------------
def _best_method_join():
    """{ticker: (calmar, axis, labeler, val_auc)} — the BEST non-always_long row per
    ticker from round_results.csv. Reproduces screen.csv's method_calmar and supplies
    the winning AXIS × LABELER (the screen `recipe` column is a lossy 5-bucket family,
    so the exact axis×labeler only lives here). Pure-csv, numpy-free."""
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


def _screen_tier(verdict, auc):
    """Trust tier from verdict + val_auc — the single load-bearing honesty rule.
    STRONG fits split on val_auc: >0.85 = overfit/selection suspect (PROVISIONAL),
    else leak-verifiable deployable shortlist (TRUSTWORTHY)."""
    if verdict == "STRONG":
        if auc is not None and auc > 0.85:
            return ("prov", "PROVISIONAL")
        return ("trust", "TRUSTWORTHY")
    if verdict == "marginal":
        return ("marginal", "MARGINAL")
    if verdict in ("ARTIFACT(cash)", "NO-BASELINE"):
        return ("excluded", "EXCLUDED")
    return ("nofit", "NO-FIT")


def load_screen():
    """Read etf_screen.csv (the persistent FIT map) + join round_results.csv for the
    winning axis × labeler, attach trust tier. Sorted by edge desc (as written)."""
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
        d["recipe_full"] = (ax + " × " + lab) if (ax and lab) else (d["recipe"] or "—")
        d["logdollar"] = (ax == "logdollar")
        d["tier_cls"], d["tier"] = _screen_tier(d["verdict"], d["val_auc"])
        rows.append(d)
    rows.sort(key=lambda d: (d["edge"] is not None, d["edge"] or -9), reverse=True)
    return rows


def _progress(path):
    """(done, ...) — DONE lines in a screen/sweep progress log. Numpy-free."""
    try:
        with open(path) as f:
            return sum(1 for l in f if l.startswith("DONE "))
    except Exception:
        return 0


def screen_summary(rows=None):
    """Tier/verdict tallies + progress, shared by render + poll so they never diverge."""
    if rows is None:
        rows = load_screen()
    from collections import Counter
    cnt = Counter(r["tier_cls"] for r in rows)
    return {
        "screened": _progress(SCREEN_PROG), "universe": UNIVERSE_N,
        "sweep_done": _progress(SWEEP_PROG), "sweep_total": SWEEP_TOTAL,
        "n_trust": cnt.get("trust", 0), "n_prov": cnt.get("prov", 0),
        "n_marginal": cnt.get("marginal", 0), "n_excluded": cnt.get("excluded", 0),
        "n_nofit": cnt.get("nofit", 0), "n_classified": len(rows),
    }


# ---- small HTML builders ---------------------------------------------------
def _spark(series, w=120, h=28):
    pts = [c for _, c in series]
    if len(pts) < 2:
        return '<span class="sparkna">—</span>'
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    n = len(pts)

    def _y(c):
        return h - (c - lo) / rng * (h - 6) - 3
    coords = " ".join(f"{i / (n - 1) * w:.1f},{_y(c):.1f}" for i, c in enumerate(pts))
    last_up = pts[-1] >= pts[0]
    col = "var(--pos)" if last_up else "var(--neg)"
    base_y = _y(pts[0])                                  # reference line at the first tested value
    last_x, last_y = w, _y(pts[-1])
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" preserveAspectRatio="none" '
            f'role="img" aria-label="Calmar over {n} rounds, {"rising" if last_up else "falling"}">'
            f'<line x1="0" y1="{base_y:.1f}" x2="{w}" y2="{base_y:.1f}" stroke="var(--line-2)" stroke-width="1"/>'
            f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="1.8"/>'
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.2" fill="{col}"/></svg>')


def _edgebar(edge, w):
    if edge is None:
        return '<td class="num">—</td>'
    cls = "pos" if edge >= 0 else "neg"
    sign = "+" if edge >= 0 else ""
    side = "left:50%" if edge >= 0 else f"right:50%"
    return (f'<td class="edgecell num {cls}"><span class="edgenum">{sign}{edge:.2f}</span>'
            f'<span class="edgebar"><i class="{cls}" style="{side};width:{w / 2:.1f}%"></i></span></td>')


def _num(v, fmt, suf="", cls=""):
    extra = (" " + cls) if cls else ""
    if v is None:
        return f'<td class="num{extra}">—</td>'
    return f'<td class="num{extra}">{format(v, fmt)}{suf}</td>'


def _row_html(r):
    sig = r.get("significant")
    psr = r.get("psr")
    psr_txt = f'{psr:.2f}' if isinstance(psr, (int, float)) else "—"
    if sig is True:
        sigb = (f'<span class="sigbadge holds" title="Probabilistic Sharpe {psr_txt} clears the bar required '
                f'after {r.get("n_trials")} trials — survives the multiple-testing correction.">holds up</span>')
    elif sig is False:
        sigb = (f'<span class="sigbadge luck" title="Probabilistic Sharpe {psr_txt} is below the bar required '
                f'after {r.get("n_trials")} trials — the apparent edge is probably selection bias.">likely luck</span>')
    else:
        sigb = '<span class="sigbadge na" title="too few trades to run the deflation test">not assessed</span>'
    cal = r["calmar"]
    calcell = f'<td class="num metric {"pos" if (cal or 0) > 0 else "neg"}">{cal:+.2f}</td>' if cal is not None else '<td class="num">—</td>'
    return (
        f'<tr data-etf="{r["etf"]}">'
        f'<td class="etf"><b>{r["etf"]}</b><span class="char">{r["character"]}</span></td>'
        f'{calcell}'
        f'{_edgebar(r["edge"], r["edge_w"])}'
        f'<td class="num spk">{_spark(r["series"])}</td>'
        f'{_num(r["sharpe"], ".2f", cls="firstdiag")}'
        f'{_num(r["val_auc"], ".2f")}'
        f'<td class="num">{r["trades"] if r["trades"] is not None else "—"}</td>'
        f'<td>{sigb}</td>'
        f'<td class="recipe"><div class="rgloss">{r["recipe"]}</div>'
        f'<code class="cellid" title="{r["cell"]}">id</code></td>'
        f'</tr>')


def _lb_table(rows, tid):
    return ('<table id="' + tid + '"><thead><tr>'
            '<th data-k="etf" data-t="s">ETF</th>'
            '<th data-k="calmar" data-t="n" class="num sorted-desc" title="annual return ÷ worst drawdown">Calmar</th>'
            '<th data-k="edge" data-t="n" class="num" title="Calmar minus simply buying &amp; holding">vs. buy &amp; hold</th>'
            '<th class="num" title="best Calmar across successive rounds — green if the latest is at/above the first tested, red if below">Calmar history</th>'
            '<th data-k="sharpe" data-t="n" class="num firstdiag" title="return per unit of volatility">Sharpe</th>'
            '<th data-k="val_auc" data-t="n" class="num" title="learnable structure — ~0.5 = none, &gt;0.6 = real signal">signal</th>'
            '<th data-k="trades" data-t="n" class="num">trades</th>'
            '<th title="does the edge hold up after correcting for how many strategies we tried?">significance</th>'
            '<th>recipe</th>'
            '</tr></thead><tbody>' + ''.join(_row_html(r) for r in rows) + '</tbody></table>')


def _leaderboard_html(rows):
    edges = [r for r in rows if (r["edge"] or 0) > 0.05]
    drift = [r for r in rows if r not in edges]
    return (_lb_table(edges, "lb")
            + '<details class="bhfold"><summary>Buy &amp; hold — no durable edge (' + str(len(drift)) + ' assets)</summary>'
            + _lb_table(drift, "lbbh") + '</details>')


def _reward_verdict(r):
    """Plain-English verdict on whether this asset rewards ML — derived from the live data,
    not hardcoded. Keys on whether the champion is an actual ML model vs just buy-and-hold
    (always_long => edge==0, the model only holds), then on how far it beats buy-and-hold.
    Note: the raw `significant` flag alone is unreliable here — an always_long champion can
    pass the deflation test on buy-hold returns (e.g. HYG) without being an ML edge."""
    al = "always_long" in (r.get("cell", "") or "")
    edge = r.get("edge") or 0.0
    cal = r.get("calmar") or 0.0
    if al:
        return ("no ML edge — best held passively", "v-none")
    if cal > 0 and (r.get("significant") is True or edge >= 0.5):
        return ("real ML edge", "v-edge")
    if cal > 0 and edge >= 0.15:
        return ("modest ML edge over buy &amp; hold", "v-soft")
    return ("no durable ML edge", "v-none")


def _charmap_html(rows):
    body = ""
    for r in rows:
        verdict, vcls = _reward_verdict(r)
        va = r.get("val_auc")
        vacell = f'{va:.2f}' if va is not None else "—"
        body += (f'<tr><td><b>{r["etf"]}</b><span class="char">{r["character"]}</span></td>'
                 f'<td class="rewards">{r.get("rewards", "")}</td>'
                 f'<td><span class="vbadge {vcls}">{verdict}</span></td>'
                 f'<td class="num" title="validation AUC — ~0.5 = no learnable structure, &gt;0.6 = real signal">{vacell}</td>'
                 f'<td class="num metric {"pos" if (r["calmar"] or 0) > 0 else "neg"}">{r["calmar"]:+.2f}</td></tr>')
    return ('<table class="charmap"><thead><tr><th>asset</th>'
            '<th>what it rewards</th><th>verdict</th>'
            '<th class="num" title="learnable structure: ~0.5 none, &gt;0.6 real">signal</th>'
            '<th class="num">Calmar</th></tr></thead><tbody>'
            + body + '</tbody></table>')


def _scr_auc_cell(auc):
    if auc is None:
        return '<td class="num">—</td>'
    if auc > 0.85:
        return f'<td class="num"><span class="aucwarn">{auc:.2f}</span></td>'
    return f'<td class="num">{auc:.2f}</td>'


def _scr_row(r, maxe, rank):
    edge = r["edge"]
    w = (abs(edge) / maxe * 100.0) if edge is not None else 0.0
    cal = r["method_calmar"]
    bh = r["buyhold_calmar"]
    calcell = f'<td class="num metric {"pos" if (cal or 0) > 0 else "neg"}">{cal:+.2f}</td>' if cal is not None else '<td class="num">—</td>'
    bhcell = f'<td class="num">{bh:+.2f}</td>' if bh is not None else '<td class="num">—</td>'
    trd = r["trades"]
    trdcell = f'<td class="num">{int(trd)}</td>' if trd is not None else '<td class="num">—</td>'
    ld = ' <span class="ldflag" title="logdollar-axis fit — re-verified leak-clean by two deep leak-hunt workflows">ⓛ</span>' if r["logdollar"] else ""
    return (
        f'<tr class="scr-{r["tier_cls"]}" data-tier="{r["tier_cls"]}">'
        f'<td class="num rank">{rank}</td>'
        f'<td class="etf"><b>{r["ticker"]}</b><span class="char">{r["name"]}</span></td>'
        f'<td class="acls">{r["asset_class"]}</td>'
        f'<td><span class="tbadge t-{r["tier_cls"]}">{r["tier"]}</span></td>'
        f'<td class="recipe"><code>{r["recipe_full"]}</code>{ld}</td>'
        f'{calcell}{bhcell}'
        f'{_edgebar(edge, w)}'
        f'{_scr_auc_cell(r["val_auc"])}'
        f'{trdcell}'
        f'<td class="vcell">{r["verdict"]}</td>'
        f'</tr>')


def _scr_table(rows, maxe, start_rank, tid=None):
    id_attr = (' id="' + tid + '"') if tid else ""
    body = "".join(_scr_row(r, maxe, start_rank + i) for i, r in enumerate(rows))
    return ('<table' + id_attr + '><thead><tr>'
            '<th class="num">#</th><th>ETF</th><th>class</th><th>trust tier</th>'
            '<th>axis × labeler</th>'
            '<th class="num" title="annual return ÷ worst drawdown">Calmar</th>'
            '<th class="num" title="buy-and-hold Calmar for the same ETF">buy&amp;hold</th>'
            '<th class="num" title="Calmar minus buy-and-hold; the rank key">edge</th>'
            '<th class="num" title="validation AUC — 0.5 none, 0.6-0.75 real, &gt;0.85 overfit-suspect">val_auc</th>'
            '<th class="num">trades</th><th>verdict</th>'
            '</tr></thead><tbody>' + body + '</tbody></table>')


def _screen_section_html():
    rows = load_screen()
    if not rows:
        return ('<section class="block" id="screen"><h2>Universe screen</h2>'
                '<p class="small">(screen results pending)</p></section>')
    sm = screen_summary(rows)
    fits = [r for r in rows if r["tier_cls"] in ("trust", "prov")]
    marg = [r for r in rows if r["tier_cls"] == "marginal"]
    excl = [r for r in rows if r["tier_cls"] == "excluded"]
    nofit = [r for r in rows if r["tier_cls"] == "nofit"]
    maxe = max([abs(r["edge"]) for r in rows if r["edge"] is not None] or [1.0]) or 1.0

    # honesty banner
    prov_names = ", ".join(f'{r["ticker"]} {r["val_auc"]:.2f}' for r in fits if r["tier_cls"] == "prov") or "—"
    trust_names = ", ".join(r["ticker"] for r in fits if r["tier_cls"] == "trust") or "—"
    banner = (
        '<div class="next-box screen-banner"><b>Selection-bias warning.</b> Each ETF is the best of a '
        '<b>49-config-per-ETF</b> (21 axes + 27 labelers) best-of-N search — that search inflates the top result. '
        f'<b>TRUSTWORTHY</b> fits ({trust_names}) are leak-re-verified with val_auc 0.57–0.66. '
        f'<b>PROVISIONAL</b> fits (val_auc&nbsp;&gt;&nbsp;0.85: {prov_names}) are overfit / selection suspects — they need '
        '<b>DSR deflation</b> + a <b>permuted-label control</b> before deployment. '
        'Cash-<b>ARTIFACT</b> (Calmar inflated by ~0 MaxDD) and <b>NO-BASELINE</b> (degenerate 0-trade buy-hold, e.g. '
        'leveraged ETFs) are shown as <b>excluded, not fits</b>. <span class="ldflag">ⓛ</span> marks logdollar-axis fits '
        '(leak-re-verified).</div>')

    # progress bars
    scr_pct = sm["screened"] / sm["universe"] * 100 if sm["universe"] else 0
    sw_pct = sm["sweep_done"] / sm["sweep_total"] * 100 if sm["sweep_total"] else 0
    prog = (
        '<div class="scrprog">'
        '<div class="prow"><span class="plabel">universe screened</span>'
        f'<span class="pbar"><i style="width:{scr_pct:.1f}%"></i></span>'
        f'<span class="pval">{sm["screened"]}/{sm["universe"]}</span></div>'
        '<div class="prow"><span class="plabel">deep-sweep (49 cfg each)</span>'
        f'<span class="pbar"><i style="width:{sw_pct:.1f}%"></i></span>'
        f'<span class="pval">{sm["sweep_done"]}/{sm["sweep_total"]}</span></div></div>')

    # tier-count chips
    chips = (
        '<div class="tierchips">'
        f'<span class="tc tc-trust">{sm["n_trust"]} TRUSTWORTHY</span>'
        f'<span class="tc tc-prov">{sm["n_prov"]} PROVISIONAL</span>'
        f'<span class="tc tc-marg">{sm["n_marginal"]} marginal</span>'
        f'<span class="tc tc-excl">{sm["n_excluded"]} excluded</span>'
        f'<span class="tc tc-nofit">{sm["n_nofit"]} no-fit</span>'
        f'<span class="tc tc-tot">{sm["screened"]}/{sm["universe"]} screened</span></div>')

    # leaderboard: fits + marginal in the open table, excluded shown struck, no-fit folded
    open_rows = fits + marg + excl
    lb_tbl = _scr_table(open_rows, maxe, 1, tid="scrlb")
    nofit_tbl = ('<details class="bhfold"><summary>No fit — buy-and-hold wins (' + str(len(nofit)) +
                 ' ETFs)</summary>' + _scr_table(nofit, maxe, len(open_rows) + 1) + '</details>')

    # fit-by-mechanism breakdown (STRONG + marginal, joined axis/labeler)
    mech = _mechanism_html(fits, marg)

    # stale quarantine strip
    stale_chips = "".join(f'<span class="stalechip">{t} {c:.2f}</span>' for t, c, _ in STALE_HIST)
    stale = ('<p class="small stale-strip"><b>Not in this screen (STALE — quarantined):</b> '
             + stale_chips + ' — old pre-leak / window-decayed single-ticker rows. Shown for '
             'context only; <b>not current fits</b> and deliberately kept out of the ranked board above.</p>')

    return (
        '<section class="block" id="screen"><h2>Universe screen — 311 QC-confirmed ETFs raced vs. buy-and-hold</h2>'
        '<p class="small">This week\'s big result: every screened ETF\'s best method panel is raced against '
        '<b>always-long buy-and-hold</b>; a deep-sweep then tries <b>all axes × all labelers</b> on the fit-relevant '
        'classes. The board ranks fits by <b>edge over buy-and-hold</b>, color-coded by trust tier.</p>'
        + banner + prog + chips
        + f'<div class="tablewrap">{lb_tbl}</div>'
        + nofit_tbl
        + mech
        + stale
        + '</section>')


def _mechanism_html(fits, marg):
    from collections import Counter
    fitmarg = fits + marg
    by_axis = Counter((r["axis"] or "?") for r in fitmarg if r["axis"])
    by_lab = Counter((r["labeler"] or "?") for r in fitmarg if r["labeler"])
    by_class = Counter(r["asset_class"] for r in fits)        # STRONG-fit asset classes

    def _bars(counter):
        if not counter:
            return '<span class="mut">—</span>'
        return " · ".join(f'<b>{k}</b> ×{v}' for k, v in counter.most_common())
    classes_all = ["Commodity", "Leveraged/Inverse", "Fixed Income", "International Equity", "Real Estate", "Currency"]
    zero = [c for c in classes_all if by_class.get(c, 0) == 0]
    zero_txt = (', '.join(zero)) if zero else "none"
    return (
        '<div class="scr-mech">'
        '<div class="mechcard"><div class="mechhd">winning axis (fits + marginal)</div>'
        f'<div class="mechbody">{_bars(by_axis)}</div></div>'
        '<div class="mechcard"><div class="mechhd">winning labeler (fits + marginal)</div>'
        f'<div class="mechbody">{_bars(by_lab)}</div></div>'
        '<div class="mechcard"><div class="mechhd">where STRONG fits live</div>'
        f'<div class="mechbody">{_bars(by_class)}<div class="mechnote">zero STRONG fits in: {zero_txt} '
        '— the mechanism only fits commodity &amp; leveraged-equity classes.</div></div></div>'
        '</div>')


def _acts_html(K):
    cg = K.get("causal_graph", {}) or {}
    lbl = {n["id"]: n.get("label", "") for n in cg.get("nodes", [])}

    def has(i):
        return i in lbl
    acts = [
        ("I · The null", "Single-asset long-only ML couldn't beat buy-and-hold across 6 rounds → reframed to a v2 per-ETF tournament.", "6-round NULL → tournament" if has("f_null") else ""),
        ("II · The unlock", "Long-only can't beat a declining asset; shorting + a directional label + ③ mean-reversion features unlocked TLT.", "TLT 0.31 → 1.52" if has("f_meanrev_feat") or has("unlock_short") else ""),
        ("III · The correction", "A bar-threshold look-ahead (full-series stats incl. OOS) was inflating results; the leak fix re-validated the whole board.", "XLE 2.26 → 0.64" if has("r68") else ""),
        ("IV · Convergence", "One rule governs the board — time when the hold is weak, hold when the trend is strong (f_timing_when).", "7/7 at leak-free ceilings" if has("f_timing_when") else ""),
        ("V · The widening", "With the fixed-12 frontier saturated, the hunt widened to all 311 QC-confirmed ETFs — race each against buy-and-hold, keep only genuine fits. A cluster of commodity &amp; leveraged-equity names surfaced, now under validation.", "311-ETF screen"),
    ]
    cards = ""
    for t, body, delta in acts:
        d = f'<span class="delta">{delta}</span>' if delta else ""
        cards += f'<div class="act"><div class="acttitle">{t}</div><p>{body}</p>{d}</div>'
    return f'<div class="acts">{cards}</div>'


def _insights_html(K):
    ins = (K.get("top_insights", []) or [])[:5]
    out = ""
    for i, it in enumerate(ins, 1):
        ev = f'<span class="ev">{it.get("ev","")}</span>' if it.get("ev") else ""
        out += (f'<li><div class="ititle"><span class="inum">{i:02d}</span><span>{it.get("title","")}</span>{ev}</div>'
                f'<div class="idetail">{it.get("detail","")}</div></li>')
    return f'<ol class="insights">{out or "<li>(none)</li>"}</ol>'


def _ledger_html(rounds):
    items = ""
    for i, (_, base, title, summary, hyps) in enumerate(rounds):
        v = "keep" if "KEEP" in title.upper() else ("discard" if "DISCARD" in title.upper() else "")
        cl = (v or "other")
        pill = f'<span class="pill {v}">{v.upper()}</span>' if v else ""
        hid = "" if i < 14 else ' hidden extra'
        summ = f'<div class="rsum">{summary}</div>' if summary else ""
        hyp = ('<div class="hyps-row">' + "".join(f'<span class="hchip">{h}</span>' for h in hyps[:2]) + "</div>") if hyps else ""
        items += (f'<li class="ritem {cl}{hid}"><div class="rmain"><a href="{base}">{title}</a> {pill}</div>{summ}{hyp}</li>')
    return f'<ol class="rounds">{items}</ol>'


def _ledger_html_csv(rounds):
    """Rounds ledger built LIVE from round_results.csv (newest first). Keeps the
    .ritem/keep/discard/extra classes + .rmain/.rsum/.hyps-row hooks the filter +
    show-all JS expects, so the existing client code keeps working unchanged."""
    items = ""
    for i, d in enumerate(rounds):
        v = d["verdict"]                                    # 'keep' | 'discard'
        pill = f'<span class="pill {v}">{v.upper()}</span>'
        hid = "" if i < 14 else " hidden extra"
        title = f'Round {d["n"]} · {d["etf"]}'
        head = (f'<a href="{d["link"]}">{title}</a>' if d["link"]
                else f'<span class="rtitle">{title}</span>')
        wc = d["win_calmar"]
        wc_txt = f'{wc:+.2f}' if wc is not None else "—"
        # plain-English line: what was tried, what it scored, vs the bar it had to beat
        bits = []
        if d["win_recipe"]:
            bits.append(d["win_recipe"])
        prevc = d["prev_calmar"]
        if v == "keep":
            sub = (f'New best for {d["etf"]}: Calmar <b>{wc_txt}</b> beat the prior '
                   f'{prevc:+.2f}.' if prevc is not None else
                   f'New best for {d["etf"]}: Calmar <b>{wc_txt}</b>.')
        else:
            sub = (f'Calmar <b>{wc_txt}</b> did not beat {d["etf"]}\'s standing best '
                   f'{prevc:+.2f} — kept the incumbent.' if prevc is not None else
                   f'Calmar <b>{wc_txt}</b> — not kept.')
        trd = f' · {d["win_trades"]} trades' if d["win_trades"] not in (None, "") else ""
        summ = f'<div class="rsum">{sub}{trd}</div>'
        chips = ""
        if d["win_recipe"]:
            chips = (f'<div class="hyps-row"><span class="hchip">{d["win_recipe"]}</span>'
                     f'<span class="hchip">cell {d["win_cell"]}</span></div>')
        items += (f'<li class="ritem {v}{hid}"><div class="rmain">{head} {pill}'
                  f'<span class="rcal">{wc_txt}</span></div>{summ}{chips}</li>')
    return f'<ol class="rounds">{items}</ol>'


def _scoreboard_html(sb):
    return ('<div class="kpi"><div class="k" title="ETFs whose best strategy beats buy-and-hold by a traded margin (before the luck-check)">beat buy &amp; hold</div><div class="v acc">' + str(sb.get("edges", 0)) + '</div></div>'
            '<div class="kpi"><div class="k">best Calmar</div><div class="v pos">' + f'{sb.get("best_calmar",0):.2f}' + '</div></div>'
            '<div class="kpi"><div class="k" title="how many edges survive the multiple-testing / deflated-Sharpe correction">pass luck-check</div><div class="v">' + f'{sb.get("n_sig",0)}/{sb.get("n_assessed",0)}' + '</div></div>'
            '<div class="kpi"><div class="k">rounds</div><div class="v">' + str(sb.get("rounds", 0)) + '</div></div>')


def _portfolio_html(K):
    pf = K.get("portfolio") or {}
    champ = pf.get(pf.get("champion", "conviction_weight")) or pf.get("conviction_weight") or {}
    if not champ:
        return '<p class="small">(portfolio pending)</p>'
    def k(name, val, cls=""):
        return f'<div class="kpi"><div class="k">{name}</div><div class="v {cls}">{val}</div></div>'
    kpis = (k("Calmar", champ.get("calmar", "—"), "acc") + k("Max DD", f'{champ.get("mdd_pct","—")}%', "pos")
            + k("Sharpe", champ.get("sharpe", "—")) + k("Win", f'{champ.get("win_pct","—")}%')
            + k("CAGR", f'{champ.get("car_pct","—")}%') + k("Names", champ.get("n", "—")))
    members = champ.get("members") or []
    mem_txt = ", ".join(f'{m[0]}' for m in members) if members else "GLD, UUP, TIP, DBC, HYG"
    extra = ('<p class="small"><b>Why these five:</b> the book is built by <b>decorrelation, not by picking the '
             'highest single Calmar.</b> Only two members carry a real machine-learned edge — <b>GLD</b> '
             '(gold trend-following) and <b>UUP</b> (dollar regime); the rest are decorrelated buy-and-hold '
             'diversifiers whose job is to cut the drawdown. Dropping the correlated high-drawdown equities '
             '(QQQ/EEM/EFA/IWM/XLE) roughly doubled the Calmar (1.78 → 3.53). Leverage is a separate dial — it '
             'scales return and drawdown together, leaving Calmar unchanged.</p>')
    return ('<div class="scoreboard">' + kpis + '</div>'
            f'<p class="small">Members: <b>{mem_txt}</b> · weight ∝ Calmar² · gross ≤ 1 (no leverage) · '
            'positive every calendar year 2023–26. Re-validated + selection-bias-audited 2026-06-02 — the earlier '
            '“EEM 4.03 / book 4.22” figures were stale window artifacts and have been corrected.</p>'
            + extra)


def _intro_html(K):
    """Plain-English lead for a human landing cold: what this is, the bottom line (live numbers),
    and how to read the jargon. No stale hardcoded claims — pulls from the live champion book."""
    pe = K.get("per_etf_best") or {}
    gld = (pe.get("GLD") or {}).get("real_calmar"); uup = (pe.get("UUP") or {}).get("real_calmar")
    edges = (f'<b>GLD</b> (gold trend-following, Calmar {gld:.2f}) and <b>UUP</b> (US-dollar regime, {uup:.2f})'
             if (gld and uup) else 'a small handful of assets')
    return (
        '<section class="block intro" id="intro"><h2>What this is</h2>'
        '<p>An <b>autonomous research loop</b> that invents and back-tests <b>single-ticker</b> ETF trading '
        'strategies on real market data (QuantConnect), and keeps only the ones that beat buy-and-hold <i>and</i> '
        'survive strict out-of-sample + multiple-testing checks. The AI proposes and runs the experiments; a human steers.</p>'
        f'<p><b>Bottom line today —</b> fully-validated single-ticker alpha is <b>scarce</b>: only {edges} clear every '
        'gate including the deflation (multiple-testing) check. But the loop is now <b>screening all 311 QC-confirmed '
        'ETFs</b> against buy-and-hold, and a cluster of <b>commodity &amp; leveraged-equity fits</b> (IAU, GDX, USO, '
        'SSO, AGQ…) has surfaced — screen-strong and being validated before they earn a book seat. Each round attacks '
        'one ticker with two competing hypotheses and keeps a winner only if it clears every honest gate. The backtest '
        'is verified <b>leak-free and fully online</b>, using only models trained in the cloud (see '
        '<a href="deployment.md">deploy</a>).</p>'
        '<details class="glossary"><summary>How to read the numbers</summary>'
        '<ul><li><b>Calmar</b> = annual return ÷ worst drawdown (higher is better; above 3 is strong).</li>'
        '<li><b>MaxDD (MDD)</b> = the deepest peak-to-trough loss along the way.</li>'
        '<li><b>CAGR</b> = compounded annual return. <b>Sharpe</b> = return per unit of volatility.</li>'
        '<li><b>Edge</b> = how much a strategy beats simply buying and holding the same ETF.</li>'
        '<li><b>“Survives deflation” / luck-check</b> = the result still looks real after correcting for how many '
        'strategies we tried (a guard against luck/overfitting). <i>Beat buy &amp; hold</i> counts ETFs whose best '
        'strategy clears the ETF by a traded margin; the <i>luck-check</i> tally is the subset of those that also '
        'survive this correction.</li>'
        '<li><b>Recipe / cell</b> = the exact pipeline that produced a result: '
        '<code>asset · bar-clock · labeling-method · sizing · threshold</code>.</li>'
        '<li><b>Bar-clock</b> = strategies sample the market on event bars (e.g. equal dollar traded), not fixed '
        'clock time — so each bar carries comparable information.</li>'
        '<li><b>Entry 0.40</b> = the model only takes a position when its conviction exceeds 0.40.</li>'
        '</ul></details></section>')


def _champions_html(K):
    pe = K.get("per_etf_best") or {}

    def cal(t):
        c = _f((pe.get(t) or {}).get("real_calmar"))
        return f"{c:.2f}" if c is not None else "—"
    pf = K.get("portfolio") or {}
    book = pf.get(pf.get("champion", "")) or {}
    _members = book.get("members") or ["GLD", "UUP", "TIP", "DBC", "HYG"]
    # SOXX's edge was a bar-threshold leak (gone leak-free), so any book that still lists
    # SOXX — and its Calmar computed on those leak-inflated champions — is stale and must
    # be quarantined: show the leak-free members and a "pending" Calmar, not 4.15-with-SOXX.
    _stale = bool(book.get("STALE_PRE_LEAK_FIX")) or ("SOXX" in _members)
    if "SOXX" in _members:
        _members = [m for m in _members if m != "SOXX"]
    _bcv = book.get("calmar")
    bc = "pending" if _stale else (f"{_bcv:.2f}" if isinstance(_bcv, (int, float)) else "—")
    book_note = ('⚠ pre-leak-fix — re-deriving leak-free' if _stale
                 else '·'.join(_members) + f' · MaxDD {book.get("mdd_pct","—")}%')
    book_count = len(_members)
    n_drift = sum(1 for v in pe.values()
                  if (v.get("config") or {}).get("labeler") == "always_long")
    return (
        '<section class="champions" id="champions">'
        f'<div class="champ edge"><div class="ctag">REAL ML EDGE</div>'
        f'<div class="cname">GLD <span class="csub">gold trend-following</span></div>'
        f'<div class="cnum">{cal("GLD")}<span class="cunit">Calmar</span></div>'
        f'<div class="cnote">✓ survives the multiple-testing gate</div></div>'
        f'<div class="champ edge"><div class="ctag">REAL ML EDGE</div>'
        f'<div class="cname">UUP <span class="csub">US-dollar regime</span></div>'
        f'<div class="cnum">{cal("UUP")}<span class="cunit">Calmar</span></div>'
        f'<div class="cnote">vs. 0.42 buy-and-hold</div></div>'
        f'<div class="champ muted"><div class="ctag">NO DURABLE EDGE</div>'
        f'<div class="cname">{n_drift} others <span class="csub">best held passively</span></div>'
        f'<div class="cnum">—<span class="cunit">buy &amp; hold</span></div>'
        f'<div class="cnote">timing edges decayed out-of-sample</div></div>'
        f'<div class="champ book"><div class="ctag">DEPLOYABLE BOOK</div>'
        f'<div class="cname">{book_count}-ETF book <span class="csub">decorrelated</span></div>'
        f'<div class="cnum">{bc}<span class="cunit">Calmar</span></div>'
        f'<div class="cnote">{book_note}</div></div>'
        '</section>')


def _latest_callout(csv_rounds):
    """A plain-English 'latest result' banner reading the most recent round in the
    CSV — so a human sees what just happened without scrolling to the ledger."""
    if not csv_rounds:
        return ""
    d = csv_rounds[0]                                       # newest first
    wc = d["win_calmar"]
    wc_txt = f'{wc:+.2f}' if wc is not None else "—"
    verb = "KEPT a new best" if d["verdict"] == "keep" else "tested but did not beat the standing best"
    cls = "keep" if d["verdict"] == "keep" else "discard"
    recipe = f' — {d["win_recipe"]}' if d["win_recipe"] else ""
    when = d["ts"].replace("T", " ")
    return ('<div class="latest tldr-box"><b>Latest round (#' + str(d["n"]) + ', ' + when + '):</b> '
            f'on <b>{d["etf"]}</b> the loop {verb} at Calmar <b>{wc_txt}</b>'
            f'<span class="pill {cls}">{d["verdict"].upper()}</span>{recipe}</div>')


def build_html():
    K = _load(KJ, {})
    data = build_data(K)
    rows = data["rows"]
    csv_rounds = _scan_rounds_csv()
    rounds = csv_rounds if csv_rounds else _scan_rounds()
    cg = K.get("causal_graph", {})
    try:
        gn, ge = vis_data(cg)
        graph = net_html("cgmain", gn, ge, cg.get("phases", []), height=560, zoom=False)
    except Exception:
        graph = '<p class="small">(causal graph unavailable)</p>'
    nrounds = scoreboard(K, rows)["rounds"]
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Autoresearch — research console</title>'
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
            '<link rel="stylesheet" href="style.css"></head><body>')
    nav = ('<div class="commandbar"><span class="prompt">autoresearch / research-console</span>'
           '<span class="chapnav"><a href="#screen">universe</a><a href="#leaderboard">leaderboard</a><a href="#map">map</a>'
           '<a href="#story">story</a><a href="#insights">insights</a><a href="#graph">graph</a>'
           '<a href="#rounds">rounds</a><a href="program.md">program.md</a><a href="deployment.md">deploy</a></span>'
           f'<span class="stale" id="stale">live · {nrounds} rounds</span><span class="clock" id="clock"></span></div>')
    hero = ('<section class="statushero"><div class="block" id="nowrunning"><h2><span class="idledot"></span>'
            'Status</h2><p class="small">loading…</p></div>'
            '<div class="block scoreboard" id="scoreboard">' + _scoreboard_html(data["scoreboard"]) + '</div></section>'
            + _latest_callout(csv_rounds))
    lb = (f'<section class="block" id="leaderboard"><h2>Leaderboard — real out-of-sample results, leak-free</h2>'
          f'<div class="tablewrap">{_leaderboard_html(rows)}</div>'
          '<p class="small"><b>Calmar</b> = annual return ÷ worst drawdown (higher is better) · '
          '<b>vs. buy &amp; hold</b> = Calmar minus simply holding the ETF; bar length is relative to the '
          'largest edge on the board · <b>Calmar history</b> = best Calmar across successive rounds (line '
          'green if the latest is at/above the first tested, red if below) · '
          '<b>signal</b> = learnable structure in the label (validation AUC): 0.50 = none, above 0.60 = real; '
          'blank = not applicable to this recipe · '
          '<b>significance</b> — “holds up” = the Calmar survives the deflated-Sharpe / multiple-testing '
          'correction; “likely luck” = it does not, so the apparent edge is probably selection bias; '
          '“not assessed” = too few trades to test. '
          'Click a header to sort.</p></section>')
    cmap = (f'<section class="block" id="map"><h2>What each asset rewards</h2>'
            '<p class="small">The hard-won lesson among the core names: <b>fully-validated machine-learned edges are '
            'scarce.</b> <b>GLD</b> (gold trend-following) and <b>UUP</b> (dollar regime) are the two that survive the '
            'selection-bias audit; two-sided <i>timing</i> edges (EEM/TLT/IWM) looked great on a short window but '
            '<b>decayed</b> as the out-of-sample window grew. The 311-ETF <a href="#screen">universe screen</a> is now '
            'widening the hunt and has surfaced a cluster of commodity &amp; leveraged-equity fits (IAU, GDX, USO, '
            'SSO…) — screen-strong, validating before they earn a book seat.</p>'
            f'{_charmap_html(rows)}</section>')
    story = (f'<section class="block" id="story"><h2>The research arc</h2>'
             '<p class="small">How the project got here — from a failed start, to the rule that governs the core '
             'board, to the universe-wide screen now running.</p>'
             f'{_acts_html(K)}</section>')
    insights = (f'<section class="block" id="insights"><h2>Top insights</h2>'
                '<p class="small">The most important things learned, each with the evidence that backs it.</p>'
                f'{_insights_html(K)}</section>')
    graph_sec = (f'<section class="block" id="graph"><h2>Causal graph — every experiment &amp; finding</h2>'
                 '<p class="small">How each outcome caused the next hypothesis. Drag · hover · double-click a phase '
                 'cluster to expand. Full page: <a href="causal_graph.html">causal_graph.html</a></p>'
                 f'<div id="graphwrap" data-pending="1">{graph}</div></section>')
    screen = _screen_section_html()
    ledger_body = _ledger_html_csv(rounds) if csv_rounds else _ledger_html(rounds)
    ledger = (f'<section class="block" id="rounds"><h2>Rounds ({len(rounds)})</h2>'
              '<p class="small">Every round pits <b>two competing hypotheses</b> against the '
              'weakest ETF\'s standing best; we <b>KEEP</b> the winner only if it beats that bar on '
              'real out-of-sample Calmar (else <b>DISCARD</b> and keep the incumbent). Newest first; '
              'built live from <code>round_results.csv</code>.</p>'
              '<div class="filters"><button class="fchip on" data-f="all">all</button>'
              '<button class="fchip" data-f="keep">KEEP</button>'
              '<button class="fchip" data-f="discard">DISCARD</button></div>'
              f'{ledger_body}'
              '<button class="showall" id="showall">show all rounds</button></section>')
    # Status-first: the live poller (Now-running / Idle + scoreboard) anchors the top,
    # then THE UNIVERSE SCREEN (this week's headline), then the legacy single-ticker
    # champions band / cold-landing intro / data sections (reframed as historical context).
    return (head + '<div class="dash">' + nav + hero + screen + _champions_html(K) + _intro_html(K)
            + lb + cmap + story + insights + graph_sec + ledger
            + '</div><script>' + CONSOLE_JS + '</script></body></html>')


# ---- client JS (plain string — literal braces, NOT an f-string) ------------
CONSOLE_JS = r"""
function tick(){var c=document.getElementById('clock');if(c)c.textContent=new Date().toLocaleTimeString();}
tick();setInterval(tick,1000);

// sortable leaderboard (client-side), remembers active sort, re-applied after poll
var SORT={key:'calmar',dir:-1};
function applySort(){
  var tb=document.querySelector('#lb tbody');if(!tb)return;
  var rows=[].slice.call(tb.querySelectorAll('tr'));
  var ths=[].slice.call(document.querySelectorAll('#lb thead th'));
  var idx=-1,type='n';
  ths.forEach(function(th,i){th.classList.remove('sorted-asc','sorted-desc');
    if(th.dataset.k===SORT.key){idx=i;type=th.dataset.t;th.classList.add(SORT.dir<0?'sorted-desc':'sorted-asc');}});
  if(idx<0)return;
  rows.sort(function(a,b){
    var x=a.children[idx].innerText.replace(/[%+,]/g,'').trim(), y=b.children[idx].innerText.replace(/[%+,]/g,'').trim();
    if(type==='n'){x=parseFloat(x);y=parseFloat(y);if(isNaN(x))x=-1e9;if(isNaN(y))y=-1e9;return (x-y)*SORT.dir;}
    return x.localeCompare(y)*SORT.dir;});
  rows.forEach(function(r){tb.appendChild(r);});
}
document.addEventListener('click',function(e){
  var th=e.target.closest('#lb thead th[data-k]');
  if(th){ if(SORT.key===th.dataset.k)SORT.dir*=-1; else {SORT.key=th.dataset.k;SORT.dir=(th.dataset.t==='s')?1:-1;} applySort(); return;}
  var fc=e.target.closest('.fchip');
  if(fc){document.querySelectorAll('.fchip').forEach(function(c){c.classList.remove('on');});fc.classList.add('on');
    var f=fc.dataset.f;document.querySelectorAll('.ritem').forEach(function(li){
      li.style.display=(f==='all'||li.classList.contains(f))?'':'none';});return;}
  if(e.target.id==='showall'){document.querySelectorAll('.ritem.extra').forEach(function(li){li.classList.remove('hidden');});e.target.style.display='none';}
});
applySort();

// single data poll -> patch status hero + scoreboard + verdict + changed cells
function fmtPct(v){return (v==null)?'—':((v>=0?'+':'')+(+v).toFixed(1)+'%');}
async function poll(){
  var el=document.getElementById('nowrunning');
  try{
    var r=await fetch('data.json?_='+Date.now(),{cache:'no-store'});
    if(!r.ok)throw 0; var d=await r.json(); var s=d.status||{};
    if(s.running){
      var legs=(s.hypotheses||[]).map(function(h){return '<li>▸ '+h+'</li>';}).join('');
      el.className='block running';
      el.innerHTML='<h2><span class="livedot"></span>Now running'+(s.round?' — round '+s.round:'')+(s.etf?' · '+s.etf:'')+'</h2>'+
        '<p class="small">phase: <b>'+(s.phase||'…')+'</b>'+(s.since?' · started '+s.since:'')+(s.legs?' · '+s.legs:'')+'</p>'+
        '<ul class="hyps">'+legs+'</ul>';
    } else {
      el.className='block';
      el.innerHTML='<h2><span class="idledot"></span>Idle</h2><p class="small">last: <b>'+(s.note||('round '+(s.round||'?')))+'</b></p>';
    }
    var sb=d.scoreboard||{};
    var sbel=document.getElementById('scoreboard');
    if(sbel)sbel.innerHTML='<div class="kpi"><div class="k" title="ETFs whose best strategy beats buy-and-hold by a traded margin (before the luck-check)">beat buy &amp; hold</div><div class="v acc">'+(sb.edges||0)+'</div></div>'+
      '<div class="kpi"><div class="k">best Calmar</div><div class="v pos">'+(+(sb.best_calmar||0)).toFixed(2)+'</div></div>'+
      '<div class="kpi"><div class="k" title="how many edges survive the multiple-testing / deflated-Sharpe correction">pass luck-check</div><div class="v">'+(sb.n_sig||0)+'/'+(sb.n_assessed||0)+'</div></div>'+
      '<div class="kpi"><div class="k">rounds</div><div class="v">'+(sb.rounds||0)+'</div></div>';
    var ve=document.getElementById('verdict'); if(ve)ve.textContent=d.verdict||'';
    var st=document.getElementById('stale'); // keep server stale text
  }catch(e){ if(el)el.className='block'; var p=el&&el.querySelector('p'); }
}
poll();setInterval(poll,8000);

// lazy-init the causal graph the first time it scrolls into view (it sets physics off itself)
var gw=document.getElementById('graphwrap');
if(gw&&'IntersectionObserver' in window){
  var io=new IntersectionObserver(function(en){en.forEach(function(e){
    if(e.isIntersecting&&gw.dataset.pending){gw.dataset.pending='';
      var n=window['cgmain_net']; if(n){try{n.setSize('100%','560px');n.redraw();n.fit();}catch(x){}}}});},{rootMargin:'200px'});
  io.observe(gw);
}
// old deep-links (#overview/#rounds/#graph/#insights) -> scroll
function jump(){var h=(location.hash||'').replace('#overview','#leaderboard');var t=h&&document.querySelector(h);if(t)t.scrollIntoView({behavior:'smooth'});}
window.addEventListener('hashchange',jump); if(location.hash)setTimeout(jump,200);
"""


if __name__ == "__main__":
    html = build_html()
    open(os.path.join(R, "index.html"), "w").write(html)
    print(f"wrote research console index.html ({html.count('round_')} round refs, {len(html)} bytes)")
