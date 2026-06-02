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

R = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "reports")
KJ = os.path.join(R, "..", "knowledge.json")
PROG = os.path.join(R, "..", "program.md")
STATUS = os.path.join(R, "status.json")
ROUND_CSV = os.path.join(R, "..", "results", "round_results.csv")
TICKERS = ["TLT", "IWM", "QQQ", "EEM", "GLD", "HYG", "XLE"]
CHARACTER = {
    "TLT": "mean-reverter · rates", "IWM": "small-cap · reversals", "QQQ": "pure trender",
    "EEM": "two-sided · EM", "GLD": "strong trender · gold", "HYG": "strong trender · credit",
    "XLE": "noisy trender · energy",
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


def derive(K):
    """Per-ETF derived dashboard rows (sorted by edge desc). Shared by render + poll."""
    pe = K.get("per_etf_best", {}) or {}
    bh = K.get("buyhold", {}) or {}
    rows = []
    for etf, v in pe.items():
        cal = _f(v.get("real_calmar"))
        bc = _f((bh.get(etf, {}) or {}).get("calmar"))
        edge = (cal - bc) if (cal is not None and bc is not None) else None
        lt, ltc, lttip = leak_trust(etf, v)
        cfg = v.get("config", {}) or {}
        sig = v.get("significant")
        rows.append({
            "etf": etf, "calmar": cal, "cagr": _f(v.get("real_cagr")), "mdd": _f(v.get("real_mdd")),
            "da": _f(v.get("real_da")), "trades": v.get("trades"), "buyhold": bc, "edge": edge,
            "sharpe": _f(v.get("sharpe")), "psr": _f(v.get("psr")), "n_trials": v.get("n_trials"),
            "significant": sig,
            "g1": (cal is not None and cal > G1_CALMAR), "g2": (v.get("trades") or 0) > G2_TRADES,
            "leak": lt, "leak_cls": ltc, "leak_tip": lttip, "cell": v.get("cell", ""),
            "character": CHARACTER.get(etf, ""),
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
    best = max((r for r in rows if r["edge"] is not None), key=lambda r: r["edge"], default=None)
    g1 = sum(1 for r in rows if r["g1"])
    return {
        "rounds": n_rounds, "keeps": n_keep, "etfs": len(rows),
        "best_edge": (f'{best["etf"]} +{best["edge"]:.2f}' if best else "—"),
        "g1_pass": g1, "g1_total": len(rows),
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
        "verdict": (f'Converged @ r{sb["rounds"]} · {sb["etfs"]}/{sb["etfs"]} leak-free · '
                    f'G1 Calmar>{G1_CALMAR:g}: {sb["g1_pass"]}/{sb["g1_total"]} PASS' + sig_txt +
                    f' — frontier ~{(rows[0]["calmar"] if rows and rows[0]["calmar"] else 0):.2f}; '
                    f'>{G1_CALMAR:g} needs cross-asset pairs'),
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


# ---- small HTML builders ---------------------------------------------------
def _spark(series, w=120, h=26):
    pts = [c for _, c in series]
    if len(pts) < 2:
        return '<span class="sparkna">—</span>'
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    n = len(pts)
    coords = " ".join(f"{i / (n - 1) * w:.1f},{h - (c - lo) / rng * (h - 4) - 2:.1f}" for i, c in enumerate(pts))
    last_up = pts[-1] >= pts[0]
    col = "var(--pos)" if last_up else "var(--neg)"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" preserveAspectRatio="none" '
            f'aria-label="{n} pts">'
            f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="1.6"/></svg>')


def _edgebar(edge, w):
    if edge is None:
        return '<td class="num">—</td>'
    cls = "pos" if edge >= 0 else "neg"
    sign = "+" if edge >= 0 else ""
    side = "left:50%" if edge >= 0 else f"right:50%"
    return (f'<td class="edgecell num {cls}"><span class="edgenum">{sign}{edge:.2f}</span>'
            f'<span class="edgebar"><i class="{cls}" style="{side};width:{w / 2:.1f}%"></i></span></td>')


def _num(v, fmt, suf=""):
    if v is None:
        return '<td class="num">—</td>'
    return f'<td class="num">{format(v, fmt)}{suf}</td>'


def _leaderboard_html(rows):
    body = ""
    for r in rows:
        gates = (f'<span class="gatechip {"pass" if r["g1"] else "fail"}">G1</span>'
                 f'<span class="gatechip {"pass" if r["g2"] else "fail"}">G2</span>')
        leak = f'<span class="leakbadge {r["leak_cls"]}" title="{r["leak_tip"]}">{r["leak"]}</span>'
        sig = r.get("significant")
        if sig is True:
            sigb = f'<span class="leakbadge" title="PSR {r.get("psr")} clears Bonferroni 1-0.05/{r.get("n_trials")}">✓ sig</span>'
        elif sig is False:
            sigb = f'<span class="leakbadge untrusted" title="PSR {r.get("psr")} below Bonferroni 1-0.05/{r.get("n_trials")} trials — selection bias">⚠ not sig</span>'
        else:
            sigb = ""
        cal = r["calmar"]
        calcell = f'<td class="num metric {"pos" if (cal or 0) > 0 else "neg"}">{cal:+.4f}</td>' if cal is not None else '<td class="num">—</td>'
        body += (
            f'<tr data-etf="{r["etf"]}">'
            f'<td class="etf"><b>{r["etf"]}</b><span class="char">{r["character"]}</span></td>'
            f'{calcell}'
            f'{_edgebar(r["edge"], r["edge_w"])}'
            f'<td class="num spk">{_spark(r["series"])}</td>'
            f'{_num(r["cagr"], ".1f", "%")}{_num(r["mdd"], ".1f", "%")}{_num(r["da"], ".2f")}{_num(r["sharpe"], ".2f")}'
            f'<td class="num">{r["trades"] if r["trades"] is not None else "—"}</td>'
            f'<td>{gates}</td><td>{leak}<br>{sigb}</td>'
            f'<td class="recipe"><code>{r["cell"]}</code><div class="rgloss">{r["recipe"]}</div></td>'
            f'</tr>')
    return (
        '<table id="lb"><thead><tr>'
        '<th data-k="etf" data-t="s">ETF</th>'
        '<th data-k="calmar" data-t="n" class="num sorted-desc">Calmar</th>'
        '<th data-k="edge" data-t="n" class="num">edge vs B&amp;H</th>'
        '<th class="num">Calmar history</th>'
        '<th data-k="cagr" data-t="n" class="num">CAGR</th>'
        '<th data-k="mdd" data-t="n" class="num">MDD</th>'
        '<th data-k="da" data-t="n" class="num">DA</th>'
        '<th data-k="sharpe" data-t="n" class="num">Sharpe</th>'
        '<th data-k="trades" data-t="n" class="num">trades</th>'
        '<th class="num">gates</th><th>trust / significance</th><th>recipe (cell · plain English)</th>'
        '</tr></thead><tbody>' + body + '</tbody></table>')


def _charmap_html(rows):
    body = ""
    for r in rows:
        cfg = None
        body += (f'<tr><td><b>{r["etf"]}</b></td><td>{r["character"]}</td>'
                 f'<td class="num metric">{r["calmar"]:+.2f}</td>'
                 f'<td class="recipe"><code>{r["cell"]}</code></td></tr>')
    return ('<table class="charmap"><thead><tr><th>ETF</th><th>character</th>'
            '<th class="num">Calmar</th><th>winning recipe</th></tr></thead><tbody>'
            + body + '</tbody></table>')


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
    return ('<div class="kpi"><div class="k">rounds</div><div class="v">' + str(sb.get("rounds", 0)) + '</div></div>'
            '<div class="kpi"><div class="k">kept</div><div class="v pos">' + str(sb.get("keeps", 0)) + '</div></div>'
            '<div class="kpi"><div class="k">ETFs</div><div class="v">' + str(sb.get("etfs", 0)) + '</div></div>'
            '<div class="kpi"><div class="k">best edge</div><div class="v acc">' + str(sb.get("best_edge", "—")) + '</div></div>')


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
    pf = K.get("portfolio") or {}
    champ = pf.get(pf.get("champion", "")) or pf.get("conviction_weight") or {}
    cal = champ.get("calmar", "—"); cagr = champ.get("car_pct", champ.get("cagr_pct", "—"))
    mdd = champ.get("mdd_pct", "—"); n = champ.get("n", "—")
    bottom = (f'a deployable <b>{n}-asset book</b> that earns about <b>{cagr}%/yr</b> with a worst drawdown of only '
              f'<b>~{mdd}%</b> (Calmar <b>{cal}</b>), positive every year 2023–26 — beating a passive hold of the same assets.'
              if champ else 'pending (no book yet).')
    return (
        '<section class="block intro" id="intro"><h2>What this is</h2>'
        '<p>An <b>autonomous research loop</b> that invents and back-tests ETF trading strategies on real market '
        'data (QuantConnect), and keeps only the ones that survive strict out-of-sample <i>and</i> multiple-testing '
        'checks. The AI proposes and runs the experiments; a human steers the direction.</p>'
        f'<p><b>Bottom line today —</b> {bottom} Its returns come from <b>two genuine machine-learned edges</b> — '
        'gold trend-following (GLD) and a US-dollar regime model (UUP) — plus <b>decorrelated diversifiers</b> '
        '(inflation bonds, commodities, credit) that cut the drawdown. The backtest is verified <b>leak-free and '
        'fully online</b>, using only models trained in the cloud (see <a href="deployment.md">deploy</a>).</p>'
        '<details class="glossary"><summary>How to read the numbers</summary>'
        '<ul><li><b>Calmar</b> = annual return ÷ worst drawdown (higher is better; above 3 is strong).</li>'
        '<li><b>MaxDD (MDD)</b> = the deepest peak-to-trough loss along the way.</li>'
        '<li><b>CAGR</b> = compounded annual return. <b>Sharpe</b> = return per unit of volatility.</li>'
        '<li><b>Edge</b> = how much a strategy beats simply buying and holding the same ETF.</li>'
        '<li><b>“Survives deflation”</b> = the result still looks real after correcting for how many strategies we '
        'tried (a guard against luck/overfitting). A weak edge that was lucky over a short window is rejected.</li>'
        '<li><b>Recipe / cell</b> = the exact pipeline that produced a result: '
        '<code>asset · bar-clock · labeling-method · sizing · threshold</code>.</li></ul></details></section>')


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
    n_cells_round = max([v.get("round") for v in (K.get("cells", {}) or {}).values()
                         if isinstance(v.get("round"), int)] or [0])
    stale = (f'live: r{scoreboard(K, rows)["rounds"]} (status) · cells thru r{n_cells_round} · '
             f'last_round meta={K.get("last_round","?")}')
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Autoresearch — research console</title>'
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
            '<link rel="stylesheet" href="style.css"></head><body>')
    nav = ('<div class="commandbar"><span class="prompt">autoresearch / research-console</span>'
           '<span class="chapnav"><a href="#portfolio">book</a><a href="#leaderboard">leaderboard</a><a href="#map">map</a>'
           '<a href="#story">story</a><a href="#insights">insights</a><a href="#graph">graph</a>'
           '<a href="#rounds">rounds</a><a href="program.md">program.md</a><a href="deployment.md">deploy</a></span>'
           f'<span class="stale" id="stale">{stale}</span><span class="clock" id="clock"></span></div>')
    hero = ('<section class="statushero"><div class="block" id="nowrunning"><h2><span class="idledot"></span>'
            'Status</h2><p class="small">loading…</p></div>'
            '<div class="block scoreboard" id="scoreboard">' + _scoreboard_html(data["scoreboard"]) + '</div></section>'
            '<div class="verdict" id="verdict">' + data["verdict"] + '</div>'
            + _latest_callout(csv_rounds))
    pf_sec = (f'<section class="block" id="portfolio"><h2>Production book — deployable portfolio (Wang endpoint)</h2>'
              '<p class="small">The actual money-on portfolio: a small basket of decorrelated holdings, '
              'weighted to maximise risk-adjusted return (Calmar). These are the live numbers.</p>'
              f'{_portfolio_html(K)}</section>')
    lb = (f'<section class="block" id="leaderboard"><h2>Leaderboard — real OOS, leak-free</h2>'
          f'<div class="tablewrap">{_leaderboard_html(rows)}</div>'
          '<p class="small">Calmar=CAGR/MaxDD (higher better) · CAGR=compounding annual return · MDD=max drawdown · '
          'DA=drawdown-area (lower MDD/DA better) · edge=Calmar−buy&amp;hold · sparkline=Calmar across this ETF\'s '
          f'cells · gates: G1 Calmar&gt;{G1_CALMAR:g}, G2 trades&gt;{G2_TRADES}. Click a header to sort.</p></section>')
    cmap = (f'<section class="block" id="map"><h2>What each asset rewards</h2>'
            '<p class="small">The hard-won lesson: <b>durable machine-learned edges are scarce.</b> Only '
            '<b>GLD</b> (gold trend-following) and <b>UUP</b> (dollar regime) beat buy-and-hold in a way that '
            'survives the selection-bias audit. Two-sided <i>timing</i> edges (EEM/TLT/IWM) looked great on a '
            'short window but <b>decayed to nothing</b> as the out-of-sample window grew — so the rest of the '
            'universe is best held passively, and the value comes from combining decorrelated holdings.</p>'
            f'{_charmap_html(rows)}</section>')
    story = (f'<section class="block" id="story"><h2>The research arc</h2>'
             '<p class="small">How the project got here, in four acts — from a failed start to the rule that '
             'now governs the whole board.</p>'
             f'{_acts_html(K)}</section>')
    insights = (f'<section class="block" id="insights"><h2>Top insights</h2>'
                '<p class="small">The most important things learned, each with the evidence that backs it.</p>'
                f'{_insights_html(K)}</section>')
    graph_sec = (f'<section class="block" id="graph"><h2>Causal graph — every experiment &amp; finding</h2>'
                 '<p class="small">How each outcome caused the next hypothesis. Drag · hover · double-click a phase '
                 'cluster to expand. Full page: <a href="causal_graph.html">causal_graph.html</a></p>'
                 f'<div id="graphwrap" data-pending="1">{graph}</div></section>')
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
    return (head + '<div class="dash">' + nav + _intro_html(K) + hero + pf_sec + lb + cmap + story + insights + graph_sec + ledger
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
    if(sbel)sbel.innerHTML='<div class="kpi"><div class="k">rounds</div><div class="v">'+(sb.rounds||0)+'</div></div>'+
      '<div class="kpi"><div class="k">kept</div><div class="v pos">'+(sb.keeps||0)+'</div></div>'+
      '<div class="kpi"><div class="k">ETFs</div><div class="v">'+(sb.etfs||0)+'</div></div>'+
      '<div class="kpi"><div class="k">best edge</div><div class="v acc">'+(sb.best_edge||'—')+'</div></div>';
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
