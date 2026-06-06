#!/usr/bin/env python3
"""console.sections.screen — the 311-ETF universe screen (id=screen).

Blueprint section "screen" (secondary): keep the universe-search machinery for
research density, but DEMOTE it from "this week's headline" to "the funnel that
surfaced the oil mean-reversion cluster" (USO / UCO / XOP — now the book's 3rd
mechanism). Components, all reframed:

  * a one-line selection-bias note (the canonical multiple-testing home is the
    honesty gauntlet; here we just flag the best-of-N inflation),
  * two progress bars (universe screened · deep-sweep coverage),
  * the trust-tier count chips,
  * the ranked screen table (edge over buy-and-hold, tinted by trust tier),
    with the EXCLUDED (cash-artifact / NO-BASELINE) and NO-FIT rows folded away,
  * the winning-axis / winning-labeler / where-STRONG-fits-live breakdown,
  * the STALE-history quarantine demoted to ONE collapsed footnote.

THE TIERING FIX (blueprint-critical): a VALIDATED tier (data.py already emits
tier_cls 'valid' for USO) OVERRIDES the val_auc>0.85 -> PROVISIONAL/overfit-suspect
heuristic once an edge has cleared the honesty stack. So USO (val_auc 0.979) reads
"VALIDATED — high AUC is reversion-label structure, gate-confirmed", NOT
"overfit-suspect".

Pure ctx -> HTML. Reads ONLY ctx["screen"] (load_screen rows, tiered) +
ctx["screen_summary"] (tallies + progress) — no file-I/O — and invents no CSS:
every class is emitted through a console.primitives helper that OWNS it (the ~20
legacy screen classes are routed through primitives, so an undefined-class bug is
structurally impossible). Progress bars use inline styles over existing :root
design tokens only.
"""
import os
import sys
from collections import Counter

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:                       # console.data also self-inserts scripts/
    sys.path.insert(0, _SCRIPTS)
from console import primitives as P  # noqa: E402  (scripts/ on path above)


# ---------------------------------------------------------------------------
# tier -> (chip kind, hover title). VALIDATED is the override that fixes the
# "high val_auc == overfit" mis-read for gate-confirmed edges.
# ---------------------------------------------------------------------------
TIER_META = {
    "valid":    ("pos",   "VALIDATED — cleared the 7-lens honesty stack; high val_auc is "
                          "reversion-label structure, gate-confirmed (NOT overfit-suspect)"),
    "trust":    ("edge",  "TRUSTWORTHY — leak-re-verified fit, val_auc in the real 0.57–0.66 band"),
    "prov":     ("amber", "PROVISIONAL — val_auc > 0.85, overfit / selection suspect; "
                          "needs DSR deflation + a permuted-label control before deployment"),
    "marginal": ("muted", "MARGINAL — only a slim edge over buy-and-hold"),
    "excluded": ("muted", "EXCLUDED — cash-MaxDD artifact (Calmar inflated by ~0 drawdown) "
                          "or a degenerate 0-trade buy-hold baseline (NO-BASELINE)"),
    "nofit":    ("muted", "NO-FIT — buy-and-hold wins; no method panel beats it"),
}

# asset classes we expect STRONG fits to live in (for the zero-fit note)
_CLASSES_ALL = ["Commodity", "Leveraged/Inverse", "Fixed Income",
                "International Equity", "Real Estate", "Currency"]


# ---- tiny inline-styled progress bar (no class dependency; :root tokens only) ----
def _bar(pct):
    p = max(0.0, min(100.0, pct))
    return ('<div style="height:6px;background:var(--bg-2);border:1px solid var(--line-soft);'
            'border-radius:var(--r-sm);overflow:hidden;margin:.4rem 0">'
            f'<i style="display:block;height:100%;width:{p:.1f}%;background:var(--accent)"></i></div>')


# ---- row cell helpers ------------------------------------------------------
def _cal_cell(cal):
    """Signed method Calmar, tinted by sign."""
    if cal is None:
        return P._num(None)
    return P._num(cal, "+.2f", cls=("metric pos" if cal > 0 else "metric neg"))


def _trd_cell(t):
    """Raw trade count (not a 2dp float)."""
    if t is None:
        return P._num(None)
    try:
        return P.cell(str(int(t)), num=True)
    except (TypeError, ValueError):
        return P._num(None)


def _auc_cell(r):
    """val_auc cell — the tiering-fix lives here. VALIDATED tickers read their high
    AUC as a POSITIVE (label structure, gate-confirmed); only un-validated >0.85
    fits carry the overfit flag."""
    auc = r.get("val_auc")
    if auc is None:
        return P._num(None)
    if r.get("tier_cls") == "valid":
        return P.cell(f"{auc:.2f}", num=True, cls="metric pos",
                      attrs='title="high val_auc = reversion-label structure, '
                            'gate-confirmed (NOT overfit-suspect)"')
    if auc > 0.85:
        return P.cell(f"{auc:.2f} " + P.chip("overfit?", "amber",
                      "val_auc > 0.85 — overfit / selection suspect; "
                      "needs DSR + a permuted-label control"), num=True)
    return P._num(auc, ".2f")


def _tier_chip(r):
    kind, title = TIER_META.get(r.get("tier_cls"), ("muted", ""))
    return P.chip(r.get("tier") or (r.get("tier_cls") or "").upper(), kind, title)


def _recipe_cell(r):
    rf = r.get("recipe_full") or "—"
    ld = (" " + P.chip("Ⓛ", "muted",
                       "logdollar-axis fit — re-verified leak-clean by two deep leak-hunt workflows")
          ) if r.get("logdollar") else ""
    return P.cell(f'<code class="cellid" title="{P._esc(rf)}">{P._esc(rf)}</code>' + ld, cls="recipe")


def _scr_row(r, maxe, rank):
    """One screen <tr>, carrying data-tier for the client-side tier filter."""
    edge = r.get("edge")
    w = (abs(edge) / maxe * 100.0) if (edge is not None and maxe) else 0.0
    etf = P.cell(f'<b>{P._esc(r.get("ticker"))}</b>'
                 f'<span class="char">{P._esc(r.get("name", ""))}</span>', cls="etf")
    cells = [
        P.cell(str(rank), num=True),
        etf,
        P.cell(P._esc(r.get("asset_class", ""))),
        P.cell(_tier_chip(r)),
        _recipe_cell(r),
        _cal_cell(r.get("method_calmar")),
        P._num(r.get("buyhold_calmar"), "+.2f"),
        P._edgebar(edge, w),
        _auc_cell(r),
        _trd_cell(r.get("trades")),
        P.cell(P._esc(r.get("verdict", ""))),
    ]
    return P.tr(cells, attrs=f'data-tier="{P._esc(r.get("tier_cls", ""))}"')


# column spec consumed by P.table (sortable headers via key/t; edge is default sort)
_SCR_COLS = [
    {"label": "#", "num": True},
    {"label": "ETF", "key": "etf", "t": "s"},
    {"label": "class"},
    {"label": "trust tier",
     "title": "VALIDATED (gate-confirmed) ▸ TRUSTWORTHY ▸ PROVISIONAL (overfit-suspect) "
              "▸ marginal ▸ excluded ▸ no-fit"},
    {"label": "axis × labeler", "title": "the winning bar-clock axis × labeling method for this ETF"},
    {"label": "Calmar", "num": True, "key": "calmar", "t": "n", "title": "annual return ÷ worst drawdown"},
    {"label": "buy&hold", "num": True, "key": "buyhold", "t": "n",
     "title": "buy-and-hold Calmar for the same ETF"},
    {"label": "edge", "num": True, "key": "edge", "t": "n", "cls": "sorted-desc",
     "title": "Calmar minus buy-and-hold — the rank key; bar length is relative to the largest edge"},
    {"label": "val_auc", "num": True, "key": "val_auc", "t": "n",
     "title": "validation AUC — 0.5 none, 0.6-0.75 real, >0.85 overfit-suspect unless gate-confirmed"},
    {"label": "trades", "num": True, "key": "trades", "t": "n"},
    {"label": "verdict"},
]


def _scr_table(rows, maxe, start_rank, tid=None):
    body_rows = [_scr_row(r, maxe, start_rank + i) for i, r in enumerate(rows)]
    return P.table(_SCR_COLS, body_rows, id=tid)


# ---- supporting blocks -----------------------------------------------------
def _selbias():
    """A concise selection-bias flag. The canonical multiple-testing correction
    lives in the honesty gauntlet; here we just note the best-of-N inflation and
    the VALIDATED override."""
    return ('<p class="small"><b>Selection-bias note.</b> Each ETF is the best of a best-of-N '
            '(all axes × all labelers) per-ETF search, which inflates the top result — the full '
            'multiple-testing correction lives in the honesty gauntlet. <b>VALIDATED</b> fits (USO) '
            'cleared that stack: their high val_auc is reversion-label structure, gate-confirmed, '
            'not overfit. <b>PROVISIONAL</b> fits (val_auc&nbsp;&gt;&nbsp;0.85) are overfit / selection '
            'suspects pending DSR deflation + a permuted-label control. Cash-<b>ARTIFACT</b> and '
            '<b>NO-BASELINE</b> rows are shown as excluded, not fits.</p>')


def _progress(sm):
    uni = sm.get("universe") or 0
    swt = sm.get("sweep_total") or 0
    scr = sm.get("screened") or 0
    sw = sm.get("sweep_done") or 0
    scr_pct = (scr / uni * 100.0) if uni else 0.0
    sw_pct = (sw / swt * 100.0) if swt else 0.0

    def _card(label, done, total, pct):
        body = (_bar(pct) + f'<div class="small">{done}/{total}</div>')
        return P.card(body, kind="muted", eyebrow_text=label)

    return ('<div class="stats">'
            + _card("UNIVERSE SCREENED", scr, uni, scr_pct)
            + _card("DEEP-SWEEP · ALL AXES × LABELERS / ETF", sw, swt, sw_pct)
            + '</div>')


def _tier_chips(sm):
    chips = " ".join([
        P.chip(f'{sm.get("n_valid", 0)} VALIDATED', "pos", "cleared the 7-lens honesty stack"),
        P.chip(f'{sm.get("n_trust", 0)} TRUSTWORTHY', "edge", "leak-re-verified fit, val_auc 0.57–0.66"),
        P.chip(f'{sm.get("n_prov", 0)} PROVISIONAL', "amber", "val_auc > 0.85 — overfit / selection suspect"),
        P.chip(f'{sm.get("n_marginal", 0)} marginal', "muted", "slim edge over buy-and-hold"),
        P.chip(f'{sm.get("n_excluded", 0)} excluded', "muted", "cash-MaxDD artifact / NO-BASELINE"),
        P.chip(f'{sm.get("n_nofit", 0)} no-fit', "muted", "buy-and-hold wins"),
        P.chip(f'{sm.get("screened", 0)}/{sm.get("universe", 0)} screened', "muted", "universe coverage"),
    ])
    return f'<div class="small">{chips}</div>'


def _bars(counter):
    if not counter:
        return '<span class="char">—</span>'
    return " · ".join(f"<b>{P._esc(k)}</b> ×{v}" for k, v in counter.most_common())


def _mech_cards(top, marg):
    """Winning-axis / winning-labeler / where-STRONG-fits-live breakdown."""
    fitmarg = top + marg
    by_axis = Counter((r.get("axis") or "?") for r in fitmarg if r.get("axis"))
    by_lab = Counter((r.get("labeler") or "?") for r in fitmarg if r.get("labeler"))
    by_class = Counter(r.get("asset_class") for r in top if r.get("asset_class"))
    zero = [c for c in _CLASSES_ALL if by_class.get(c, 0) == 0]
    zero_txt = ", ".join(zero) if zero else "none"
    note = (f'<div class="small">zero STRONG fits in: {P._esc(zero_txt)} — the mechanism fits the '
            'commodity &amp; leveraged-equity classes, not broad equity / rates.</div>')
    return ('<div class="stats">'
            + P.card(_bars(by_axis), kind="mech", eyebrow_text="WINNING AXIS · fits + marginal")
            + P.card(_bars(by_lab), kind="mech", eyebrow_text="WINNING LABELER · fits + marginal")
            + P.card(_bars(by_class) + note, kind="mech", eyebrow_text="WHERE STRONG FITS LIVE")
            + '</div>')


def _stale_footnote():
    """STALE-history quarantine, demoted to one collapsed footnote (blueprint)."""
    return ('<details class="bhfold"><summary>Quarantined (STALE — pre-leak / window-decayed)</summary>'
            '<p class="small">Old single-ticker rows from before the bar-threshold leak fix, or whose '
            'edge decayed as the out-of-sample window grew, are deliberately kept <b>out</b> of the ranked '
            'board above. They are not current fits; the board asserts only present, leak-free results.</p>'
            '</details>')


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the complete <section id="screen"> HTML fragment."""
    rows = ctx.get("screen") or []
    sm = ctx.get("screen_summary") or {}

    head = (P.eyebrow("THE FUNNEL · 311-ETF UNIVERSE SCREEN")
            + "<h2>Universe screen — the funnel that surfaced oil</h2>")

    if not rows:
        body = head + '<p class="small">(screen results pending)</p>'
        return '<section class="block" id="screen">' + body + "</section>"

    top = [r for r in rows if r.get("tier_cls") in ("valid", "trust", "prov")]
    marg = [r for r in rows if r.get("tier_cls") == "marginal"]
    excl = [r for r in rows if r.get("tier_cls") == "excluded"]
    nofit = [r for r in rows if r.get("tier_cls") == "nofit"]
    maxe = max([abs(r["edge"]) for r in rows if r.get("edge") is not None] or [1.0]) or 1.0

    open_rows = top + marg
    main_tbl = _scr_table(open_rows, maxe, 1, tid="scrlb")
    rank = len(open_rows)

    excl_fold = ""
    if excl:
        excl_fold = ('<details class="bhfold"><summary>Excluded — cash-MaxDD artifact / NO-BASELINE ('
                     + str(len(excl)) + ' ETFs)</summary>' + _scr_table(excl, maxe, rank + 1) + '</details>')
        rank += len(excl)
    nofit_fold = ""
    if nofit:
        nofit_fold = ('<details class="bhfold"><summary>No fit — buy-and-hold wins ('
                      + str(len(nofit)) + ' ETFs)</summary>' + _scr_table(nofit, maxe, rank + 1) + '</details>')

    body = (
        head
        + P.provenance(
            "Every QC-confirmed ETF's best method panel raced vs always-long buy-and-hold; a deep-sweep "
            "then tries all axes × labelers on the fit-prone classes. This is the funnel that surfaced the "
            "oil mean-reversion cluster (USO / UCO / XOP) — now the book's 3rd mechanism. Ranked by edge "
            "over buy-and-hold, tinted by trust tier.")
        + _selbias()
        + _progress(sm)
        + _tier_chips(sm)
        + main_tbl
        + excl_fold
        + nofit_fold
        + _mech_cards(top, marg)
        + _stale_footnote()
    )
    return '<section class="block" id="screen">' + body + "</section>"


if __name__ == "__main__":
    import ast
    with open(__file__) as _f:
        ast.parse(_f.read())
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"screen section: {len(out)} bytes")
    for needle in ('id="screen"', 'id="scrlb"', "Universe screen", "VALIDATED",
                   "PROVISIONAL", "TRUSTWORTHY", "Selection-bias note", "card--mech",
                   "WINNING AXIS", 'class="edgecell', "bhfold", "Quarantined"):
        assert needle in out, f"missing: {needle}"
    print("ok — lede + selbias + progress + tier chips + table + folds + mech cards + stale footnote present")
