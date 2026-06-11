#!/usr/bin/env python3
"""console.sections.leaderboard — the single-ticker leaderboard (id=leaderboard).

Blueprint section "leaderboard" (secondary): the per-ticker evidence behind the
edges — the strongest existing component, kept and HARDENED. Three merges land here:

  1. The existing _lb_table, verbatim in spirit but routed entirely through
     console.primitives (sortable headers, inline _edgebar diverging bars, _spark
     Calmar sparklines, signal / val_auc, plain-English recipe).
  2. _charmap_html FOLDED IN as two added columns — the data-derived 'what it
     rewards' prose + a verdict badge (the standalone "What each asset rewards"
     section is deleted; _reward_verdict survives here as a row helper).
  3. The old single-ticker scoreboard (beat-BH / best-Calmar / luck-check / rounds)
     DEMOTED to a thin research-stats strip in this section's caption — no longer
     the page lead.

The canonical 'how to read the numbers' glossary lives ONLY in the read/colophon
section (cut-plan dedup); this caption keeps just the operational significance gloss
and a sort hint, pointing at the colophon for the full glossary.

Pure ctx -> HTML. Reads ONLY ctx["rows"] (derive) + ctx["scoreboard"] — no file-I/O —
and emits no class a primitive (or style.css) does not already own.
"""
from lb.console import primitives as P

EDGE_FLOOR = 0.05   # split: a "real edge" beats buy-and-hold by > 0.05 Calmar


# ---- row helpers -----------------------------------------------------------
def _reward_verdict(r):
    """Plain-English verdict on whether this asset rewards an ML edge — derived from
    the LIVE row, never hardcoded. Keys on whether the champion is an actual model vs
    a passive hold (always_long => the model only holds, edge==0), then on how far it
    beats buy-and-hold. Returns (text, chip_kind). NOTE: the raw `significant` flag
    alone is unreliable — an always_long champion can pass deflation on buy-hold
    returns (e.g. HYG) without being an ML edge, so always_long is checked first."""
    al = "always_long" in (r.get("cell", "") or "")
    edge = r.get("edge") or 0.0
    cal = r.get("calmar") or 0.0
    trades = r.get("trades")
    if al:
        # a degenerate always-long champion (≤1 trade) is a decorrelator seat, NOT a
        # timing edge — flag it honestly (e.g. IWM / EEM, trades=1).
        if trades is not None and trades <= 1:
            return ("degenerate always-long (trades=1) — decorrelator seat", "muted")
        return ("no ML edge — held passively", "muted")
    if cal > 0 and (r.get("significant") is True or edge >= 0.5):
        return ("real ML edge", "edge")
    if cal > 0 and edge >= 0.15:
        return ("modest ML edge", "amber")
    return ("no durable ML edge", "muted")


def _sig_chip(r):
    """Significance pill — does the edge hold up under the multiple-testing / deflated
    -Sharpe correction? 'holds up' (pos) / 'likely luck' (amber) / 'not assessed' (na)."""
    sig = r.get("significant")
    psr = r.get("psr")
    psr_txt = f"{psr:.2f}" if isinstance(psr, (int, float)) else "—"
    nt = r.get("n_trials")
    if sig is True:
        return P.chip("holds up", "pos",
                      f"Probabilistic Sharpe {psr_txt} clears the bar required after {nt} "
                      "trials — survives the multiple-testing correction.")
    if sig is False:
        return P.chip("likely luck", "amber",
                      f"Probabilistic Sharpe {psr_txt} is below the bar required after {nt} "
                      "trials — the apparent edge is probably selection bias.")
    return P.chip("not assessed", "na", "too few trades to run the deflation test")


def _verdict_chip(r):
    """The merged charmap verdict badge (data-derived)."""
    text, kind = _reward_verdict(r)
    return P.chip(text, kind, "does this asset reward a real ML edge, or is it best held passively?")


def _calmar_cell(r):
    """Signed Calmar metric cell, tinted by sign (.num.metric.pos/.neg)."""
    cal = r.get("calmar")
    cls = "metric pos" if (cal or 0) > 0 else "metric neg"
    return P._num(cal, "+.2f", cls=cls)


def _trades_cell(r):
    """Raw trade count (NOT a 2dp float) — right-aligned tabular num cell."""
    t = r.get("trades")
    return P.cell("—" if t is None else str(t), num=True)


def _lb_row(r):
    """One leaderboard <tr>, in the column order declared by _LB_COLS."""
    etf = P.cell(f'<b>{P._esc(r.get("etf"))}</b>'
                 f'<span class="char">{P._esc(r.get("character", ""))}</span>', cls="etf")
    return P.tr([
        etf,
        _calmar_cell(r),
        P._edgebar(r.get("edge"), r.get("edge_w", 0.0)),
        P.cell(P._spark(r.get("series") or []), num=True, cls="spk"),
        P._num(r.get("sharpe"), ".2f", cls="firstdiag"),
        P._num(r.get("val_auc"), ".2f"),
        _trades_cell(r),
        P.cell(_sig_chip(r)),
        P.cell(_verdict_chip(r)),
        P.cell(f'<div class="rgloss">{P._esc(r.get("recipe", ""))}</div>'
               f'<code class="cellid" title="{P._esc(r.get("cell", ""))}">id</code>', cls="recipe"),
        P.cell(P._esc(r.get("rewards", "")), cls="rewards"),
    ], attrs=f'data-etf="{P._esc(r.get("etf"))}"')


# column spec consumed by P.table (sortable headers via key/t; Calmar is default sort)
_LB_COLS = [
    {"label": "ETF", "key": "etf", "t": "s"},
    {"label": "Calmar", "num": True, "key": "calmar", "t": "n", "cls": "sorted-desc",
     "title": "annual return ÷ worst drawdown"},
    {"label": "vs. buy & hold", "num": True, "key": "edge", "t": "n",
     "title": "Calmar minus simply buying & holding; bar length is relative to the largest edge on the board"},
    {"label": "Calmar history", "num": True,
     "title": "best Calmar across successive rounds — green if the latest is at/above the first tested, red if below"},
    {"label": "Sharpe", "num": True, "key": "sharpe", "t": "n", "cls": "firstdiag",
     "title": "return per unit of volatility"},
    {"label": "signal", "num": True, "key": "val_auc", "t": "n",
     "title": "learnable structure — ~0.5 = none, >0.6 = real signal; blank = not applicable to this recipe"},
    {"label": "trades", "num": True, "key": "trades", "t": "n"},
    {"label": "significance",
     "title": "does the edge hold up after correcting for how many strategies we tried?"},
    {"label": "verdict", "title": "does this asset reward a real ML edge, or is it best held passively?"},
    {"label": "recipe", "title": "the exact pipeline: asset · bar-clock · labeling-method · sizing · threshold"},
    {"label": "what it rewards", "title": "the edge type this asset's structure rewards (or doesn't)"},
]


def _lb_table(rows, tid):
    return P.table(_LB_COLS, [_lb_row(r) for r in rows], id=tid)


# ---- the demoted research-stats strip --------------------------------------
def _stats_strip(sb):
    """The old single-ticker scoreboard, demoted to a thin caption strip."""
    best = sb.get("best_calmar", 0) or 0
    kpis = "".join([
        P.kpi("beat buy & hold", str(sb.get("edges", 0)), tone="acc",
              title="ETFs whose best strategy beats buy-and-hold by a traded margin (before the luck-check)"),
        P.kpi("best Calmar", f"{best:.2f}", tone="pos"),
        P.kpi("pass luck-check", f'{sb.get("n_sig", 0)}/{sb.get("n_assessed", 0)}',
              title="how many edges survive the multiple-testing / deflated-Sharpe correction"),
        P.kpi("G1 Calmar>3", f'{sb.get("g1_pass", 0)}/{sb.get("g1_total", 0)}',
              title="members clearing the hard Calmar>3 gate"),
        P.kpi("rounds", str(sb.get("rounds", 0)), title="distinct experiment rounds raced to date"),
    ])
    return '<div class="stats">' + kpis + "</div>"


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the full <section id="leaderboard"> HTML fragment."""
    rows = ctx.get("rows") or []
    sb = ctx.get("scoreboard") or {}
    sm = ctx.get("screen_summary") or {}
    universe = sm.get("universe", 311)
    strong = (sm.get("n_valid", 0) + sm.get("n_trust", 0) + sm.get("n_prov", 0)) or 8
    edges = [r for r in rows if (r.get("edge") or 0) > EDGE_FLOOR]
    drift = [r for r in rows if (r.get("edge") or 0) <= EDGE_FLOOR]

    bhfold = (f'<details class="bhfold"><summary>Buy &amp; hold — no durable edge '
              f'({len(drift)} assets)</summary>{_lb_table(drift, "lbbh")}</details>')

    caption = ('<p class="small"><b>significance</b> — "holds up" = the Calmar survives the '
               'deflated-Sharpe / multiple-testing correction; "likely luck" = it does not, so the '
               'apparent edge is probably selection bias; "not assessed" = too few trades to test. '
               '<b>verdict / what it rewards</b> distil whether each asset carries a real ML edge or '
               'is best held passively; the edge types are defined in '
               '<a href="#mechanisms">→ #mechanisms</a>. Full number glossary lives in the colophon. '
               'Click a header to sort.</p>')

    body = (
        P.eyebrow("PER-TICKER EVIDENCE · REAL OOS · LEAK-FREE")
        + "<h2>Single-ticker leaderboard</h2>"
        + P.provenance(
            f"Of the {universe}-ETF QC-confirmed universe screened down to {strong} strong fits, "
            "only GLD / UUP / USO clear every gate; the two-sided timing edges (EEM / TLT / IWM) "
            "decayed as the out-of-sample window grew.")
        + _stats_strip(sb)
        + _lb_table(edges, "lb")
        + bhfold
        + caption
    )
    return '<section class="block" id="leaderboard">' + body + "</section>"


if __name__ == "__main__":
    from lb.console.data import build_ctx
    out = render(build_ctx())
    print(f"leaderboard section: {len(out)} bytes")
    for needle in ('id="leaderboard"', 'id="lb"', 'id="lbbh"', "bhfold",
                   "what it rewards", "significance", "Single-ticker leaderboard",
                   'class="edgecell', 'class="stats"'):
        assert needle in out, f"missing: {needle}"
    print("ok — table + merged charmap cols + demoted stats strip + buy-hold fold present")
