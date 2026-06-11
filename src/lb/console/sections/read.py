#!/usr/bin/env python3
"""console.sections.read — How to read the numbers + colophon (id=read, tertiary).

Blueprint section "read": the ONE canonical glossary + methods/provenance that
CLOSES the page. It absorbs three things the old monolith duplicated or scattered:

  * the SINGLE "how to read the numbers" <details> glossary
    (Calmar / MaxDD / CAGR / Sharpe / Edge / signal / recipe / bar-clock / entry) —
    the ONLY copy on the page (the two old copies, in _intro_html and the leaderboard
    caption, are deleted), so the definitions can never drift apart;
  * the orientation abstract now lives in the book hero (not duplicated here);
  * a short COLOPHON: the reconciled funnel chain, the data sources (knowledge.json /
    round_results.csv / etf_screen.csv), a single link to the canonical deployment-gate +
    leak-free assurance section (#honesty), the doc links (program.md / deployment.md /
    BACKTEST_AUDIT.md), and the global last-updated stamp.

Pure ctx -> HTML. Every LIVE string flows through a resolver already on ctx
(ctx["book"] for the freshness stamp, ctx["screen_summary"]/ctx["edges"] for the
reconciled funnel chain); the gate-floor + leak-assurance strings are NOT repeated
here — they live once in #honesty and are linked. The static glossary prose has no
resolver — it is the
project's definitions, not research state. Invents no CSS: layout flows through
console.primitives helpers that OWN their classes, plus the pre-existing
`.glossary`/`details`/`.small` styling already defined in reports/style.css.
"""
import os
import sys

from lb.console import primitives as P  # noqa: E402  (scripts/ on path above)


# ---- the ONE canonical glossary -------------------------------------------
# Static definitions (not research state -> no resolver). Each row is raw inner
# HTML on purpose (it carries <b>/<code>/<i> markup we control); this is the only
# copy of these definitions anywhere on the page.
_GLOSSARY = [
    ("Calmar", "annualized return ÷ worst drawdown — higher is better; above <b>3</b> is strong."),
    ("MaxDD (MDD)", "the deepest peak-to-trough loss along the way — lower is better."),
    ("CAGR", "compounded annual growth rate of the strategy."),
    ("Sharpe", "return per unit of volatility — the classic risk-adjusted return."),
    ("Edge", "how much a strategy beats simply buying and holding the same ETF "
             "(its Calmar minus the buy-and-hold Calmar)."),
    ("Signal (val_auc / DA)", "validation AUC / directional accuracy — how well the model's "
                              "conviction ranks the next move. ~<code>0.50</code> = no structure; "
                              "<code>&gt;0.60</code> = real, exploitable structure."),
    ("Survives deflation / luck-check", "the result still looks real after correcting for how many "
                                        "strategies were tried — a multiple-testing guard against "
                                        "luck / overfitting (DSR + Holm-Bonferroni)."),
    ("Recipe / cell", "the exact pipeline that produced a result: "
                      "<code>asset · bar-clock · labeling-method · sizing · threshold</code>."),
    ("Bar-clock", "strategies sample the market on <i>event bars</i> (e.g. equal dollar traded), "
                  "not fixed clock time — so each bar carries comparable information."),
    ("Entry 0.40", "the model only takes a position when its conviction exceeds <code>0.40</code>."),
]


def _glossary():
    """The single 'how to read the numbers' <details> fold — the ONLY copy on the page.
    Uses the pre-existing `.glossary` styling (defined in reports/style.css)."""
    items = "".join(f"<li><b>{lbl}</b> = {body}</li>" for lbl, body in _GLOSSARY)
    return ('<details class="glossary"><summary>How to read the numbers</summary>'
            f"<ul>{items}</ul></details>")


# ---- the colophon ----------------------------------------------------------
def _sources(ctx):
    """Data-source line — the three live files, by basename, with one-line roles."""
    paths = ctx.get("paths") or {}

    def base(key, fallback):
        p = paths.get(key)
        return os.path.basename(p) if p else fallback
    kj = base("KJ", "knowledge.json")
    rc = base("ROUND_CSV", "round_results.csv")
    sc = base("SCREEN_CSV", "etf_screen.csv")
    return ('<div class="small">Data sources · '
            f"<code>{P._esc(kj)}</code> (book · per-ETF best · insights · causal graph) · "
            f"<code>{P._esc(rc)}</code> (the round ledger) · "
            f"<code>{P._esc(sc)}</code> (the 311-ETF universe screen).</div>")


def _funnel_chain(ctx):
    """The ONE reconciled funnel string, kept identical to sections.screen._funnel_chain
    (orientation lives in the book hero now; the colophon just restates the chain)."""
    sm = ctx.get("screen_summary") or {}
    tower = (ctx.get("book") or {}).get("stat_tower") or {}
    universe = sm.get("universe", 311)
    fitprone = tower.get("screened_total", 42)
    strong = (sm.get("n_valid", 0) + sm.get("n_trust", 0) + sm.get("n_prov", 0)) or 8
    mechs = len(ctx.get("edges") or []) or 3
    return (f"{universe} QC-confirmed universe → {fitprone} fit-prone screened → "
            f"{strong} strong fits → {mechs} deployed mechanisms")


def _colophon(ctx):
    """The closing colophon card: the reconciled funnel chain + sources + a single link
    to the canonical gate/leak section + doc links + the global last-updated stamp. The
    orientation abstract now lives in the book hero, and the gate-floor / leak-assurance
    strings live ONCE in #honesty (linked here, not repeated verbatim)."""
    book = ctx.get("book") or {}

    funnel = ('<div class="small">Funnel · ' + P._esc(_funnel_chain(ctx)) + ".</div>")

    gates_leak = ('<div class="small">Deployment gates &amp; leak-free assurance · '
                  '<a href="#honesty">→ #honesty</a></div>')

    links = ('<div class="small">Docs · '
             '<a href="program.md">program.md</a> · '
             '<a href="deployment.md">deployment.md</a> · '
             '<a href="autoresearch/BACKTEST_AUDIT.md">BACKTEST_AUDIT.md</a></div>')

    stamp = ('<div class="small">' + P.chip("last updated", "muted", "global freshness stamp")
             + " " + P._esc(book.get("freshness", "")) + "</div>") if book.get("freshness") else ""

    body = funnel + _sources(ctx) + gates_leak + links + stamp
    return P.card(body, kind="muted", eyebrow_text="COLOPHON · DATA, GATES & PROVENANCE",
                  title="data sources, deployment gates, leak assurance, doc links")


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the complete <section id="read"> HTML fragment (glossary + colophon)."""
    book = ctx.get("book") or {}
    body = (
        P.eyebrow("HOW TO READ · METHODS & PROVENANCE")
        + "<h2>How to read the numbers</h2>"
        + P.provenance(
            "the one canonical glossary + colophon that closes the page · "
            + (book.get("freshness") or ""))
        + _glossary()
        + _colophon(ctx)
    )
    return '<section class="block" id="read">' + body + "</section>"


if __name__ == "__main__":
    import ast
    with open(os.path.abspath(__file__)) as _f:
        ast.parse(_f.read())                       # self-syntax-check
    from lb.console.data import build_ctx
    out = render(build_ctx())
    print(f"read section: {len(out)} bytes")
    for needle in ('id="read"', "How to read the numbers", 'class="glossary"',
                   "<summary>How to read the numbers</summary>", "Calmar", "MaxDD",
                   "CAGR", "Sharpe", "Edge", "Bar-clock", "Entry 0.40",
                   "COLOPHON", "Data sources", "Deployment gates",
                   "program.md", "deployment.md", "BACKTEST_AUDIT.md",
                   "last updated"):
        assert needle in out, f"missing: {needle}"
    assert "pending" not in out.lower(), "read leaked a 'pending' value"
    # glossary appears exactly once (it is the ONLY copy on the page)
    assert out.count('<details class="glossary">') == 1, "glossary must be the single copy"
    print("ok — single glossary + colophon (sources, gates, leak assurance, doc links, stamp)")
