#!/usr/bin/env python3
"""console.sections.appendix — the SINGLE collapsed evidence drawer (id=appendix).

The 2026-06-09 reconstruction folds the page into a one-page narrative
(book -> mechanisms -> honesty -> what's next) and moves every dense evidence
surface into ONE appendix of <details> drawers, collapsed by default:

  leaderboard · universe screen · book-construction lab · causal graph ·
  rounds ledger · glossary & colophon

Each drawer renders the EXISTING section builder verbatim (so the data contract,
the live #lb sort, the #graphwrap lazy-mount and every anchor id are preserved);
the appendix CSS neutralizes the nested section chrome so there is no box-in-a-box.
Opening a drawer reveals the full instrument; the main page stays calm.

Pure ctx -> HTML. Each nested render is wrapped defensively so one bad builder
degrades to a small note instead of blanking the appendix.
"""
from lb.console import primitives as P
from lb.console.sections import (
    leaderboard, screen, book_lab, graph, rounds, read, arc,
)

# (drawer id, summary title, render_fn, one-line hint) — in evidence order.
_DRAWERS = [
    ("leaderboard", "Single-ticker leaderboard", leaderboard.render,
     "per-ETF Calmar vs buy-and-hold, leak status, significance"),
    ("screen", "Universe screen", screen.render,
     "311 ETFs raced against buy-and-hold — which names have structure"),
    ("book-lab", "Book construction & sizing", book_lab.render,
     "every portfolio variant + the decorrelation sweep"),
    ("graph", "Causal / provenance graph", graph.render,
     "the experiment lineage as an interactive graph"),
    ("rounds", "Rounds ledger", rounds.render,
     "every A/B round, newest first"),
    ("acts", "History — six acts", arc.render_acts,
     "from the founding null to the third mechanism"),
    ("read", "Glossary & colophon", read.render,
     "how to read the numbers + data sources"),
]

# Honest record of what was tried and REJECTED at the mechanism level (moved off
# the main page in the 2026-06-10 simplification; detail in the rounds ledger).
_REJECTED = ("Rejected mechanisms — leveraged-equity reversion (vol artifact) · "
             "sliced_wasserstein high-Calmars (no-baseline, unverifiable) · "
             "sticky-HMM (predictable but not profitable) · "
             "β200 (a buy-hold pre-filter, no timing edge)")


def _drawer(ctx, sid, title, fn, hint):
    try:
        inner = fn(ctx)
    except Exception as e:  # one bad builder must never blank the appendix
        inner = f'<p class="small">(unavailable: {type(e).__name__})</p>'
    return (f'<details class="appx" id="appx-{sid}">'
            f'<summary>{P._esc(title)}<span class="appx-hint">{P._esc(hint)}</span></summary>'
            f'<div class="appx-body">{inner}</div></details>')


def render(ctx):
    """ctx -> the complete <section id="appendix"> HTML fragment."""
    drawers = "".join(_drawer(ctx, *d) for d in _DRAWERS)
    body = (
        P.eyebrow("APPENDIX · EVIDENCE & LEDGER")
        + "<h2>Appendix — evidence &amp; ledger</h2>"
        + P.provenance(
            "The full leaderboard, universe screen, composition lab, causal graph, "
            "rounds ledger, history and glossary — collapsed. Open any drawer for "
            "the detail behind the headline.")
        + drawers
        + P.provenance(_REJECTED)
    )
    return '<section class="block" id="appendix">' + body + "</section>"


if __name__ == "__main__":
    from lb.console.data import build_ctx
    out = render(build_ctx())
    print(f"appendix section: {len(out)} bytes")
    for needle in ('id="appendix"', 'class="appx"', 'id="leaderboard"', 'id="screen"',
                   'id="book-lab"', 'id="graphwrap"', 'id="read"', 'class="glossary"'):
        assert needle in out, f"missing: {needle}"
    print("ok — all six evidence drawers present, anchors + graphwrap preserved")
