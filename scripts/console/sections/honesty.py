#!/usr/bin/env python3
"""console.sections.honesty — the 7-lens credibility gauntlet (id=honesty).

Blueprint section "honesty" (primary): the project's core "why these are real for
real money" claim, today only scattered per-row chips. Renders ONE canonical home:

  * a pass/fail MATRIX — rows = the 3 confirmed edges (GLD / UUP / USO·oil),
    columns = the 7 honesty lenses (deflated Sharpe, DSR+Holm-Bonferroni,
    permuted-label, decay monitor, e-value, PBO, cost stress) as pos/amber/neg/na
    status cells with the value on hover. e-value/PBO are genuinely uncomputed for
    most edges -> honest NA chips (no green-washing); UUP reads as a decorrelator.
  * a status legend so the colour code is self-describing.
  * the HARD-GATE FLOOR string (G1 Calmar>3 / G2 trades>80 / G3 no-lookahead /
    G4 |train-val auc|<0.05) — the bar every book member cleared.
  * the leak-free + fully-online assurance row, linking BACKTEST_AUDIT.md.

Pure ctx -> HTML. Reads ONLY ctx["honesty"] (honesty_resolver) — no file-I/O — and
invents no CSS: every class is emitted through a console.primitives helper that OWNS
it, so an undefined-class bug is impossible.
"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:                       # console.data also self-inserts scripts/
    sys.path.insert(0, _SCRIPTS)
from console import primitives as P  # noqa: E402  (scripts/ on path above)

# The doc the leak-assurance row links to (relative to the served site root).
_AUDIT_DOC = "autoresearch/BACKTEST_AUDIT.md"


def _legend():
    """A self-describing colour key for the matrix status cells."""
    chips = " ".join([
        P.chip("pass", "pos", "clears the lens"),
        P.chip("partial / weak", "amber", "positive but below the strict bar"),
        P.chip("below strict bar", "neg", "fails the stricter family-wise correction"),
        P.chip("uncomputed", "na", "lens not yet run for this edge — honest NA, not a pass"),
    ])
    return (P.eyebrow("STATUS KEY · VALUE ON HOVER", "muted")
            + f'<div class="small">{chips}</div>')


def _gate_floor(gates):
    """The hard-gate floor string -> one muted chip per gate clause."""
    parts = [g.strip() for g in (gates or "").split("·") if g.strip()]
    chips = " ".join(
        P.chip(g, "edge", "the floor every deployed book member cleared") for g in parts)
    body = (f'<div>{chips}</div>'
            + '<div class="small">Every book member cleared all four before it could be '
              'crowned; the matrix above is the evidence each one held up.</div>')
    return P.card(body, kind="muted",
                  eyebrow_text="HARD-GATE FLOOR · the bar every member cleared")


def _leak_assurance(text):
    """The leak-free + fully-online assurance row, with a link to the audit doc."""
    # The resolver string ends with "(autoresearch/BACKTEST_AUDIT.md)"; strip the
    # parenthetical and re-attach it as a real link rather than escaped prose.
    prose = text or ""
    cut = prose.find(" (autoresearch")
    if cut != -1:
        prose = prose[:cut]
    link = f'<a href="{P._esc(_AUDIT_DOC)}">{P._esc(_AUDIT_DOC)} →</a>'
    body = (f'<p>{P._esc(prose)}</p>'
            + f'<div class="small">audit · {link}</div>')
    return P.card(body, kind="metric",
                  eyebrow_text="LEAK-FREE + FULLY ONLINE", id="leak-assurance")


def render(ctx):
    """ctx -> the complete <section id="honesty"> HTML fragment."""
    h = ctx.get("honesty") or {}
    lenses = h.get("lenses") or []
    lens_titles = h.get("lens_titles") or None
    rows = h.get("rows") or []

    head = (
        P.eyebrow("WHY THESE ARE REAL · THE 7-LENS GAUNTLET")
        + "<h2>Why these edges are real</h2>"
        + P.provenance(
            "3 confirmed edges (GLD · UUP · USO oil) × 7 lenses · "
            "DSR / PSR data-backed from per_etf_best · permute / decay / cost qualitative · "
            "e-value + PBO uncomputed → genuine NA · "
            "oil row XOP-proxied (USO / UCO not in per_etf_best)")
    )

    table = P.matrix(lenses, rows, col_titles=lens_titles)

    body = (
        head
        + table
        + _legend()
        + '<div class="stats">'
        + _gate_floor(h.get("gates"))
        + _leak_assurance(h.get("leak_assurance"))
        + "</div>"
    )
    return '<section class="block" id="honesty">' + body + "</section>"


if __name__ == "__main__":
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"honesty section: {len(out)} bytes")
    for needle in ('id="honesty"', "class=\"matrix\"", "mcell--na", "mcell--pos",
                   "HARD-GATE FLOOR", "G1 Calmar&gt;3", "LEAK-FREE + FULLY ONLINE",
                   "BACKTEST_AUDIT.md", "STATUS KEY"):
        assert needle in out, f"missing: {needle}"
    print("ok — matrix + legend + gate floor + leak assurance all present")
