#!/usr/bin/env python3
"""console.sections.honesty — the 7-lens credibility gauntlet (id=honesty).

Blueprint section "honesty" (primary): the project's core "why these are real for
real money" claim, today only scattered per-row chips. Renders ONE canonical home:

  * a pass/fail MATRIX — rows = the 3 confirmed edges (GLD / UUP / USO·oil),
    columns = the 7 honesty lenses (deflated Sharpe, DSR+Holm-Bonferroni,
    permuted-label, decay monitor, e-value, PBO, cost stress) as pos/amber/neg/na
    status cells with the value on hover. e-value/PBO are genuinely uncomputed for
    most edges -> honest NA chips (no green-washing); UUP reads as a decorrelator.
  * ONE compact bar+audit card: the HARD-GATE FLOOR chips (G1..G4) + the leak-free
    fully-online assurance with the BACKTEST_AUDIT.md link. (The colour legend was
    dropped in the simplification — hover titles carry value + meaning.)

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


def _bar_and_audit(gates):
    """ONE compact card: the hard-gate floor chips + the leak-free audit link.
    (Colour key dropped — the matrix cells carry their value + meaning on hover.)"""
    parts = [g.strip() for g in (gates or "").split("·") if g.strip()]
    chips = " ".join(
        P.chip(g, "edge", "the floor every deployed book member cleared") for g in parts)
    cred = P.chip("train-only bar-threshold leak fix", "pos",
                  "magnitude thresholds are re-fit on TRAIN-only stats (previously full-series "
                  "incl. OOS) — the 2026-06-03 correction that re-validated the whole board")
    link = f'<a href="{P._esc(_AUDIT_DOC)}">{P._esc(_AUDIT_DOC)} →</a>'
    body = (f'<div>{chips}</div>'
            + f'<div style="margin-top:.55rem">{cred} '
            + f'<span class="small">backtest is leak-free + fully online — ObjectStore-'
              f'replay-only, online==saved, embargo-bounded · audit · {link}</span></div>')
    return P.card(body, kind="muted", id="leak-assurance",
                  eyebrow_text="THE BAR EVERY MEMBER CLEARED · LEAK-FREE + FULLY ONLINE")


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
            "3 confirmed edges × 7 honesty lenses — hover any cell for the value; "
            "uncomputed lenses show an honest NA, not a pass")
    )

    table = P.matrix(lenses, rows, col_titles=lens_titles)

    body = head + table + _bar_and_audit(h.get("gates"))
    return '<section class="block" id="honesty">' + body + "</section>"


if __name__ == "__main__":
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"honesty section: {len(out)} bytes")
    for needle in ('id="honesty"', "class=\"matrix\"", "mcell--na", "mcell--pos",
                   "G1 Calmar&gt;3", "LEAK-FREE + FULLY ONLINE", "BACKTEST_AUDIT.md"):
        assert needle in out, f"missing: {needle}"
    assert "STATUS KEY" not in out, "legend dropped — hover titles carry the meaning"
    print("ok — matrix + single bar/audit card present, legend gone")
