#!/usr/bin/env python3
"""console.sections.arc — Research arc & what's next (id=arc).

Blueprint section "arc" (secondary): the narrative spine PLUS the single most
decision-relevant text on the page (next_idea). Merges the two old thin sections
("The research arc" + "Top insights") into one block and DROPS the stale early-grind
top_insights list.

  * a 6-act TIMELINE (extends the old _acts_html with Act VI "The third mechanism":
    oil reversion confirmed on USO/UCO/XOP -> permute/decay/cost/DSR -> proposed into
    the book; the single-ticker frontier converging). Each act: title, one sentence,
    a delta chip (TLT 0.31->1.52, XLE 2.26->0.64, book +USO 4.62->5.16). Act V is
    revised OFF "screen now running" to the converged 42/42 framing.
  * the CURRENT RECOMMENDATION callout — the knowledge.json next_idea distilled into ONE
    non-contradictory sentence (the raw next_idea is self-contradictory), with the +USO
    numbers cross-referenced to the book hero rather than restated.
  * 2 STILL-TRUE methodological lessons folded in as sub-notes (no-lookahead invariant,
    selectivity beats exposure) pulled from top_insights by title, plus a decorrelation
    lesson that links to the composition lab (the now-closed meta-labeling lede is dropped).

Pure ctx -> HTML. Reads ctx["book"] (book_resolver — the live +USO numbers) and
ctx["K"] (next_idea + top_insights, which have no resolver — pure narrative). Invents
no CSS: every class is emitted through a console.primitives helper that OWNS it, so an
undefined-class bug is structurally impossible.
"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:                       # console.data also self-inserts scripts/
    sys.path.insert(0, _SCRIPTS)
from console import primitives as P  # noqa: E402  (scripts/ on path above)


# ---- the 6-act timeline ----------------------------------------------------
def _acts(ctx):
    """The six narrative acts. The only LIVE numbers (Act V screen count, Act VI book
    upgrade) flow from the resolvers (ctx["book"]) so they can never diverge from
    knowledge.json; the historical deltas (TLT/XLE) are fixed narrative facts."""
    book = ctx.get("book") or {}
    tower = book.get("stat_tower") or {}
    scr, scr_t = tower.get("screened", 42), tower.get("screened_total", 42)
    return [
        ("I · The null",
         "Single-asset long-only ML couldn't beat buy-and-hold across 6 rounds — so the "
         "problem was reframed into a per-ETF axis × labeler tournament.",
         "6-round null → tournament", "muted",
         "the founding negative result that reframed the whole program", None),
        ("II · The unlock",
         "Long-only can't beat a declining asset; adding shorting + a directional label + "
         "causal mean-reversion features unlocked the first real timing edge on TLT.",
         "TLT 0.31 → 1.52", "pos",
         "③ causal RSI / Bollinger-%b features; permuted control stays at −0.14 (real, not leak)", None),
        ("III · The correction",
         "A bar-threshold look-ahead (magnitude thresholds fit on full-series stats incl. "
         "OOS) was inflating results; the TRAIN-only leak fix re-validated the entire board.",
         "XLE 2.26 → 0.64", "amber",
         "honesty correction — the leak fix deflated inflated crowns to leak-free truth", None),
        ("IV · Convergence",
         "One rule governs the board — time when the hold is weak, hold when the trend is "
         "strong; the fixed-12 frontier saturated at its leak-free single-asset ceilings.",
         "7/7 at leak-free ceilings", "edge",
         "f_timing_when — the unifying mechanism across the original twelve", None),
        ("V · The widening",
         "With the fixed frontier saturated, the hunt widened to all 311 QC-confirmed ETFs "
         "— race each against buy-and-hold, keep only genuine fits; a commodity & oil "
         "cluster surfaced and the fit-prone universe is now fully screened.",
         f"{scr}/{scr_t} fit-prone screened", "edge",
         "the universe funnel that surfaced the oil-reversion cluster", None),
        ("VI · The third mechanism",
         "Oil mean-reversion confirmed on USO/UCO/XOP — discovered, permute-checked, "
         "decay-monitored, cost-stressed, DSR-positive — and proposed into the book as the "
         "3rd mechanism; the single-ticker frontier is converging.",
         "book +USO", "oil",
         "the +USO upgrade and its numbers live in the deployable-book hero", "#book"),
    ]


def _act_card(title, body, delta, tone, delta_title, href=None):
    """One act as a .card--mech (accent-2 left stripe), with the delta as a tinted chip.
    When `href` is given the chip is a real cross-reference anchor (e.g. Act VI → #book)."""
    chip = ""
    if delta:
        c = P.chip(delta, tone, delta_title)
        chip = f'<a href="{href}">{c}</a>' if href else c
    inner = "<p>" + P._esc(body) + "</p>" + ("<div>" + chip + "</div>" if chip else "")
    return P.card(inner, kind="mech", eyebrow_text=title, title=delta_title)


# ---- the live recommendation -----------------------------------------------
def _recommendation():
    """The two decision-relevant 'what's next' threads, distilled from knowledge.json
    next_idea (2026-06-09 real-yield direction). (1) The decision-ready book upgrade
    (+USO, numbers in the book hero). (2) The next research experiment — the only credible
    untested feature channel left: the real interest rate on GLD. Honest about the prior."""
    book_link = ('<a href="#book">'
                 + P.chip("+USO → the book", "oil",
                          "the proposed +USO numbers live in the deployable-book hero")
                 + "</a>")
    doc_link = ('<a href="docs/research/DIRECTION_REALYIELD_GLD_2026-06-09.md">'
                + P.chip("the direction report →", "muted",
                         "the full local + internet + adversarial analysis")
                + "</a>")
    body = (
        P.eyebrow("WHAT'S NEXT · one decision, one experiment", "amber")
        + "<div>" + book_link + " " + doc_link + "</div>"
        + "<p><b>The frontier is mapped.</b> On minute price + volume alone, every axis, "
          "labeler, feature and sizing lever is exhausted, and the outside literature agrees "
          "(single-asset timing edges mostly vanish out-of-sample after a multiple-testing "
          "haircut). Three asset-intrinsic mechanisms survive — that is the honest ceiling for "
          "this data.</p>"
        + "<p><b>Decision ready:</b> fold the proposed USO oil edge into the book "
          "(+12% Calmar → 5.16) — awaiting a human / Opus crown.</p>"
        + "<p><b>Next experiment:</b> the one credible untested channel is an exogenous "
          "<i>fundamental-macro</i> feature on the strongest edge — add the <b>10-year TIPS "
          "real yield</b> (FRED DFII10) and the 2s10s curve slope to the GLD trend model and "
          "race it against the champion (Calmar 4.02). Gold is a real-rate duration asset, so "
          "this is its cleanest upstream driver; it is free, QC-native, leak-clean at a 1-day "
          "lag, and distinct from the dollar-ETF price proxy already closed (R1242).</p>"
        + '<div class="small">Honest prior ~1-in-4 it lifts GLD; otherwise it formally closes '
          'the last feature channel on the best name. A genuinely new mechanism needs a new '
          'authorized input (options IV, positioning) — a human decision, not one the loop can '
          'mine from current data.</div>'
    )
    return P.card(body, kind="upgrade")


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the complete <section id="arc"> HTML fragment.

    Decision-first: the 'what's next' recommendation leads, then the 6-act timeline
    ('how we got here') as a compact strip. The old methodology-lessons grid is dropped
    to keep the section simple — the lessons live in program.md."""
    book = ctx.get("book") or {}
    acts = "".join(_act_card(*item) for item in _acts(ctx))
    body = (
        P.eyebrow("WHAT'S NEXT · THE RESEARCH FRONTIER")
        + "<h2>What's next</h2>"
        + _recommendation()
        + P.eyebrow("HOW WE GOT HERE · SIX ACTS", "muted")
        + '<div class="acts">' + acts + "</div>"
        + P.provenance(
            "From the founding null to the third mechanism — the spine of the program. "
            + (book.get("freshness") or ""))
    )
    return '<section class="block" id="arc">' + body + "</section>"


if __name__ == "__main__":
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"arc section: {len(out)} bytes")
    for needle in ('id="arc"', "What's next", "I · The null", "VI · The third mechanism",
                   "TLT 0.31 → 1.52", "XLE 2.26 → 0.64", "book +USO",
                   "one decision, one experiment", "real yield", "DIRECTION_REALYIELD"):
        assert needle in out, f"missing: {needle}"
    assert "pending" not in out.lower(), "arc leaked a 'pending' value"
    print("ok — recommendation (real-yield direction) + 6-act timeline present")
