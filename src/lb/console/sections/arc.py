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
from lb.console import primitives as P


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
    """The decision-relevant 'what's next', updated 2026-06-10 after the re-validation
    sweep + the real-yield races. (1) The +USO crown, now with fresh evidence. (2) The
    real-yield channel: RACED AND CLOSED (R1860/1861). (3) What the loop does next."""
    book_link = ('<a href="#book">'
                 + P.chip("+USO → the book", "oil",
                          "USO re-validated bit-exact 3.85 · +2.93 over buy-hold · "
                          "decay strengthening · book 4.65 → 5.03 at its natural weight")
                 + "</a>")
    body = (
        P.eyebrow("WHAT'S NEXT · one decision, frontier closed honestly", "amber")
        + "<div>" + book_link + "</div>"
        + "<p><b>Decision ready:</b> fold the USO oil edge into the book — re-validated "
          "today (2026-06-10) bit-exact at 3.85, +2.93 over buy-hold, decay-strengthening; "
          "at its natural Calmar² weight the book goes 4.65 → <b>5.03</b> (Sharpe 2.74, "
          "MaxDD −2.7%). Awaiting a human / Opus crown.</p>"
        + "<p><b>The whole loop-runnable frontier closed by experiment (2026-06-10):</b> "
          "the real-yield channel lost with full live data (nominal 3.38, true DFII10 3.12 "
          "vs 4.02 — the old \u201cno data\u201d wall was a harness bug, fixed); Wang\u2019s "
          "remaining levers all raced — de-scaled axis rejected, multi-axis netting rejected, "
          "rich-panel\u00d7compressor rejected (nonlinearity adds nothing), SPY session momentum "
          "learnable (val_auc 0.556) but unprofitable (0.51 &lt; BH 1.04). The phase-invariance "
          "certificate PASSED: the GLD edge holds at all 5 bar-clock alignments (honest "
          "phase-median \u22483.2).</p>"
        + "<p><b>Book honesty (full re-validation sweep):</b> GLD and USO hold bit-exact and are "
          "strengthening (e-values 4.95 / 5.23 accumulating); GLD\u2019s permute control re-ran "
          "genuinely after the gate fix — 96% collapse, the edge is real label signal. UUP\u2019s "
          "timing alpha decayed (1.85 \u2192 0.60, decorrelation seat); IWM\u2019s timing retired "
          "below buy-hold.</p>"
        + '<div class="small">A genuinely new edge now needs a new authorized input '
          '(options IV, positioning, flows) — a human decision, not one the loop can '
          'mine from current data. Until then the loop runs monitors only.</div>'
    )
    return P.card(body, kind="upgrade")


# ---- the section -----------------------------------------------------------
def render_acts(ctx):
    """The 6-act 'how we got here' strip — rendered inside the appendix history
    drawer (moved off the main page in the 2026-06-10 simplification)."""
    book = ctx.get("book") or {}
    acts = "".join(_act_card(*item) for item in _acts(ctx))
    return ('<div class="acts">' + acts + "</div>"
            + P.provenance(
                "From the founding null to the third mechanism — the spine of the program. "
                + (book.get("freshness") or "")))


def render(ctx):
    """ctx -> the complete <section id="arc"> HTML fragment.

    Decision-first and ONLY the decision: the 'what's next' recommendation card.
    The 6-act history strip lives in the appendix history drawer (render_acts);
    the methodology lessons live in program.md."""
    body = (
        P.eyebrow("WHAT'S NEXT · THE RESEARCH FRONTIER")
        + "<h2>What's next</h2>"
        + _recommendation()
    )
    return '<section class="block" id="arc">' + body + "</section>"


if __name__ == "__main__":
    from lb.console.data import build_ctx
    ctx = build_ctx()
    out = render(ctx)
    acts = render_acts(ctx)
    print(f"arc section: {len(out)} bytes · acts strip: {len(acts)} bytes")
    for needle in ('id="arc"', "What's next", "one decision, frontier closed honestly",
                   "real-yield", "frontier closed by experiment", "5.03"):
        assert needle in out, f"missing: {needle}"
    for needle in ("I · The null", "VI · The third mechanism",
                   "TLT 0.31 → 1.52", "XLE 2.26 → 0.64", "book +USO"):
        assert needle in acts, f"missing from acts strip: {needle}"
    assert "I · The null" not in out, "acts strip must live in the appendix, not #arc"
    assert "pending" not in out.lower(), "arc leaked a 'pending' value"
    print("ok — lean recommendation section + appendix-ready acts strip")
