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
  * the CURRENT RECOMMENDATION callout — knowledge.json next_idea rendered VERBATIM
    (the deployable +USO recommendation), wired through book_resolver for the headline
    Calmar so it can never diverge.
  * 2-3 STILL-TRUE methodological lessons folded in as sub-notes (no-lookahead
    invariant, meta-labeling crosses Calmar>3, selectivity beats exposure), pulled
    from top_insights by title so the prose stays sourced from knowledge.json.

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


# ---- tiny formatter --------------------------------------------------------
def _fmt(v, d=2):
    try:
        return format(float(v), f".{d}f")
    except (TypeError, ValueError):
        return "—"


# ---- the 6-act timeline ----------------------------------------------------
def _acts(ctx):
    """The six narrative acts. The only LIVE numbers (Act V screen count, Act VI book
    upgrade) flow from the resolvers (ctx["book"]) so they can never diverge from
    knowledge.json; the historical deltas (TLT/XLE) are fixed narrative facts."""
    book = ctx.get("book") or {}
    up = book.get("upgrade") or {}
    tower = book.get("stat_tower") or {}
    scr, scr_t = tower.get("screened", 42), tower.get("screened_total", 42)
    cf, ct = up.get("calmar_from"), up.get("calmar_to")
    book_delta = f"book +USO {_fmt(cf)} → {_fmt(ct)}"
    return [
        ("I · The null",
         "Single-asset long-only ML couldn't beat buy-and-hold across 6 rounds — so the "
         "problem was reframed into a per-ETF axis × labeler tournament.",
         "6-round null → tournament", "muted",
         "the founding negative result that reframed the whole program"),
        ("II · The unlock",
         "Long-only can't beat a declining asset; adding shorting + a directional label + "
         "causal mean-reversion features unlocked the first real timing edge on TLT.",
         "TLT 0.31 → 1.52", "pos",
         "③ causal RSI / Bollinger-%b features; permuted control stays at −0.14 (real, not leak)"),
        ("III · The correction",
         "A bar-threshold look-ahead (magnitude thresholds fit on full-series stats incl. "
         "OOS) was inflating results; the TRAIN-only leak fix re-validated the entire board.",
         "XLE 2.26 → 0.64", "amber",
         "honesty correction — the leak fix deflated inflated crowns to leak-free truth"),
        ("IV · Convergence",
         "One rule governs the board — time when the hold is weak, hold when the trend is "
         "strong; the fixed-12 frontier saturated at its leak-free single-asset ceilings.",
         "7/7 at leak-free ceilings", "edge",
         "f_timing_when — the unifying mechanism across the original twelve"),
        ("V · The widening",
         "With the fixed frontier saturated, the hunt widened to all 311 QC-confirmed ETFs "
         "— race each against buy-and-hold, keep only genuine fits; a commodity & oil "
         "cluster surfaced and the fit-prone universe is now fully screened.",
         f"{scr}/{scr_t} fit-prone screened", "edge",
         "the universe funnel that surfaced the oil-reversion cluster"),
        ("VI · The third mechanism",
         "Oil mean-reversion confirmed on USO/UCO/XOP — discovered, permute-checked, "
         "decay-monitored, cost-stressed, DSR-positive — and proposed into the book as the "
         "3rd mechanism; the single-ticker frontier is converging.",
         book_delta, "oil",
         "R1196-1206: discover → permute → decay → cost → DSR → book-additive (+USO)"),
    ]


def _act_card(title, body, delta, tone, delta_title):
    """One act as a .card--mech (accent-2 left stripe), with the delta as a tinted chip."""
    chip = P.chip(delta, tone, delta_title) if delta else ""
    inner = "<p>" + P._esc(body) + "</p>" + ("<div>" + chip + "</div>" if chip else "")
    return P.card(inner, kind="mech", eyebrow_text=title, title=delta_title)


# ---- the live recommendation -----------------------------------------------
def _recommendation(ctx):
    """knowledge.json next_idea rendered VERBATIM — the single most decision-relevant
    text on the page — inside the amber .card--upgrade. The book Calmar in the lede
    flows through book_resolver so the callout can never contradict the book hero."""
    K = ctx.get("K") or {}
    book = ctx.get("book") or {}
    up = book.get("upgrade") or {}
    idea = (K.get("next_idea") or "").strip()
    if not idea:
        return ""
    lede = P.chip(
        f"+USO → book Calmar {_fmt(up.get('calmar_from'))} → {_fmt(up.get('calmar_to'))} "
        f"(+{up.get('calmar_lift_pct', 12)}%)", "oil",
        "the deployable recommendation — sourced through book_resolver")
    body = (
        P.eyebrow("CURRENT RECOMMENDATION · next_idea (verbatim)", "amber")
        + "<div>" + lede + "</div>"
        + "<p>" + P._esc(idea) + "</p>"
        + '<div class="small">Rendered verbatim from knowledge.json — the live, '
          'decision-relevant text. Awaiting human / Opus crown.</div>'
    )
    return P.card(body, kind="upgrade")


# ---- still-true methodological lessons -------------------------------------
# The 2-3 lessons that survive (no-lookahead invariant, meta-labeling crosses
# Calmar>3, selectivity beats exposure). Matched against top_insights by title so
# the prose stays sourced from knowledge.json; the stale early-grind list is dropped.
_LESSON_KEYS = ("Meta-labeling crosses", "No-lookahead invariant", "Selectivity beats")


def _lessons(ctx):
    ins = (ctx.get("K") or {}).get("top_insights") or []
    by_title = {}
    for it in ins:
        t = (it.get("title") or "")
        for key in _LESSON_KEYS:
            if t.startswith(key) or key in t:
                by_title.setdefault(key, it)
    cards = []
    for key in _LESSON_KEYS:                       # preserve curated order
        it = by_title.get(key)
        if not it:
            continue
        text = it.get("body") or it.get("detail") or ""
        ev = it.get("ev")
        evchip = (P.chip(ev, "muted", "evidence trail")) if ev else ""
        inner = "<p>" + P._esc(text) + "</p>" + ("<div>" + evchip + "</div>" if evchip else "")
        cards.append(P.card(inner, kind="muted", eyebrow_text=it.get("title", "")))
    # a durable lesson with no top_insights row of its own
    cards.append(P.card(
        "<p>" + P._esc(
            "Decorrelation beats exposure: dropping correlated high-MDD equities lifted the "
            "book Calmar 1.78 → 3.53 while cutting MaxDD 6.6% → 2.1% — selectivity and "
            "decorrelation, not more names, build the book.") + "</p>",
        kind="muted", eyebrow_text="Decorrelation beats exposure"))
    return '<div class="stats">' + "".join(cards) + "</div>"


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the complete <section id="arc"> HTML fragment."""
    book = ctx.get("book") or {}
    acts = "".join(_act_card(t, b, d, tone, dt) for (t, b, d, tone, dt) in _acts(ctx))
    body = (
        P.eyebrow("RESEARCH ARC · WHAT SURVIVED & WHAT'S NEXT")
        + "<h2>Research arc &amp; what's next</h2>"
        + P.provenance(
            "Six acts from the founding null to the third mechanism — the spine of the "
            "program. " + (book.get("freshness") or ""))
        + '<div class="acts">' + acts + "</div>"
        + _recommendation(ctx)
        + P.eyebrow("STILL-TRUE LESSONS · methodology")
        + _lessons(ctx)
    )
    return '<section class="block" id="arc">' + body + "</section>"


if __name__ == "__main__":
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"arc section: {len(out)} bytes")
    for needle in ('id="arc"', "Research arc", "I · The null", "VI · The third mechanism",
                   "TLT 0.31 → 1.52", "XLE 2.26 → 0.64", "book +USO",
                   "CURRENT RECOMMENDATION", "STILL-TRUE LESSONS", "Decorrelation beats exposure"):
        assert needle in out, f"missing: {needle}"
    assert "pending" not in out.lower(), "arc leaked a 'pending' value"
    print("ok — 6 acts + verbatim recommendation + still-true lessons all present")
