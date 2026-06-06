#!/usr/bin/env python3
"""console.sections.book — THE deployable-book HERO band (id=book).

The single above-the-fold answer: what to deploy, how good, and the pending
upgrade. Pure ctx-in -> HTML-out; every headline number flows through
book_resolver() (ctx["book"]), so the band can never show 'pending' or contradict
knowledge.json. It is the ONLY surviving home of the live /data.json poller (the
#nowrunning block the 8s poll patches).

Layout (two-column desktop / stacked mobile via the existing .statushero grid):
  LEFT  — eyebrow · headline · KPI row · member chips · verdict · +USO upgrade pill
  RIGHT — live status block (#nowrunning) · "what backs it" stat tower
  below — one freshness/provenance stamp
"""
import os
import sys
import html as _html

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:                       # console.data also self-inserts scripts/
    sys.path.insert(0, _SCRIPTS)
from console import primitives as P  # noqa: E402


def _esc(s):
    return _html.escape("" if s is None else str(s))


def _fmt(v, suf="", dash="—"):
    """Tabular 2dp formatter that degrades to an em-dash, never crashing on None."""
    if v is None:
        return dash
    try:
        return format(float(v), ".2f") + suf
    except (TypeError, ValueError):
        return _esc(v)


def render(ctx):
    """ctx -> the complete <section class="hero" id="book"> band (string)."""
    b = ctx["book"]
    up = b.get("upgrade", {}) or {}

    # ---- LEFT: the headline ------------------------------------------------
    eyebrow = P.eyebrow("DEPLOYABLE BOOK · leak-free · re-derived 2026-06-06", "pos")
    title = (f'<h1>The deployable book '
             f'{P.chip(str(b["n"]) + " names", "pos")} '
             f'{P.chip("leak-free", "edge")}</h1>')

    # KPI row — large tabular numerals, EVERY value from book_resolver.
    kpis = "".join([
        P.kpi("Calmar", _fmt(b.get("calmar")), tone="pos",
              sub="annual return ÷ worst drawdown",
              title=f'deployable-book Calmar ({b.get("source_key", "")})'),
        P.kpi("Sharpe", _fmt(b.get("sharpe")), sub="risk-adjusted return"),
        P.kpi("MaxDD", _fmt(b.get("mdd_pct"), "%"), tone="pos", sub="worst peak-to-trough"),
        P.kpi("CAGR", _fmt(b.get("cagr_pct"), "%"), tone="pos", sub="annualized growth"),
        P.kpi("Positive yrs", _esc(b.get("positive_years")), sub="every calendar year"),
        P.kpi("Names", str(b.get("n", "")), sub="decorrelated members"),
    ])
    kpirow = f'<div class="stats">{kpis}</div>'

    # member chips, tinted by role: edge (GLD) / decorr (UUP) / muted (diversifiers)
    chips = " ".join(
        P.chip(m.get("ticker", ""), m.get("tint", "muted"), title=m.get("role", ""))
        for m in b.get("members", []))
    members = (P.eyebrow("MEMBERS · TINTED BY ROLE", "muted") + f"<div>{chips}</div>")

    verdict = f'<p class="tldr">{_esc(b.get("verdict"))}</p>'

    # +USO upgrade pill (amber .card--upgrade) — numbers via book_resolver.upgrade.
    up_body = (
        P.chip("PROPOSED", "amber") + " "
        + P.chip("+" + str(up.get("add_member", "USO")), "new") + " "
        + _esc(str(up.get("text", "")).replace("->", "→"))
        + f'<div class="small">{_esc(up.get("vehicle_note", ""))}</div>')
    upgrade = (f'<div style="margin-top:1.1rem">'
               + P.card(up_body, kind="upgrade",
                        eyebrow_text="PROPOSED UPGRADE · awaiting human/Opus crown")
               + "</div>")

    left = f"<div>{eyebrow}{title}{kpirow}{members}{verdict}{upgrade}</div>"

    # ---- RIGHT: live poller + "what backs it" stat tower -------------------
    # #nowrunning is the live target the existing 8s /data.json poll overwrites;
    # it sits INSIDE the card so the poller's className churn can't strip the surface.
    live_inner = ('<div id="nowrunning"><h2><span class="idledot"></span>Status</h2>'
                  '<p class="small">loading live state… <code>data.json</code> polled every 8s</p></div>')
    live = P.card(live_inner, kind="metric", eyebrow_text="LIVE")

    t = b.get("stat_tower", {}) or {}
    tower_body = " ".join([
        P.chip(f'{t.get("mechanisms", 3)} confirmed mechanisms', "edge"),
        P.chip(f'{t.get("screened", 42)}/{t.get("screened_total", 42)} fit-prone ETFs screened', "muted"),
        P.chip(f'{t.get("lenses", 7)} honesty lenses', "muted"),
    ])
    tower = (f'<div style="margin-top:1.1rem">'
             + P.card(tower_body, kind="muted", eyebrow_text="WHAT BACKS IT")
             + "</div>")

    right = f"<div>{live}{tower}</div>"

    # ---- assemble the band -------------------------------------------------
    band = f'<div class="statushero">{left}{right}</div>'
    freshness = P.provenance(b.get("freshness", ""))
    return f'<section class="hero" id="book">{band}{freshness}</section>'
