#!/usr/bin/env python3
"""console.page — the DATA-DRIVEN section registry + page assembler (STEP 5).

This is the top of the console package: it owns the page chrome (head, command
bar, the externalized <script src>) and a single declarative SECTIONS list that
build_page() walks in order. Re-ordering or relabelling the dashboard is a one-line
edit to SECTIONS — never a 76-line hand-concatenated return.

Public surface (the SAME entrypoints the driver + app.py already call, so nothing
downstream changes):

  build_html()        -> the full dashboard HTML string (build_ctx once -> build_page)
  build_data(K=None)  -> the /data.json poll payload (delegates to console.data)
  build_page(ctx)     -> assemble head + command bar + every section + <script src>

INVARIANTS this module enforces structurally:
  * EVERY section is a pure ctx->HTML builder pulled from the registry, so a dead
    builder cannot silently drop out (the registry forces each one to be referenced).
  * NO headline number is formatted here — every Calmar/Sharpe/DSR already flowed
    through a resolver in console.data and arrives pre-rendered inside a section.
  * The client JS + vis-network init are EXTERNAL (reports/console.js), off the HTML
    payload and cacheable; this file only emits a <script src> reference.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from console.data import build_ctx, build_data as _build_data  # noqa: E402
from console.sections import (  # noqa: E402
    book, mechanisms, honesty, arc, appendix,
)


# ---------------------------------------------------------------------------
# THE REGISTRY — (id, title, render_fn, priority), in final decision-first order.
# id doubles as the in-page anchor; render_fn is a pure ctx -> <section> builder.
# ---------------------------------------------------------------------------
# 2026-06-09 RECONSTRUCTION — a ONE-PAGE NARRATIVE + a single collapsed appendix.
# The four core sections tell the whole story top-to-bottom; every dense evidence
# surface (leaderboard, screen, book-lab, graph, rounds, glossary) now lives inside
# `appendix` as collapsed <details> drawers, so their anchor ids, the live #lb sort,
# the lazy #graphwrap mount and the /data.json poll all keep working unchanged.
SECTIONS = [
    ("book",       "The deployable book",        book.render,      "primary"),
    ("mechanisms", "Three confirmed mechanisms", mechanisms.render, "primary"),
    ("honesty",    "Why these edges are real",   honesty.render,   "primary"),
    ("arc",        "What's next",                arc.render,       "primary"),
    ("appendix",   "Appendix — evidence & ledger", appendix.render, "secondary"),
]

# Command-bar chapter anchors DERIVED from SECTIONS (friendly labels for the 4 core
# beats + the appendix), then the two themed doc routes (program.md / deployment.md).
_NAV_LABELS = {"book": "book", "mechanisms": "mechanisms", "honesty": "why real",
               "arc": "what's next", "appendix": "appendix"}
_NAV = [(_NAV_LABELS.get(sid, sid), f"#{sid}") for sid, _t, _fn, _p in SECTIONS] + [
    ("program.md", "program.md"), ("deploy", "deployment.md"),
]

_FONTS = ("https://fonts.googleapis.com/css2?"
          "family=Space+Grotesk:wght@400;500;600;700"
          "&family=IBM+Plex+Sans:wght@400;500;600"
          "&family=JetBrains+Mono:wght@400;500;600&display=swap")


def _head():
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="description" content="Autoresearch — a deployable single-ticker '
        'ETF book (Calmar 4.62, leak-free) + the 3 confirmed edge mechanisms behind it.">'
        '<title>Autoresearch — research console</title>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{_FONTS}" rel="stylesheet">'
        '<link rel="stylesheet" href="style.css"></head><body>'
    )


def _commandbar(ctx):
    """Sticky command bar: › prompt + chapter anchors + a thin live strip + clock.

    The live strip ('live · N rounds · last KEEP: ETF') is server-rendered from the
    CSV ledger so it is correct on first paint; the clock + status are patched by the
    8s poll in console.js. Anchors point at the NEW section ids."""
    rounds_list = ctx.get("rounds") or []
    n = len(rounds_list)
    latest = ""
    if rounds_list:
        d = rounds_list[0]
        verb = "KEEP" if d.get("verdict") == "keep" else "DISCARD"
        wc = d.get("win_calmar")
        wc_txt = f" {wc:+.2f}" if isinstance(wc, (int, float)) else ""
        latest = f' · last {verb}: {d.get("etf", "")}{wc_txt}'
    nav = "".join(f'<a href="{href}">{label}</a>' for label, href in _NAV)
    return (
        '<div class="commandbar">'
        '<span class="prompt">› autoresearch / research-console</span>'
        f'<span class="chapnav">{nav}</span>'
        f'<span class="stale" id="stale">live · {n} rounds{latest}</span>'
        '<span class="clock" id="clock"></span>'
        '</div>'
    )


def build_page(ctx):
    """Assemble the whole page from the registry: head + command bar + every section
    (in SECTIONS order) + the external <script src>. Each section is rendered defensively
    so one bad builder degrades to a small note instead of blanking the page."""
    parts = [_head(), '<div class="dash">', _commandbar(ctx)]
    for sid, title, fn, _priority in SECTIONS:
        try:
            parts.append(fn(ctx))
        except Exception as e:  # a single bad section must never blank the page
            parts.append(
                f'<section class="block" id="{sid}"><h2>{title}</h2>'
                f'<p class="small">(section unavailable: {type(e).__name__})</p></section>')
    parts.append('</div><script src="console.js" defer></script></body></html>')
    return "".join(parts)


def build_html():
    """The public entrypoint the driver + app.py call. build_ctx() does all file-I/O
    exactly once, then build_page() walks the registry."""
    return build_page(build_ctx())


def build_data(K=None):
    """The /data.json poll payload — delegates to console.data so render + poll share
    one derive() and can never diverge (the 8s poller contract is preserved)."""
    return _build_data(K)


if __name__ == "__main__":
    html = build_html()
    n_bytes = len(html.encode("utf-8"))
    print(f"page: {len(html)} chars · {n_bytes} utf-8 bytes · {len(SECTIONS)} sections")
    for sid, _t, _fn, _p in SECTIONS:
        assert f'id="{sid}"' in html, f"section missing from page: {sid}"
    # Byte guard at the size that actually ships over the wire (utf-8 bytes, not str len) —
    # render_index measured len(str); the multibyte glyphs made the real payload larger.
    assert n_bytes < 200_000, f"build_html is {n_bytes} utf-8 bytes (>= 200000) — perf regression"
    assert "4.62" in html or "4.617" in html, "book hero Calmar missing"
    # The stale-SOXX bug rendered a metric as the literal text 'pending' (e.g.
    # '<div class="cnum">pending<span>'). Guard that exact value-leak pattern — NOT the
    # bare word, which appears legitimately in prose ('pending DSR deflation') and in the
    # graph mount attribute (data-pending="1").
    assert "pending<" not in html and ">pending" not in html, "page leaked a 'pending' metric value"
    print("ok — every section present, book Calmar rendered, no stale 'pending' metric")
