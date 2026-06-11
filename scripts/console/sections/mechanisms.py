#!/usr/bin/env python3
"""console.sections.mechanisms — the THREE confirmed edge-mechanism triptych.

Blueprint section "mechanisms" (primary): the mature taxonomy of *what actually
works*, currently rendered nowhere.

  [0] TREND-MOMENTUM      GLD            — the one true standalone ML edge.
  [1] MACRO-REGIME        UUP            — framed honestly as a DECORRELATOR
                                           (DSR 0.46 / significant=False), not a
                                           co-equal standalone Calmar edge.
  [2] OIL MEAN-REVERSION  USO/UCO/XOP    — the NEW 3rd mechanism, with the
                                           R1196-1206 discover->...->book-additive
                                           arc and the +USO 4.62->5.16 upgrade pill.

Pure ctx -> HTML. Reads ONLY ctx["edges"] (edges_resolver) + ctx["book"]["upgrade"]
(book_resolver) — no file-I/O — and invents no CSS: every class is emitted through a
console.primitives helper that OWNS it, so an undefined-class bug is impossible.
"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from console import primitives as P  # noqa: E402  (scripts/ on path above)


# ---- tiny formatters -------------------------------------------------------
def _fmt(v, d=2):
    try:
        return format(float(v), f".{d}f")
    except (TypeError, ValueError):
        return "—"


def _trades(t):
    try:
        return str(int(float(t)))
    except (TypeError, ValueError):
        return "—"


# ---- card pieces -----------------------------------------------------------
def _asset_chips(e):
    """Asset pills, tinted by the edge's role; oil also flies the NEW badge."""
    tint = e.get("tint", "muted")
    kind = tint if tint in {"edge", "decorr", "oil"} else "muted"
    chips = [P.chip(a, kind, e.get("character", "")) for a in (e.get("assets") or [])]
    if e.get("new"):
        chips.append(P.chip("NEW · 3rd mechanism", "new",
                            "oil mean-reversion — confirmed across the R1196-1206 arc"))
    return " ".join(chips)


def _kpis(e):
    """The two headline numerals per card: Calmar (teal) + DSR (green if significant)."""
    is_oil = e.get("id") == "oil"
    cal = P.kpi("Calmar", _fmt(e.get("calmar")), tone="acc",
                sub=("USO screen" if is_oil else ""),
                title="real out-of-sample Calmar (CAGR / |MaxDD|)")
    dsr_v = e.get("dsr")
    dsr_tone = "pos" if (e.get("significant") is True and dsr_v is not None) else ""
    dsr_title = "Deflated Sharpe Ratio · " + (e.get("dsr_source") or "from per_etf_best")
    dsr = P.kpi("DSR", _fmt(dsr_v), tone=dsr_tone,
                sub=("XOP proxy" if is_oil else ""), title=dsr_title)
    return '<div class="stats">' + cal + dsr + '</div>'


def _role_chip(e):
    """ONE honesty-framing pill per mechanism (detail lives in the hover title +
    the honesty matrix + the leaderboard drawer)."""
    eid = e.get("id")
    if eid == "oil":
        return P.chip("gate-confirmed real", "pos",
                      "permute collapses to ~-0.09 · decay + cost survive · DSR-positive · "
                      f"{_trades(e.get('trades'))} trades · val_auc {_fmt(e.get('val_auc'))}")
    if eid == "regime":
        return P.chip("regime decorrelator", "decorr",
                      "significant=False — its job is decorrelation, not standalone Calmar; "
                      f"{_trades(e.get('trades'))} trades")
    if e.get("significant") is True:
        ti = "survives the multiple-testing correction"
        if e.get("psr") is not None:
            ti += " · PSR " + _fmt(e.get("psr"))
        if e.get("n_trials") is not None:
            ti += f" · n_trials {e.get('n_trials')}"
        return P.chip("survives Bonferroni", "pos", ti + f" · {_trades(e.get('trades'))} trades")
    return ""


def _edge_card(e):
    """One .card--edge per mechanism: chips · two numbers · one sentence. Recipes,
    trade counts and the +USO upgrade story live in the leaderboard drawer / #arc."""
    body = (
        "<div>" + _asset_chips(e) + "</div>"
        + _kpis(e)
        + "<div>" + _role_chip(e) + "</div>"
        + "<p>" + P._esc(e.get("why", "")) + "</p>"
    )
    return P.card(body, kind="edge", eyebrow_text=e.get("mechanism", ""),
                  id="mech-" + str(e.get("id", "")), title=e.get("character", ""))


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the full <section id="mechanisms"> HTML fragment."""
    edges = ctx.get("edges") or []
    book = ctx.get("book") or {}
    cards = "".join(_edge_card(e) for e in edges)
    body = (
        P.eyebrow("THE TAXONOMY · WHAT ACTUALLY WORKS")
        + "<h2>Three confirmed edge mechanisms</h2>"
        + '<div class="stats">' + cards + "</div>"
        + P.provenance(
            "Which mechanism wins is a property of the asset, not a choice — "
            "each labeler fails on the other mechanisms' assets. "
            + (book.get("freshness") or ""))
    )
    return '<section class="block" id="mechanisms">' + body + "</section>"


if __name__ == "__main__":
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"mechanisms section: {len(out)} bytes")
    for needle in ("TREND-MOMENTUM", "MACRO-REGIME", "OIL MEAN-REVERSION",
                   "NEW · 3rd mechanism"):
        assert needle in out, f"missing: {needle}"
    assert "recipe ·" not in out, "recipes belong in the leaderboard drawer now"
    print("ok — three tight mechanism cards, no recipe small-print")
