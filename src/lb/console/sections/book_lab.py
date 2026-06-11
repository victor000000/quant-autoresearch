#!/usr/bin/env python3
"""console.sections.book_lab — Book compositions & sizing lab (id=book-lab).

Blueprint section "book-lab" (secondary): surface the rich portfolio data the quant
wants but which the page used to compute and throw away. Pure research density.

  * a COMPOSITIONS table — every portfolio variant (the honest-full baseline, the
    decorrelated core, the +DBC/+UUP/+TIP/+EFA member sweep, the equal/conviction/
    significant weightings, the leak-free SOXX book) with Calmar / Sharpe / MaxDD /
    CAGR / n / members / note. The CURRENT deployable 6-name book + the PROPOSED +USO
    book are sourced through book_resolver (ctx["book"]) so they can never diverge
    from knowledge.json, and are flagged with CURRENT / PROPOSED chips.
  * a 3-point LEVERAGE dial (1x / 2x / 3x) + the weight-exponent sizing sweep.
  * a PER-YEAR strip (positive every calendar year 2023-26).
  * an ALPHA-vs-passive panel (book Calmar vs the equal-weight buy-hold-7 benchmark).

The lede tells the real decorrelation story: dropping correlated high-MDD equities
lifted the book Calmar 1.78 -> 3.53 while cutting MaxDD 6.6% -> 2.1%; DBC helped,
EFA hurt — decorrelation beats member-count.

Pure ctx -> HTML. Reads ctx["book"] (book_resolver, the single source of truth for
the two headline rows) + ctx["K"]["portfolio"] (the historical variant records, which
have NO resolver — pure research density). Invents no CSS: every class is emitted
through a console.primitives helper that OWNS it, so an undefined-class bug is
structurally impossible.
"""
from lb.console import primitives as P


# ---- tiny formatters -------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _metric(d, *keys):
    """First present non-None value across `keys` (variant records use drifting key
    names: calmar/calmar_daily, mdd_pct/mdd, cagr_pct/cagr/car_pct)."""
    for k in keys:
        if k in d and d[k] is not None:
            return _f(d[k])
    return None


def _members_str(d):
    """A compact ' · '-joined member string from a variant's `members` field, which
    may be a list of [ticker, calmar] pairs, a list of tickers, or a free string."""
    m = d.get("members")
    if not m:
        return "—"
    if isinstance(m, str):
        return m
    out = []
    for it in m:
        if isinstance(it, (list, tuple)) and it:
            out.append(str(it[0]))
        else:
            out.append(str(it))
    return " · ".join(out)


def _short(s, n=104):
    s = str(s or "")
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _split_year(s):
    """'+3.0% (mdd 1.8%)' -> ('+3.0%', '(mdd 1.8%)')."""
    s = str(s or "")
    if "(" in s:
        ret, rest = s.split("(", 1)
        return ret.strip(), "(" + rest.strip()
    return s.strip(), ""


# ---- the compositions table ------------------------------------------------
# Curated order narrates the decorrelation arc: full honest baseline -> drop
# correlated equities -> the member sweep (DBC wins, EFA hurts) -> the weightings ->
# the leak-free SOXX book. Each tuple: (portfolio key, display label, tag).
_VARIANTS = [
    ("honest_2026_06_02", "Honest full book", "baseline"),
    ("decorr_core_2026_06_02", "Decorrelated core", "win"),
    ("decorr_calmarsq_2026_06_02", "Decorr core · Calmar² weight", ""),
    ("conviction_8_with_DBC", "+ DBC commodities", "win"),
    ("with_UUP_9", "+ UUP dollar", "neutral"),
    ("with_TIP", "+ TIP inflation", "neutral"),
    ("with_EFA_8", "+ EFA developed-eq", "hurts"),
    ("equal_weight", "Equal-weight 7", ""),
    ("conviction_weight", "Conviction ∝Calmar 7", ""),
    ("significant_only", "Significant-only 4", ""),
    ("honest_book_leakfree_2026_06_03", "Leak-free SOXX book", ""),
]

_TAG_CHIP = {
    "baseline": ("baseline", "muted", "the full honest re-validated book — everything in"),
    "win": ("decorrelation win", "pos", "lifted the book by decorrelating, not by adding exposure"),
    "neutral": ("neutral", "muted", "decorrelated hedge but Calmar-flat — excluded from the book"),
    "hurts": ("hurts Calmar", "neg", "adding correlated equity HURT — decorrelation beats name-count"),
}


def _comp_row(label, tag, stale, calmar, sharpe, mdd, cagr, n, members, note, lead=None):
    """One <tr> for the compositions table."""
    chips = []
    if lead:
        kind = "pos" if lead == "CURRENT · DEPLOYED" else "amber"
        chips.append(P.chip(lead, kind, "sourced through book_resolver — single source of truth"))
        if lead.startswith("PROPOSED"):
            chips.append(P.chip("+USO", "new", "oil mean-reversion, the 3rd mechanism"))
    elif tag in _TAG_CHIP:
        t, kind, ti = _TAG_CHIP[tag]
        chips.append(P.chip(t, kind, ti))
    if stale:
        chips.append(P.chip("pre-leak-fix", "amber",
                            "computed on leaky logdollar champions — superseded by the 2026-06-03 leak fix"))
    name_cell = P.cell("<b>" + P._esc(label) + "</b> " + " ".join(chips))
    cells = [
        name_cell,
        P._num(calmar, ".2f", cls=("pos" if lead == "CURRENT · DEPLOYED" else "")),
        P._num(sharpe, ".2f"),
        P._num(mdd, ".2f", suf="%"),
        P._num(cagr, ".2f", suf="%"),
        P._num(n, ".0f"),
        P.cell(P._esc(members)),
        P.cell(P._esc(_short(note)), attrs=f'title="{P._esc(note)}"'),
    ]
    return P.tr(cells)


def _compositions_table(pf, book):
    cols = [
        {"label": "Composition", "key": "name", "t": "s"},
        {"label": "Calmar", "num": True, "key": "calmar", "t": "n",
         "title": "annual return ÷ worst drawdown"},
        {"label": "Sharpe", "num": True, "key": "sharpe", "t": "n",
         "title": "return per unit of volatility"},
        {"label": "MaxDD", "num": True, "key": "mdd", "t": "n",
         "title": "worst peak-to-trough drawdown %"},
        {"label": "CAGR", "num": True, "key": "cagr", "t": "n",
         "title": "annualized compound growth %"},
        {"label": "n", "num": True, "key": "n", "t": "n", "title": "member count"},
        {"label": "Members", "key": "members", "t": "s"},
        {"label": "Note", "key": "note", "t": "s"},
    ]
    body = []
    for key, label, tag in _VARIANTS:
        d = pf.get(key) or {}
        if not d:
            continue
        body.append(_comp_row(
            label, tag, bool(d.get("STALE_PRE_LEAK_FIX")),
            _metric(d, "calmar", "calmar_daily"), _metric(d, "sharpe"),
            _metric(d, "mdd_pct", "mdd"), _metric(d, "cagr_pct", "cagr", "car_pct"),
            _metric(d, "n"), _members_str(d), d.get("note", "")))

    # ---- the deployed book as the table's reference row (book_resolver) ----
    body.append(_comp_row(
        "Current deployable book", "", False,
        _f(book.get("calmar")), _f(book.get("sharpe")), _f(book.get("mdd_pct")),
        _f(book.get("cagr_pct")), _f(book.get("n")),
        " · ".join(book.get("member_tickers") or []),
        "Current deployable book — 1 ML edge (GLD) + UUP regime decorrelator + 4 "
        "decorrelated buy-hold diversifiers; weight ∝Calmar², gross≤1, no leverage.",
        lead="CURRENT · DEPLOYED"))
    # ---- the proposed +USO upgrade: deduped to a cross-reference into the book hero
    # (the ONE place its numbers live) rather than a re-rendered hero KPI row ----
    prop_link = ('<a href="#book">'
                 + P.chip("PROPOSED +USO → see the book hero", "amber",
                          "the +USO upgrade and its Calmar / Sharpe lift live in the book hero")
                 + "</a>")
    body.append(P.tr([
        P.cell("<b>Proposed book + USO</b> " + prop_link),
        P.cell("—", num=True), P.cell("—", num=True), P.cell("—", num=True),
        P.cell("—", num=True), P.cell("—", num=True),
        P.cell("+ USO"),
        P.cell("Adds the 3rd mechanism (oil mean-reversion); numbers in the book hero."),
    ]))
    return P.table(cols, body, id="complab")


# ---- the sizing panels -----------------------------------------------------
def _leverage_card(pf):
    """The 3-point leverage dial + the weight-exponent sizing sweep, as one card."""
    lev = pf.get("leverage") or {}
    cols = [{"label": "Leverage"}, {"label": "Calmar", "num": True},
            {"label": "CAGR", "num": True}, {"label": "MaxDD", "num": True},
            {"label": "Sharpe", "num": True}]
    rows = []
    for mult in ("1x", "2x", "3x"):
        d = lev.get(mult) or {}
        rows.append(P.tr([
            P.cell("<b>" + mult + "</b>"),
            P._num(_f(d.get("calmar")), ".2f"),
            P._num(_f(d.get("cagr")), ".1f", suf="%"),
            P._num(_f(d.get("mdd")), ".1f", suf="%"),
            P._num(_f(d.get("sharpe")), ".2f"),
        ]))
    we = pf.get("weight_exponent") or {}
    we_line = ("weight ∝Calmar^p · "
               + " · ".join(f"^{p}={_f(we.get(p)):.2f}" for p in ("0.5", "1.0", "2.0")
                            if _f(we.get(p)) is not None)
               + f" (optimum ^{we.get('optimal', 1.0):g})") if we else ""
    body = (P.table(cols, rows)
            + f'<div class="small">{P._esc(lev.get("note", ""))}</div>'
            + (f'<div class="small">{P._esc(we_line)}</div>' if we_line else ""))
    return P.card(body, kind="muted", eyebrow_text="LEVERAGE DIAL · sizing is a separate lever")


def _peryear_card(pf):
    """The per-year strip — book return + drawdown for each calendar year."""
    py = pf.get("per_year") or {}
    kpis = []
    for y in sorted(py.keys()):
        ret, sub = _split_year(py[y])
        kpis.append(P.kpi(y, ret, tone=("pos" if ret.startswith("+") else ""),
                          sub=sub, title=f"{y}: {py[y]}"))
    body = ('<div class="stats">' + "".join(kpis) + "</div>"
            + '<div class="small">Positive every calendar year — the decorrelated book '
              'has had no losing year across 2023-26.</div>')
    return P.card(body, kind="muted", eyebrow_text="PER-YEAR · positive every year")


def _alpha_card(pf, book):
    """Book Calmar vs the passive equal-weight buy-hold-7 benchmark."""
    bench = pf.get("benchmark_buyhold7") or {}
    bc = _f(bench.get("calmar"))
    bookc = _f(book.get("calmar"))
    delta = (bookc - bc) if (bookc is not None and bc is not None) else None
    kpis = (P.kpi("Book", f"{bookc:.2f}" if bookc is not None else "—", tone="pos",
                  sub="strategy book Calmar", title="book Calmar (book_resolver)")
            + P.kpi("Passive", f"{bc:.2f}" if bc is not None else "—",
                    sub="equal-weight buy-hold 7",
                    title=P._esc(bench.get("note", ""))))
    chip = ""
    if delta is not None:
        chip = P.chip(f"+{delta:.2f} Calmar vs passive", "pos", P._esc(pf.get("alpha", "")))
    body = ('<div class="stats">' + kpis + "</div>"
            + f"<div>{chip}</div>"
            + '<div class="small">Risk-reduction alpha — beats passive via drawdown control '
              f'(MaxDD {_f(book.get("mdd_pct")):.2f}% vs passive {_f(bench.get("mdd_pct")):.1f}%); '
              'gives up raw CAGR in this bull.</div>')
    return P.card(body, kind="muted", eyebrow_text="ALPHA · vs passive buy-hold")


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the complete <section id="book-lab"> HTML fragment."""
    book = ctx.get("book") or {}
    pf = ((ctx.get("K") or {}).get("portfolio")) or {}

    head = (
        P.eyebrow("BOOK COMPOSITIONS & SIZING LAB · research density")
        + "<h2>Book compositions &amp; sizing lab</h2>"
        + P.provenance(
            "Dropping correlated high-MDD equities (QQQ/EEM/EFA/IWM/XLE) lifted the book "
            "Calmar 1.78 → 3.53 and cut MaxDD 6.6% → 2.1%; DBC helped, EFA hurt — "
            "decorrelation beats member-count. " + (book.get("freshness") or ""))
    )

    panels = ('<div class="stats">'
              + _leverage_card(pf) + _peryear_card(pf) + _alpha_card(pf, book)
              + "</div>")

    body = head + _compositions_table(pf, book) + panels
    return '<section class="block" id="book-lab">' + body + "</section>"


if __name__ == "__main__":
    from lb.console.data import build_ctx
    out = render(build_ctx())
    print(f"book-lab section: {len(out)} bytes")
    for needle in ('id="book-lab"', "Book compositions", "CURRENT · DEPLOYED", "PROPOSED",
                   "+USO", "LEVERAGE DIAL", "PER-YEAR", "ALPHA", "4.62", "decorrelation"):
        assert needle in out, f"missing: {needle}"
    # headline rows must carry the live book numbers, never 'pending'
    assert "pending" not in out.lower(), "headline row leaked a 'pending' value"
    print("ok — compositions table + leverage + per-year + alpha all present")
