#!/usr/bin/env python3
"""console.primitives — the HTML building blocks every section builder uses.

DESIGN RULE: a primitive OWNS its CSS classes. A section builder must never emit a
raw class string; it calls a primitive, and the primitive emits classes that are
guaranteed to exist in reports/style.css. This makes the "unstyled-class" bug
(the 20 undefined screen classes) structurally impossible.

Two families:
  data viz   : _spark, _edgebar, _num   (ported verbatim from the monolith)
  layout     : card, table, chip, kpi, matrix, eyebrow, provenance
  supporting : cell, tr, tablewrap, _esc

Every function is pure (args -> HTML fragment string); no I/O, no globals.
"""
import html as _html

# ---------------------------------------------------------------------------
# CLASS OWNERSHIP MAP — the exact CSS classes each primitive emits.
# (Documentation only; style.css must define every class listed here.)
# ---------------------------------------------------------------------------
OWNS = {
    "_spark": [".spark", ".sparkna"],
    "_edgebar": [".edgecell", ".edgenum", ".edgebar", ".edgebar i.pos", ".edgebar i.neg"],
    "_num": [".num", ".num.pos", ".num.neg"],
    "eyebrow": [".eyebrow"],
    "provenance": [".provenance"],
    "chip": [".chip", ".chip--pos", ".chip--neg", ".chip--amber", ".chip--muted",
             ".chip--na", ".chip--edge", ".chip--decorr", ".chip--oil", ".chip--new"],
    "kpi": [".kpi", ".kpi .k", ".kpi .v", ".kpi .v.pos", ".kpi .v.neg", ".kpi .v.acc", ".kpi .sub"],
    "card": [".card", ".card--metric", ".card--edge", ".card--book", ".card--upgrade",
             ".card--mech", ".card--muted", ".card .card-eyebrow", ".card .card-body"],
    "table": [".tablewrap", "table", "thead", "tbody", "th", "td", ".num",
              ".sorted-asc", ".sorted-desc"],
    "matrix": [".matrix", ".matrix .mlabel", ".matrix .msub", ".mcell",
               ".mcell--pos", ".mcell--amber", ".mcell--neg", ".mcell--na"],
}

_TONES = {"pos", "neg", "amber", "muted", "na", "edge", "decorr", "oil", "new", "acc"}


def _esc(s):
    return _html.escape("" if s is None else str(s))


# ===========================================================================
# DATA-VIZ PRIMITIVES (ported from the monolith, unchanged behaviour)
# ===========================================================================
def _spark(series, w=120, h=28):
    """Calmar-over-rounds sparkline. `series` = [(round, calmar), ...].
    OWNS: .spark, .sparkna. Colour from --pos/--neg; baseline from --line-2."""
    pts = [c for _, c in series]
    if len(pts) < 2:
        return '<span class="sparkna">—</span>'
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    n = len(pts)

    def _y(c):
        return h - (c - lo) / rng * (h - 6) - 3
    coords = " ".join(f"{i / (n - 1) * w:.1f},{_y(c):.1f}" for i, c in enumerate(pts))
    last_up = pts[-1] >= pts[0]
    col = "var(--pos)" if last_up else "var(--neg)"
    base_y = _y(pts[0])
    last_x, last_y = w, _y(pts[-1])
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" preserveAspectRatio="none" '
            f'role="img" aria-label="Calmar over {n} rounds, {"rising" if last_up else "falling"}">'
            f'<line x1="0" y1="{base_y:.1f}" x2="{w}" y2="{base_y:.1f}" stroke="var(--line-soft)" stroke-width="1"/>'
            f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="1.8"/>'
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.2" fill="{col}"/></svg>')


def _edgebar(edge, w):
    """Diverging edge bar as a full <td>. `w` = bar width 0..100 (% of half-width).
    OWNS: .edgecell, .edgenum, .edgebar, .edgebar i.pos/.neg."""
    if edge is None:
        return '<td class="num">—</td>'
    cls = "pos" if edge >= 0 else "neg"
    sign = "+" if edge >= 0 else ""
    side = "left:50%" if edge >= 0 else "right:50%"
    return (f'<td class="edgecell num {cls}"><span class="edgenum">{sign}{edge:.2f}</span>'
            f'<span class="edgebar"><i class="{cls}" style="{side};width:{w / 2:.1f}%"></i></span></td>')


def _num(v, fmt=".2f", suf="", cls=""):
    """Right-aligned tabular-numeral <td>. OWNS: .num (+ optional .pos/.neg tint)."""
    extra = (" " + cls) if cls else ""
    if v is None:
        return f'<td class="num{extra}">—</td>'
    return f'<td class="num{extra}">{format(v, fmt)}{_esc(suf)}</td>'


# ===========================================================================
# LAYOUT PRIMITIVES
# ===========================================================================
def eyebrow(text, tone=""):
    """Uppercase mono section label carrying the narrative beat. OWNS: .eyebrow."""
    t = (" " + tone) if tone in _TONES else ""
    return f'<div class="eyebrow{t}">{_esc(text)}</div>'


def provenance(text):
    """Faint per-section freshness / source stamp. OWNS: .provenance."""
    return f'<div class="provenance">{_esc(text)}</div>'


def chip(text, kind="", title=""):
    """Status / role pill. `kind` in {pos,neg,amber,muted,na,edge,decorr,oil,new}.
    OWNS: .chip + .chip--<kind>. Replaces the bespoke .pill/.tc/.tbadge/.sigbadge/.vbadge."""
    k = (" chip--" + kind) if kind in _TONES else ""
    ti = f' title="{_esc(title)}"' if title else ""
    return f'<span class="chip{k}"{ti}>{_esc(text)}</span>'


def kpi(label, value, tone="", sub="", title=""):
    """One big-numeral KPI block (label + JetBrains-Mono tabular value).
    `tone` in {pos,neg,acc}. OWNS: .kpi, .kpi .k/.v/.sub, .v.<tone>."""
    tv = (" " + tone) if tone in _TONES else ""
    ti = f' title="{_esc(title)}"' if title else ""
    sub_html = f'<div class="sub">{_esc(sub)}</div>' if sub else ""
    return (f'<div class="kpi"{ti}><div class="k">{_esc(label)}</div>'
            f'<div class="v{tv}">{_esc(value)}</div>{sub_html}</div>')


def card(body, kind="", eyebrow_text="", id=None, title=""):
    """The single card primitive + modifiers. `kind` in
    {metric, edge, book, upgrade, mech, muted}. `body` is raw inner HTML (already
    built from other primitives). OWNS: .card + .card--<kind>, .card-eyebrow, .card-body."""
    k = (" card--" + kind) if kind in {"metric", "edge", "book", "upgrade", "mech", "muted"} else ""
    idattr = f' id="{_esc(id)}"' if id else ""
    ti = f' title="{_esc(title)}"' if title else ""
    eb = f'<div class="card-eyebrow">{_esc(eyebrow_text)}</div>' if eyebrow_text else ""
    return f'<div class="card{k}"{idattr}{ti}>{eb}<div class="card-body">{body}</div></div>'


def cell(inner, num=False, cls="", attrs=""):
    """A single <td>. Supporting helper for table()/matrix()."""
    classes = ((["num"] if num else []) + ([cls] if cls else []))
    c = (' class="' + " ".join(classes) + '"') if classes else ""
    a = (" " + attrs) if attrs else ""
    return f'<td{c}{a}>{inner}</td>'


def tr(cells, cls="", attrs=""):
    """A <tr> from an iterable of <td> strings. Supporting helper."""
    c = f' class="{cls}"' if cls else ""
    a = (" " + attrs) if attrs else ""
    return f'<tr{c}{a}>{"".join(cells)}</tr>'


def tablewrap(inner):
    """Horizontal-scroll wrapper for wide tables. OWNS: .tablewrap."""
    return f'<div class="tablewrap">{inner}</div>'


def table(cols, body_rows, id=None, cls="", wrap=True):
    """A <table>. `cols` = list of dicts {label, num(bool), title, key, t('n'|'s')};
    `key`+`t` make a header client-sortable. `body_rows` = list of complete <tr> strings
    (build them with tr()/cell()/_num()/_edgebar()/chip()). OWNS: .tablewrap, table head/body, .num."""
    ths = []
    for c in cols:
        cc = []
        if c.get("num"):
            cc.append("num")
        if c.get("cls"):
            cc.append(c["cls"])
        cls_attr = (' class="' + " ".join(cc) + '"') if cc else ""
        title = f' title="{_esc(c["title"])}"' if c.get("title") else ""
        data = ""
        if c.get("key"):
            data = f' data-k="{_esc(c["key"])}" data-t="{c.get("t", "n")}"'
        ths.append(f"<th{cls_attr}{data}{title}>{_esc(c['label'])}</th>")
    idattr = f' id="{_esc(id)}"' if id else ""
    clsattr = f' class="{_esc(cls)}"' if cls else ""
    html = (f"<table{idattr}{clsattr}><thead><tr>{''.join(ths)}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>")
    return tablewrap(html) if wrap else html


def matrix(col_labels, rows, col_titles=None):
    """The honesty pass/fail matrix. `rows` = list of
    {"label":str, "sub":str, "cells":[{"status":pos|amber|neg|na, "value":str, "title":str}]}.
    OWNS: .matrix, .matrix .mlabel/.msub, .mcell + .mcell--<status>."""
    col_titles = col_titles or [""] * len(col_labels)
    head = "<th></th>" + "".join(
        f'<th class="num" title="{_esc(t)}">{_esc(lbl)}</th>'
        for lbl, t in zip(col_labels, col_titles))
    body = ""
    for r in rows:
        tds = (f'<td class="mlabel"><b>{_esc(r.get("label", ""))}</b>'
               f'<span class="msub">{_esc(r.get("sub", ""))}</span></td>')
        for c in r.get("cells", []):
            st = c.get("status", "na")
            st = st if st in {"pos", "amber", "neg", "na"} else "na"
            tds += (f'<td class="mcell mcell--{st}" title="{_esc(c.get("title") or c.get("value"))}">'
                    f'{_esc(c.get("value", "—"))}</td>')
        body += f"<tr>{tds}</tr>"
    return tablewrap(f'<table class="matrix"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')
