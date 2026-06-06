#!/usr/bin/env python3
"""console.sections.graph — the causal / provenance graph PREVIEW (tertiary).

Blueprint section "graph": the full experiment lineage on demand — NOT the
~517 KB inline vis.DataSet that taxed every single page load (40% of the old
1.3 MB page). This builder emits ONLY a lightweight preview card:

  * summary counts (nodes / edges / phases) read from causal_graph,
  * a phase-cluster legend + a node-type legend (tints mirror the vis palette),
  * an EMPTY lazy-load mount (id="graphwrap", data-pending="1",
    data-graph-src="/graph.json") that console.js builds vis.Network into on the
    existing IntersectionObserver expand — fetching /graph.json only then,
  * an "open full graph" link to the standalone causal_graph.html page.

ZERO vis dataset is inlined here (the whole point of the perf cut). Pure
ctx -> HTML: reads ONLY ctx["K"]["causal_graph"] counts + ctx["book"]["freshness"]
(no file-I/O), and invents no CSS — every class is emitted through a
console.primitives helper that OWNS it.
"""
import os
import sys
from collections import Counter

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from console import primitives as P  # noqa: E402  (scripts/ on path above)


# Full-page standalone graph + the lazy JSON endpoint console.js fetches.
_FULL_PAGE = "causal_graph.html"
_GRAPH_JSON = "/graph.json"

# node-type -> (legend label, chip tint, hover). Tints mirror the vis palette:
#   finding = amber mechanism hub · milestone = green KEEP · decision = teal unlock
#   · round = muted collapsed experiment.
_TYPE_LEGEND = [
    ("finding", "amber",
     "finding / mechanism hub — a confirmed result that spawned the next hypothesis"),
    ("milestone", "pos", "milestone / KEEP — a kept best-in-cell"),
    ("decision", "edge",
     "decision / unlock — a pipeline change that opened a new regime"),
    ("round", "muted",
     "experiment — one tournament round (collapsed by phase by default)"),
]


# ---- card pieces -----------------------------------------------------------
def _counts(cg):
    """(n_nodes, n_edges, phases, by_type Counter) from the causal_graph dict."""
    nodes = cg.get("nodes") or []
    edges = cg.get("edges") or []
    phases = cg.get("phases") or []
    by_type = Counter(n.get("type") for n in nodes)
    return len(nodes), len(edges), phases, by_type


def _kpis(n_nodes, n_edges, n_phases):
    """The three headline numerals: nodes / edges / phases (tabular-nums via .kpi)."""
    return ('<div class="stats">'
            + P.kpi("nodes", str(n_nodes),
                    title="experiments + findings + milestones + decisions")
            + P.kpi("edges", str(n_edges),
                    title="causal links — each outcome -> the next hypothesis")
            + P.kpi("phases", str(n_phases),
                    title="research phases the lineage is clustered into")
            + "</div>")


def _phase_legend(phases):
    chips = " ".join(P.chip(p, "muted", f"{p} phase cluster") for p in phases)
    return '<div class="small">phases · ' + (chips or "—") + "</div>"


def _type_legend(by_type):
    chips = [P.chip(f"{t} · {by_type.get(t, 0)}", tint, ti)
             for t, tint, ti in _TYPE_LEGEND]
    return '<div class="small">node types · ' + " ".join(chips) + "</div>"


def _mount():
    """The empty lazy-load mount. console.js fetches /graph.json on the
    IntersectionObserver expand and builds vis.Network here — NOTHING is inlined,
    so the initial page carries zero graph payload."""
    return (f'<div id="graphwrap" data-pending="1" data-graph-src="{_GRAPH_JSON}" '
            'role="img" aria-label="causal graph — loads interactively on scroll">'
            '<div class="small">Interactive lineage loads on scroll — fetched lazily '
            f'from {_GRAPH_JSON}, kept off the initial page weight.</div>'
            "</div>")


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the full <section id="graph"> HTML fragment (preview card only)."""
    K = ctx.get("K") or {}
    cg = K.get("causal_graph") or {}
    book = ctx.get("book") or {}
    n_nodes, n_edges, phases, by_type = _counts(cg)

    link = f'<a href="{_FULL_PAGE}">open full graph → {_FULL_PAGE}</a>'
    body = (
        _kpis(n_nodes, n_edges, len(phases))
        + _phase_legend(phases)
        + _type_legend(by_type)
        + _mount()
        + '<div class="small">' + link + "</div>"
    )
    preview = P.card(body, kind="muted", eyebrow_text="CAUSAL GRAPH",
                     title="full experiment lineage, loaded on demand")

    sec = (
        P.eyebrow("PROVENANCE · FULL EXPERIMENT LINEAGE")
        + "<h2>Causal / provenance graph</h2>"
        + P.provenance(
            "how each outcome caused the next hypothesis · the full vis network "
            "loads on demand (kept off the page weight) · " + (book.get("freshness") or ""))
        + preview
    )
    return '<section class="block" id="graph">' + sec + "</section>"


if __name__ == "__main__":
    from console.data import build_ctx
    out = render(build_ctx())
    print(f"graph section: {len(out)} bytes")
    # preview-only contract: counts + legends + mount + full-page link, NO vis dataset.
    for needle in ('id="graph"', "Causal / provenance graph", "nodes", "edges",
                   "phases", 'id="graphwrap"', 'data-graph-src="/graph.json"',
                   _FULL_PAGE):
        assert needle in out, f"missing: {needle}"
    for banned in ("vis.DataSet", "new vis.Network", "cgmain"):
        assert banned not in out, f"inlined graph leaked into preview: {banned}"
    print("ok — preview card only (counts + legends + lazy mount + full link), zero vis dataset")
