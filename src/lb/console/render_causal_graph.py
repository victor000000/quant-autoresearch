#!/usr/bin/env python3
"""Render knowledge.json['causal_graph'] -> an INTERACTIVE causal DAG (vis-network).

  python3 scripts/render_causal_graph.py                                 # regen reports/causal_graph.html
  python3 scripts/render_causal_graph.py --inject FILE [--label L]        # embed graph into a round report
        [--highlight id1,id2] [--note "reasoning path"]

Interactivity (vis-network, CDN):
  * The MAIN FINDINGS (finding/milestone/decision hubs) are always shown, large.
  * The many round/experiment nodes are the "less important" ones -> collapsed into
    one cluster per phase by default. Double-click a cluster to expand it (and a node
    to re-collapse its phase). Drag / zoom / pan / hover-for-full-text throughout.
  * This round's new/changed nodes get a red ring.

The graph is the SINGLE SOURCE OF TRUTH in knowledge.json. Each round: append
nodes/edges, then run this (with --inject) to regenerate + embed the up-to-date graph.
"""
import json, re, sys

from lb.paths import KNOWLEDGE_JSON, REPORTS_DIR

KJ = str(KNOWLEDGE_JSON)
OUT = str(REPORTS_DIR / "causal_graph.html")

PHASE_TITLE = {"Landscape": "A · Landscape / the 6-round null",
               "TLT": "B · TLT (declining / two-sided)",
               "IWM": "C · IWM (trending-up small-cap)",
               "XLE": "D · XLE (trending-up energy)"}
GROUP_COLOR = {"finding": ("#13231f", "#38e0c8"), "milestone": ("#0f241a", "#3fd07a"),
               "decision": ("#1c1832", "#9a86ff"), "round": ("#0d1320", "#33415a")}
VIS_CDN = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def _short(n):
    t = n["label"]
    if n["type"] == "round":
        m = re.match(r"(R\d+)", t)
        return m.group(1) if m else t[:16]
    for pre in ("FINDING: ", "FINDING ", "USER POLICY: ", "UNLOCK (user): ", "REFRAME -> "):
        if t.startswith(pre):
            t = t[len(pre):]
            break
    return t if len(t) <= 70 else t[:68] + "…"


def vis_data(cg, highlight=None):
    highlight = set(highlight or [])
    nodes, edges = [], []
    for n in cg["nodes"]:
        nd = {"id": n["id"], "label": _short(n), "title": n["label"],
              "group": n["type"], "phase": n["phase"]}
        if n["type"] == "round":
            nd["shape"] = "dot"; nd["size"] = 11
        if n["id"] in highlight:
            nd["borderWidth"] = 4
            nd["color"] = {"border": "#d62728", "background": GROUP_COLOR.get(n["type"], ("#eee", "#999"))[0]}
        nodes.append(nd)
    for e in cg["edges"]:
        ed = {"from": e["src"], "to": e["dst"], "arrows": "to"}
        if e.get("label"):
            ed["label"] = e["label"]
        edges.append(ed)
    return nodes, edges


def net_html(cid, nodes, edges, phases, height=620, zoom=True):
    nj, ej, pj = json.dumps(nodes), json.dumps(edges), json.dumps(phases)
    zoomjs = "true" if zoom else "false"
    return f"""<div id="{cid}" style="height:{height}px;border:1px solid #222b3a;border-radius:9px;background:#070a10"></div>
<div style="margin:.4em 0"><button onclick="{cid}_collapse()">⊟ Collapse experiments by phase</button>
<button onclick="{cid}_expand()">⊞ Expand all</button>
<span style="color:#7c8aa0;font-size:1rem"> · double-click a phase cluster to expand · drag · hover a node for full text</span></div>
<script src="{VIS_CDN}"></script>
<script>
(function(){{
  var nodes=new vis.DataSet({nj}); var edges=new vis.DataSet({ej}); var phases={pj};
  var groups={{
    finding:{{shape:'box',color:{{background:'#13231f',border:'#38e0c8'}},font:{{color:'#cdeee7',size:13}},borderWidth:2}},
    milestone:{{shape:'box',color:{{background:'#0f241a',border:'#3fd07a'}},font:{{color:'#bfe9cd',size:13,bold:true}},borderWidth:2}},
    decision:{{shape:'box',color:{{background:'#1c1832',border:'#9a86ff'}},font:{{color:'#d8d0ff',size:13}},borderWidth:2}},
    round:{{shape:'dot',color:{{background:'#0d1320',border:'#33415a'}},font:{{size:11,color:'#9aa6b8'}}}}
  }};
  var opts={{
    nodes:{{shape:'box',margin:8,widthConstraint:{{maximum:190}},shadow:false}},
    groups:groups,
    edges:{{arrows:{{to:{{scaleFactor:.6}}}},color:{{color:'#33415a',highlight:'#38e0c8'}},
      font:{{size:11,color:'#9aa6b8',strokeWidth:4,strokeColor:'#070a10',align:'middle'}},smooth:{{type:'cubicBezier',roundness:.4}}}},
    physics:{{stabilization:{{iterations:300}},barnesHut:{{gravitationalConstant:-14000,springLength:150,springConstant:.02,avoidOverlap:.5}}}},
    interaction:{{hover:true,tooltipDelay:120,navigationButtons:true,keyboard:false,zoomView:{zoomjs},dragView:true}},
    layout:{{improvedLayout:true}}
  }};
  var net=new vis.Network(document.getElementById('{cid}'),{{nodes:nodes,edges:edges}},opts);
  function collapse(){{
    phases.forEach(function(p){{
      net.cluster({{
        joinCondition:function(o){{return o.group==='round'&&o.phase===p;}},
        processProperties:function(c,kids){{c.label=p+' · '+kids.length+' experiments ▸';return c;}},
        clusterNodeProperties:{{shape:'box',borderWidth:2,shapeProperties:{{borderDashes:[4,3]}},
          color:{{background:'#f1f3f6',border:'#94a0ad'}},font:{{size:12,color:'#3a4250'}}}}
      }});
    }});
  }}
  function expand(){{ net.body.nodeIndices.slice().forEach(function(id){{ if(net.isCluster(id)) net.openCluster(id); }}); }}
  window['{cid}_collapse']=collapse; window['{cid}_expand']=expand; window['{cid}_net']=net;
  net.on('doubleClick',function(p){{ if(p.nodes.length&&net.isCluster(p.nodes[0])) net.openCluster(p.nodes[0]); }});
  net.once('stabilizationIterationsDone',function(){{ net.setOptions({{physics:false}}); collapse(); }});
}})();
</script>"""


# Chips use the dashboard's DARK-badge convention (soft dark fill + bright text +
# hairline border) so the per-round legend matches the dark report palette instead
# of clashing with light pastel pills. All keep >=4.5:1 text contrast on their fills.
_CHIP = "padding:.12em .5em;border-radius:5px;font-weight:600"
LEGEND = ('<p style="font-size:.95em;color:#aab6c8"><b style="color:#e6edf6">Legend:</b> '
          f'<span style="background:#241d0c;color:#f2c14e;border:1px solid #4d3f18;{_CHIP}">FINDING (mechanism hub)</span> '
          f'<span style="background:#0f241a;color:#3fd07a;border:1px solid #1f5236;{_CHIP}">milestone / KEEP</span> '
          f'<span style="background:#0e2a2a;color:#38e0c8;border:1px solid #1d4d49;{_CHIP}">decision / unlock</span> '
          f'<span style="background:#0d1320;color:#aab6c8;border:1px solid #222b3a;{_CHIP}">• experiment (collapsed)</span> '
          f'<span style="border:1px dashed #ff6b6b;color:#ff6b6b;{_CHIP}">this round\'s new nodes</span>. '
          'Each new hypothesis is reasoned from a FINDING node.</p>')


def leaderboard(d):
    pe = d.get("per_etf_best", {})
    rows = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))
    tr = "".join(f"<tr><td>{k}</td><td>{v.get('real_calmar'):+.4f}</td><td>{v.get('trades')}</td>"
                 f"<td>{v.get('cell','')}</td></tr>" for k, v in rows)
    return ('<h2>Leaderboard (current per-ETF best REAL OOS Calmar)</h2>'
            '<table><tr><th>ETF</th><th>Calmar</th><th>trades</th><th>cell</th></tr>' + tr + '</table>')


def section(cid, nodes, edges, phases, title, note=""):
    # Per-round reports keep the result front-and-centre: the readable reasoning-path
    # summary stays visible; the heavy interactive graph collapses behind a click
    # (resize-on-open since vis-network can't size itself while hidden).
    cap = (f'<p class="cg-note"><b>This round\'s reasoning path:</b> {note}</p>' if note else "")
    body = f'{LEGEND}{net_html(cid, nodes, edges, phases)}'
    toggle = (f"if(this.open&&window['{cid}_net']){{var n=window['{cid}_net'];"
              f"n.setSize('100%','620px');n.redraw();n.fit();}}")
    return (f'<section class="causalgraph"><h2>{title}</h2>{cap}'
            f'<details class="cg-details" ontoggle="{toggle}">'
            f'<summary>Show the interactive causal graph &nbsp;<span class="cg-count">{len(nodes)} nodes</span></summary>'
            f'<div class="cg-body">{body}</div></details></section>')


def standalone(d, cg):
    nodes, edges = vis_data(cg)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch - Interactive causal graph</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>body{{font:18px/1.7 "IBM Plex Sans",system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:2.4rem 1.2rem 5rem;color:#e6edf6;
  background:#0a0e14;background-image:radial-gradient(120% 60% at 50% -8%,rgba(56,224,200,.07),rgba(56,224,200,0) 60%);background-attachment:fixed}}
h1{{font:600 2rem/1.15 "Space Grotesk",sans-serif;letter-spacing:-.02em;border-bottom:1px solid #222b3a;padding-bottom:.35em;color:#e6edf6}}
table{{border-collapse:separate;border-spacing:0;margin:1.2em 0;border:1px solid #222b3a;border-radius:9px;overflow:hidden;font-size:1rem}}
th,td{{border-bottom:1px solid #1b2230;padding:.6em .9em;text-align:left}}
th{{background:#161d2b;font:600 .8rem/1.3 "JetBrains Mono",monospace;text-transform:uppercase;letter-spacing:.06em;color:#7c8aa0}}
td{{font:500 1rem/1.4 "JetBrains Mono",monospace;color:#aab6c8;font-variant-numeric:tabular-nums}} tbody tr:last-child td{{border-bottom:0}}
h2{{font:600 1.3rem/1.2 "Space Grotesk",sans-serif;color:#e6edf6;margin-top:1.4em}}
a{{color:#38e0c8;text-decoration:none}} p{{color:#aab6c8}}
code{{font:500 .9em "JetBrains Mono",monospace;background:#0d1320;border:1px solid #222b3a;padding:.1em .4em;border-radius:6px;color:#38e0c8}}
button{{cursor:pointer;font:600 .85rem "Space Grotesk",sans-serif;padding:.5em 1em;margin-right:.4em;border:1px solid #222b3a;border-radius:9px;background:#161d2b;color:#aab6c8}}
button:hover{{color:#38e0c8;border-color:#1d4d49}}</style></head><body>
<p><a href="index.html">&larr; all reports</a></p>
<h1>Interactive causal graph - every autoresearch experiment</h1>
<p>How each experiment's outcome <i>caused</i> the next hypothesis. Built from <code>knowledge.json.causal_graph</code> and
regenerated every round. The yellow <b>FINDING</b> hubs drive design choices; the many experiments are collapsed by phase —
double-click to expand. Drag, zoom, and hover for full text.</p>
{leaderboard(d)}
{LEGEND}
{net_html("cgmain", nodes, edges, cg.get("phases", []), height=720)}
</body></html>"""


def inject(path, cid, nodes, edges, phases, round_label, note=""):
    html = open(path).read()
    sec = section(cid, nodes, edges, phases,
                  f"Causal graph (as of {round_label}) — interactive: collapse/expand, drag, hover", note)
    pat = re.compile(r'<section class="causalgraph">.*?</section>', re.S)
    if pat.search(html):
        html = pat.sub(sec, html, count=1)
    elif "</div></body>" in html:                 # new .wrap layout: keep graph inside wrap
        html = html.replace("</div></body>", sec + "\n</div></body>", 1)
    else:
        html = html.replace("</body>", sec + "\n</body>", 1)
    open(path, "w").write(html)


def main():
    d = json.load(open(KJ))
    cg = d.get("causal_graph")
    if not cg:
        print("no causal_graph in knowledge.json; run seed_causal_graph.py first"); sys.exit(1)
    open(OUT, "w").write(standalone(d, cg))
    print(f"wrote {OUT} ({len(cg['nodes'])} nodes, {len(cg['edges'])} edges)")
    if "--inject" in sys.argv:
        f = sys.argv[sys.argv.index("--inject") + 1]
        lab = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "latest round"
        hl = sys.argv[sys.argv.index("--highlight") + 1].split(",") if "--highlight" in sys.argv else []
        note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
        nodes, edges = vis_data(cg, highlight=hl)
        cid = "cg" + re.sub(r"\W", "", lab)
        inject(f, cid, nodes, edges, cg.get("phases", []), lab, note)
        print(f"injected interactive causal graph into {f} (highlight={hl})")


if __name__ == "__main__":
    main()
