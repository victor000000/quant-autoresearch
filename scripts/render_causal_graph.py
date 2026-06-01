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
import json, os, re, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
KJ = os.path.join(ROOT, "autoresearch", "knowledge.json")
OUT = os.path.join(ROOT, "autoresearch", "reports", "causal_graph.html")

PHASE_TITLE = {"Landscape": "A · Landscape / the 6-round null",
               "TLT": "B · TLT (declining / two-sided)",
               "IWM": "C · IWM (trending-up small-cap)",
               "XLE": "D · XLE (trending-up energy)"}
GROUP_COLOR = {"finding": ("#fff3cd", "#e0a800"), "milestone": ("#d4edda", "#28a745"),
               "decision": ("#e2dcff", "#6f42c1"), "round": ("#eef1f6", "#9aa5b1")}
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


def net_html(cid, nodes, edges, phases, height=620):
    nj, ej, pj = json.dumps(nodes), json.dumps(edges), json.dumps(phases)
    return f"""<div id="{cid}" style="height:{height}px;border:1px solid #d0d7de;border-radius:6px;background:#fcfcfd"></div>
<div style="margin:.4em 0"><button onclick="{cid}_collapse()">⊟ Collapse experiments by phase</button>
<button onclick="{cid}_expand()">⊞ Expand all</button>
<span style="color:#666;font-size:.85em"> · double-click a phase cluster to expand · drag / scroll-zoom · hover a node for full text</span></div>
<script src="{VIS_CDN}"></script>
<script>
(function(){{
  var nodes=new vis.DataSet({nj}); var edges=new vis.DataSet({ej}); var phases={pj};
  var groups={{
    finding:{{shape:'box',color:{{background:'#fff3cd',border:'#e0a800'}},font:{{color:'#5c4500',size:13}},borderWidth:2}},
    milestone:{{shape:'box',color:{{background:'#d4edda',border:'#28a745'}},font:{{color:'#0b3d1a',size:13,bold:true}},borderWidth:2}},
    decision:{{shape:'box',color:{{background:'#e2dcff',border:'#6f42c1'}},font:{{color:'#2d1a5c',size:13}},borderWidth:2}},
    round:{{shape:'dot',color:{{background:'#eef1f6',border:'#9aa5b1'}},font:{{size:11,color:'#444'}}}}
  }};
  var opts={{
    nodes:{{shape:'box',margin:8,widthConstraint:{{maximum:190}},shadow:false}},
    groups:groups,
    edges:{{arrows:{{to:{{scaleFactor:.6}}}},color:{{color:'#c3cad3',highlight:'#d62728'}},
      font:{{size:10,color:'#667',strokeWidth:4,strokeColor:'#fff',align:'middle'}},smooth:{{type:'cubicBezier',roundness:.4}}}},
    physics:{{stabilization:{{iterations:300}},barnesHut:{{gravitationalConstant:-14000,springLength:150,springConstant:.02,avoidOverlap:.5}}}},
    interaction:{{hover:true,tooltipDelay:120,navigationButtons:true,keyboard:false,zoomView:true,dragView:true}},
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
  window['{cid}_collapse']=collapse; window['{cid}_expand']=expand;
  net.on('doubleClick',function(p){{ if(p.nodes.length&&net.isCluster(p.nodes[0])) net.openCluster(p.nodes[0]); }});
  net.once('stabilizationIterationsDone',function(){{ net.setOptions({{physics:false}}); collapse(); }});
}})();
</script>"""


LEGEND = ('<p style="font-size:.9em"><b>Legend:</b> '
          '<span style="background:#fff3cd;border:1px solid #e0a800;padding:.05em .4em;border-radius:3px">FINDING (mechanism hub)</span> '
          '<span style="background:#d4edda;border:1px solid #28a745;padding:.05em .4em;border-radius:3px">milestone / KEEP</span> '
          '<span style="background:#e2dcff;border:1px solid #6f42c1;padding:.05em .4em;border-radius:3px">decision / unlock</span> '
          '<span style="background:#eef1f6;border:1px solid #9aa5b1;padding:.05em .4em;border-radius:3px">• experiment (collapsed)</span> '
          '<span style="border:1px dashed #d62728;padding:.05em .4em;border-radius:3px;color:#d62728">this round\'s new nodes</span>. '
          'Each new hypothesis is reasoned from a FINDING node.</p>')


def leaderboard(d):
    pe = d.get("per_etf_best", {})
    rows = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))
    tr = "".join(f"<tr><td>{k}</td><td>{v.get('real_calmar'):+.4f}</td><td>{v.get('trades')}</td>"
                 f"<td>{v.get('cell','')}</td></tr>" for k, v in rows)
    return ('<h2>Leaderboard (current per-ETF best REAL OOS Calmar)</h2>'
            '<table><tr><th>ETF</th><th>Calmar</th><th>trades</th><th>cell</th></tr>' + tr + '</table>')


def section(cid, nodes, edges, phases, title, note=""):
    cap = (f'<p style="font-size:.92em;color:#444"><b>This round\'s reasoning path:</b> {note}</p>' if note else "")
    return (f'<section class="causalgraph"><h2>{title}</h2>{cap}{LEGEND}'
            f'{net_html(cid, nodes, edges, phases)}</section>')


def standalone(d, cg):
    nodes, edges = vis_data(cg)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch - Interactive causal graph</title>
<style>body{{font:16px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1.2rem;color:#1b1f23}}
h1{{border-bottom:2px solid #eaecef;padding-bottom:.3em}}table{{border-collapse:collapse;margin:1em 0}}th,td{{border:1px solid #d0d7de;padding:.4em .7em}}th{{background:#f6f8fa}}
a{{color:#0969da;text-decoration:none}}code{{background:#f3f4f6;padding:.1em .35em;border-radius:4px}}button{{cursor:pointer;padding:.25em .6em;margin-right:.4em;border:1px solid #d0d7de;border-radius:5px;background:#f6f8fa}}</style></head><body>
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
