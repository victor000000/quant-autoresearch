#!/usr/bin/env python3
"""Render knowledge.json['causal_graph'] -> a Mermaid causal DAG.

  python3 scripts/render_causal_graph.py                 # regen reports/causal_graph.html
  python3 scripts/render_causal_graph.py --inject FILE    # also embed the graph into a round report

The graph is the SINGLE SOURCE OF TRUTH in knowledge.json. Each round: append nodes/edges
(seed_causal_graph.py shows the schema), then run this to regenerate the visual + inject it
into that round's report. So every round_N.html carries the causal graph as of that round.
"""
import json, os, sys, re

ROOT = os.path.join(os.path.dirname(__file__), "..")
KJ = os.path.join(ROOT, "autoresearch", "knowledge.json")
OUT = os.path.join(ROOT, "autoresearch", "reports", "causal_graph.html")

PHASE_TITLE = {"Landscape": "Phase A - Landscape / the 6-round null",
               "TLT": "Phase B - TLT (declining / two-sided)",
               "IWM": "Phase C - IWM (trending-up small-cap)",
               "XLE": "Phase D - XLE (trending-up energy)"}


def _esc(s):
    # Mermaid node/edge text is quoted with "..."; keep it clean.
    return (s or "").replace('"', "'").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")


def mermaid(cg):
    running = set(cg.get("running", []))
    phases = cg.get("phases", [])
    by_phase = {p: [] for p in phases}
    for n in cg["nodes"]:
        by_phase.setdefault(n["phase"], []).append(n)
    L = ["flowchart TB"]
    for p in phases:
        pid = "sg_" + re.sub(r"\W", "", p)
        L.append(f'  subgraph {pid}["{_esc(PHASE_TITLE.get(p, p))}"]')
        L.append("    direction TB")
        for n in by_phase.get(p, []):
            cls = n["type"] + ("_run" if n["id"] in running else "")
            L.append(f'    {n["id"]}["{_esc(n["label"])}"]:::{cls}')
        L.append("  end")
    for e in cg["edges"]:
        lbl = _esc(e.get("label", ""))
        if lbl:
            L.append(f'  {e["src"]} -->|"{lbl}"| {e["dst"]}')
        else:
            L.append(f'  {e["src"]} --> {e["dst"]}')
    # styling
    L += [
        "  classDef finding fill:#fff3cd,stroke:#e0a800,stroke-width:2px,color:#5c4500;",
        "  classDef round fill:#eef1f6,stroke:#9aa5b1,color:#1b1f23;",
        "  classDef milestone fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#0b3d1a;",
        "  classDef decision fill:#e2dcff,stroke:#6f42c1,stroke-width:2px,color:#2d1a5c;",
        "  classDef round_run fill:#eef1f6,stroke:#9aa5b1,color:#1b1f23,stroke-dasharray:5 4;",
        "  classDef finding_run fill:#fff3cd,stroke:#e0a800,color:#5c4500,stroke-dasharray:5 4;",
        "  classDef milestone_run fill:#d4edda,stroke:#28a745,color:#0b3d1a,stroke-dasharray:5 4;",
        "  classDef decision_run fill:#e2dcff,stroke:#6f42c1,color:#2d1a5c,stroke-dasharray:5 4;",
    ]
    return "\n".join(L)


MERMAID_CDN = ("<script type=\"module\">import mermaid from "
               "'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
               "mermaid.initialize({startOnLoad:true,securityLevel:'loose',"
               "flowchart:{useMaxWidth:true,htmlLabels:true},maxTextSize:200000});</script>")

LEGEND = ('<p style="font-size:.9em"><b>Legend:</b> '
          '<span style="background:#fff3cd;border:1px solid #e0a800;padding:.05em .4em;border-radius:3px">FINDING (mechanism hub)</span> '
          '<span style="background:#d4edda;border:1px solid #28a745;padding:.05em .4em;border-radius:3px">milestone / KEEP</span> '
          '<span style="background:#e2dcff;border:1px solid #6f42c1;padding:.05em .4em;border-radius:3px">decision / unlock</span> '
          '<span style="background:#eef1f6;border:1px solid #9aa5b1;padding:.05em .4em;border-radius:3px">experiment</span> '
          '<span style="border:1px dashed #9aa5b1;padding:.05em .4em;border-radius:3px">running</span>. '
          'Edges read cause &rarr; effect (the label is the mechanism). '
          'FINDING nodes are the hubs: each new hypothesis is reasoned from them.</p>')


def leaderboard(d):
    pe = d.get("per_etf_best", {})
    rows = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))
    tr = "".join(f"<tr><td>{k}</td><td>{v.get('real_calmar'):+.4f}</td><td>{v.get('trades')}</td>"
                 f"<td>{v.get('cell','')}</td></tr>" for k, v in rows)
    return ('<h2>Leaderboard (current per-ETF best REAL OOS Calmar)</h2>'
            '<table><tr><th>ETF</th><th>Calmar</th><th>trades</th><th>cell</th></tr>' + tr + '</table>')


def section(graph, title):
    """Self-contained embeddable section (its own mermaid include)."""
    return (f'<section class="causalgraph"><h2>{title}</h2>{LEGEND}'
            f'<div class="mermaid">\n{graph}\n</div>{MERMAID_CDN}</section>')


def standalone(d, graph):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch - Causal Graph of all experiments</title>
<style>body{{font:16px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1.2rem;color:#1b1f23}}
h1{{border-bottom:2px solid #eaecef;padding-bottom:.3em}}table{{border-collapse:collapse;margin:1em 0}}th,td{{border:1px solid #d0d7de;padding:.4em .7em}}th{{background:#f6f8fa}}
a{{color:#0969da;text-decoration:none}}.mermaid{{margin:1.5em 0}}code{{background:#f3f4f6;padding:.1em .35em;border-radius:4px}}</style></head><body>
<p><a href="index.html">&larr; all reports</a></p>
<h1>Causal graph - every autoresearch experiment</h1>
<p>How each experiment's outcome <i>caused</i> the next hypothesis. Built from <code>knowledge.json.causal_graph</code>
(the single source of truth) and regenerated every round. The yellow <b>FINDING</b> nodes are the mechanism hubs that drive
design choices; the arc of each ETF reads top-to-bottom within its phase.</p>
{leaderboard(d)}
{LEGEND}
<div class="mermaid">
{graph}
</div>
{MERMAID_CDN}
<h2>The four mechanism hubs in one line each</h2>
<ul>
<li><b>overlay washes the label</b> - inverse-vol sizing is label-independent (R2/R5).</li>
<li><b>6-round null</b> - long-only single-asset ML can't beat buy-hold; a label must CHANGE exposure (R6).</li>
<li><b>directional + short for two-sided assets; long-bias for trending</b> - asset character sets the sizing (R9/R19/R21).</li>
<li><b>edge needs a DENSE axis</b> to clear G2&gt;80 trades - tick has edge but under-trades; range/logdollar are dense (R13/R15/R20).</li>
</ul>
</body></html>"""


def inject(path, graph, round_label):
    html = open(path).read()
    sec = section(graph, f"Causal graph (as of {round_label})")
    pat = re.compile(r'<section class="causalgraph">.*?</section>', re.S)
    if pat.search(html):
        html = pat.sub(sec, html, count=1)
    else:
        html = html.replace("</body>", sec + "\n</body>")
    open(path, "w").write(html)


def main():
    d = json.load(open(KJ))
    cg = d.get("causal_graph")
    if not cg:
        print("no causal_graph in knowledge.json; run seed_causal_graph.py first"); sys.exit(1)
    g = mermaid(cg)
    open(OUT, "w").write(standalone(d, g))
    print(f"wrote {OUT} ({len(cg['nodes'])} nodes, {len(cg['edges'])} edges)")
    if "--inject" in sys.argv:
        f = sys.argv[sys.argv.index("--inject") + 1]
        lab = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "latest round"
        inject(f, g, lab)
        print(f"injected causal graph into {f}")


if __name__ == "__main__":
    main()
