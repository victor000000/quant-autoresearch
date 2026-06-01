#!/usr/bin/env python3
"""Render ONE readable round report from a small JSON spec (consistent + low-effort).

  python3 scripts/render_round.py '{"round":26,"etf":"TLT","verdict":"DISCARD",
     "tldr":"...","rows":[{"cfg":"logdollar/triple_barrier/ls_cdf","calmar":"+0.7537",
       "da":"5.5","trades":"292","note":"re-verify","win":true}, ...],
     "why":"<p>plain-language mechanism…</p>","reasoning":"finding -> hypotheses",
     "next":"next step"}'

Writes autoresearch/reports/round_{N}.html (links shared style.css). Then run
scripts/render_causal_graph.py --inject to embed the interactive causal graph.
"""
import json, os, sys

R = json.loads(sys.argv[1])
OUT = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "reports", f"round_{R['round']}.html")
keep = str(R.get("verdict", "")).upper() == "KEEP"
badge = f'<span class="badge {"keep" if keep else "discard"}">{R.get("verdict","")}</span>'


def numcell(v):
    s = str(v); cls = ""
    try:
        f = float(s)
        cls = " pos" if f > 0 else (" neg" if f < 0 else "")
    except ValueError:
        pass
    return f'<td class="num{cls}">{s}</td>'


rows_html = ""
for r in R.get("rows", []):
    cls = ' class="win"' if r.get("win") else ""
    rows_html += (f"<tr{cls}><td><code>{r.get('cfg','')}</code></td>"
                  f"{numcell(r.get('calmar',''))}{numcell(r.get('da',''))}{numcell(r.get('trades',''))}"
                  f"<td class='small'>{r.get('note','')}</td></tr>")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Round {R['round']} — {R['etf']} ({R.get('verdict','')})</title>
<link rel="stylesheet" href="style.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script></head><body>
<div class="nav"><a href="index.html">← all rounds</a> · <a href="causal_graph.html">interactive causal graph</a></div>
<h1>Round {R['round']} — {R['etf']} &nbsp;{badge}</h1>
<div class="tldr"><b>TL;DR.</b> {R.get('tldr','')}</div>
<h2>Results — real OOS</h2>
<table><thead><tr><th>config</th><th class="num">Calmar</th><th class="num">DA</th><th class="num">trades</th><th>note</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<h2>Why</h2><div class="why">{R.get('why','')}</div>
<h2>Reasoning path (from the causal graph)</h2><div class="why">{R.get('reasoning','')}</div>
<h2>Next</h2><div class="next">{R.get('next','')}</div>
</body></html>"""

open(OUT, "w").write(html)
print(f"wrote {OUT}")
