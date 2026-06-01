#!/usr/bin/env python3
"""Render ONE beautiful, readable round report from a small JSON spec.

  python3 scripts/render_round.py '{"round":27,"etf":"IWM","verdict":"DISCARD",
     "tldr":"…","rows":[{"cfg":"logdollar/bgm/cdf_overlay","calmar":"+0.71","da":"12",
        "trades":"593","note":"…","win":true}, …],
     "why":"<p>…</p>","reasoning":"…","next":"…"}'

Hero header + stat cards (from the winning row) + results table + why/reasoning/next,
on the shared style.css. Then run scripts/render_causal_graph.py --inject to embed the
interactive causal graph. Writes autoresearch/reports/round_{N}.html.
"""
import json, os, sys

R = json.loads(sys.argv[1])
OUT = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "reports", f"round_{R['round']}.html")
keep = str(R.get("verdict", "")).upper().startswith("KEEP")
vcls = "keep" if keep else "discard"
rows = R.get("rows", [])
win = next((r for r in rows if r.get("win")), (rows[0] if rows else {}))


def signcls(v):
    try:
        f = float(str(v))
        return "pos" if f > 0 else ("neg" if f < 0 else "")
    except ValueError:
        return ""


def numcell(v):
    return f'<td class="num {signcls(v)}">{v}</td>'


rows_html = ""
for r in rows:
    cls = ' class="win"' if r.get("win") else ""
    rows_html += (f"<tr{cls}><td><code>{r.get('cfg','')}</code></td>"
                  f"{numcell(r.get('calmar',''))}{numcell(r.get('da',''))}"
                  f"<td class='num'>{r.get('trades','')}</td>"
                  f"<td class='small'>{r.get('note','')}</td></tr>")

# stat cards from the winning row (+ optional extra card)
cards = [("Winner Calmar", win.get("calmar", "—"), signcls(win.get("calmar", "")), "real OOS"),
         ("Drawdown area", win.get("da", "—"), "", "lower = better"),
         ("OOS trades", win.get("trades", "—"), "", "G2 needs >80")]
if R.get("headline"):
    cards.append((R["headline"].get("k", ""), R["headline"].get("v", ""),
                  signcls(R["headline"].get("v", "")), R["headline"].get("sub", "")))
cards_html = "".join(
    f'<div class="stat"><div class="k">{k}</div><div class="v {c}">{v}</div><div class="sub">{s}</div></div>'
    for (k, v, c, s) in cards)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Round {R['round']} — {R['etf']} ({R.get('verdict','')})</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script></head>
<body><div class="wrap">
<div class="nav"><a href="index.html">← all rounds</a><span class="sep">·</span><a href="causal_graph.html">interactive causal graph</a></div>

<div class="hero">
  <div class="eyebrow">Autoresearch · Round {R['round']}</div>
  <h1><span class="chip">{R['etf']}</span> <span class="pill {vcls}">{R.get('verdict','')}</span></h1>
  <p class="tldr"><b>TL;DR.</b> {R.get('tldr','')}</p>
</div>

<div class="stats">{cards_html}</div>

<section class="block"><h2>Results — real OOS</h2>
<table><thead><tr><th>config</th><th class="num">Calmar</th><th class="num">DA</th><th class="num">trades</th><th>note</th></tr></thead>
<tbody>{rows_html}</tbody></table></section>

<section class="block"><h2>Why</h2>{R.get('why','')}</section>
<section class="block"><h2>Reasoning path (from the causal graph)</h2><p>{R.get('reasoning','')}</p></section>
<section class="block"><h2>Next</h2><div class="next-box">{R.get('next','')}</div></section>
</div></body></html>"""

open(OUT, "w").write(html)
print(f"wrote {OUT}")
