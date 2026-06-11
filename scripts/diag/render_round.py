#!/usr/bin/env python3
"""Render ONE beautiful, readable round report from a small JSON spec.

  python3 scripts/render_round.py '{"round":27,"etf":"IWM","verdict":"DISCARD",
     "tldr":"…","rows":[{"cfg":"logdollar/bgm/cdf_overlay","calmar":"+0.71","da":"12",
        "trades":"593","note":"…","win":true}, …],
     "why":"<p>…</p>","reasoning":"…","next":"…"}'

Hero header + stat cards (from the winning row) + results table + why/reasoning/next,
on the shared style.css. Then run src/lb/console/render_causal_graph.py --inject to embed the
interactive causal graph. Writes autoresearch/reports/round_{N}.html.
"""
import json, os, sys, re

from lb.describe import describe                       # config -> plain-English hypothesis

R = json.loads(sys.argv[1])
OUT = os.path.join(os.path.dirname(__file__), "..", "reports", f"round_{R['round']}.html")
keep = str(R.get("verdict", "")).upper().startswith("KEEP")
vcls = "keep" if keep else "discard"
# Clean one-word verdict keyword for the <title>/tab (the ledger parser matches
# KEEP/DISCARD here); the full rationale stays in the on-page verdict pill + TL;DR.
vkey = "KEEP" if keep else "DISCARD"
rows = R.get("rows", [])
win = next((r for r in rows if r.get("win")), (rows[0] if rows else {}))


def nl_of(cfg):
    """Plain-English gloss of a 'axis/labeler/sizing [@thresh]' config string."""
    base = (cfg or "").split()[0]
    parts = base.split("/")
    if len(parts) < 3:
        return ""
    m = re.search(r"[@t]\s*([0-9]*\.?[0-9]+)", cfg)
    return describe(parts[0], parts[1], m.group(1) if m else None, parts[2])


def signcls(v):
    try:
        f = float(str(v))
        return "pos" if f > 0 else ("neg" if f < 0 else "")
    except ValueError:
        return ""


def numcell(v):
    return f'<td class="num {signcls(v)}">{v}</td>'


rows_html = ""
hyps_nl = []
for r in rows:
    cls = ' class="win"' if r.get("win") else ""
    nl = nl_of(r.get("cfg", ""))
    hyps_nl.append(nl or r.get("cfg", ""))
    nlrow = f'<div class="cfg-nl">{nl}</div>' if nl else ""
    rows_html += (f"<tr{cls}><td><code>{r.get('cfg','')}</code>{nlrow}</td>"
                  f"{numcell(r.get('calmar',''))}{numcell(r.get('da',''))}"
                  f"<td class='num'>{r.get('trades','')}</td>"
                  f"<td class='small'>{r.get('note','')}</td></tr>")
# Machine-readable plain-English hypotheses (the index reads this to show them per round).
hyps_comment = "<!--HYPS_NL:" + " ||| ".join(h.replace("\n", " ") for h in hyps_nl) + "-->"

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
<title>Round {R['round']} — {R['etf']} · {vkey}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script></head>
<body><div class="wrap">{hyps_comment}
<div class="nav"><a href="index.html">← all rounds</a><span class="sep">·</span><a href="causal_graph.html">interactive causal graph</a></div>

<div class="hero">
  <div class="eyebrow">Autoresearch · Round {R['round']}</div>
  <h1><span class="chip">{R['etf']}</span> <span class="pill {vcls}">{R.get('verdict','')}</span></h1>
  <p class="tldr"><b>TL;DR.</b> {R.get('tldr','')}</p>
  <div class="hyps-nl"><span class="hyps-lbl">Hypotheses tested</span><ol>{"".join(f"<li>{h}</li>" for h in hyps_nl)}</ol></div>
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
