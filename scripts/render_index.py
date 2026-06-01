#!/usr/bin/env python3
"""Regenerate reports/index.html — dark TERMINAL theme, current leaderboard, ALL rounds.

Replaces the old hand-edited index (which listed only rounds 1-6 and whose </ol>-append
silently no-op'd because it used <ul>). Scans reports/round_*.html for titles, reads
knowledge.json for per_etf_best, links style.css so it shares the report theme."""
import os, re, json, glob

R = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "reports")
KJ = os.path.join(R, "..", "knowledge.json")

rounds = []
for f in glob.glob(os.path.join(R, "round_*.html")):
    base = os.path.basename(f)
    m = re.match(r"round_(\d+)([a-z]?)\.html$", base)
    if not m:
        continue
    try:
        mt = re.search(r"<title>(.*?)</title>", open(f).read())
        title = re.sub(r"\s+", " ", mt.group(1)).strip() if mt else base
    except Exception:
        title = base
    key = int(m.group(1)) + (0.5 if m.group(2) else 0)
    rounds.append((key, base, title))
rounds.sort()

pe = {}
try:
    pe = json.load(open(KJ)).get("per_etf_best", {})
except Exception:
    pass
lb = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))
lb_rows = "".join(
    f'<tr><td><b>{k}</b></td>'
    f'<td class="num {"pos" if v.get("real_calmar",0)>0 else "neg"}">{v.get("real_calmar",0):+.4f}</td>'
    f'<td class="num">{v.get("trades","")}</td><td><code>{v.get("cell","")}</code></td></tr>'
    for k, v in lb)


def li(base, title):
    v = "keep" if "KEEP" in title.upper() else ("discard" if "DISCARD" in title.upper() else "")
    tag = f'<span class="pill {v}">{v.upper()}</span>' if v else ""
    return f'<li><a href="{base}">{title}</a> {tag}</li>'


items = "\n".join(li(b, t) for _, b, t in rounds)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch — reports</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<style>
ol.rounds{{list-style:none;padding:0;margin:0;counter-reset:r}}
ol.rounds li{{border:1px solid var(--line);border-radius:9px;background:var(--card);margin:.5rem 0;
  padding:.85rem 1.1rem;display:flex;align-items:center;gap:.7rem;justify-content:space-between;
  transition:border-color .18s, transform .18s}}
ol.rounds li:hover{{border-color:var(--accent-line);transform:translateX(3px)}}
ol.rounds a{{font:500 1.05rem/1.4 "IBM Plex Sans",sans-serif;color:var(--ink)}}
ol.rounds a:hover{{color:var(--accent)}}
</style></head><body><div class="wrap">
<div class="nav">autoresearch / reports</div>
<div class="hero">
  <div class="eyebrow">Autoresearch · Quant Pipeline</div>
  <h1>Experiment reports</h1>
  <p class="tldr">Per-round tech reports — each races two hypotheses on the weakest ETF, keeps the winner if it beats
  that ETF's best (real OOS Calmar+DA), and updates the <a href="causal_graph.html">interactive causal graph</a>.</p>
</div>
<section class="block"><h2>Leaderboard — best real OOS Calmar per ETF</h2>
<table><thead><tr><th>ETF</th><th class="num">Calmar</th><th class="num">trades</th><th>cell</th></tr></thead>
<tbody>{lb_rows}</tbody></table></section>
<section class="block"><h2>Rounds ({len(rounds)})</h2>
<ol class="rounds">
{items}
</ol></section>
</div></body></html>"""

open(os.path.join(R, "index.html"), "w").write(html)
print(f"wrote index.html ({len(rounds)} rounds)")
