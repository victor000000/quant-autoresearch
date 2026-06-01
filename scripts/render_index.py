#!/usr/bin/env python3
"""Regenerate reports/index.html — dark TERMINAL dashboard.

Shows: a "now running" status panel (reports/status.json), the leaderboard with each
ETF's best vs its BUY-AND-HOLD baseline (knowledge.json['buyhold']) and the edge, and
ALL rounds newest-first. Links style.css so it shares the report theme."""
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
rounds.sort(reverse=True)                      # NEWEST FIRST (descending)

K = {}
try:
    K = json.load(open(KJ))
except Exception:
    pass
pe = K.get("per_etf_best", {})
bh = K.get("buyhold", {})                      # {ETF: {calmar, da}} pure 1-trade buy-and-hold, OOS
lb = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))


def cls(x):
    return "pos" if x > 0 else ("neg" if x < 0 else "")


lb_rows = ""
for k, v in lb:
    best = v.get("real_calmar", 0.0)
    b = bh.get(k, {})
    bc = b.get("calmar")
    bh_cell = f'<td class="num {cls(bc)}">{bc:+.3f}</td>' if isinstance(bc, (int, float)) else '<td class="num">—</td>'
    edge = (best - bc) if isinstance(bc, (int, float)) else None
    edge_cell = (f'<td class="num {cls(edge)}">{edge:+.3f}</td>' if edge is not None else '<td class="num">—</td>')
    lb_rows += (f'<tr><td><b>{k}</b></td>'
                f'<td class="num {cls(best)}">{best:+.4f}</td>'
                f'<td class="num">{v.get("trades","")}</td>'
                f'{bh_cell}{edge_cell}'
                f'<td><code>{v.get("cell","")}</code></td></tr>')

# "Now running" panel from status.json (written by the round driver/loop)
status_html = ""
try:
    st = json.load(open(os.path.join(R, "status.json")))
    if st.get("running"):
        legs = "".join(f'<li><code>{h}</code></li>' for h in st.get("hypotheses", []))
        status_html = (f'<section class="block running"><h2><span class="livedot"></span>Now running '
                       f'— round {st.get("round","?")} · {st.get("etf","?")}</h2>'
                       f'<p class="small">started {st.get("since","?")} · racing 2 hypotheses on the 2 QC nodes:</p>'
                       f'<ul class="hyps">{legs}</ul></section>')
    else:
        status_html = (f'<section class="block"><h2><span class="idledot"></span>Idle</h2>'
                       f'<p class="small">last completed: <b>round {st.get("last_round","?")}</b> · {st.get("note","")}</p></section>')
except Exception:
    pass


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
ol.rounds{{list-style:none;padding:0;margin:0}}
ol.rounds li{{border:1px solid var(--line);border-radius:9px;background:var(--card);margin:.5rem 0;
  padding:.85rem 1.1rem;display:flex;align-items:center;gap:.7rem;justify-content:space-between;
  transition:border-color .18s, transform .18s}}
ol.rounds li:hover{{border-color:var(--accent-line);transform:translateX(3px)}}
ol.rounds a{{font:500 1.05rem/1.4 "IBM Plex Sans",sans-serif;color:var(--ink)}}
ol.rounds a:hover{{color:var(--accent)}}
section.running{{border-color:var(--accent-line);box-shadow:var(--glow)}}
ul.hyps{{list-style:none;padding:0;margin:.4rem 0 0}} ul.hyps li{{margin:.3rem 0}}
.livedot{{display:inline-block;width:.6em;height:.6em;border-radius:50%;background:var(--accent);
  box-shadow:0 0 8px var(--accent);animation:pulse 1.3s ease-in-out infinite}}
.idledot{{display:inline-block;width:.6em;height:.6em;border-radius:50%;background:var(--mut)}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
</style></head><body><div class="wrap">
<div class="nav">autoresearch / reports</div>
<div class="hero">
  <div class="eyebrow">Autoresearch · Quant Pipeline</div>
  <h1>Experiment reports</h1>
  <p class="tldr">Per-round tech reports — each races two hypotheses on the weakest ETF, keeps the winner if it beats
  that ETF's best (real OOS Calmar+DA), and updates the <a href="causal_graph.html">interactive causal graph</a>.</p>
</div>
{status_html}
<section class="block"><h2>Leaderboard — best strategy Calmar vs buy-and-hold (real OOS)</h2>
<table><thead><tr><th>ETF</th><th class="num">best Calmar</th><th class="num">trades</th><th class="num">buy&amp;hold</th><th class="num">edge</th><th>cell</th></tr></thead>
<tbody>{lb_rows}</tbody></table>
<p class="small">buy&amp;hold = pure 1-trade hold over the OOS window (2023-08 → 2026-06); edge = best − buy&amp;hold.</p></section>
<section class="block"><h2>Rounds ({len(rounds)}) — newest first</h2>
<ol class="rounds">
{items}
</ol></section>
</div></body></html>"""

open(os.path.join(R, "index.html"), "w").write(html)
print(f"wrote index.html ({len(rounds)} rounds, newest-first, buy&hold + status)")
