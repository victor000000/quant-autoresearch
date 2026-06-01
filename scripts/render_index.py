#!/usr/bin/env python3
"""Build the full-width dark TERMINAL dashboard HTML.

`build_html()` reads knowledge.json + scans reports/round_*.html and returns the page
string — called LIVE by the Flask app (scripts/app.py) on every request, and by
`__main__` here to write a static reports/index.html fallback.

Layout uses the WHOLE page: full-width hero, a highlighted LATEST-IMPROVEMENT banner, a
live "now running" panel (status.json, auto-polled), then a 2-column grid — leaderboard +
top-5 insights on the left, all rounds (newest first, plain-English hypotheses) on the right."""
import os, re, json, glob

R = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "reports")
KJ = os.path.join(R, "..", "knowledge.json")
TICKERS = ["TLT", "IWM", "QQQ", "EEM", "GLD", "HYG", "XLE"]


def _cls(x):
    return "pos" if x > 0 else ("neg" if x < 0 else "")


def _scan_rounds():
    rounds = []
    for f in glob.glob(os.path.join(R, "round_*.html")):
        base = os.path.basename(f)
        m = re.match(r"round_(\d+)([a-z]?)\.html$", base)
        if not m:
            continue
        title, summary, hyps = base, "", []
        try:
            txt = open(f).read()
            mt = re.search(r"<title>(.*?)</title>", txt)
            if mt:
                title = re.sub(r"\s+", " ", mt.group(1)).strip()
            ms = re.search(r'class="tldr"[^>]*>(.*?)</(?:p|div)>', txt, re.S)
            if ms:
                s = re.sub(r"<[^>]+>", "", ms.group(1))
                s = re.sub(r"\s+", " ", s).strip()
                s = re.sub(r"^TL;DR\.?\s*", "", s, flags=re.I)
                summary = (s[:260].rstrip() + "…") if len(s) > 260 else s
            mh = re.search(r"<!--HYPS_NL:(.*?)-->", txt, re.S)
            if mh:
                hyps = [h.strip() for h in mh.group(1).split("|||") if h.strip()]
        except Exception:
            pass
        key = int(m.group(1)) + (0.5 if m.group(2) else 0)
        rounds.append((key, base, title, summary, hyps))
    rounds.sort(reverse=True)
    return rounds


def build_html():
    rounds = _scan_rounds()
    K = {}
    try:
        K = json.load(open(KJ))
    except Exception:
        pass
    pe = K.get("per_etf_best", {})
    bh = K.get("buyhold", {})
    lb = sorted(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0))

    latest = rounds[0] if rounds else None
    latest_etf, latest_keep = None, False
    if latest:
        _t = latest[2]
        latest_keep = "KEEP" in _t.upper()
        for tk in TICKERS:
            if re.search(rf"\b{tk}\b", _t):
                latest_etf = tk
                break

    lb_rows = ""
    for k, v in lb:
        best = v.get("real_calmar", 0.0)
        b = bh.get(k, {})
        bc = b.get("calmar")
        bh_cell = f'<td class="num {_cls(bc)}">{bc:+.3f}</td>' if isinstance(bc, (int, float)) else '<td class="num">—</td>'
        edge = (best - bc) if isinstance(bc, (int, float)) else None
        edge_cell = (f'<td class="num {_cls(edge)}">{edge:+.3f}</td>' if edge is not None else '<td class="num">—</td>')
        hot = ' class="justimproved"' if (latest_keep and k == latest_etf) else ""
        star = ' <span class="hotdot" title="improved this round">▲</span>' if (latest_keep and k == latest_etf) else ""
        lb_rows += (f'<tr{hot}><td><b>{k}</b>{star}</td>'
                    f'<td class="num {_cls(best)}">{best:+.4f}</td>'
                    f'<td class="num">{v.get("trades","")}</td>'
                    f'{bh_cell}{edge_cell}'
                    f'<td><code>{v.get("cell","")}</code></td></tr>')

    status_html = ('<section class="block" id="nowrunning"><h2><span class="idledot"></span>Status</h2>'
                   '<p class="small">loading live status…</p></section>')

    ins = (K.get("top_insights", []) or [])[:5]
    ins_html = ""
    for i, it in enumerate(ins, 1):
        ev = f'<span class="ev">{it.get("ev","")}</span>' if it.get("ev") else ""
        ins_html += (f'<li><div class="ititle"><span class="inum">{i:02d}</span>'
                     f'<span>{it.get("title","")}</span>{ev}</div>'
                     f'<div class="idetail">{it.get("detail","")}</div></li>')
    if not ins_html:
        ins_html = '<li class="small">(no insights recorded yet)</li>'

    if latest:
        _, lbase, ltitle, lsum, _ = latest
        banner = (f'<a class="latest-banner{" keepglow" if latest_keep else ""}" href="{lbase}">'
                  f'<span class="lb-tag">★ LATEST{" · KEPT" if latest_keep else ""}</span>'
                  f'<span class="lb-title"><mark>{ltitle}</mark></span>'
                  f'<span class="lb-sum">{lsum}</span></a>')
    else:
        banner = ""

    def li(base, title, summary, hyps, is_latest=False):
        v = "keep" if "KEEP" in title.upper() else ("discard" if "DISCARD" in title.upper() else "")
        tag = f'<span class="pill {v}">{v.upper()}</span>' if v else ""
        lt = '<span class="pill latest">★ LATEST</span>' if is_latest else ""
        summ = f'<div class="rsum">{summary}</div>' if summary else ""
        hyp = ""
        if hyps:
            hyp = '<div class="hyps-row">' + "".join(f'<span class="hchip">{h}</span>' for h in hyps[:2]) + "</div>"
        klass = ' class="latest-item"' if is_latest else ""
        return f'<li{klass}><div class="rmain"><a href="{base}">{title}</a> {lt}{tag}</div>{summ}{hyp}</li>'

    items = "\n".join(li(b, t, s, h, is_latest=(i == 0)) for i, (_, b, t, s, h) in enumerate(rounds))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Autoresearch — reports</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<style>
.dash{{max-width:1560px;margin:0 auto;padding:1.4rem 2.2rem 4rem}}
.dgrid{{display:grid;grid-template-columns:minmax(0,1.02fr) minmax(0,1.25fr);gap:1.5rem;align-items:start;margin-top:1.2rem}}
@media(max-width:1080px){{.dgrid{{grid-template-columns:1fr}}}}
.dgrid .block{{margin:0 0 1.4rem}}
.latest-banner{{display:block;border:1px solid var(--accent-line);border-radius:12px;
  background:linear-gradient(120deg,rgba(56,224,200,.10),rgba(56,224,200,.02));
  padding:1rem 1.3rem;margin:1.1rem 0 .2rem;text-decoration:none;box-shadow:var(--glow);transition:transform .18s}}
.latest-banner:hover{{transform:translateY(-2px)}}
.latest-banner.keepglow{{border-color:var(--accent)}}
.lb-tag{{display:inline-block;font:700 .76rem/1 "JetBrains Mono",monospace;letter-spacing:.08em;
  color:var(--accent);background:rgba(56,224,200,.12);border:1px solid var(--accent-line);
  border-radius:999px;padding:.25em .7em;margin-right:.7rem}}
.lb-title{{font:700 1.12rem/1.3 "Space Grotesk",sans-serif;color:var(--ink)}}
.lb-title mark{{background:rgba(56,224,200,.22);color:var(--ink);padding:.05em .25em;border-radius:4px}}
.lb-sum{{display:block;margin-top:.5rem;color:var(--ink-2);font-size:.98rem;line-height:1.55}}
ol.rounds{{list-style:none;padding:0;margin:0}}
ol.rounds li{{border:1px solid var(--line);border-radius:9px;background:var(--card);margin:.5rem 0;
  padding:.85rem 1.1rem;transition:border-color .18s, transform .18s}}
ol.rounds li:hover{{border-color:var(--accent-line);transform:translateX(3px)}}
ol.rounds li.latest-item{{border-color:var(--accent);background:linear-gradient(120deg,rgba(56,224,200,.07),var(--card));box-shadow:var(--glow)}}
.rmain{{display:flex;align-items:center;gap:.6rem;justify-content:space-between;flex-wrap:wrap}}
ol.rounds a{{font:600 1.04rem/1.35 "Space Grotesk",sans-serif;color:var(--ink)}}
ol.rounds a:hover{{color:var(--accent)}}
.rsum{{margin-top:.45rem;color:var(--ink-2);font-size:.95rem;line-height:1.55}}
.pill.latest{{background:rgba(56,224,200,.16);color:var(--accent);border:1px solid var(--accent-line)}}
.hyps-row{{margin-top:.5rem;display:flex;flex-wrap:wrap;gap:.4rem}}
.hchip{{font:500 .8rem/1.3 "JetBrains Mono",monospace;color:var(--ink-2);background:var(--bg-2,#0d1219);
  border:1px solid var(--line);border-radius:7px;padding:.28em .6em}}
tr.justimproved td{{background:rgba(56,224,200,.10)}}
.hotdot{{color:var(--accent);font-size:.8em}}
ol.insights{{list-style:none;padding:0;margin:0}}
ol.insights li{{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:9px;
  background:var(--card);margin:.55rem 0;padding:.8rem 1.05rem}}
.ititle{{font:600 1.02rem/1.4 "Space Grotesk",sans-serif;color:var(--ink);display:flex;align-items:baseline;gap:.55rem}}
.inum{{font:700 .82rem/1 "JetBrains Mono",monospace;color:var(--accent);opacity:.85}}
.idetail{{margin-top:.4rem;color:var(--ink-2);font-size:.94rem;line-height:1.55}}
.ev{{font:500 .76rem/1 "JetBrains Mono",monospace;color:var(--mut);margin-left:auto;white-space:nowrap}}
section.running{{border-color:var(--accent-line);box-shadow:var(--glow)}}
ul.hyps{{list-style:none;padding:0;margin:.4rem 0 0}} ul.hyps li{{margin:.3rem 0;color:var(--ink-2)}}
.livedot{{display:inline-block;width:.6em;height:.6em;border-radius:50%;background:var(--accent);
  box-shadow:0 0 8px var(--accent);animation:pulse 1.3s ease-in-out infinite}}
.idledot{{display:inline-block;width:.6em;height:.6em;border-radius:50%;background:var(--mut)}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
</style></head><body><div class="dash">
<div class="nav">autoresearch / reports · <span style="color:var(--accent)">served by Flask</span></div>
<div class="hero">
  <div class="eyebrow">Autoresearch · Quant Pipeline</div>
  <h1>Experiment reports</h1>
  <p class="tldr">Each round races two hypotheses on the weakest ETF, keeps the winner if it beats that ETF's
  best (real OOS Calmar + DA). Full <a href="causal_graph.html">interactive causal graph</a>. Live; auto-refreshing.</p>
</div>
{banner}
{status_html}
<div class="dgrid">
  <div class="dcol">
    <section class="block"><h2>Leaderboard — best Calmar vs buy-and-hold (real OOS)</h2>
    <table><thead><tr><th>ETF</th><th class="num">best Calmar</th><th class="num">trades</th><th class="num">buy&amp;hold</th><th class="num">edge</th><th>cell</th></tr></thead>
    <tbody>{lb_rows}</tbody></table>
    <p class="small">buy&amp;hold = pure 1-trade hold over OOS (2023-08 → 2026-06); edge = best − buy&amp;hold. ▲ = improved this round.</p></section>
    <section class="block"><h2>Top {len(ins)} insights</h2>
    <ol class="insights">{ins_html}</ol></section>
  </div>
  <div class="dcol">
    <section class="block"><h2>Rounds ({len(rounds)}) — newest first</h2>
    <ol class="rounds">
{items}
    </ol></section>
  </div>
</div>
</div>
<script>
async function pollStatus(){{
  try{{
    const r = await fetch('status.json?_=' + Date.now(), {{cache:'no-store'}});
    if(!r.ok) return;
    const s = await r.json();
    const el = document.getElementById('nowrunning');
    if(!el) return;
    if(s.running){{
      const legs = (s.hypotheses||[]).map(h => '<li>▸ '+h+'</li>').join('');
      el.className = 'block running';
      el.innerHTML = '<h2><span class="livedot"></span>Now running'
        + (s.round ? ' — round ' + s.round : '') + (s.etf ? ' \\u00b7 ' + s.etf : '') + '</h2>'
        + '<p class="small">phase: <b>' + (s.phase || '…') + '</b>'
        + (s.since ? ' \\u00b7 started ' + s.since : '')
        + (s.legs ? ' \\u00b7 ' + s.legs : '') + '</p>'
        + '<ul class="hyps">' + legs + '</ul>';
    }} else {{
      el.className = 'block';
      el.innerHTML = '<h2><span class="idledot"></span>Idle</h2><p class="small">last completed: <b>'
        + (s.note || ('round ' + (s.round || '?'))) + '</b></p>';
    }}
  }} catch(e){{}}
}}
pollStatus(); setInterval(pollStatus, 8000);
</script>
</body></html>"""


if __name__ == "__main__":
    html = build_html()
    open(os.path.join(R, "index.html"), "w").write(html)
    print(f"wrote static index.html ({html.count('round_')} round refs)")
