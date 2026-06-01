#!/usr/bin/env python3
"""Build the full-width dark TERMINAL dashboard HTML (tabbed, openclue.net style).

`build_html()` reads knowledge.json + scans reports/round_*.html and returns the page
string — called LIVE by the Flask app (scripts/app.py) on every request, and by
`__main__` here to write a static reports/index.html fallback.

Tabs: OVERVIEW (default — overall results: live status, latest improvement, leaderboard,
top-5 insights) · ROUNDS (all rounds newest-first, plain-English hypotheses) · CAUSAL
GRAPH (embedded) · PROGRAM.MD (embedded). Smallest text is 18px; tables never clip."""
import os, re, json, glob, sys
import html as _htmllib

sys.path.insert(0, os.path.dirname(__file__))
from render_causal_graph import net_html, vis_data, LEGEND   # embed the graph INLINE (no iframe)

R = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "reports")
KJ = os.path.join(R, "..", "knowledge.json")
PROG = os.path.join(R, "..", "program.md")
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
                summary = (s[:280].rstrip() + "…") if len(s) > 280 else s
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

    # KPI strip
    n_keep = sum(1 for _, _, t, _, _ in rounds if "KEEP" in t.upper())
    best_etf = max(pe.items(), key=lambda kv: kv[1].get("real_calmar", 0)) if pe else None

    lb_rows = ""
    for k, v in lb:
        best = v.get("real_calmar", 0.0)
        cagr = v.get("real_cagr")
        mdd = v.get("real_mdd")
        cagr_cell = f'<td class="num {_cls(cagr)}">{cagr:+.1f}%</td>' if isinstance(cagr, (int, float)) else '<td class="num">—</td>'
        mdd_cell = f'<td class="num">{mdd:.1f}%</td>' if isinstance(mdd, (int, float)) else '<td class="num">—</td>'
        da = v.get("real_da")
        da_cell = f'<td class="num">{da:.2f}</td>' if isinstance(da, (int, float)) else '<td class="num">—</td>'
        b = bh.get(k, {})
        bc = b.get("calmar")
        bh_cell = f'<td class="num {_cls(bc)}">{bc:+.3f}</td>' if isinstance(bc, (int, float)) else '<td class="num">—</td>'
        edge = (best - bc) if isinstance(bc, (int, float)) else None
        edge_cell = (f'<td class="num {_cls(edge)}">{edge:+.3f}</td>' if edge is not None else '<td class="num">—</td>')
        hot = ' class="justimproved"' if (latest_keep and k == latest_etf) else ""
        star = ' <span class="hotdot" title="improved this round">▲</span>' if (latest_keep and k == latest_etf) else ""
        if v.get("leak_pending"):
            flag = ' <span class="leakwarn" title="pre-leak-fix number — pending re-validation under TRAIN-only bar thresholds">⚠ pre-fix</span>'
        elif v.get("leak_fixed"):
            flag = ' <span class="leakok" title="re-validated under leak-fixed bars">✓ leak-fixed</span>'
        else:
            flag = ' <span class="leakok" title="clean axis (imbalance/range) — unaffected by the threshold leak">✓ clean</span>'
        lb_rows += (f'<tr{hot}><td><b>{k}</b>{star}</td>'
                    f'<td class="num {_cls(best)}">{best:+.4f}</td>'
                    f'{cagr_cell}{mdd_cell}{da_cell}'
                    f'<td class="num">{v.get("trades","")}</td>'
                    f'{bh_cell}{edge_cell}'
                    f'<td><code>{v.get("cell","")}</code>{flag}</td></tr>')

    status_html = ('<section class="block" id="nowrunning"><h2><span class="idledot"></span>Status</h2>'
                   '<p>loading live status…</p></section>')

    ins = (K.get("top_insights", []) or [])[:5]
    ins_html = ""
    for i, it in enumerate(ins, 1):
        ev = f'<span class="ev">{it.get("ev","")}</span>' if it.get("ev") else ""
        ins_html += (f'<li><div class="ititle"><span class="inum">{i:02d}</span>'
                     f'<span>{it.get("title","")}</span>{ev}</div>'
                     f'<div class="idetail">{it.get("detail","")}</div></li>')
    if not ins_html:
        ins_html = '<li>(no insights recorded yet)</li>'

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

    # Inline causal graph (reuse vis-network renderer — NO iframe) and inline program.md.
    cg = K.get("causal_graph", {})
    try:
        g_nodes, g_edges = vis_data(cg)
        graph_inline = LEGEND + net_html("cgtab", g_nodes, g_edges, cg.get("phases", []), height=660)
    except Exception:
        graph_inline = '<p>(causal graph unavailable)</p>'
    try:
        prog_inline = '<pre class="mdblock">' + _htmllib.escape(open(PROG).read()) + '</pre>'
    except Exception:
        prog_inline = '<p>(program.md not found)</p>'

    kpis = ""
    if pe:
        kpis = (
            f'<div class="kpi"><div class="k">rounds</div><div class="v">{len(rounds)}</div></div>'
            f'<div class="kpi"><div class="k">kept (wins)</div><div class="v pos">{n_keep}</div></div>'
            f'<div class="kpi"><div class="k">ETFs live</div><div class="v">{len(pe)}</div></div>'
            + (f'<div class="kpi"><div class="k">top ETF</div><div class="v">{best_etf[0]} '
               f'<span class="kpi-sub">{best_etf[1].get("real_calmar",0):+.2f}</span></div></div>' if best_etf else ""))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Autoresearch — reports</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<style>
.dash{{max-width:1560px;margin:0 auto;padding:1.2rem 2.2rem 4rem}}
.hero{{margin-bottom:1rem}}
/* KPI strip */
.kpis{{display:flex;flex-wrap:wrap;gap:.8rem;margin:1rem 0 0}}
.kpi{{flex:1 1 150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.85rem 1.1rem}}
.kpi .k{{font:600 1rem/1 "JetBrains Mono",monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}}
.kpi .v{{font:600 1.9rem/1.1 "JetBrains Mono",monospace;color:var(--ink);margin-top:.35rem;font-variant-numeric:tabular-nums}}
.kpi-sub{{font-size:1.1rem;color:var(--accent)}}
/* TAB BAR — openclue style: sticky, monospace, accent underline on active */
.tabbar{{display:flex;gap:.35rem;flex-wrap:wrap;border-bottom:1px solid var(--line);
  margin:1.3rem 0 1.5rem;position:sticky;top:0;z-index:30;
  background:rgba(10,14,20,.93);backdrop-filter:blur(9px);padding-top:.4rem}}
.tab{{font:600 1rem/1 "JetBrains Mono",monospace;letter-spacing:.02em;color:var(--mut);
  padding:.9em 1.15em;border:1px solid transparent;border-bottom:2px solid transparent;
  border-radius:9px 9px 0 0;text-decoration:none;transition:all .15s;white-space:nowrap}}
.tab:hover{{color:var(--ink);background:var(--bg-2);text-decoration:none}}
.tab.active{{color:var(--accent);border-bottom-color:var(--accent);
  background:linear-gradient(180deg,rgba(56,224,200,.07),transparent)}}
.tabpanel{{display:none;animation:rise .4s ease both}}
.tabpanel.active{{display:block}}
iframe.report{{width:100%;height:80vh;border:1px solid var(--line);border-radius:12px;background:var(--card);display:block}}
/* latest-improvement banner */
.latest-banner{{display:block;border:1px solid var(--accent-line);border-radius:12px;
  background:linear-gradient(120deg,rgba(56,224,200,.10),rgba(56,224,200,.02));
  padding:1.05rem 1.35rem;margin:0 0 1.2rem;text-decoration:none;box-shadow:var(--glow);transition:transform .18s}}
.latest-banner:hover{{transform:translateY(-2px);text-decoration:none}}
.latest-banner.keepglow{{border-color:var(--accent)}}
.lb-tag{{display:inline-block;font:700 1rem/1 "JetBrains Mono",monospace;letter-spacing:.06em;
  color:var(--accent);background:rgba(56,224,200,.12);border:1px solid var(--accent-line);
  border-radius:999px;padding:.3em .8em;margin-right:.7rem}}
.lb-title{{font:700 1.22rem/1.35 "Space Grotesk",sans-serif;color:var(--ink)}}
.lb-title mark{{background:rgba(56,224,200,.22);color:var(--ink);padding:.05em .25em;border-radius:4px}}
.lb-sum{{display:block;margin-top:.55rem;color:var(--ink-2);font-size:1.04rem;line-height:1.6}}
/* rounds list */
ol.rounds{{list-style:none;padding:0;margin:0}}
ol.rounds li{{border:1px solid var(--line);border-radius:10px;background:var(--card);margin:.6rem 0;
  padding:1rem 1.2rem;transition:border-color .18s, transform .18s}}
ol.rounds li:hover{{border-color:var(--accent-line);transform:translateX(3px)}}
ol.rounds li.latest-item{{border-color:var(--accent);background:linear-gradient(120deg,rgba(56,224,200,.07),var(--card));box-shadow:var(--glow)}}
.rmain{{display:flex;align-items:center;gap:.6rem;justify-content:space-between;flex-wrap:wrap}}
ol.rounds a{{font:600 1.12rem/1.4 "Space Grotesk",sans-serif;color:var(--ink)}}
ol.rounds a:hover{{color:var(--accent)}}
.rsum{{margin-top:.5rem;color:var(--ink-2);font-size:1.02rem;line-height:1.6}}
.pill.latest{{background:rgba(56,224,200,.16);color:var(--accent);border:1px solid var(--accent-line)}}
.hyps-row{{margin-top:.55rem;display:flex;flex-wrap:wrap;gap:.45rem}}
.hchip{{font:500 1rem/1.4 "JetBrains Mono",monospace;color:var(--ink-2);background:var(--bg-2);
  border:1px solid var(--line);border-radius:8px;padding:.35em .65em}}
/* leaderboard highlight */
tr.justimproved td{{background:rgba(56,224,200,.10)}}
.hotdot{{color:var(--accent);font-size:.85em}}
/* top insights */
ol.insights{{list-style:none;padding:0;margin:0}}
ol.insights li{{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;
  background:var(--card);margin:.65rem 0;padding:1rem 1.25rem}}
.ititle{{font:600 1.16rem/1.45 "Space Grotesk",sans-serif;color:var(--ink);display:flex;align-items:baseline;gap:.6rem}}
.inum{{font:700 1rem/1 "JetBrains Mono",monospace;color:var(--accent)}}
.idetail{{margin-top:.45rem;color:var(--ink-2);font-size:1.02rem;line-height:1.62}}
.ev{{font:500 1rem/1 "JetBrains Mono",monospace;color:var(--mut);margin-left:auto;white-space:nowrap}}
/* live status */
section.running{{border-color:var(--accent-line);box-shadow:var(--glow)}}
#nowrunning p{{font-size:1.04rem}}
ul.hyps{{list-style:none;padding:0;margin:.5rem 0 0}} ul.hyps li{{margin:.35rem 0;color:var(--ink-2);font-size:1.04rem}}
.livedot{{display:inline-block;width:.6em;height:.6em;border-radius:50%;background:var(--accent);
  box-shadow:0 0 8px var(--accent);animation:pulse 1.3s ease-in-out infinite}}
.idledot{{display:inline-block;width:.6em;height:.6em;border-radius:50%;background:var(--mut)}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.grid2{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1.4rem;align-items:start}}
@media(max-width:1000px){{.grid2{{grid-template-columns:1fr}}}}
.mdblock{{background:var(--bg-2);border:1px solid var(--line);border-radius:10px;padding:1.5rem 1.8rem;
  font:500 1rem/1.78 "JetBrains Mono",monospace;color:var(--ink);white-space:pre-wrap;overflow-wrap:anywhere;margin:0}}
.leakwarn{{color:var(--amber);font:600 1rem/1 "JetBrains Mono",monospace;white-space:nowrap;margin-left:.4em}}
.leakok{{color:var(--mut);font:600 1rem/1 "JetBrains Mono",monospace;white-space:nowrap;margin-left:.4em}}
.tabpanel .causalgraph,.tabpanel .block button{{margin-top:.6rem}}
</style></head><body><div class="dash">
<div class="nav">autoresearch / reports · <span style="color:var(--accent)">Flask</span></div>
<div class="hero">
  <div class="eyebrow">Autoresearch · Quant Pipeline</div>
  <h1>Experiment reports</h1>
  <p class="tldr">Each round races two hypotheses on the weakest ETF and keeps the winner if it beats that ETF's
  best (real OOS Calmar + DA). Live; auto-refreshing every 30s.</p>
  <div class="kpis">{kpis}</div>
</div>

<nav class="tabbar" role="tablist">
  <a class="tab" href="#overview" data-tab="overview">▸ Overview</a>
  <a class="tab" href="#rounds" data-tab="rounds">Rounds ({len(rounds)})</a>
  <a class="tab" href="#insights" data-tab="insights">Insights</a>
  <a class="tab" href="#graph" data-tab="graph">Causal graph</a>
  <a class="tab" href="#program" data-tab="program">program.md</a>
</nav>

<section class="tabpanel" id="tab-overview" data-panel="overview">
  {status_html}
  {banner}
  <section class="block"><h2>Leaderboard — best strategy vs buy-and-hold (real OOS)</h2>
  <div class="tablewrap"><table><thead><tr><th>ETF</th><th class="num">best Calmar</th><th class="num">CAGR</th><th class="num">MDD</th><th class="num">DA</th><th class="num">trades</th><th class="num">buy&amp;hold</th><th class="num">edge</th><th>cell</th></tr></thead>
  <tbody>{lb_rows}</tbody></table></div>
  <p class="small"><b>Calmar</b> = CAGR/MaxDD (higher better); <b>CAGR</b> = compounding annual return; <b>MDD</b> = max drawdown; <b>DA</b> = drawdown area = Σ(1−equity/peak) (all three: lower MDD/DA better). buy&amp;hold = pure 1-trade hold over OOS (2023-08 → 2026-06); edge = best − buy&amp;hold. ▲ = improved this round. CAGR/MDD show “—” until an ETF is (re)validated under the leak-fixed bars.</p></section>
</section>

<section class="tabpanel" id="tab-insights" data-panel="insights">
  <section class="block"><h2>Top {len(ins)} insights — what the research has learned</h2>
  <ol class="insights">{ins_html}</ol></section>
</section>

<section class="tabpanel" id="tab-rounds" data-panel="rounds">
  <section class="block"><h2>Rounds ({len(rounds)}) — newest first</h2>
  <ol class="rounds">
{items}
  </ol></section>
</section>

<section class="tabpanel" id="tab-graph" data-panel="graph">
  <section class="block"><h2>Causal graph — every experiment &amp; finding</h2>
  <p class="small">How each outcome <i>caused</i> the next hypothesis. Drag / scroll-zoom / hover for full text; double-click a phase cluster to expand. Full page: <a href="causal_graph.html">causal_graph.html</a></p>
  {graph_inline}</section>
</section>

<section class="tabpanel" id="tab-program" data-panel="program">
  <section class="block"><h2>program.md — the loop</h2>
  <p class="small">The Karpathy-minimal spec the research follows each round. Raw: <a href="program.md">program.md</a></p>
  {prog_inline}</section>
</section>
</div>
<script>
function showTab(name){{
  var tabs=document.querySelectorAll('.tab'), panels=document.querySelectorAll('.tabpanel');
  var found=false;
  panels.forEach(function(p){{var on=p.dataset.panel===name;p.classList.toggle('active',on);if(on)found=true;}});
  tabs.forEach(function(t){{t.classList.toggle('active',t.dataset.tab===name);}});
  if(!found){{showTab('overview');return;}}
  // vis-network renders at 0px while its tab is hidden — resize/fit on activation.
  if(name==='graph' && window['cgtab_net']){{
    var n=window['cgtab_net'];
    setTimeout(function(){{try{{n.setSize('100%','660px');n.redraw();n.fit();}}catch(e){{}}}},60);
  }}
}}
function curTab(){{return (location.hash||'#overview').replace('#','');}}
window.addEventListener('hashchange',function(){{showTab(curTab());}});
showTab(curTab());

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
        + '<p>phase: <b>' + (s.phase || '…') + '</b>'
        + (s.since ? ' \\u00b7 started ' + s.since : '')
        + (s.legs ? ' \\u00b7 ' + s.legs : '') + '</p>'
        + '<ul class="hyps">' + legs + '</ul>';
    }} else {{
      el.className = 'block';
      el.innerHTML = '<h2><span class="idledot"></span>Idle</h2><p>last completed: <b>'
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
