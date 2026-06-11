#!/usr/bin/env python3
"""Flask report server for autoresearch — one small app, rendered LIVE per request.

  GET /  ·  /index.html      -> the dashboard (console.page.build_html, live from knowledge.json)
  GET /data.json             -> the 8s status-poll payload (build_data)
  GET /graph.json            -> causal-graph nodes/edges/phases (lazy-fetched by console.js)
  GET /rounds.json?page=     -> paginated rounds ledger (the "show all" lazy loader)
  GET /round/<n>             -> dynamic per-round drill-down (replaces the deleted round_*.html)
  GET /causal_graph.html     -> the full interactive graph, rendered live (no stored file)
  GET /program.md · /deployment.md  -> themed docs
  GET /style.css · /console.js · /status.json  -> static assets (allowlisted)

Caching: dynamic routes are no-store (live); style.css/console.js get a short max-age
+ ETag so they cache across the poll loop. Text responses are gzipped on the fly
(no flask-compress dependency): the ~170 KB page ships ~20-25 KB over the wire.

Run directly (python3 scripts/app.py); the systemd unit autoresearch-reports binds :80.
"""
import re as _re
import gzip
import json as _json
import html as _html

from flask import Flask, Response, request, send_from_directory, abort

from lb.paths import REPORTS_DIR, ROOT
from lb.console.render_index import build_html, build_data
from lb.console.data import _scan_rounds_csv, _load, KJ
from lb.console.sections import rounds as rounds_sec       # reuse _item / SHOWN

REPORTS = str(REPORTS_DIR)
PROGRAM = str(ROOT / "program.md")
DEPLOY = str(ROOT / "docs" / "analysis" / "DEPLOYMENT.md")

app = Flask(__name__, static_folder=None)

# Routes that reflect LIVE research state -> never cache.
_DYNAMIC = {"/", "/index.html", "/data.json", "/status.json",
            "/graph.json", "/rounds.json", "/causal_graph.html"}


@app.after_request
def _headers(resp):
    """Scope no-store to dynamic routes; give static assets a short max-age; gzip text."""
    p = request.path
    if p in _DYNAMIC or p.startswith("/round/") or p.endswith(".md"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    elif p.endswith((".css", ".js")):
        # overwrite send_from_directory's default 'no-cache' — these cache across the
        # 8s poll loop, with the ETag / Last-Modified it already set for revalidation.
        resp.headers["Cache-Control"] = "public, max-age=300"
    return _maybe_gzip(resp)


def _maybe_gzip(resp):
    """Gzip text/json responses the client accepts — skips streamed/static passthrough
    responses (style.css/console.js are small + already cached) and tiny bodies."""
    try:
        if resp.direct_passthrough or resp.status_code != 200:
            return resp
        if "gzip" not in (request.headers.get("Accept-Encoding") or ""):
            return resp
        if resp.headers.get("Content-Encoding"):
            return resp
        ct = resp.headers.get("Content-Type", "")
        if not (ct.startswith("text/") or "json" in ct or "javascript" in ct):
            return resp
        data = resp.get_data()
        if len(data) < 1024:
            return resp
        resp.set_data(gzip.compress(data, 6))
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Vary"] = "Accept-Encoding"
    except Exception:
        pass
    return resp


@app.route("/")
@app.route("/index.html")
def index():
    return Response(build_html(), mimetype="text/html")


@app.route("/data.json")
def data_json():
    return Response(_json.dumps(build_data()), mimetype="application/json")


@app.route("/graph.json")
def graph_json():
    """Causal-graph dataset for the lazy vis.Network build (console.js)."""
    from lb.console.render_causal_graph import vis_data
    k = _load(KJ, {})
    cg = k.get("causal_graph", {}) or {}
    try:
        nodes, edges = vis_data(cg)
    except Exception:
        nodes, edges = [], []
    return Response(_json.dumps({"nodes": nodes, "edges": edges,
                                 "phases": cg.get("phases", [])}),
                    mimetype="application/json")


@app.route("/rounds.json")
def rounds_json():
    """Paginated rounds ledger. page=1 is the newest `SHOWN` (already server-rendered
    on the page); the "show all" button starts at page=2 and appends each chunk."""
    try:
        page = max(1, int(request.args.get("page", 1) or 1))
    except (TypeError, ValueError):
        page = 1
    size = rounds_sec.SHOWN
    allr = _scan_rounds_csv()
    total = len(allr)
    start, end = (page - 1) * size, (page - 1) * size + size
    chunk = allr[start:end]
    items = "".join(rounds_sec._item(d) for d in chunk)
    pages = (total + size - 1) // size if size else 1
    return Response(_json.dumps({"items": items, "page": page, "pages": pages,
                                 "total": total, "count": len(chunk),
                                 "has_more": end < total}),
                    mimetype="application/json")


@app.route("/round/<int:n>")
def round_n(n):
    """Dynamic per-round drill-down from round_results.csv — replaces the 131 deleted
    stored round_*.html files (covers all rounds, always fresh, zero stored files)."""
    d = next((r for r in _scan_rounds_csv() if r.get("n") == n), None)
    if not d:
        abort(404)
    def esc(x):
        return _html.escape("" if x is None else str(x))
    wc = d.get("win_calmar")
    wc_txt = f"{wc:+.2f}" if isinstance(wc, (int, float)) else "—"
    pv = d.get("prev_calmar")
    pv_txt = f"{pv:+.2f}" if isinstance(pv, (int, float)) else "—"
    verdict = "KEEP" if d.get("verdict") == "keep" else "DISCARD"
    body = (
        f'<div class="nav"><a href="/">← dashboard</a><span class="sep">·</span>'
        f'<a href="/#rounds">rounds ledger</a></div>'
        f'<h1>Round {esc(d.get("n"))} · {esc(d.get("etf"))} '
        f'<span class="pill {d.get("verdict","")}">{verdict}</span></h1>'
        f'<p class="small">{esc(d.get("ts"))} · recipe <code>{esc(d.get("win_recipe"))}</code></p>'
        f'<table><tbody>'
        f'<tr><th>winner Calmar</th><td>{wc_txt}</td></tr>'
        f'<tr><th>prior best</th><td>{pv_txt} <code>{esc(d.get("prev_cell"))}</code></td></tr>'
        f'<tr><th>trades</th><td>{esc(d.get("win_trades"))}</td></tr>'
        f'<tr><th>directional acc.</th><td>{esc(d.get("win_da"))}</td></tr>'
        f'<tr><th>cell</th><td><code>{esc(d.get("win_cell"))}</code></td></tr>'
        f'<tr><th>loser Calmar</th><td>{esc(d.get("lose_calmar"))}</td></tr>'
        f'</tbody></table>')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Round {esc(n)} · {esc(d.get("etf"))}</title>'
            '<link rel="stylesheet" href="/style.css"></head><body>'
            f'<div class="dash"><section class="block">{body}</section></div></body></html>')
    return Response(page, mimetype="text/html")


@app.route("/causal_graph.html")
def causal_graph():
    """The full interactive causal graph, rendered live from knowledge.json (replaces
    the deleted stored reports/causal_graph.html — always fresh, off the page weight)."""
    from lb.console.render_causal_graph import standalone
    k = _load(KJ, {})
    cg = k.get("causal_graph", {}) or {}
    try:
        return Response(standalone(k, cg), mimetype="text/html")
    except Exception:
        abort(404)


def _md_page(path, title):
    try:
        with open(path) as fh:
            txt = fh.read()
    except Exception:
        txt = "(" + title + " not found)"
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>' + title + '</title>'
            '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
            '<link rel="stylesheet" href="style.css">'
            '<style>.mdwrap{max-width:920px;margin:0 auto;padding:1.6rem 1.4rem 5rem}'
            'pre.md{background:var(--card);border:1px solid var(--line);border-radius:12px;'
            'padding:1.7rem 1.9rem;font:500 18px/1.78 "JetBrains Mono",monospace;color:var(--ink);'
            'white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;box-shadow:var(--shadow)}'
            '</style></head><body><div class="mdwrap">'
            '<div class="nav"><a href="/">← dashboard</a><span class="sep">·</span>'
            '<a href="/program.md">program.md</a><span class="sep">·</span>'
            '<a href="/deployment.md">deployment.md</a></div>'
            '<pre class="md">' + _html.escape(txt) + '</pre></div></body></html>')
    return Response(page, mimetype="text/html")


@app.route("/program.md")
def program():
    return _md_page(PROGRAM, "program.md")


@app.route("/deployment.md")
def deployment():
    return _md_page(DEPLOY, "deployment.md")


# Static allowlist — the round_*.html + causal_graph.html files are GONE (now routes),
# and /data is route-shadowed; only style.css, console.js and status.json remain static.
_ALLOW = _re.compile(r"^(style|console|status)\.(css|js|json)$")


@app.route("/<path:fname>")
def static_file(fname):
    # send_from_directory blocks ../ traversal but would otherwise serve every file under
    # REPORTS; the allowlist restricts it to the intended static assets only.
    if not _ALLOW.fullmatch(fname):
        abort(404)
    return send_from_directory(REPORTS, fname)


def main():
    app.run(host="0.0.0.0", port=80, threaded=True)


if __name__ == "__main__":
    main()
