#!/usr/bin/env python3
"""Flask report server for autoresearch — simple is better, one small app.

  GET /  and  /index.html   -> the dashboard, rendered LIVE from knowledge.json on
                               every request (no stale file; reflects the latest round)
  GET /<file>               -> static reports: round_*.html, causal_graph.html,
                               style.css, status.json (served no-cache so the live
                               poller + auto-refresh always see fresh data)

Run directly (python3 scripts/app.py); the systemd unit autoresearch-reports binds :80.
"""
import os, sys, html as _html
from flask import Flask, Response, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPORTS = os.path.abspath(os.path.join(HERE, "..", "autoresearch", "reports"))
PROGRAM = os.path.abspath(os.path.join(HERE, "..", "autoresearch", "program.md"))
DEPLOY = os.path.abspath(os.path.join(HERE, "..", "autoresearch", "DEPLOYMENT.md"))
from render_index import build_html   # live dashboard renderer

app = Flask(__name__, static_folder=None)


@app.after_request
def _nocache(resp):
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.route("/")
@app.route("/index.html")
def index():
    return Response(build_html(), mimetype="text/html")


@app.route("/data.json")
def data_json():
    from render_index import build_data
    return Response(__import__("json").dumps(build_data()), mimetype="application/json")


def _md_page(path, title):
    try:
        txt = open(path).read()
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


import re as _re
from flask import abort
_ALLOW = _re.compile(r"^(round_\d+[a-z]?|causal_graph|style|status|data)\.(html|json|css)$")


@app.route("/<path:fname>")
def static_file(fname):
    # Allowlist the intended output files only — send_from_directory blocks ../ traversal but would
    # otherwise serve every file under REPORTS (templates, _md_legacy, etc.) with no auth.
    if not _ALLOW.fullmatch(fname):
        abort(404)
    return send_from_directory(REPORTS, fname)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True)
