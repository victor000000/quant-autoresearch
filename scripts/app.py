#!/usr/bin/env python3
"""Flask report server for autoresearch — simple is better, one small app.

  GET /  and  /index.html   -> the dashboard, rendered LIVE from knowledge.json on
                               every request (no stale file; reflects the latest round)
  GET /<file>               -> static reports: round_*.html, causal_graph.html,
                               style.css, status.json (served no-cache so the live
                               poller + auto-refresh always see fresh data)

Run directly (python3 scripts/app.py); the systemd unit autoresearch-reports binds :80.
"""
import os, sys
from flask import Flask, Response, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPORTS = os.path.abspath(os.path.join(HERE, "..", "autoresearch", "reports"))
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


@app.route("/<path:fname>")
def static_file(fname):
    return send_from_directory(REPORTS, fname)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True)
