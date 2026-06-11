#!/usr/bin/env python3
"""render_index — the autoresearch "Research Console" entrypoint (thin shim).

The former 1041-line monolith was decomposed (STEP 1-5) into the scripts/console/
package: console.data (file-I/O + derive + the three RESOLVERS), console.primitives
(class-owning HTML helpers), console.sections.* (one pure ctx->HTML builder per
section), and console.page (the data-driven SECTIONS registry + assembler).

This module now only RE-EXPORTS the two public entrypoints the rest of the repo
already imports, so nothing downstream changes:

    from render_index import build_html      # app.py, run_round.py
    from render_index import build_data       # app.py /data.json

  python3 scripts/render_index.py            # write reports/index.html (asserts <200KB)

Every headline number flows through a resolver in console.data, so the page can
never again show 'pending' or contradict knowledge.json.
"""
from lb.paths import REPORTS_DIR

# Public entrypoints (the SAME names app.py + the driver import) ---------------
from lb.console.page import build_html, build_data, build_page  # noqa: F401
# Re-export the most-used data helpers so any legacy `render_index.X` keeps working.
from lb.console.data import (  # noqa: F401
    build_ctx, derive, scoreboard, load_screen, screen_summary,
    book_resolver, edges_resolver, honesty_resolver, _scan_rounds_csv,
)

MAX_BYTES = 200_000  # perf-regression guard: a future inline blob must not silently return


def main():
    html = build_html()
    out = str(REPORTS_DIR / "index.html")
    with open(out, "w") as f:
        f.write(html)
    n = len(html.encode("utf-8"))
    assert n < MAX_BYTES, f"index.html is {n} bytes (>= {MAX_BYTES}) — perf regression"
    print(f"wrote research console index.html ({n} bytes, < {MAX_BYTES} guard)")


if __name__ == "__main__":
    main()
