"""console — the autoresearch dashboard rendering package.

Decomposes the former render_index.py monolith into testable units:

  console.data        — ALL file-I/O + derive + the three RESOLVERS (single
                        source of truth for every headline number).
  console.primitives  — HTML primitives (_spark/_edgebar/_num + card/table/chip/
                        kpi/matrix/eyebrow/provenance); each helper OWNS its CSS
                        classes so a renderer can never emit an undefined class.
  console.sections.*  — (STEP 4) one pure ctx->HTML builder per page section.
  console.page        — (STEP 5) data-driven SECTIONS registry + build_html.

The resolver pattern (book_resolver / edges_resolver / honesty_resolver) is the
anti-divergence mechanism: every Calmar/Sharpe/DSR flows from knowledge.json
through a resolver, so the page can never again show 'pending' or contradict the
research state.
"""
