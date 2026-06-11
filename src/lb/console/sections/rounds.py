#!/usr/bin/env python3
"""console.sections.rounds — the rounds ledger (id=rounds).

Blueprint section "rounds" (tertiary): the full experiment audit trail, PAGINATED.
This is provenance, not front-page weight — so the perf plan applies here:

  * render ONLY the newest ~25 rounds server-side (the old monolith serialized all
    1206 distinct-timestamp rounds as <li>, ~1180 carried a "hidden extra" class but
    still fully downloaded — ~620 KB / half the 1.3 MB page). We stop emitting the
    ~1180 hidden <li> entirely.
  * the header count is the REAL CSV total (e.g. "Rounds (1206)"), not the 25 shown.
  * "show all" no longer reveals hidden rows; it lazy-fetches the rest from the new
    /rounds.json?page= endpoint (data-src hook below; the wiring lives in console.js).
  * we drop the dead deep-link <a> spans for rounds with no backing page (the 131
    stored round_*.html are being deleted; drill-down is a future dynamic /round/<n>).

Kept verbatim in spirit: the KEEP/DISCARD filter chips (.fchip[data-f] / .fchip.on),
the per-round recipe gloss + cell chips (.hyps-row/.hchip), and the signed round-Calmar
(.rcal) — the same .ritem/.keep/.discard/.rmain/.rsum hooks the client filter + show-all
JS already target, so nothing in console.js has to change.

Pure ctx -> HTML. Reads ONLY ctx["rounds"] (console.data._scan_rounds_csv, already
newest-first) — no file-I/O. Status verdicts route through P.chip (a primitive that
OWNS its CSS); the surviving ledger containers (.rounds/.ritem/.rmain/.rsum/.hyps-row/
.hchip/.rcal/.filters/.fchip/.showall) are all defined in reports/style.css, so no
undefined-class is emitted.
"""
from lb.console import primitives as P

# How many rounds to render server-side. The rest are lazy-fetched from /rounds.json.
SHOWN = 25


# ---- tiny formatters -------------------------------------------------------
def _signed(v):
    """Signed 2-dp Calmar, or an em-dash when missing."""
    try:
        return f"{float(v):+.2f}"
    except (TypeError, ValueError):
        return "—"


# ---- one ledger row --------------------------------------------------------
def _summary(d):
    """Plain-English line: what beat (or failed to beat) the weakest ETF's standing
    best, on real OOS Calmar — derived from the live round dict, never hardcoded."""
    etf = P._esc(d.get("etf", ""))
    wc = _signed(d.get("win_calmar"))
    prevc = d.get("prev_calmar")
    if d.get("verdict") == "keep":
        if prevc is not None:
            sub = (f"New best for {etf}: Calmar <b>{wc}</b> beat the prior "
                   f"{_signed(prevc)}.")
        else:
            sub = f"New best for {etf}: Calmar <b>{wc}</b>."
    else:
        if prevc is not None:
            sub = (f"Calmar <b>{wc}</b> did not beat {etf}'s standing best "
                   f"{_signed(prevc)} — kept the incumbent.")
        else:
            sub = f"Calmar <b>{wc}</b> — not kept."
    trd = d.get("win_trades")
    trd_txt = f" · {P._esc(trd)} trades" if trd not in (None, "") else ""
    return f'<div class="rsum">{sub}{trd_txt}</div>'


def _recipe_chips(d):
    """The recipe gloss + the exact cell id, as the existing mono .hchip code-chips
    (distinct from the status-pill primitive; .hyps-row/.hchip are defined in style.css)."""
    recipe = d.get("win_recipe")
    if not recipe:
        return ""
    return (f'<div class="hyps-row"><span class="hchip">{P._esc(recipe)}</span>'
            f'<span class="hchip">cell {P._esc(d.get("win_cell", ""))}</span></div>')


def _item(d):
    """One ledger <li>. The .ritem + verdict class are the client filter / show-all
    JS hooks; ol.rounds li carries the visual style. The verdict badge routes through
    P.chip (KEEP -> pos, DISCARD -> muted). No dead deep-link span: rounds with no
    backing page render a plain .rtitle (drill-down is the future dynamic /round/<n>)."""
    v = d.get("verdict", "discard")                 # 'keep' | 'discard'
    badge = (P.chip("KEEP", "pos", "the winner beat the weakest ETF's standing best")
             if v == "keep"
             else P.chip("DISCARD", "muted", "did not beat the incumbent — kept the standing best"))
    title = f'Round {P._esc(d.get("n", ""))} · {P._esc(d.get("etf", ""))}'
    head = f'<span class="rtitle">{title}</span>'
    rcal = f'<span class="rcal">{_signed(d.get("win_calmar"))}</span>'
    return (f'<li class="ritem {v}"><div class="rmain">{head} {badge}{rcal}</div>'
            f'{_summary(d)}{_recipe_chips(d)}</li>')


# ---- the filter chips ------------------------------------------------------
def _filters():
    """KEEP / DISCARD / all filter chips — the .fchip[data-f] hooks the client JS toggles."""
    return ('<div class="filters">'
            '<button class="fchip on" data-f="all">all</button>'
            '<button class="fchip" data-f="keep">KEEP</button>'
            '<button class="fchip" data-f="discard">DISCARD</button></div>')


# ---- the section -----------------------------------------------------------
def render(ctx):
    """ctx -> the complete <section id="rounds"> HTML fragment (newest ~25 only)."""
    rounds = ctx.get("rounds") or []
    total = len(rounds)
    keeps = sum(1 for d in rounds if d.get("verdict") == "keep")
    shown = rounds[:SHOWN]                           # already newest-first from data.py

    if not rounds:
        ledger = '<p class="small">No rounds recorded yet.</p>'
        showall = ""
    else:
        ledger = '<ol class="rounds">' + "".join(_item(d) for d in shown) + "</ol>"
        # "show all" lazy-fetches the rest from /rounds.json (wired in console.js);
        # data-src is the endpoint hook, data-shown the server-rendered offset.
        rest = max(total - len(shown), 0)
        showall = (f'<button class="showall" id="showall" data-src="rounds.json" '
                   f'data-shown="{len(shown)}">show all {total} rounds'
                   + (f" ({rest} more)" if rest else "")
                   + "</button>") if rest else ""

    body = (
        P.eyebrow("ROUNDS LEDGER · FULL AUDIT TRAIL, PAGINATED")
        + f"<h2>Rounds ({total})</h2>"
        + P.provenance(
            f"Newest {len(shown)} of {total} rounds shown · {keeps} KEEPs · built live "
            "from results/round_results.csv. Each round races one new recipe against the "
            "weakest ETF's standing best — KEEP only if it beats that bar on real "
            "out-of-sample Calmar, else DISCARD and keep the incumbent.")
        + _filters()
        + ledger
        + showall
    )
    return '<section class="block" id="rounds">' + body + "</section>"


if __name__ == "__main__":
    from lb.console.data import build_ctx
    out = render(build_ctx())
    print(f"rounds section: {len(out)} bytes")
    n_li = out.count('<li class="ritem')
    assert n_li <= SHOWN, f"emitted {n_li} rows — must render only newest {SHOWN}"
    for needle in ('id="rounds"', "<h2>Rounds (", 'class="filters"',
                   'data-f="keep"', 'data-f="discard"', 'class="rounds"',
                   "ROUNDS LEDGER"):
        assert needle in out, f"missing: {needle}"
    assert "hidden extra" not in out, "rounds leaked the hidden-extra <li> blob (perf regression)"
    assert "pending" not in out.lower(), "rounds leaked a 'pending' value"
    print(f"ok — {n_li} rows (<= {SHOWN}), filter chips + show-all + no hidden blob")
