#!/usr/bin/env python3
"""Render autoresearch markdown reports -> styled HTML with MathJax-rendered math.

Stdlib only (no pip). Handles the report subset: # headers, **bold**, `code`,
fenced ``` blocks, > blockquotes, GitHub | tables |, --- rules, -/1. lists,
and $inline$ / $$display$$ LaTeX (passed through verbatim for MathJax in-browser).

Run after writing each round's report (it regenerates ALL of reports/*.md):
    python3 scripts/reports_to_html.py
Output: autoresearch/reports/html/<name>.html  +  index.html
"""
import os, re, html, glob

REPORTS = "/home/ubuntu/lb/autoresearch/reports"
OUT = os.path.join(REPORTS, "html")

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}},svg:{{fontCache:'global'}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
 body{{font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1b1f23;
   max-width:900px;margin:2rem auto;padding:0 1.2rem;background:#fff}}
 h1,h2,h3{{line-height:1.25;margin-top:1.6em}} h1{{border-bottom:2px solid #eaecef;padding-bottom:.3em}}
 h2{{border-bottom:1px solid #eaecef;padding-bottom:.2em}}
 code{{background:#f3f4f6;padding:.15em .35em;border-radius:4px;font-size:.88em;
   font-family:SFMono-Regular,Consolas,monospace}}
 pre{{background:#f6f8fa;padding:1em;border-radius:8px;overflow:auto}} pre code{{background:none;padding:0}}
 table{{border-collapse:collapse;margin:1em 0;width:100%;font-size:.94em}}
 th,td{{border:1px solid #d0d7de;padding:.45em .7em;text-align:left}} th{{background:#f6f8fa}}
 tr:nth-child(even) td{{background:#fbfcfd}}
 blockquote{{border-left:4px solid #d0d7de;color:#57606a;margin:1em 0;padding:.2em 1em;background:#f6f8fa}}
 hr{{border:0;border-top:1px solid #eaecef;margin:1.6em 0}}
 a{{color:#0969da;text-decoration:none}} a:hover{{text-decoration:underline}}
 .nav{{font-size:.9em;color:#57606a;margin-bottom:1.5em}}
</style></head><body>
<div class="nav"><a href="index.html">&larr; all reports</a></div>
"""
FOOT = "\n</body></html>\n"


def _inline(t):
    """Inline markdown -> HTML, protecting code + math from escaping/formatting."""
    keep = []
    def stash(m):
        keep.append(m.group(0)); return f"\x00{len(keep)-1}\x00"
    t = re.sub(r"\$\$.+?\$\$", stash, t)       # display math
    t = re.sub(r"\$[^$]+?\$", stash, t)         # inline math
    t = re.sub(r"`[^`]+?`", stash, t)           # inline code
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    def restore(m):
        s = keep[int(m.group(1))]
        if s.startswith("`"):
            return "<code>" + html.escape(s[1:-1], quote=False) + "</code>"
        return s  # math: verbatim for MathJax
    return re.sub(r"\x00(\d+)\x00", restore, t)


def md_to_html(md):
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                # fenced code
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf), quote=False) + "</code></pre>")
            continue
        if re.match(r"^\|.*\|\s*$", ln) and i+1 < len(lines) and re.match(r"^\|[\s:|-]+\|\s*$", lines[i+1]):
            rows = []                            # table
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            head, body = rows[0], rows[2:]
            t = "<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            for r in body:
                t += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</tbody></table>"); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1)); out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>"); i += 1; continue
        if re.match(r"^---+\s*$", ln):
            out.append("<hr>"); i += 1; continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip()); i += 1
            out.append("<blockquote>" + _inline(" ".join(buf)) + "</blockquote>"); continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            ordered = bool(re.match(r"\d+\.", m.group(2))); tag = "ol" if ordered else "ul"; items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm: break
                items.append(f"<li>{_inline(mm.group(3))}</li>"); i += 1
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>"); continue
        if ln.strip() == "":
            i += 1; continue
        out.append("<p>" + _inline(ln) + "</p>"); i += 1
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    mds = sorted(glob.glob(os.path.join(REPORTS, "*.md")))
    made = []
    for md in mds:
        name = os.path.splitext(os.path.basename(md))[0]
        with open(md) as f:
            body = md_to_html(f.read())
        with open(os.path.join(OUT, name + ".html"), "w") as f:
            f.write(HEAD.format(title=name) + body + FOOT)
        made.append(name)
    # index
    links = "".join(f'<li><a href="{n}.html">{n}</a></li>' for n in made)
    idx = (HEAD.format(title="Autoresearch reports").replace(
            '<div class="nav"><a href="index.html">&larr; all reports</a></div>', "")
           + "<h1>Autoresearch — tech reports</h1><ul>" + links + "</ul>" + FOOT)
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(idx)
    print(f"wrote {len(made)} reports + index -> {OUT}")
    for n in made:
        print("  ", n + ".html")


if __name__ == "__main__":
    main()
