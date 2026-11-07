#!/usr/bin/env python3
"""Check public canonical links and inventory owned documentation without network I/O."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

LINK = re.compile(r'!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
SUFFIXES = {'.md', '.rst', '.txt'}


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)[0]
            if fence is None:
                fence = token
            elif token == fence:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return '\n'.join(lines)


def anchors(text):
    found = set()
    counts = Counter()
    for line in prose(text).splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if match:
            slug = re.sub(r'[^\w\- ]', '', match.group(1).lower()).replace(' ', '-')
            n = counts[slug]
            counts[slug] += 1
            found.add(slug + ('-'+str(n) if n else ''))
    found.update(re.findall(r'<a\s+(?:name|id)=["\']([^"\']+)', text))
    return found


def owned(root, native=False):
    if not native and (root/'.git').exists():
        output = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
        return sorted({Path(os.fsdecode(v)) for v in output.split(b'\0') if v and Path(os.fsdecode(v)).suffix.lower() in SUFFIXES and not Path(os.fsdecode(v)).name.startswith('requirements')})
    result = []
    excluded = {'.git', '.local', '.venv', 'venv', 'target', '__pycache__', 'node_modules', 'tools', 'data', 'workspaces', '.pytest_cache'}
    for directory, dirs, files in os.walk(root):
        current = Path(directory)
        if current != root and '.git' in dirs:
            dirs[:] = []  # independent nested repositories are separate ownership
            continue
        dirs[:] = [name for name in dirs if name not in excluded and not name.startswith('.') and not name.endswith('.egg-info')]
        result.extend((current/name).relative_to(root) for name in files if Path(name).suffix.lower() in SUFFIXES and not name.startswith('requirements'))
    return sorted(set(result))


def inspect(root, relative, canonical):
    text = (root/relative).read_text(errors='replace')
    body = prose(text)
    missing = []
    external_local = 0
    count = 0
    for match in LINK.finditer(body):
        target = unquote(match.group(1).strip('<>'))
        split = urlsplit(target)
        if split.scheme or split.netloc:
            continue
        count += 1
        if split.path.startswith('/'):
            external_local += 1
            continue
        dest = (root/relative).parent/split.path if split.path else root/relative
        if not dest.exists():
            missing.append(target)
        elif split.fragment and dest.is_file() and dest.suffix.lower() == '.md':
            if split.fragment not in anchors(dest.read_text(errors='replace')):
                missing.append(target)
    title = next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith('# ')), relative.name)
    role = 'canonical-current' if str(relative) in canonical else 'retained-history-reference-or-evidence'
    return dict(path=str(relative), role=role, title=title, local_links=count,
                workstation_local_references=external_local, unavailable_links=sorted(set(missing)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-inventory', action='store_true')
    parser.add_argument('--native-root', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = json.loads((root/'docs/doc-check.json').read_text())
    canonical = set(config['canonical'])
    mixed = set(config.get('mixed', []))
    if args.write_inventory and not (root/'docs/documentation-inventory.json').exists():
        (root/'docs/documentation-inventory.json').write_text('{}\n')
    paths = owned(root)
    rows = [inspect(root, path, canonical) for path in paths if (root/path).is_file()]
    for row in rows:
        if row['path'] in mixed:
            row['role'] = 'current-guide-with-retained-historical-evidence'
    errors = [f"{row['path']}: {link}" for row in rows if row['role']=='canonical-current' for link in row['unavailable_links']]
    errors += [name+': missing canonical document' for name in canonical if not (root/name).is_file()]
    errors += [f"{row['path']}: workstation-local link in canonical public guide" for row in rows if row['role']=='canonical-current' and row['workstation_local_references']]
    inventory = dict(scope='Tracked/nonignored owned prose; network URLs not fetched; fenced code not interpreted as links.',
                     counts=dict(documents=len(rows), canonical=sum(row['role']=='canonical-current' for row in rows), mixed=sum(row['role']=='current-guide-with-retained-historical-evidence' for row in rows), retained=sum(row['role']=='retained-history-reference-or-evidence' for row in rows)), documents=rows)
    if args.native_root:
        native = args.native_root.resolve()
        private_paths = owned(native, native=True)
        inventory['native_scope'] = dict(documents=len(private_paths),
            by_area=dict(sorted(Counter(str(p).split('/')[0] for p in private_paths).items())),
            exclusions='Runtime data, .local, environments, target, tools/vendor, workspaces and independent nested Git repositories. Native historical results are counted but not republished or economically revalidated.',
            canonical_comparison={name: ('same' if (native/name).exists() and (native/name).read_bytes()==(root/name).read_bytes() else 'publication revision supersedes native copy' if (native/name).exists() else 'new public guide') for name in sorted(canonical)})
    if args.write_inventory:
        (root/'docs/documentation-inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(**inventory['counts'], canonical_errors=errors,
                         retained_unavailable_links=sum(len(row['unavailable_links']) for row in rows if row['role']!='canonical-current'))))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
