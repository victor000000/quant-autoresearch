#!/usr/bin/env python3
"""
QC ObjectStore listing/cleanup tool.

Reads credentials from the QC server's settings DB and talks to the QC REST API.
List-only by default; pass --delete to actually delete keys matching --pattern.
"""
import argparse
import base64
import hashlib
import sqlite3
import sys
import time
from collections import defaultdict

import requests

API_BASE = "https://www.quantconnect.com/api/v2"
SETTINGS_DB = "/home/txy/qc_work/data/S003.db"


def load_creds(db_path=SETTINGS_DB):
    con = sqlite3.connect(db_path)
    rows = dict(con.execute(
        "SELECT key, value FROM settings WHERE category IN ('credentials','project')"
    ).fetchall())
    return rows["qc_user_id"], rows["qc_api_token"], rows["organization_id"]


def auth_headers(user_id, api_token):
    ts = str(int(time.time()))
    hash_hex = hashlib.sha256(f"{api_token}:{ts}".encode()).hexdigest()
    basic = base64.b64encode(f"{user_id}:{hash_hex}".encode()).decode()
    return {"Authorization": f"Basic {basic}", "Timestamp": ts}


def list_path(user_id, api_token, org_id, path):
    r = requests.post(
        f"{API_BASE}/object/list",
        headers=auth_headers(user_id, api_token),
        json={"organizationId": org_id, "path": path},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(f"list_objects failed at {path!r}: {j.get('errors')}")
    return j.get("objects", [])


def delete_key(user_id, api_token, org_id, key):
    r = requests.post(
        f"{API_BASE}/object/delete",
        headers=auth_headers(user_id, api_token),
        json={"organizationId": org_id, "key": key},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    return j.get("success", False), j.get("errors", [])


def walk(user_id, api_token, org_id, root=""):
    """Yield (key, size, is_folder) for every object under root."""
    stack = [root]
    while stack:
        p = stack.pop()
        try:
            items = list_path(user_id, api_token, org_id, p)
        except Exception as e:
            print(f"  [warn] list failed at {p!r}: {e}", file=sys.stderr)
            continue
        for it in items:
            key = it.get("key", "")
            size = int(it.get("size", 0) or 0)
            folder = bool(it.get("folder")) or key.endswith("/")
            yield key, size, folder
            if folder:
                stack.append(key)


def categorize(key):
    if key.startswith("bars_zn_v1/"):
        return "bars_zn_v1"
    if key.startswith("bars_zn_v"):
        return f"bars_zn_other ({key.split('/')[0]})"
    if key.startswith("bars_"):
        return f"bars_other ({key.split('/')[0]})"
    if key.startswith("models_"):
        return key.split("/")[0]
    if key.startswith("oos_") or key.startswith("series_"):
        return "oos_series"
    if "/" in key:
        return key.split("/")[0]
    return "top-level"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="ObjectStore prefix to walk")
    ap.add_argument("--delete", action="store_true", help="actually delete matched keys")
    ap.add_argument("--pattern", action="append", default=[], help="prefix to delete (repeatable, e.g. /dollar_bars_30k_)")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--detail", action="store_true", help="print every key")
    ap.add_argument("--top", type=int, default=20, help="show top-N categories by size")
    args = ap.parse_args()

    uid, tok, org = load_creds()
    print(f"# user_id={uid} org={org}")

    cat_count = defaultdict(int)
    cat_size = defaultdict(int)
    all_keys = []

    print(f"# walking {args.root!r}…")
    for key, size, folder in walk(uid, tok, org, args.root):
        if folder:
            continue
        cat = categorize(key)
        cat_count[cat] += 1
        cat_size[cat] += size
        all_keys.append((key, size))
        if args.detail:
            print(f"  {size:>12,d}  {key}")

    print()
    print(f"=== ObjectStore summary ===")
    print(f"  total keys: {sum(cat_count.values()):,}")
    print(f"  total size: {sum(cat_size.values())/1e6:.1f} MB")
    print()
    print(f"{'category':<40}  {'count':>8}  {'size_MB':>10}")
    for cat, sz in sorted(cat_size.items(), key=lambda x: -x[1])[: args.top]:
        print(f"{cat:<40}  {cat_count[cat]:>8,d}  {sz/1e6:>10.1f}")

    if args.delete and args.pattern:
        matches = [(k, s) for k, s in all_keys if any(k.startswith(p) for p in args.pattern)]
        total_mb = sum(s for _, s in matches) / 1e6
        print()
        print(f"# DELETE: {len(matches)} keys matching {args.pattern} ({total_mb:.1f} MB)")
        if not args.yes:
            ok = input("type 'DELETE' to confirm: ").strip()
            if ok != "DELETE":
                print("aborted.")
                return
        n_ok = n_fail = 0
        for k, _ in matches:
            success, errs = delete_key(uid, tok, org, k)
            if success:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  fail: {k}: {errs}")
        print(f"# done: {n_ok} deleted, {n_fail} failed")


if __name__ == "__main__":
    main()
