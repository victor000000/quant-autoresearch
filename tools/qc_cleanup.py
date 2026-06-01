#!/usr/bin/env python3
"""
QC ObjectStore targeted cleanup.

Deletes well-defined deprecated path prefixes. Walks each prefix recursively,
deletes every leaf key, reports counts. Idempotent.

Usage:
  qc_cleanup.py --dry-run         # show what would be deleted
  qc_cleanup.py --execute         # actually delete
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/txy/lb/tools")
from qc_objstore import load_creds, list_path, delete_key  # noqa: E402

# Confirmed deprecated — all from pre-ZN-axis exploration phases.
# These were superseded by the v1 Wang pipeline (bars_zn_v1, features_zn_v1, etc).
DEPRECATED_HIGH_CONFIDENCE = [
    # 30K dollar-bar PA variants (Round 10-30 explorations, replaced by ZN axis)
    "/dollar_bars_30k_norm",
    "/dollar_bars_30k_figarch_norm",
    "/dollar_bars_30k_var_norm",
    "/dollar_bars_30k_figarch_z",
    "/dollar_bars_30k_iid_gauss",
    "/dollar_bars_30k_rank_only",
    "/dollar_bars_30k_mle_iid",
    "/dollar_bars_30k_acf_min",
    "/dollar_bars_30k_gacf_min",
    "/dollar_bars_30k_iid_compare",
    "/dollar_bars_30k_simple_iid",
    # Early dollar-bar tests (single-asset, superseded)
    "/dollar_bars_norm",
    "/dollar_bars_d07",
    "/dollar_bars_slx",
    "/dollar_bars_vxx",
    # 30K HMM/VMD models (superseded by models_zn_v1)
    "/models_30k_iid_hmm",
    "/models_30k_iid_oos",
    "/models_v3_30k_hmm",
    "/models_vmd_30k_hmm",
    "/models_vmd_30k_oos",
    "/models_vmd_9k_hmm",
    "/models_vmd_9k_oos",
    "/features_30k",
    "/labels_30k_test",
    # Finished sweeps (results extracted into stage docs)
    "/models_tau_sweep",
    "/models_512_freq",
    "/models_axis_sweep",
    "/models_multibar_sweep",
    "/models_crossaxis_stack",
    "/models_regime_features",
    "/models_slx_bar_sweep",
    "/models_vxx_bar_sweep",
    "/models_screen_ls",
    "/models_l2_meta",
    "/models_ensemble",
    # Pre-Wang baseline placeholders
    "/baselines",
    # Old S001 strategy artifacts (separate project)
    "/s001",
    "/s001_model_rf_v1.json",
    "/s001_model_tree_v1.json",
    "/s001_model_tree_v2.json",
    "/s001_model_tree_v1.bin",
]


def list_recursive(uid, tok, org, root):
    """List all leaf keys under a prefix (no folders). Bare files are returned as-is."""
    out = []
    stack = [root]
    while stack:
        p = stack.pop()
        try:
            items = list_path(uid, tok, org, p)
        except Exception:
            # Likely a bare-file key — treat as leaf.
            out.append((p, 0))
            continue
        if items is None:
            out.append((p, 0))
            continue
        for it in items:
            key = it.get("key", "")
            size = int(it.get("size", 0) or 0)
            if it.get("folder") or key.endswith("/"):
                stack.append(key)
            else:
                out.append((key, size))
    return out


def delete_one(args):
    uid, tok, org, key = args
    try:
        ok, errs = delete_key(uid, tok, org, key)
        return key, ok, errs
    except Exception as e:
        return key, False, [str(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.dry_run or args.execute):
        ap.error("specify --dry-run or --execute")

    uid, tok, org = load_creds()
    total_keys = 0
    total_bytes = 0
    all_targets = []
    print(f"# enumerating {len(DEPRECATED_HIGH_CONFIDENCE)} deprecated prefixes…")
    for prefix in DEPRECATED_HIGH_CONFIDENCE:
        keys = list_recursive(uid, tok, org, prefix)
        sz = sum(s for _, s in keys)
        print(f"  {prefix:<42}  {len(keys):>6,d} keys   {sz/1e6:>8.2f} MB")
        total_keys += len(keys)
        total_bytes += sz
        all_targets.extend(keys)

    print()
    print(f"=== TOTAL: {total_keys:,} keys, {total_bytes/1e6:.1f} MB ===")

    if args.dry_run:
        print("# dry-run, no deletions performed")
        return

    print(f"# deleting {len(all_targets)} keys with {args.workers} workers…")
    n_ok = n_fail = 0
    payload = [(uid, tok, org, k) for k, _ in all_targets]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (k, ok, errs) in enumerate(ex.map(delete_one, payload), 1):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  fail: {k}: {errs}")
            if i % 100 == 0:
                print(f"  …{i}/{len(payload)} ok={n_ok} fail={n_fail}")
    print(f"# done: {n_ok} deleted, {n_fail} failed")


if __name__ == "__main__":
    main()
