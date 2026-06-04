"""ObjectStore cleanup — uses qc/api.py auth. Lists then deletes stale prefixes."""
import sys, json
sys.path.insert(0, "/home/txy/lb")
from qc.api import qc_post

ORG = "c14b34052fc7a34207917af5db51e6f5"

# Keep paths matching CURRENT recipes (v4, v6, v16, v18). Delete everything else under models/.
KEEP_PREFIXES = (
    "v4_", "v6_", "v14_", "v16_", "v17_", "v18_",
    "v3",  # potential v3 we still need
)
# Hardcoded list of stale prefixes from prior memory
DEPRECATED = [
    "/dollar_bars_30k_norm", "/dollar_bars_30k_figarch_norm", "/dollar_bars_30k_var_norm",
    "/dollar_bars_30k_figarch_z", "/dollar_bars_30k_iid_gauss", "/dollar_bars_30k_rank_only",
    "/dollar_bars_30k_mle_iid", "/dollar_bars_30k_acf_min", "/dollar_bars_30k_gacf_min",
    "/dollar_bars_30k_iid_compare", "/dollar_bars_30k_simple_iid",
    "/dollar_bars_norm", "/dollar_bars_d07", "/dollar_bars_slx", "/dollar_bars_vxx",
    "/models_30k_iid_hmm", "/models_30k_iid_cpd", "/models_30k_iid_joint",
    "/labels_30k_iid", "/features_30k_iid", "/vae_30k_iid", "/pca_30k_iid",
]


def list_keys(prefix):
    """List ObjectStore keys under prefix. Returns list of key strings."""
    r = qc_post("/object/list", {"organizationId": ORG, "path": prefix})
    if not r.get("success"):
        return []
    return [o["key"] for o in r.get("objects", [])]


def delete_key(key):
    r = qc_post("/object/delete", {"organizationId": ORG, "key": key})
    return r.get("success", False)


def walk_and_delete(prefix, dry=True):
    keys = list_keys(prefix)
    if not keys:
        return 0
    n = 0
    for k in keys:
        if k.endswith("/"):
            n += walk_and_delete(k.rstrip("/"), dry=dry)
        else:
            if dry:
                print(f"  would delete: {k}")
            else:
                if delete_key(k):
                    n += 1
    return n


if __name__ == "__main__":
    dry = "--execute" not in sys.argv

    print("=== Root-level paths ===")
    root_keys = list_keys("/")
    for k in sorted(root_keys):
        print(f"  {k}")

    print(f"\n=== {'DRY RUN' if dry else 'EXECUTING'} cleanup ===")
    total = 0
    for prefix in DEPRECATED:
        n = walk_and_delete(prefix, dry=dry)
        if n > 0:
            print(f"{prefix}: {n} keys")
        total += n
    print(f"\nTotal {'would delete' if dry else 'deleted'}: {total}")
