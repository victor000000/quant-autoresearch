"""Cell-key collision guard (the 3-time bug class: inline _PSUF copies, the vacuous
permute gate, and the _fx/_va shared-token collisions fixed 2026-06-12).

Every distinct config knob combination MUST map to a distinct cell suffix —
otherwise two different configs silently share an ObjectStore cell and gates
(especially the permute control) read the wrong predictions.
"""
import itertools

from lb.harness.psuf import cell_suffix


def test_every_knob_combination_unique():
    reduces = ["correlation", "infogain", "pca", "vae", "vae_rl", "pls", "spca",
               "minor_pca", "whiten", "spearman"]
    features = ["base", "rich", "termstruct", "realyield", "vix", "calendar",
                "ddstate", "regime", "oilbasis"]
    models = ["xgb", "lgbm", "catboost", "lgbm_bag", "xgb_bag"]
    cals = ["isotonic", "venn_abers", "beta"]
    seen = {}
    for r, f, m, c, perm in itertools.product(reduces, features, models, cals,
                                              (False, True)):
        cfg = {"reduce": r, "features": f, "model": m, "calibration": c,
               "permute_labels": perm}
        suf = cell_suffix(cfg)
        key = (r, f, m, c, perm)
        assert suf not in seen, (
            f"cell-suffix COLLISION: {key} and {seen[suf]} both -> {suf!r}")
        seen[suf] = key


def test_permute_always_distinct():
    for r in ("correlation", "vae", "pls"):
        a = cell_suffix({"reduce": r, "permute_labels": False})
        b = cell_suffix({"reduce": r, "permute_labels": True})
        assert a != b


def test_defaults_empty_suffix():
    # A fully-default config must keep the historical bare cell key.
    assert cell_suffix({}) == ""
