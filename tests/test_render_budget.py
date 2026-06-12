"""QC 64,000-byte render-budget guard.

main.py overflows have burned three round-trips (nullimp 64055, the first two
lgbm renders at 65256/64399). Any template/module edit that pushes the CHAMPION
configs over budget must fail here, locally, before a QC submission does.
"""
import pytest

from lb.harness.orchestrator import render_train_config

LIMIT = 64000

CHAMPION_CFGS = [
    # GLD crown — the largest render in the project (ensemble labeler + vae + bag)
    {"ticker": "GLD", "axis": "logdollar", "labeler": "trend_leg+regime_gmm",
     "thresh": 0.40, "sizing": "dd_overlay", "n_components": 16, "reduce": "vae",
     "model": "lgbm_bag", "max_depth": 3, "permute_labels": False,
     "rebal_band": 0.01, "features": "base"},
    # USO crown
    {"ticker": "USO", "axis": "logdollar", "labeler": "revert", "thresh": 0.45,
     "sizing": "cdf_plain", "n_components": 20, "reduce": "correlation",
     "max_depth": 3, "permute_labels": False, "rebal_band": 0.01,
     "features": "base"},
    # DIA candidate (imbalance ensemble)
    {"ticker": "DIA", "axis": "imbalance", "labeler": "bgm+ker", "thresh": 0.50,
     "sizing": "cdf_overlay", "max_depth": 3, "permute_labels": False,
     "rebal_band": 0.01, "features": "base"},
]


@pytest.mark.parametrize("cfg", CHAMPION_CFGS, ids=lambda c: c["ticker"])
def test_champion_renders_fit_budget(cfg):
    out = render_train_config(cfg)
    files = out if isinstance(out, dict) else {"main.py": out[0], **(out[1] or {})}
    for name, code in files.items():
        if not isinstance(code, str):
            continue
        size = len(code.encode("utf-8"))
        assert size < LIMIT, f"{cfg['ticker']} {name}: {size} >= {LIMIT}"
