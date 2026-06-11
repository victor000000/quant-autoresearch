"""I3/I4 guard: render_train_config must produce compilable QC files under 64k."""
import ast

from lb.harness import orchestrator

SAMPLE_CFG = {
    "ticker": "GLD", "axis": "vol", "labeler": "tertile", "sizing": "binary",
    "thresh": 0.45, "max_depth": 3, "permute_labels": False, "n_components": 20,
    "rebal_band": 0.01, "horizons": None, "reduce": "correlation",
    "features": "base", "calibration": "isotonic", "train_purge": False,
}

def _files(cfg):
    """Return {filename: source} for the rendered main + extra files.

    render_train_config returns (main_code, extra_dict) where extra_dict contains
    {"bar_builder.py": ..., "bar_ext.py": ..., "ml_ext.py": ..., "sizing_ext.py": ...}.
    """
    main, extra = orchestrator.render_train_config(cfg)
    out = {"main.py": main}
    out.update(extra or {})
    return out

def test_render_compiles_and_under_64k():
    files = _files(dict(SAMPLE_CFG))
    assert "main.py" in files
    for name, src in files.items():
        ast.parse(src, filename=name)       # syntactically valid
        nbytes = len(src.encode("utf-8"))
        assert nbytes < 64000, f"{name} is {nbytes} bytes (>= 64000 QC limit)"
