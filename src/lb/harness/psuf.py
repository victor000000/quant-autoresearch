"""Canonical ObjectStore cell-key SUFFIX builder (invariant I5).

The ObjectStore cell key has a suffix that is computed in TWO places that MUST be
byte-identical, or inference reads a non-existent cell -> 0 trades -> Calmar 0.0:

  - QC side: templates/header.py.tmpl `_PSUF` (executed on QuantConnect).
  - Local side: scripts/run_round.py `_cell_key` (the local mirror).

Historically these inline copies DIVERGED twice (2026-06-08 `_fx{set}`, 2026-06-10
the `_n15_b3_ig` permute-control no-op). This module is now the SINGLE definition:
the driver imports `cell_suffix`, and the orchestrator INJECTS this exact source into
the rendered QC header (replacing the inline `_PSUF` expression). One source -> they
can never drift again.

stdlib-only on purpose: `inspect.getsource(cell_suffix)` is injected verbatim into the
QC main.py, so the body must use nothing QuantConnect can't run.

`.get(...)` (rather than `cfg["..."]`) is used for key access so the driver — whose
config dicts do not always carry every lever key (e.g. `train_purge`, `calibration`) —
behaves exactly as the previous `_cell_key` did (defaults, no KeyError). For a COMPLETE
CONFIG (always the case on the QC side) the output is byte-identical to the old inline
`_PSUF` expression.
"""


def cell_suffix(cfg) -> str:
    """Canonical ObjectStore cell-key suffix. stdlib-only so it can run on QC.
    MUST stay the single definition (header injection + driver both use this)."""
    return (
        ("_perm" if cfg.get("permute_labels") else "")
        + ("" if cfg.get("n_components", 20) == 20 else "_n" + str(cfg.get("n_components", 20)))
        + ("" if cfg.get("rebal_band", 0.01) == 0.01 else "_b" + str(int(round(cfg.get("rebal_band", 0.01) * 100))))
        + ("" if cfg.get("horizons") is None else "_hz" + "x".join(str(_h) for _h in cfg["horizons"]))
        + ("" if cfg.get("reduce", "correlation") == "correlation" else "_ig" if cfg.get("reduce") == "infogain" else "_rd" + str(cfg.get("reduce")))
        + ("" if cfg.get("features", "base") == "base" else "_fr" if cfg.get("features") == "rich" else "_ts" if cfg.get("features") == "termstruct" else "_ry" if cfg.get("features") == "realyield" else "_fx")
        + ("" if cfg.get("calibration", "isotonic") == "isotonic" else "_va")
        + ("_tp" if cfg.get("train_purge") else "")
    )
