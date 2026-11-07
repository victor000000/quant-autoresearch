# Quant autoresearch workspace

A Python research/rendering pipeline for ETF experiments executed through QuantConnect. Package metadata is **0.1.1**. The independent qc_work executor/storage platform is a different project; its credential, queue and recovery contracts do not automatically apply here.

[Research contract](program.md) preserves scoring/selection gates and dated findings. [Source and operations guide](docs/GUIDE.md) explains current paths, rendering, configuration and account effects. [CI](CI.md) gives safe offline checks; [documentation inventory](docs/DOCUMENTATION_AUDIT.md) classifies every owned guide, analysis and archived plan.

## Install for offline development

Python 3.12 is the tested CI runtime (package minimum is 3.10). From the checkout root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . pytest
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts
```

Pyproject declares pandas, NumPy, XGBoost, scikit-learn, SciPy and requests. For the CPU-only CI dependency ranges, follow `.github/workflows/ci.yml`, which installs xgboost-cpu before `pip install --no-deps -e .`. This repo has no complete dependency lock; historical numerical replay needs its original pinned environment and data separately. A source/editable checkout is required for the root-level scripts/templates/state layout.

The current 19 unit tests use synthetic data and renderer fixtures without an account, market downloads or production training. They cover drawdown/ratio edge cases, rendering budget/smoke, real-yield features, tokens/parity and activity-threshold leakage. They do not certify real OOS performance, prospective alpha, numerical model parity in QC, or live readiness.

## Code and state ownership

| Path | Purpose |
|---|---|
| src/lb/harness/ | Scoring constants, rendering/orchestration, QC API client and cell suffix conventions |
| src/lb/modules/ | Axis, labels, features, training and sizing implementations |
| src/lb/templates/ | QC scripts rendered from .py.tmpl files |
| src/lb/metrics.py | Drawdown shape and Martin ratio helpers |
| src/lb/paths.py | Checkout-root resolution and LB_ROOT override |
| scripts/ | Round/sweep drivers, research/audit/diagnostic utilities |
| program.md | Historical research contract and immutable scoring boundaries |
| knowledge.json, hypotheses.json | Research state/configuration, not proof of validated current results |
| docs/analysis, docs/research, docs/legacy | Dated analyses, specifications and archived sessions |

Private data, complete historical outputs/reports and third-party reference material can be absent. Source-visible score tables and archived “champion” claims are research records, not independently revalidated deployment recommendations. The scorer remains locked by the research contract; this documentation revision changes no scoring/selection rules.

## Account effects and limitations

`lb-round` directly invokes the round driver; it is not a harmless help-only CLI. Do not use `lb-round --help`, import arbitrary drivers, or run round scripts as smoke tests. Drivers can upload source, start/delete cloud backtests, overwrite research state and interact with ObjectStore. Read the source/operations guide before a separately scoped account run. No new job or provider acquisition is implied by installation/testing.

Version 0.1.1 fixes flat/insolvent Martin reporting and finite-input/annualization validation. The source's legacy `lb.__version__` string still says 0.1.0; package metadata is the distribution authority, and that discrepancy is recorded rather than disguised. Legacy scripts may embed workstation paths or project-specific assumptions. There is no CD pipeline or certified live-trading deployment.
