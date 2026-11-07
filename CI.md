# Offline validation and CI

GitHub Actions uses Python 3.12, CPU dependency ranges from `.github/workflows/ci.yml`, editable package installation and the complete current 19-test suite, followed by compileall for src/scripts. Initial installation needs registry access. Pyproject minimum Python is 3.10; CI does not currently test every supported interpreter. There is no complete dependency lock.

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts
```

Tests do not upload models, contact QC, download market data, start real account jobs or invoke arbitrary round/research drivers. Rendering/compile checks use selected source fixtures; this is not batch/online numerical or QC runtime parity. `lb-round --help` is not a safe smoke path because the entrypoint directly runs the round script.

Version 0.1.1 has 19 tests, superseding the prior 17-test description. [Current guide](docs/GUIDE.md) documents finite/flat/insolvent metric semantics and source/version discrepancies. There is no configured deployment workflow; historical deployment analyses are not CD. The documentation audit checks public navigation and records retained historical link/data gaps separately.
