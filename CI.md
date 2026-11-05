# Validation

GitHub Actions uses Python 3.12 with CPU-only dependencies and runs the 17 self-contained unit tests and Python compilation checks. It does not upload models, call QuantConnect, download market datasets, or run training/backtests against an account. Package installation requires registry access.

Local check: `PYTHONPATH=src python -m pytest -q` after installing the workflow's dependency ranges. Templates render and compile against mocked/runtime-free tests; this is not QC cloud parity or live-model validation.

No deployment workflow or target is configured; continuous deployment is not claimed.
