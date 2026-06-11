"""Compat shim — the systemd unit `autoresearch-reports` runs `python3 scripts/app.py`.
Real server lives in lb.console.app. Keeps the service working without a unit edit."""
from lb.console.app import main
if __name__ == "__main__":
    main()
