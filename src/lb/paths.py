"""Single source of truth for filesystem paths. Derives the repo root from this
file's location; override with LB_ROOT for unusual deployments."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LB_ROOT", Path(__file__).resolve().parents[2])).resolve()

PKG = ROOT / "src" / "lb"
TEMPLATES_DIR = PKG / "templates"
MODULES_DIR = PKG / "modules"
QC_SCRIPTS_DIR = ROOT / "_autoresearch_scripts"
QC_CREDS_PATH = ROOT / "qc" / ".creds.json"

KNOWLEDGE_JSON = ROOT / "knowledge.json"
TECHNIQUES_JSON = ROOT / "techniques.json"
HYPOTHESES_JSON = ROOT / "hypotheses.json"
RESULTS_TSV = ROOT / "results.tsv"
RESULTS_DIR = ROOT / "results"
ROUND_RESULTS_CSV = RESULTS_DIR / "round_results.csv"
REPORTS_DIR = ROOT / "reports"
STATUS_JSON = REPORTS_DIR / "status.json"
