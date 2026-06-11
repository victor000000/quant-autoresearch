from datetime import datetime

# === ETF Universe ===
CORE_7_ETFS = ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD"]

# === Time Splits ===
TRAIN_END = datetime(2021, 8, 1)
VAL_END = datetime(2023, 8, 1)
TEST_END = datetime(2026, 6, 11)   # advanced 2026-06-11 (was 06-01; ~8 trading days of new OOS)

# === QC Cloud ===
QC_PROJECT_ID = 31338454
QC_CREDS_PATH = "/home/ubuntu/lb/qc/.creds.json"
QC_POLL_INTERVAL = 30  # seconds between status checks
TIME_BUDGET = 300       # 5 minutes max per backtest

# === Gate Thresholds ===
GATE_CALMAR_MIN = 3.0       # OOS Calmar must exceed this
GATE_TRADES_MIN = 80        # OOS trade count must exceed this
GATE_AUC_DIVERGENCE_MAX = 0.05  # |train_AUC - val_AUC| must be below this

# === Rendering ===
TEMPLATES_DIR = "/home/ubuntu/lb/templates"
MODULES_DIR = "/home/ubuntu/lb/modules"
# lean_workspace is archived on this machine; use the standalone scripts dir instead.
QC_SCRIPTS_DIR = "/home/ubuntu/lb/_autoresearch_scripts"

# === Asset Fingerprinting (for ETF selection) ===
ASSET_AFFINITY = {
    "trend_following": ["QQQ", "IWM", "XLE", "GLD"],
    "mean_reversion": ["HYG", "TLT", "EEM"],
    "volatility_regime": ["GLD", "XLE", "EEM"],
}

# Target bars per 17-year history (2009-2026) at minute granularity
TARGET_BARS = 15000
