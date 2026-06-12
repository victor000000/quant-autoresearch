from datetime import datetime

from lb.paths import (
    TEMPLATES_DIR, MODULES_DIR, QC_SCRIPTS_DIR, QC_CREDS_PATH,
    KNOWLEDGE_JSON, HYPOTHESES_JSON,
    RESULTS_DIR, ROUND_RESULTS_CSV, STATUS_JSON,
)

# === ETF Universe ===
CORE_7_ETFS = ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD"]

# === Time Splits ===
TRAIN_END = datetime(2021, 8, 1)
VAL_END = datetime(2023, 8, 1)
TEST_END = datetime(2026, 7, 1)    # advanced 2026-06-12 (user: OOS real-trading backtest end -> 2026-07-01)

# === QC Cloud ===
QC_PROJECT_ID = 31338454
QC_POLL_INTERVAL = 30  # seconds between status checks
TIME_BUDGET = 300       # 5 minutes max per backtest

# === Gate Thresholds ===
GATE_CALMAR_MIN = 3.0       # OOS Calmar must exceed this
GATE_TRADES_MIN = 80        # OOS trade count must exceed this
GATE_AUC_DIVERGENCE_MAX = 0.05  # |train_AUC - val_AUC| must be below this

# === Rendering ===
# TEMPLATES_DIR / MODULES_DIR / QC_SCRIPTS_DIR / QC_CREDS_PATH now come from
# lb.paths (single source of truth); imported above as pathlib.Path objects.
# lean_workspace is archived on this machine; use the standalone scripts dir instead.

# === Asset Fingerprinting (for ETF selection) ===
ASSET_AFFINITY = {
    "trend_following": ["QQQ", "IWM", "XLE", "GLD"],
    "mean_reversion": ["HYG", "TLT", "EEM"],
    "volatility_regime": ["GLD", "XLE", "EEM"],
}

# Target bars per 17-year history (2009-2026) at minute granularity
TARGET_BARS = 15000
