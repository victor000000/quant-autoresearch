from datetime import datetime

# === ETF Universe ===
CORE_7_ETFS = ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD"]

# === Time Splits ===
TRAIN_END = datetime(2021, 8, 1)
VAL_END = datetime(2023, 8, 1)
TEST_END = datetime(2026, 6, 1)

# === QC Cloud ===
QC_PROJECT_ID = 31338454
QC_CREDS_PATH = "/Users/liyuanjun/ai_work/lb/qc/.creds.json"
QC_POLL_INTERVAL = 30  # seconds between status checks
TIME_BUDGET = 300       # 5 minutes max per backtest

# === Gate Thresholds ===
GATE_CALMAR_MIN = 3.0       # OOS Calmar must exceed this
GATE_TRADES_MIN = 80        # OOS trade count must exceed this
GATE_AUC_DIVERGENCE_MAX = 0.05  # |train_AUC - val_AUC| must be below this

# === Rendering ===
TEMPLATES_DIR = "/Users/liyuanjun/ai_work/lb/autoresearch/templates"
MODULES_DIR = "/Users/liyuanjun/ai_work/lb/autoresearch/modules"
QC_SCRIPTS_DIR = "/Users/liyuanjun/ai_work/lb/lean_workspace/_autoresearch"

# === Asset Fingerprinting (for ETF selection) ===
ASSET_AFFINITY = {
    "trend_following": ["QQQ", "IWM", "XLE", "GLD"],
    "mean_reversion": ["HYG", "TLT", "EEM"],
    "volatility_regime": ["GLD", "XLE", "EEM"],
}

# Target bars per 17-year history (2009-2026) at minute granularity
TARGET_BARS = 15000
