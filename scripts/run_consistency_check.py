"""XLE v4 with stop-loss 5% — 3-phase consistency."""
import json, time
from qc.api import submit_backtest

INFER_T = open("lean_workspace/_pipeline_v4_infer/main.py").read()
inflight = []

JOBS = [
    (31338456, "XLE", "2014-08-01", "2021-08-01", "XLE-v4d-SL5-TRAIN"),
    (31338460, "XLE", "2023-08-01", "2026-05-21", "XLE-v4d-SL5-TEST"),
]
for pid, ticker, ps, pe, lbl in JOBS:
    code = (INFER_T.replace("__TICKER__", ticker).replace("__PHASE_START__", ps)
            .replace("__PHASE_END__", pe).replace("__TRADE_THRESH__", "0.5")
            .replace("__USE_ISO__", "0").replace("__INVERSE__", "0")
            .replace("__RECIPE__", "v4_dol5000_TS40_T25").replace("__CONTINUOUS__", "0")
            .replace("__STOP_LOSS_PCT__", "0.05"))
    bid = submit_backtest(pid, code, f"{lbl}-{int(time.time())%10000}")
    print(f"{lbl} -> {bid}")
    inflight.append({"pid": pid, "bid": bid, "label": lbl})

json.dump(inflight, open("qc/inflight.json", "w"))
