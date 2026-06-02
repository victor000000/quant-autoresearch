"""Submit q2_val_winner for TLT (last of 7)."""
import json, time
from qc.api import submit_backtest

TEMPLATE = open("lean_workspace/_q2_val_winner/main.py").read()
JOBS = [(31338461, "TLT")]

inflight = []
for pid, ticker in JOBS:
    code = TEMPLATE.replace("__TICKER__", ticker)
    bid = submit_backtest(pid, code, f"q2val-{ticker}-{int(time.time())%10000}")
    print(f"submitted {ticker} -> pid={pid} bid={bid}")
    inflight.append({"pid": pid, "bid": bid, "label": ticker})

json.dump(inflight, open("qc/inflight.json", "w"))
