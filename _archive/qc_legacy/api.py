"""Raw QC Cloud API client — no qc_work dependency.

Reads credentials from /home/txy/lb/qc/.creds.json.
Provides: submit_backtest, read_backtest, poll_until_done, stat.
"""
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path

_CREDS_PATH = Path("/home/txy/lb/qc/.creds.json")


def get_creds():
    with _CREDS_PATH.open() as f:
        c = json.load(f)
    return {"user_id": c["user_id"], "token": c["token"]}


def qc_post(path, body, creds=None):
    if creds is None: creds = get_creds()
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{creds['token']}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{creds['user_id']}:{digest}".encode()).decode()
    url = f"https://www.quantconnect.com/api/v2{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Basic {auth}",
        "Timestamp": ts,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def update_main(project_id, code, creds=None):
    return qc_post("/files/update",
                   {"projectId": project_id, "name": "main.py", "content": code}, creds)


def compile_project(project_id, creds=None, max_wait=90):
    r = qc_post("/compile/create", {"projectId": project_id}, creds)
    if not r.get("success"):
        raise RuntimeError(f"compile/create failed: {r}")
    cid = r["compileId"]
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = qc_post("/compile/read", {"projectId": project_id, "compileId": cid}, creds)
        state = r.get("state")
        if state == "BuildSuccess": return cid
        if state == "BuildError":
            raise RuntimeError(f"BuildError: {(r.get('logs') or '')[:600]}")
        time.sleep(3)
    raise RuntimeError(f"compile timed out after {max_wait}s")


def submit_backtest(project_id, code, name, creds=None):
    """Push code → compile → create backtest. Returns backtestId."""
    update_main(project_id, code, creds)
    cid = compile_project(project_id, creds)
    r = qc_post("/backtests/create",
                {"projectId": project_id, "compileId": cid, "backtestName": name}, creds)
    if not r.get("success"):
        raise RuntimeError(f"backtests/create failed: {r}")
    return r["backtest"]["backtestId"]


def read_backtest(project_id, backtest_id, creds=None):
    r = qc_post("/backtests/read",
                {"projectId": project_id, "backtestId": backtest_id}, creds)
    return r["backtest"]


def is_complete(bt):
    s = bt.get("status", "")
    return s.startswith("Completed") or "Error" in s or s == "Canceled"


def stat(bt, key, default=""):
    return (bt.get("statistics") or {}).get(key) or (bt.get("runtimeStatistics") or {}).get(key) or default


def poll_until_done(jobs, creds=None, interval=15, max_wait=1800):
    """jobs = list of (project_id, backtest_id, label).
    Polls each until terminal. Returns dict label -> backtest dict.
    """
    t0 = time.time()
    results = {}
    pending = [(p, b, l) for p, b, l in jobs]
    while pending and time.time() - t0 < max_wait:
        time.sleep(interval)
        still = []
        for pid, bid, label in pending:
            bt = read_backtest(pid, bid, creds)
            if is_complete(bt):
                results[label] = bt
                cagr = stat(bt, "Compounding Annual Return", "0%").replace("%","").strip()
                mdd = stat(bt, "Drawdown", "0%").replace("%","").strip()
                try:
                    cal = float(cagr)/float(mdd) if float(mdd) > 0.01 else 0.0
                except: cal = 0.0
                print(f"[{int(time.time()-t0)}s] DONE {label}: status={bt.get('status')} Cal={cal:.2f} orders={stat(bt, 'Total Orders', '-')}")
            else:
                still.append((pid, bid, label))
                print(f"[{int(time.time()-t0)}s] WAIT {label}: status={bt.get('status')} progress={bt.get('progress', 0):.2f}")
        pending = still
    for pid, bid, label in pending:
        bt = read_backtest(pid, bid, creds)
        results[label] = bt
        print(f"[TIMEOUT] {label}: {bt.get('status')}")
    return results


if __name__ == "__main__":
    creds = get_creds()
    r = qc_post("/projects/read", {}, creds)
    print(f"Authenticated as user {creds['user_id']}. {len(r.get('projects', []))} projects visible.")
    pools = [p for p in r.get("projects", []) if "bt_pool" in (p.get("name") or "")]
    print(f"bt_pool projects: {len(pools)}")
    for p in sorted(pools, key=lambda p: p.get("name", "")):
        print(f"  pid={p['projectId']} name={p['name']}")
