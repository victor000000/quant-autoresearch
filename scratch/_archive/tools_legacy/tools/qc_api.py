"""Raw QC Cloud API harness — submit/poll backtests directly, no local queue."""
import json
import sys
import time
sys.path.insert(0, '/home/txy/lb/tools')
from qc_get_logs import get_creds, qc_post


def update_main(project_id, code, creds):
    """Push code to project's main.py."""
    return qc_post("/files/update",
                   {"projectId": project_id, "name": "main.py", "content": code}, creds)


def compile_project(project_id, creds, max_wait=60):
    """Trigger compile and wait for BuildSuccess. Returns compile_id."""
    r = qc_post("/compile/create", {"projectId": project_id}, creds)
    if not r.get("success"):
        raise RuntimeError(f"compile/create failed: {r}")
    compile_id = r["compileId"]
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = qc_post("/compile/read", {"projectId": project_id, "compileId": compile_id}, creds)
        state = r.get("state")
        if state == "BuildSuccess":
            return compile_id
        if state == "BuildError":
            logs = r.get("logs", "")
            raise RuntimeError(f"compile error: {logs[:500]}")
        time.sleep(2)
    raise RuntimeError(f"compile timed out after {max_wait}s")


def submit_backtest(project_id, code, name, creds):
    """Push code → compile → create backtest. Returns backtestId."""
    update_main(project_id, code, creds)
    compile_id = compile_project(project_id, creds)
    r = qc_post("/backtests/create",
                {"projectId": project_id, "compileId": compile_id, "name": name}, creds)
    if not r.get("success"):
        raise RuntimeError(f"backtests/create failed: {r}")
    return r["backtest"]["backtestId"]


def read_backtest(project_id, backtest_id, creds):
    """Read backtest state + stats."""
    r = qc_post("/backtests/read",
                {"projectId": project_id, "backtestId": backtest_id}, creds)
    return r["backtest"]


def is_complete(bt):
    """True if backtest reached terminal state."""
    return bt.get("status") in ("Completed.", "Completed", "Runtime Error", "Build Error", "Canceled")


def poll_until_done(jobs, creds, interval=15, max_wait=1800):
    """jobs = list of (project_id, backtest_id, label). Returns dict label → backtest result."""
    t0 = time.time()
    results = {}
    pending = list(jobs)
    while pending and time.time() - t0 < max_wait:
        time.sleep(interval)
        still = []
        for pid, bid, label in pending:
            bt = read_backtest(pid, bid, creds)
            status = bt.get("status", "?")
            progress = bt.get("progress", 0)
            print(f"[{int(time.time()-t0)}s] {label}: {status} progress={progress:.2f}")
            if is_complete(bt):
                results[label] = bt
            else:
                still.append((pid, bid, label))
        pending = still
    for pid, bid, label in pending:
        bt = read_backtest(pid, bid, creds)
        results[label] = bt
        print(f"[TIMEOUT] {label}: still {bt.get('status')}")
    return results


def stat(bt, key, default=""):
    """Lookup statistic or runtime statistic by key."""
    return (bt.get("statistics") or {}).get(key) or (bt.get("runtimeStatistics") or {}).get(key) or default


if __name__ == "__main__":
    # Quick test: list authenticated projects
    creds = get_creds()
    r = qc_post("/projects/read", {}, creds)
    if r.get("success"):
        print(f"Authenticated. {len(r.get('projects', []))} projects visible.")
        for p in r.get("projects", [])[:10]:
            print(f"  {p.get('projectId')}: {p.get('name')}")
    else:
        print(f"Failed: {r}")
