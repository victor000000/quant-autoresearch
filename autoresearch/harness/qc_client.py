"""QC Cloud API client with timeout enforcement and backtest lifecycle management.
Reuses the curl-based auth pattern from experiment_summary/tools/api_curl.py."""

import json, subprocess, time, base64, hashlib, os

def _get_creds():
    """Load QC credentials from QC_CREDS_PATH."""
    from .constants import QC_CREDS_PATH
    with open(QC_CREDS_PATH) as f:
        return json.load(f)

def _qc_post(path, body=None, max_time=120):
    """POST to QC API v2. Returns parsed JSON or {} on failure.
    Tries Python requests first, falls back to curl."""
    c = _get_creds()
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{c['token']}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{c['user_id']}:{digest}".encode()).decode()
    url = f"https://www.quantconnect.com/api/v2{path}"
    data_str = json.dumps(body or {})

    # Try Python requests first, fall back to curl on ANY error (SSL, network, etc.)
    try:
        import requests
        r = requests.post(url,
            headers={"Authorization": f"Basic {auth}", "Timestamp": ts,
                     "Content-Type": "application/json"},
            data=data_str, timeout=max_time)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass  # Fall through to curl

    # Curl fallback
    tmp = f"/tmp/qc_api_{os.getpid()}_{ts}.json"
    cmd = ["curl", "-s", "-w", "%{http_code}",
           "-X", "POST", url,
           "-H", f"Authorization: Basic {auth}",
           "-H", f"Timestamp: {ts}",
           "-H", "Content-Type: application/json",
           "-d", data_str,
           "--connect-timeout", "30", "--max-time", str(max_time),
           "-o", tmp]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 10)
        try:
            with open(tmp, 'r') as f:
                return json.load(f)
        except:
            return {}
    finally:
        try: os.unlink(tmp)
        except: pass
    return {}

def submit_backtest(code, name, extra_files=None):
    """Upload code, compile, create backtest. Returns backtestId or raises RuntimeError.

    extra_files: optional {filename: content} of ADDITIONAL project files (e.g. a
    separate bar_builder.py) that main.py imports — QC's per-file 64,000-char limit is
    sidestepped by splitting large modules into their own files. Uploaded BEFORE main.py
    so they're present at compile. Idempotent: create-then-update (create no-ops if the
    file already exists, update sets the latest content)."""
    from .constants import QC_PROJECT_ID
    pid = QC_PROJECT_ID

    # 0. Upload any extra module files first (create-then-update = exists-safe).
    for fname, fcontent in (extra_files or {}).items():
        _qc_post("/files/create", {"projectId": pid, "name": fname, "content": fcontent})
        ru = _qc_post("/files/update", {"projectId": pid, "name": fname, "content": fcontent})
        if not ru.get("success"):
            raise RuntimeError(f"Extra-file upload failed for {fname}: {ru}")

    # 1. Upload
    r = _qc_post("/files/update", {"projectId": pid, "name": "main.py", "content": code})
    if not r.get("success"):
        raise RuntimeError(f"Upload failed: {r}")

    # 2. Compile
    r = _qc_post("/compile/create", {"projectId": pid})
    if not r.get("success"):
        raise RuntimeError(f"Compile create failed: {r}")
    cid = r["compileId"]

    # 3. Wait for compile
    t0 = time.time()
    while time.time() - t0 < 120:
        r = _qc_post("/compile/read", {"projectId": pid, "compileId": cid})
        state = r.get("state", "")
        if state == "BuildSuccess":
            break
        if state == "BuildError":
            raise RuntimeError(f"BuildError: {(r.get('logs') or '')[:500]}")
        time.sleep(5)

    # 4. Create backtest
    r = _qc_post("/backtests/create",
                 {"projectId": pid, "compileId": cid, "backtestName": name})
    if not r.get("success"):
        raise RuntimeError(f"Backtest create failed: {r.get('errors', r)}")
    return r["backtest"]["backtestId"]

def read_backtest(bid):
    """Read full backtest result."""
    from .constants import QC_PROJECT_ID
    r = _qc_post("/backtests/read", {"projectId": QC_PROJECT_ID, "backtestId": bid}, max_time=120)
    return r.get("backtest", {})

def read_backtest_status(bid):
    """Lightweight status check. Returns (status, runtime_stats, logs)."""
    bt = read_backtest(bid)
    if not bt:
        return "?", {}, ""
    return bt.get("status", "?"), bt.get("runtimeStatistics") or {}, bt.get("logs") or ""

def delete_backtest(bid):
    """Cancel/delete a running or completed backtest."""
    from .constants import QC_PROJECT_ID
    r = _qc_post("/backtests/delete", {"projectId": QC_PROJECT_ID, "backtestId": bid})
    return r.get("success", False)

def is_done(status):
    """True if backtest is in a terminal state."""
    return status.startswith("Completed") or "Error" in status or status == "Canceled"

def submit_and_wait(code, name, timeout_s=None, extra_files=None):
    """Full lifecycle: submit -> poll -> read result. Auto-deletes on timeout.
    Returns (result_dict, status_string) where status is one of:
      'completed', 'timeout', 'crash'
    extra_files: optional {filename: content} of separate project modules (see submit_backtest).
    """
    from .constants import TIME_BUDGET, QC_POLL_INTERVAL
    if timeout_s is None:
        timeout_s = TIME_BUDGET

    # Submit
    try:
        bid = submit_backtest(code, name, extra_files=extra_files)
    except Exception as e:
        return {"error": str(e)}, "crash"

    # Poll
    t_start = time.time()
    while True:
        elapsed = time.time() - t_start
        status, rt, logs = read_backtest_status(bid)

        if is_done(status):
            if status.startswith("Completed"):
                bt = read_backtest(bid)
                return bt, "completed"
            else:
                return {"status": status, "logs": logs, "runtime": rt}, "crash"

        if elapsed > timeout_s:
            delete_backtest(bid)
            return {"timeout_s": elapsed, "runtime": rt}, "timeout"

        time.sleep(QC_POLL_INTERVAL)
