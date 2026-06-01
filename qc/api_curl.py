"""QC API via curl (resolves Python SSL issue with old LibreSSL)."""
import json, subprocess, time, base64, hashlib, os
from pathlib import Path

CREDS_PATH = Path("/Users/liyuanjun/ai_work/lb/qc/.creds.json")

def get_creds():
    with CREDS_PATH.open() as f:
        c = json.load(f)
    return c

def qc_post(path, body=None, max_time=60):
    c = get_creds()
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{c['token']}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{c['user_id']}:{digest}".encode()).decode()
    url = f"https://www.quantconnect.com/api/v2{path}"
    data_str = json.dumps(body or {})
    tmp = f"/tmp/qc_api_{os.getpid()}_{ts}.json"
    cmd = ["curl", "-s", "-w", "%{http_code}:%{size_download}",
           "-X", "POST", url,
           "-H", f"Authorization: Basic {auth}",
           "-H", f"Timestamp: {ts}",
           "-H", "Content-Type: application/json",
           "-d", data_str,
           "--connect-timeout", "30", "--max-time", str(max_time),
           "-o", tmp]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time+10)
    # Parse curl's -w output: HTTP_CODE:SIZE_DOWNLOAD
    try:
        parts = result.stdout.strip().split(":")
        http_code = int(parts[0])
        size = int(parts[1]) if len(parts) > 1 else 0
    except:
        http_code = 0; size = 0

    fsize = 0
    try:
        fsize = os.path.getsize(tmp)
        if fsize > 0:
            with open(tmp, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        if fsize > 100:
            with open(tmp, 'r') as f:
                partial = f.read(200)
            print(f"[qc_post] JSON parse error: http={http_code} downloaded={size} filesize={fsize} partial={partial}")
        return {}
    finally:
        try: os.unlink(tmp)
        except: pass

def submit(pid, code, name):
    """Upload, compile, create backtest. Returns backtestId."""
    r = qc_post("/files/update", {"projectId": pid, "name": "main.py", "content": code})
    if not r.get("success"):
        raise RuntimeError(f"update failed: {r}")
    r = qc_post("/compile/create", {"projectId": pid})
    if not r.get("success"):
        raise RuntimeError(f"compile/create: {r}")
    cid = r["compileId"]
    t0 = time.time()
    while time.time() - t0 < 120:
        r = qc_post("/compile/read", {"projectId": pid, "compileId": cid})
        state = r.get("state", "")
        if state == "BuildSuccess": break
        if state == "BuildError":
            raise RuntimeError(f"BuildError: {(r.get('logs') or '')[:500]}")
        time.sleep(5)
    r = qc_post("/backtests/create",
                {"projectId": pid, "compileId": cid, "backtestName": name})
    if not r.get("success"):
        raise RuntimeError(f"bt_create: {r.get('errors', r)}")
    return r["backtest"]["backtestId"]

def read_bt(pid, bid):
    r = qc_post("/backtests/read", {"projectId": pid, "backtestId": bid}, max_time=120)
    return r.get("backtest", {})

def read_bt_status(pid, bid):
    """Lightweight status-only read. Returns (status, runtime_stats_dict, error_logs)."""
    bt = read_bt(pid, bid)
    if not bt:
        return "?", {}, ""
    status = bt.get("status", "?")
    rt = bt.get("runtimeStatistics") or {}
    logs = bt.get("logs") or ""
    return status, rt, logs

def is_done(bt):
    s = bt.get("status", "")
    return s.startswith("Completed") or "Error" in s or s == "Canceled"
