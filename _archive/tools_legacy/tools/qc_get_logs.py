"""Fetch backtest logs from QC API."""
import base64
import hashlib
import json
import sqlite3
import sys
import time
import urllib.request


def get_creds():
    con = sqlite3.connect("/home/txy/qc_work/data/S001.db")
    cur = con.cursor()
    cur.execute("SELECT key, value FROM settings WHERE category IN ('credentials','project')")
    return dict(cur.fetchall())


def qc_post(path, body, creds):
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{creds['qc_api_token']}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{creds['qc_user_id']}:{digest}".encode()).decode()
    url = f"https://www.quantconnect.com/api/v2{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Basic {auth}",
        "Timestamp": ts,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    creds = get_creds()
    name = sys.argv[1]
    con = sqlite3.connect("/home/txy/qc_work/data/S003.db")
    cur = con.cursor()
    cur.execute("SELECT backtest_id, project_id FROM backtest_queue WHERE name LIKE ?", (f"%{name}%",))
    row = cur.fetchone()
    if not row:
        print(f"not found: {name}"); return
    bid, pid = row
    # paged pull
    all_lines = []
    for start in range(0, 5000, 200):
        body = {"projectId": int(pid), "backtestId": bid, "query": "log", "start": start, "end": start + 200}
        try:
            r = qc_post("/backtests/read/log", body, creds)
            lines = r.get("log") or r.get("logs") or []
            if not lines: break
            all_lines.extend(lines)
            if len(lines) < 200: break
        except Exception as e:
            print(f"err at {start}: {e}"); break
    print(f"# {len(all_lines)} lines")
    for ln in all_lines[-150:]:
        print(ln)


if __name__ == "__main__":
    main()
