"""Fetch runtime_statistics from QC API for a backtest_id."""
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


def qc_get(path, project_id, backtest_id, creds):
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{creds['qc_api_token']}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{creds['qc_user_id']}:{digest}".encode()).decode()
    url = f"https://www.quantconnect.com/api/v2{path}"
    if "?" in path:
        url += f"&projectId={project_id}&backtestId={backtest_id}"
    else:
        url += f"?projectId={project_id}&backtestId={backtest_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}", "Timestamp": ts})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    creds = get_creds()
    db = "/home/txy/qc_work/data/S003.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    pattern = sys.argv[1] if len(sys.argv) > 1 else "%_ms_tb"
    cur.execute("SELECT name, backtest_id, project_id FROM backtest_queue WHERE name LIKE ? AND status='done' AND backtest_id IS NOT NULL ORDER BY id", (pattern,))
    rows = cur.fetchall()
    print(f"# {len(rows)} backtests")
    print(f"# ticker\tmean_auc\tcalmar\tann\tmdd\tsharpe\truntime")
    for name, bid, pid in rows:
        ticker = name.split("_")[2]
        try:
            r = qc_get("/backtests/read", pid, bid, creds)
            stats = r.get("backtest", {}).get("runtimeStatistics", {}) or r.get("backtest", {}).get("runtime_statistics", {})
            if not stats:
                stats = r.get("runtimeStatistics") or {}
            ma = stats.get("mean_auc", "-")
            cal = stats.get("calmar", "-")
            ann = stats.get("ann", "-")
            mdd = stats.get("mdd", "-")
            shp = stats.get("sharpe", "-")
            rt = stats.get("runtime_s", "-")
            print(f"{ticker}\t{ma}\t{cal}\t{ann}\t{mdd}\t{shp}\t{rt}")
        except Exception as e:
            print(f"{ticker}\tERR: {e}")


if __name__ == "__main__":
    main()
