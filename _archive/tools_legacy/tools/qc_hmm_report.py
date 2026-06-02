"""Aggregate HMM-label and balanced-label results vs BH baseline."""
import base64
import hashlib
import json
import sqlite3
import time
import urllib.request

BH = {
    "QQQ":1.17,"SPY":1.38,"IWM":0.61,"EFA":1.47,"EEM":1.35,"AGG":0.96,"TLT":-0.19,
    "GLD":2.12,"XLE":0.95,"VXX":-0.60,"VNQ":0.41,"HYG":1.72,"TIP":1.33,"DBC":1.12,
    "XLU":1.83,"XLP":0.90,"SHY":4.67,"UUP":0.41,"EWJ":1.12,
}


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
        "Authorization": f"Basic {auth}", "Timestamp": ts, "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def report(batch, label_stat_keys, name_keys):
    creds = get_creds()
    con = sqlite3.connect("/home/txy/qc_work/data/S003.db")
    cur = con.cursor()
    cur.execute(f"SELECT name, backtest_id, project_id FROM backtest_queue WHERE batch=? AND status='done' AND backtest_id IS NOT NULL ORDER BY id", (batch,))
    rows = cur.fetchall()
    print(f"\n=== {batch} — {len(rows)} done ===")
    print(f"{'ETF':4s}  " + "  ".join(f"{k[:9]:9s}" for k in label_stat_keys) + "  best  bestV  bestO  longC  lsC    confC  BH    Alpha")
    for name, bid, pid in rows:
        ticker = name.split("_")[2]
        try:
            r = qc_post("/backtests/read", {"projectId": int(pid), "backtestId": bid}, creds)
            stats = r["backtest"].get("runtimeStatistics", {}) or {}
        except Exception as e:
            print(f"{ticker}  ERR"); continue
        cells = []
        for k in label_stat_keys:
            v = stats.get(f"{k}_v", "-")
            o = stats.get(f"{k}_o", "-")
            cells.append(f"{v}/{o}")
        bl = stats.get(name_keys[0], "-")
        bv = stats.get(name_keys[1], "-")
        bo = stats.get(name_keys[2], "-")
        lc = stats.get("long_cal", "-")
        sc = stats.get("ls_cal", "-")
        cc = stats.get("conf_cal", "-")
        bh = BH.get(ticker, 0.0)
        try:
            alpha = float(lc) - bh
            alpha_s = f"{alpha:+.2f}"
        except (ValueError, TypeError):
            alpha_s = "-"
        cells_str = "  ".join(f"{c:9s}" for c in cells)
        print(f"{ticker:4s}  {cells_str}  {str(bl)[:4]:4s}  {bv[:5]}  {bo[:5]}  {lc:5s}  {sc:5s}  {cc:5s}  {bh:+.2f} {alpha_s}")


if __name__ == "__main__":
    report("hmm_label", ["tau05", "tau07", "tau09"], ["best_tau", "best_val", "best_oos"])
    report("bal_label", ["dm400", "md400", "tbbal", "dm200"], ["best", "best_val", "best_oos"])
