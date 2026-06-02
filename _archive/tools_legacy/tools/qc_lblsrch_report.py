"""Aggregate label-search results across 19 ETFs."""
import base64
import hashlib
import json
import sqlite3
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
        "Authorization": f"Basic {auth}", "Timestamp": ts, "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    creds = get_creds()
    con = sqlite3.connect("/home/txy/qc_work/data/S003.db")
    cur = con.cursor()
    cur.execute("SELECT name, backtest_id, project_id FROM backtest_queue WHERE batch='lbl_search' AND status='done' AND backtest_id IS NOT NULL ORDER BY id")
    rows = cur.fetchall()
    print(f"# {len(rows)} done\n")
    label_names = ["sign_400", "tb_200_1", "tb_400_1", "tb_200_1_NT", "tb_200_15", "zsign_200_05"]
    short = {n: n[:14] for n in label_names}
    print(f"{'ETF':4s}  " + "  ".join(f"{n[:11]:11s}" for n in label_names) + "  best     bestVAU  bestOAU  Long_C  LS_C")
    for name, bid, pid in rows:
        ticker = name.split("_")[2]
        try:
            r = qc_post("/backtests/read", {"projectId": int(pid), "backtestId": bid}, creds)
            stats = r["backtest"].get("runtimeStatistics", {}) or {}
        except Exception as e:
            print(f"{ticker}  ERR: {e}"); continue
        row = [ticker]
        for n in label_names:
            v = stats.get(f"{short[n]}_v", "-")
            o = stats.get(f"{short[n]}_o", "-")
            row.append(f"{v}/{o}")
        bl = stats.get("best_label", "-")
        bv = stats.get("best_val_auc", "-")
        bo = stats.get("best_oos_auc", "-")
        lc = stats.get("long_cal", "-")
        sc = stats.get("ls_cal", "-")
        cells = "  ".join(f"{c:11s}" for c in row[1:])
        print(f"{ticker:4s}  {cells}  {bl[:8]:8s} {bv:8s} {bo:8s} {lc:6s}  {sc:6s}")


if __name__ == "__main__":
    main()
