#!/usr/bin/env python3
"""Self-healing watchdog for the deep-sweep loop.

The per-round driver (run_autoresearch_round.py) intermittently HANGS after its QC
backtests complete — a QC API call that never returns, so the round never records and the
coordinator (deep_sweep_etfs.py) blocks on the subprocess. Observed 2026-06-05/06 on QID
(~13 min), SPXS, QLD (~10 min). This watchdog watches results/round_results.csv: if the
last round timestamp does NOT advance for > STALL_S while a driver exists, it kills the
hung driver; the coordinator then respawns the next round.

STALL_S=420 (7 min) is above the max legit round (2 backtests x 300s timeout + overhead),
so it won't kill a merely-slow round. Logs EVERY check so its behavior is observable.
Kills ONLY run_autoresearch_round.py (never the coordinator). Safe to restart.

Run:  nohup python3 scripts/sweep_watchdog.py > /tmp/arlogs/watchdog.log 2>&1 &
"""
import time, os, csv, subprocess

ROUND_CSV = "/home/ubuntu/lb/results/round_results.csv"
STALL_S = 420
CHECK_S = 60


def last_ts():
    try:
        rows = list(csv.reader(open(ROUND_CSV)))
        return rows[-1][0] if len(rows) > 1 else ""
    except Exception:
        return ""


def driver_pids():
    # split the literal so the watchdog's own cmdline can't self-match the pgrep pattern
    pat = "run_autoresearch" + "_round.py"
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout
    return [p for p in out.split() if p.strip().isdigit()]


def log(msg):
    print(f"[watchdog] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    last, last_change = last_ts(), time.time()
    log(f"start; last_ts={last!r} STALL_S={STALL_S} CHECK_S={CHECK_S}")
    while True:
        time.sleep(CHECK_S)
        try:
            cur = last_ts()
            if cur != last:
                last, last_change = cur, time.time()
                log(f"advance -> {cur}")
                continue
            stalled = int(time.time() - last_change)
            pids = driver_pids()
            log(f"check ts={cur} stalled={stalled}s drivers={pids}")
            if stalled > STALL_S and pids:
                for p in pids:
                    try:
                        os.kill(int(p), 15)
                    except Exception as e:
                        log(f"kill {p} failed: {e}")
                log(f"KILLED hung driver(s) {pids} after {stalled}s stall; coordinator will respawn")
                last_change = time.time()
        except Exception as e:
            log(f"loop error: {e}")


if __name__ == "__main__":
    main()
