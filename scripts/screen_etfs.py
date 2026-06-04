#!/usr/bin/env python3
"""UNIVERSE SCREEN (user directive 2026-06-04: "explore the 311 ETFs, find which fit my methodology").

For each untested QC-confirmed ETF, race the two CHAMPION methodologies head-to-head via the normal
driver (leak-safe, gated, logged): the GLD TREND recipe (logdollar + trend_leg+regime_gmm + dd_overlay
+ IG) vs the UUP REGIME recipe (imbalance + bgm+ker + cdf_overlay). The driver logs each leg's val_auc
+ real OOS Calmar to results/round_results.csv; classify FIT (val_auc>0.55 AND deployable edge that
beats the ticker's buy-hold) vs NO-FIT (buy-hold drifter / no structure) from that.

Prioritises the high-fit asset classes first (Commodity/Currency like GLD/UUP, then Fixed Income / Real
Estate / Leveraged), then the broad-equity names (mostly expected NO-FIT). ONE driver call at a time
(single coordinator — no main.py collision). Resumable (skips tickers already marked DONE).

Run in background:  nohup python3 scripts/screen_etfs.py > /tmp/arlogs/screen.log 2>&1 &
"""
import sys, os, csv, json, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "etf_qc_confirmed_pre2009.csv")
DRIVER = os.path.join(HERE, "scripts", "run_autoresearch_round.py")
PROG = os.path.join(HERE, "results", "etf_screen_progress.log")

TESTED = set("DBC EEM EFA FXY GLD HYG ITB IWM KRE QQQ SLV SMH SOXX SPY TIP TLT UUP VIXY XBI XLE XME".split())
PRIORITY = {"Commodity": 0, "Currency": 0, "Fixed Income": 1, "Real Estate": 1,
            "Leveraged/Inverse": 2, "International Equity": 3, "US Equity": 4}


def _aum_rank(r):
    try:
        return int(float(r.get("AUM_Rank") or 9999))
    except Exception:
        return 9999


def already_done():
    done = set()
    if os.path.exists(PROG):
        for line in open(PROG):
            if line.startswith("DONE "):
                done.add(line.split()[1])
    return done


def main():
    rows = list(csv.DictReader(open(CSV)))
    done = already_done()
    todo = [r for r in rows
            if r["Ticker"].strip() not in TESTED
            and r["Ticker"].strip() not in done
            and r.get("Asset_Class", "") in PRIORITY]
    todo.sort(key=lambda r: (PRIORITY[r["Asset_Class"]], _aum_rank(r)))
    log = open(PROG, "a")
    log.write(f"START screen: {len(todo)} ETFs todo ({len(done)} already done)\n")
    log.flush()
    for r in todo:
        tk = r["Ticker"].strip()
        ac = r.get("Asset_Class", "?")
        trend = json.dumps({"ticker": tk, "axis": "logdollar", "labeler": "trend_leg+regime_gmm",
                            "thresh": 0.40, "sizing": "dd_overlay", "reduce": "infogain",
                            "n_components": 15, "rebal_band": 0.03})
        regime = json.dumps({"ticker": tk, "axis": "imbalance", "labeler": "bgm+ker",
                             "thresh": 0.50, "sizing": "cdf_overlay"})
        log.write(f"SCREEN {tk} ({ac})\n")
        log.flush()
        try:
            subprocess.run(["python3", DRIVER, trend, regime], cwd=HERE, timeout=1000,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log.write(f"ERR {tk}: {e}\n")
            log.flush()
        log.write(f"DONE {tk}\n")
        log.flush()
    log.write("SCREEN COMPLETE\n")
    log.flush()


if __name__ == "__main__":
    main()
