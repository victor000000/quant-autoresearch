#!/usr/bin/env python3
"""UNIVERSE SCREEN (user 2026-06-04: "explore the 311 ETFs, find which fit; do NOT forget buy-and-hold").

For each untested QC-confirmed ETF, race the GLD TREND methodology against ALWAYS_LONG (buy-and-hold) via
the normal leak-safe driver, so EVERY ETF gets its buy-hold baseline. FIT = the methodology's real OOS
Calmar BEATS buy-and-hold AND val_auc>0.55 AND deployable (>80 trades). A high Calmar that only matches
buy-hold is NO-FIT (gold rallied -> a 2.8 trend Calmar means nothing if buy-hold is 3.0).

  leg A = TREND   : logdollar + trend_leg+regime_gmm + dd_overlay + IG   (the GLD champion recipe)
  leg B = BUYHOLD : logdollar + always_long + cdf_overlay                 (vol-targeted hold = the book's
                                                                           buy-hold baseline convention)

Prioritises high-fit asset classes first (Commodity/Currency, then Fixed Income/REIT/Leveraged, then broad
equity). ONE driver call at a time (single coordinator). Resumable (skips DONE). The REGIME champion recipe
(imbalance+bgm+ker, which fit GDX) is a documented PASS-2 re-screen of the trend-no-fits.

Run:  nohup python3 scripts/screen_etfs.py > /tmp/arlogs/screen.log 2>&1 &
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
    log.write(f"START screen (trend-vs-buyhold): {len(todo)} ETFs todo ({len(done)} already done)\n")
    log.flush()
    for r in todo:
        tk = r["Ticker"].strip()
        ac = r.get("Asset_Class", "?")
        trend = json.dumps({"ticker": tk, "axis": "logdollar", "labeler": "trend_leg+regime_gmm",
                            "thresh": 0.40, "sizing": "dd_overlay", "reduce": "infogain",
                            "n_components": 15, "rebal_band": 0.03})
        buyhold = json.dumps({"ticker": tk, "axis": "logdollar", "labeler": "always_long",
                             "thresh": 0.50, "sizing": "cdf_overlay"})
        log.write(f"SCREEN {tk} ({ac})\n")
        log.flush()
        try:
            subprocess.run(["python3", DRIVER, trend, buyhold], cwd=HERE, timeout=1000,
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
