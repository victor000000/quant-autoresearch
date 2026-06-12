#!/usr/bin/env python3
"""Precompute daily ATM implied vol for an underlying into ObjectStore.

QC option chains are minute-only, so a 15-year IV series is harvested in 5-year
chunk backtests (each ~10 min). Each chunk saves {"YYYY-MM-DD": atm_iv} to
ObjectStore key autoresearch/<TK>/iv_<start>_<end>. The footer's features='iv'
block merges all chunks into one causal exogenous series (shifted 1 day).

Usage: python3 scripts/research/precompute_iv.py SPY 2010-01-04 2015-01-02 ...
"""
import sys

from lb.harness.qc_client import submit_and_wait

TMPL = '''from AlgorithmImports import *
import json


class IVHarvest(QCAlgorithm):
    def initialize(self):
        self.set_start_date(__Y0__, __M0__, __D0__)
        self.set_end_date(__Y1__, __M1__, __D1__)
        self.set_cash(100000)
        eq = self.add_equity("__TK__", Resolution.MINUTE)
        self.eq_sym = eq.symbol
        opt = self.add_option("__TK__", Resolution.MINUTE)
        opt.set_filter(lambda u: u.strikes(-1, 1).expiration(20, 45))
        self.opt_sym = opt.symbol
        self.series = {}
        self._chain = None
        self.schedule.on(self.date_rules.every_day("__TK__"),
                         self.time_rules.before_market_close("__TK__", 30), self.snap)

    def on_data(self, slice):
        ch = slice.option_chains.get(self.opt_sym)
        if ch is not None:
            self._chain = ch

    def snap(self):
        ch = self._chain
        if ch is None:
            return
        spot = self.securities[self.eq_sym].price
        best = None
        bdist = 1e9
        for c in ch:
            d = abs(float(c.strike) - float(spot))
            if d < bdist:
                bdist = d
                best = c
        if best is not None:
            iv = float(best.implied_volatility)
            if iv > 0:
                self.series[str(self.time.date())] = round(iv, 6)

    def on_end_of_algorithm(self):
        key = "autoresearch/__TK__/iv___K0_____K1__"
        self.object_store.save(key, json.dumps(self.series))
        self.set_runtime_statistic("iv_n", str(len(self.series)))
        self.set_runtime_statistic("iv_key", key)
'''


def main(argv):
    tk = argv[0]
    dates = argv[1:]
    for i in range(0, len(dates) - 1, 1):
        d0, d1 = dates[i], dates[i + 1]
        y0, m0, dd0 = d0.split("-")
        y1, m1, dd1 = d1.split("-")
        code = (TMPL.replace("__TK__", tk)
                .replace("__Y0__", str(int(y0))).replace("__M0__", str(int(m0))).replace("__D0__", str(int(dd0)))
                .replace("__Y1__", str(int(y1))).replace("__M1__", str(int(m1))).replace("__D1__", str(int(dd1)))
                .replace("__K0__", d0).replace("__K1__", d1))
        print(f"[iv] {tk} chunk {d0} -> {d1} ...", flush=True)
        res, st = submit_and_wait(code, f"iv_{tk}_{d0}", timeout_s=2400)
        rt = (res or {}).get("runtimeStatistics") or {}
        print(f"  status {st} | iv_n {rt.get('iv_n')} | key {rt.get('iv_key')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
