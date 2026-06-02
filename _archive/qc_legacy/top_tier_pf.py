"""Top-Tier Portfolio: S+A tier ETFs, optimal configs, real SetHoldings."""
from AlgorithmImports import *
import json, math, numpy as np

TOP = {
    "SOXX": ("v264", datetime(2023,8,1)),
    "GDX": ("v270", datetime(2023,8,1)),
    "QQQ": ("v246", datetime(2022,8,1)),
    "GLD": ("v270", datetime(2023,8,1)),
    "XME": ("v270", datetime(2023,8,1)),
    "SMH": ("v264", datetime(2023,8,1)),
    "VGT": ("v246", datetime(2022,8,1)),
    "KBE": ("v264", datetime(2023,8,1)),
    "XLF": ("v264", datetime(2023,8,1)),
    "KRE": ("v264", datetime(2023,8,1)),
    "XTN": ("v264", datetime(2023,8,1)),
}

class TopTierPF(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2021,1,1); self.set_end_date(2026,6,1); self.set_cash(100000)
        self.syms = {}; self.preds = {}; self.idxs = {}; self.probs = {}; self.cls = {}
        for t, (r, ts) in TOP.items():
            self.syms[t] = self.add_equity(t, Resolution.MINUTE).symbol
            d = json.loads(self.object_store.read(r + "/" + t + "/test_preds.json"))
            self.preds[t] = d["predictions"]; self.idxs[t] = 0; self.probs[t] = 0.5; self.cls[t] = []
        self.starts = {t: ts for t, (_, ts) in TOP.items()}
        self.last_reb = datetime(2021,1,1)
    def on_data(self, slice):
        for t, sym in self.syms.items():
            if sym not in slice.bars: continue
            self.cls[t].append(float(slice.bars[sym].close))
            if len(self.cls[t]) > 200: self.cls[t] = self.cls[t][-200:]
            if self.time < self.starts[t]: continue
            if self.idxs[t] >= len(self.preds[t]): continue
            p = self.preds[t][self.idxs[t]]
            dt = datetime(int(p["time"][:4]),int(p["time"][5:7]),int(p["time"][8:10]),int(p["time"][11:13]),int(p["time"][14:16]),int(p["time"][17:19]))
            if self.time >= dt: self.probs[t] = float(p["pred"]); self.idxs[t] += 1
        if (self.time - self.last_reb).days >= 60:
            self.last_reb = self.time
            selected = []
            for t in TOP:
                if self.probs.get(t, 0.5) > 0.5 and len(self.cls.get(t, [])) >= 20:
                    if self.cls[t][-1] / self.cls[t][-20] - 1 > 0:
                        selected.append(t)
            for t, sym in self.syms.items():
                w = 1.0 / len(selected) if selected and t in selected else 0.0
                self.set_holdings(sym, min(1.0, w))
