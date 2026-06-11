#!/usr/bin/env python3
"""Book weighting robustness + decorrelation evidence — ZERO new backtests (uses cached OOS series).
Closes two open items from the honest re-derivation: (1) QUANTIFY UUP's decorrelation (does its regime
edge actually decorrelate from the GLD/SOXX trend edges?), (2) confirm the honest 6-name book is robust
to the WEIGHTING choice (equal / Calmar^2 / DSR-aware / inverse-variance), not an artifact of Calmar^2.
"""
import json, os, math
from lb.paths import ROOT as _LBROOT

CACHE = str(_LBROOT / "results" / "series_cache.json")
OUT = str(_LBROOT / "docs" / "analysis" / "HONEST_AUDIT.md")

NAMES = ["GLD", "SOXX", "UUP", "TIP", "DBC", "HYG"]
# UPDATED 2026-06-04 to the leak-free, this-session-improved edges: GLD trend_leg+regime_gmm 3.47
# (was ker+regime_gmm; +38%) and UUP bgm+sadf_explosive+ker 1.85 (was bgm+ker 1.30; +42%, permute-real,
# provisional/Bonferroni-boundary). SOXX leak-free ~0.81 (edge GONE; kept only as a weak decorrelator).
CAL = {"GLD": 3.47, "SOXX": 0.807, "UUP": 1.85, "TIP": 1.146, "DBC": 0.912, "HYG": 1.828}
DSR = {"GLD": 0.931, "SOXX": 0.959, "UUP": 0.600, "TIP": 1.0, "DBC": 1.0, "HYG": 1.0}  # buy-hold = no selection bias; GLD/UUP DSR are STALE (pre-leak/pre-trend_leg/pre-sadf) — re-derivation needs leak-free trials


def main():
    cache = json.load(open(CACHE))
    common = sorted(set.intersection(*[set(cache[n].keys()) for n in NAMES]), key=lambda x: int(x))
    ppy = len(common) / ((int(common[-1]) - int(common[0])) / (365.25 * 86400.0))
    rets = {n: [cache[n][common[i]] / cache[n][common[i - 1]] - 1.0 for i in range(1, len(common))] for n in NAMES}
    T = len(common) - 1

    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        ca = [x - ma for x in a]; cb = [x - mb for x in b]
        num = sum(x * y for x, y in zip(ca, cb))
        da = math.sqrt(sum(x * x for x in ca)); db = math.sqrt(sum(y * y for y in cb))
        return num / (da * db) if da > 0 and db > 0 else 0.0

    def metrics(w):
        port = [sum(w[n] * rets[n][i] for n in NAMES) for i in range(T)]
        eq = [1.0]
        for r in port:
            eq.append(eq[-1] * (1 + r))
        peak = mdd = 0.0
        peak = eq[0]
        for v in eq:
            peak = max(peak, v); mdd = max(mdd, (peak - v) / peak)
        yrs = T / ppy
        cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else 0
        m = sum(port) / T; sd = (sum((x - m) ** 2 for x in port) / (T - 1)) ** 0.5
        return cagr / mdd if mdd > 1e-6 else 0, cagr * 100, mdd * 100, (m / sd * math.sqrt(ppy) if sd > 0 else 0)

    def norm(d):
        s = sum(d.values()); return {k: v / s for k, v in d.items()}

    var = {n: (lambda a: sum((x - sum(a) / len(a)) ** 2 for x in a) / len(a))(rets[n]) for n in NAMES}
    schemes = {
        "equal": norm({n: 1.0 for n in NAMES}),
        "Calmar^2": norm({n: CAL[n] ** 2 for n in NAMES}),
        "Calmar^2 x DSR": norm({n: CAL[n] ** 2 * DSR[n] for n in NAMES}),
        "inverse-variance": norm({n: 1.0 / var[n] if var[n] > 0 else 0 for n in NAMES}),
    }

    lines = ["", "## Book weighting robustness + decorrelation (cached series, zero backtests)", "",
             "Return correlation matrix (OOS): is UUP's regime edge decorrelated from the GLD/SOXX trend edges?", "", "```",
             "corr   " + "".join(f"{n:>6s}" for n in NAMES)]
    print("correlation matrix:")
    for a in NAMES:
        row = f"{a:5s}  " + "".join(f"{corr(rets[a], rets[b]):6.2f}" for b in NAMES)
        print(row); lines.append(row)
    lines.append("")
    lines.append(f"{'scheme':18s} {'Calmar':>7s} {'CAGR%':>6s} {'MaxDD%':>7s} {'Sharpe':>7s}   UUP_wt")
    print(f"\n{'scheme':18s} {'Calmar':>7s} {'CAGR%':>6s} {'MaxDD%':>7s} {'Sharpe':>7s}   UUP_wt")
    for nm, w in schemes.items():
        c, cg, dd, sh = metrics(w)
        row = f"{nm:18s} {c:7.3f} {cg:6.2f} {dd:7.2f} {sh:7.3f}   {w['UUP']*100:4.0f}%"
        print(row); lines.append(row)
    lines.append("```")
    # avg correlation of UUP and SOXX to the trend core (GLD) as decorrelation evidence
    uup_gld = corr(rets["UUP"], rets["GLD"]); uup_soxx = corr(rets["UUP"], rets["SOXX"])
    lines.append("")
    lines.append(f"UUP↔GLD corr = {uup_gld:.2f}, UUP↔SOXX corr = {uup_soxx:.2f} "
                 f"→ {'LOW correlation confirms UUP decorrelates the trend edges (earns its book seat despite individual fragility).' if max(abs(uup_gld), abs(uup_soxx)) < 0.3 else 'correlation not low — decorrelation weaker than assumed.'}")
    prev = open(OUT).read() if os.path.exists(OUT) else ""
    marker = "## Book weighting robustness"
    if marker in prev:
        prev = prev[:prev.index(marker)].rstrip() + "\n"
    open(OUT, "w").write(prev + "\n".join(lines) + "\n")
    print(f"\nUUP-GLD corr {uup_gld:.2f}, UUP-SOXX corr {uup_soxx:.2f}")
    print("written:", OUT)


if __name__ == "__main__":
    main()
