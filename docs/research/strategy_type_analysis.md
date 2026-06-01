# Module ⓪ — Strategy-Type Router: Spec for `modules/analyzer.py`

Purpose: a per-asset, **TRAIN-only, strictly causal** front-end that profiles each ETF's statistical
signature and ROUTES the downstream pipeline (which axis, which labeling polarity/density, which sizing
mode). This is the missing upstream module the current harness lacks — today every asset runs the
identical trend/TS pipeline regardless of whether it is structurally trending, mean-reverting, or a
random walk. Per the sources (Wang transcripts; Lopez de Prado CFI §6.3), routing each asset to the
class whose mechanism is actually present is the single highest-leverage protection against OOS collapse.

Design constraints (inherited from the harness):
- **Causality:** every statistic is computed on minute/daily bars with timestamp `< TRAIN_END`
  (`datetime(2021,8,1)` in `harness/constants.py`). No `.shift(-N)`, no reversal, no bfill.
- **Concatenation:** this module is concatenated into one QC script sharing one global namespace.
  Top-level imports ONLY `numpy/pandas/math`; `statsmodels`/`sklearn` imported INSIDE functions so
  `python3 -m py_compile` always succeeds.
- **Budget:** all statistics below run in milliseconds per asset on daily-aggregated TRAIN data; this is
  a one-time profiling call at the start of a backtest, not a per-bar cost.
- **Aggregation:** stationarity/memory/ARCH tests need ≥ ~100-250 stable points and are noisy on raw
  minute bars, so compute them on **daily-resampled** TRAIN returns (resample minute→daily log-returns),
  except kurtosis/peak which are computed on the candidate **bar** series being evaluated for axis choice.

---

## 1. Per-asset statistics to compute (causal, TRAIN only)

All inputs derive from TRAIN-segment minute closes/volumes. Build:
- `r_d` = daily log-returns (resample minute closes to daily last, diff log).
- `p_d` = daily log-close level.
- `rv_d` = daily realized variance = sum of intraday squared minute log-returns.

| # | Statistic | Definition (causal) | Library | Output |
|---|---|---|---|---|
| S1 | **Excess kurtosis** of `r_d` | `scipy.stats.kurtosis(r_d, fisher=True)` | scipy | fat-tailness; high ⇒ jump-prone |
| S1b | **Jarque-Bera** of `r_d` | `scipy.stats.jarque_bera(r_d)` → stat, p | scipy | departure from normality |
| S1c | **Peak fraction** of candidate-axis bar returns | hist of bar log-returns, fraction of mass in the central bin / Gaussian-overlay peak height | numpy | whipsaw frequency proxy (Wang) |
| S2 | **ACF(1..L)** of `r_d` | `statsmodels.tsa.stattools.acf(r_d, nlags=L, fft=True)` | statsmodels | memory / trend evidence |
| S2b | **Durbin-Watson** on `r_d` | `statsmodels.stats.stattools.durbin_watson(r_d)` ∈ [0,4]; <2 ⇒ +autocorr | statsmodels | trend-tradability (Wang DW gate) |
| S3 | **Hurst exponent** `H` | bias-corrected: average of (a) variance-scaling slope, (b) adjusted-R/S, (c) DFA, over log-price `p_d`; require agreement | numpy (polyfit, cumdev) | H>0.5 trend, <0.5 MR, ~0.5 random |
| S4 | **Lo-MacKinlay variance ratio** `VR(q)` + het-robust z | VR(q)=Var(q-period ret)/(q·Var(1-period ret)); z from het-robust estimator; q∈{2,5,10,20} | numpy (~20 lines) | VR>1 trend, <1 MR; significant if |z|>1.96 |
| S5 | **ADF unit-root test** on `p_d` | `statsmodels.tsa.stattools.adfuller(p_d, autolag='AIC')` → stat, p, crit | statsmodels | reject (p<0.05) ⇒ stationary ⇒ MR-candidate |
| S6 | **ARCH-LM test** on `r_d` | `statsmodels.stats.diagnostic.het_arch(r_d, nlags=5)` → LM stat, p | statsmodels | p<0.05 ⇒ vol clustering present |
| S6b | **GARCH persistence** α+β | `arch.arch_model(r_d, vol='Garch', p=1, q=1).fit()`; persistence = α+β | arch 8.0 | high (→1) ⇒ strong clustering |
| S7 | **OU half-life** | AR(1): regress `Δp_d` on lagged `p_d` → θ; halflife = `ln(2)/θ` (only if θ>0) | numpy | endogenous MR holding horizon (no fitted lookback to overfit) |
| S8 | **Permutation entropy** `PE` + JS complexity | Bandt-Pompe ordinal patterns d=4, delay=1 on `r_d`; `scipy.spatial.distance.jensenshannon` vs uniform | numpy + scipy | high PE ⇒ unforecastable; gate off directional trading |
| S9 | **Prevailing-mean OOS-R²** (warm-up) | recursive mean forecast vs candidate signal: `1 - Σ(r-f)²/Σ(r-mean)²` | numpy | admit asset/signal only if >0 |
| S10 | **Regime count K** (label-balance check) | for K∈{2,3,4}: KMeans/JM on `[r_d,|r_d|]`, report label balance | sklearn | pick K giving balance ∈ (0.2,0.8) |

### Recommended implementation skeleton

```python
def profile_asset(close_min, vol_min, ts_min, train_end):
    """Return a dict of TRAIN-only statistics S1..S10 for one asset.
    All causal: only minutes with ts < train_end are used."""
    import numpy as np, pandas as pd, math
    tr = np.asarray(ts_min) < np.datetime64(train_end)
    c = np.asarray(close_min, float)[tr]; v = np.asarray(vol_min, float)[tr]
    t = pd.to_datetime(np.asarray(ts_min)[tr])
    s = pd.Series(np.log(np.where(c > 0, c, np.nan)), index=t).dropna()
    # daily aggregation
    p_d = s.resample("1D").last().dropna()
    r_d = p_d.diff().dropna().to_numpy()
    rv_d = (s.diff()**2).resample("1D").sum().dropna()
    out = {}
    out["kurt"] = _kurtosis(r_d)                       # S1
    out["jb_p"] = _jarque_bera_p(r_d)                  # S1b
    out["acf1"] = _acf(r_d, 1)                         # S2
    out["dw"]   = _durbin_watson(r_d)                  # S2b
    out["hurst"]= _hurst_ensemble(p_d.to_numpy())      # S3 (mean of VS/adjR-S/DFA)
    out["vr"], out["vr_z"] = _variance_ratio(r_d, q=5) # S4
    out["adf_p"]= _adf_p(p_d.to_numpy())               # S5
    out["arch_p"]= _arch_lm_p(r_d, nlags=5)            # S6
    out["garch_persist"] = _garch_persistence(r_d)     # S6b (guard ImportError)
    out["halflife"] = _ou_halflife(p_d.to_numpy())     # S7
    out["perm_ent"] = _perm_entropy(r_d, d=4)          # S8
    out["n_daily"] = len(r_d)
    return out
```

Each `_helper` imports `statsmodels`/`sklearn`/`arch` INSIDE the function and returns a NaN-safe default
if the library is missing or there is insufficient data (`len(r_d) < 100` ⇒ return a `"insufficient"`
profile that routes to the safe default = current trend pipeline with conservative sizing).

---

## 2. Decision rules → Wang's 5 strategy classes

Classify with a **consensus** (Wang/Letian-Quant emphasis: require multiple tests to agree, which is the
anti-overfit mechanism). Evaluate in this priority order; first match wins. Thresholds are the defaults
from the sources and should themselves be validated, not treated as sacred.

```
INPUTS: stats dict from profile_asset(); plus availability of a partner asset (for pairs).

# --- Gate 0: forecastability (applies to ALL classes) ---
if perm_ent > PE_NOISE (≈0.95) and (vr_z is insignificant) and acf1 ≈ 0:
        -> class = "stand_aside"   # high entropy, no exploitable memory; do not trade
        # (Permutation-entropy / AMH gate. Catalog ⓪ #PE, #AVR.)

# --- Class A: VOLATILITY ---
elif arch_p < 0.05 and garch_persist > 0.85:
        -> class = "volatility"    # strong, persistent vol clustering
        # routes to vol-targeting + EGARCH gate; directional edge optional

# --- Class B: MEAN_REVERSION ---
elif (adf_p < 0.05)            # stationary level (reject unit root)
     and (hurst < 0.45)        # anti-persistent
     and (vr < 1.0 and abs(vr_z) > 1.96)   # significant VR<1
     and (0 < halflife < HL_MAX (≈ 0.5 * n_daily)):  # finite, sane half-life
        -> class = "mean_reversion"   # require 3-of-4 agreement; see consensus note

# --- Class C: TREND ---
elif (hurst > 0.55)            # persistent
     and (dw < 2.0 and acf1 > 0)        # positive return autocorrelation (Wang DW/ACF gate)
     and (vr > 1.0 and abs(vr_z) > 1.96):   # significant VR>1
        -> class = "trend"

# --- Class D: CROSS_SECTIONAL --- (only if a universe is available)
elif have_universe and asset_is_high_beta_member:
        -> class = "cross_sectional"   # rank/relative-value after stripping beta

# --- Class E: ARBITRAGE --- (only if a cointegrated partner is available)
elif have_partner and spread_acf1 > 0:   # spread/ratio shows positive autocorr
        -> class = "arbitrage"            # spread-as-trend transform (Wang)

# --- Default ---
else:
        -> class = "trend"   # but flag low-confidence; conservative sizing
```

**Consensus rule (anti-overfit):** for MEAN_REVERSION and TREND, require ≥3 of the 4 listed tests to
agree before committing the class; if only 1-2 agree, downgrade to `"weak_<class>"` which keeps the
class but forces conservative sizing (half target-vol, higher meta-label threshold). Letian-Quant's
USDCAD example (ADF p=0.278, H=0.555 ⇒ NOT MR) is the motivating case where a single test would
misclassify — consensus refuses to trade the wrong mechanism.

**Confidence score:** `conf = (# agreeing tests)/(# applicable tests)` ∈ [0,1], passed downstream to
scale sizing aggressiveness. Re-validate the class on a rolling basis (CFI §6.3 falsifiability): if the
live regime-health monitor (catalog ⓪ circuit-breaker) shows the class's defining statistic has drifted
out of its TRAIN envelope, cut exposure rather than waiting for a structural-break test.

For the **7-ETF set specifically:** cross_sectional needs the cross-section (use the 7 as a small one or
skip), arbitrage needs a cointegrated partner (candidates: QQQ/IWM, HYG/TLT, GLD vs others; no
physical-ratio commodity pairs). The first production version of `analyzer.py` should implement
`stand_aside / volatility / mean_reversion / trend` robustly and stub cross_sectional/arbitrage.

---

## 3. How the class ROUTES the pipeline

The router returns `{class, confidence, halflife, hurst, axis_hint, label_hint, sizing_hint}`.
Downstream modules consume these hints instead of sweeping blindly.

| Class | ① Axis | ② Labeling | ⑧ Sizing | Notes |
|---|---|---|---|---|
| **trend** | dynamic dollar bar or vol-clock; per-asset axis chosen by lowest kurtosis/peak at target bar-count | **trend-leg labels** (label whole leg, ignore pullbacks; binary) and/or vol-scaled triple-barrier with |t| weighting; positive polarity | prob→size CDF × vol-target; meta-label gate; vol-targeting overlay | Wang flagship path; the harness's current default — now applied only where ACF/Hurst/VR confirm trend. |
| **mean_reversion** | range/price-action bar (no need for vol clock); **lookback = int(halflife)** | reversion labels: long when z-score(price, window=halflife) is low; **negative** polarity vs trend | size ∝ −z-score, capped; tighter stops (half-life sets max holding) | Half-life is the endogenous, non-overfit lookback (Letian-Quant). Bonds/EM (TLT, HYG, EEM) likely land here — explains their poor trend-pipeline Calmar today. |
| **volatility** | vol-clock axis (equal realized-vol per bar) | optional direction labels; primary edge is exposure scaling | **inverse-vol / vol-targeting dominant**; EGARCH top-decile de-risk gate; size capped | Route here when ARCH-LM significant and GARCH persistence high. |
| **cross_sectional** | dollar bar on each name | rank labels (top/bottom decile fwd-return) after beta-stripping | dollar-neutral / rank-weighted | Needs a universe; stub for 7-ETF set. |
| **arbitrage** | price-action axis on the **spread** series (no volume on a spread) | **spread-as-trend**: run trend labels on spread A−k·B (k from economic ratio or reverse-fit); standardized/log/ratio variants | trend-following stops on the synthetic spread (no N-sigma band) | Reuses the entire trend stack on a transformed 1-D input (Wang). |
| **stand_aside** | n/a | n/a | **size = 0** | High-entropy / no-memory regime; protects Calmar by not trading noise. |
| **weak_\<class\>** | as parent class | as parent class | **conservative**: 0.5× target-vol, higher meta-label probability threshold | Low-consensus classification; trade small or not at all. |

### Sizing-hint plumbing
`confidence`, `halflife`, `hurst` flow into Module ⑧:
- `target_vol_eff = target_vol * confidence` (low-confidence ⇒ smaller risk budget).
- mean_reversion sizing uses `window = int(halflife)` for its z-score; trend uses the axis/label cfg.
- `stand_aside` and zero-mask periods set position to 0 (no churn, no cost bleed).

---

## 4. Causality & validation checklist (must hold before this module ships)

1. Every statistic uses only minutes with `ts < TRAIN_END`; daily resample is causal (no future bars).
2. No statistic uses VAL/TEST data; the class label is frozen at TRAIN time and applied forward.
   (Optional rolling re-validation re-fits on an expanding/rolling TRAIN window, never on future data.)
3. Library calls (`statsmodels`, `arch`, `sklearn`) are imported inside functions and degrade gracefully
   (NaN-safe defaults → `stand_aside`/conservative route) so `py_compile` and lib-absent runs never crash.
4. Thresholds (PE_NOISE, HL_MAX, Hurst bands, VR z-cutoff) are config constants, swept under purged CV at
   research time, NOT hard-tuned on the test window. Treat the whole router as a falsifiable hypothesis.
5. The router's class assignment is logged per asset so its effect on REAL OOS Calmar (vs the current
   route-everything-as-trend baseline) is measurable; keep the router only if it raises real Calmar.

---

## 5. Minimal first deliverable

`modules/analyzer.py` exposing:

```python
def profile_asset(close_min, vol_min, ts_min, train_end) -> dict          # S1..S10
def classify_asset(stats: dict, have_partner=False, have_universe=False) -> dict
    # -> {"class","confidence","halflife","hurst","axis_hint","label_hint","sizing_hint"}
ROUTE = classify_asset(profile_asset(...))
```

Implement `stand_aside / volatility / mean_reversion / trend` end-to-end; stub
`cross_sectional / arbitrage`. Wire `ROUTE` into the orchestrator so each ETF's axis/label/sizing sweep
is restricted to the hinted options rather than the full generic grid. Expected first win: bonds/EM
(TLT, HYG, EEM) reclassified out of the trend pipeline into mean_reversion (half-life lookback) or
stand_aside, removing the structurally-wrong trend bets that depress their current real Calmar.
