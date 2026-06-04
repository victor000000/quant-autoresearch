# autoresearch

Single-ticker ETF ML on QuantConnect (project 31338454), Wang's pipeline. Each round: pick the **weakest** ticker, race two hypotheses on the 2 QC nodes, KEEP the winner iff it clears every gate on real OOS Calmar. This file is the whole operating manual + honest results.

## The loop, rules & backtest contract

**Core rules**
- **ALWAYS attack the WEAKEST ETF each round** (lowest real OOS Calmar) — never drift to the strong/structured names (GLD/UUP). Don't be lazy: co-design the WHOLE config for that ticker, don't just swap one knob.
- **Single-ticker only.** No cross-ticker ensembling. Simple > complex.
- **Priority modules = custom bar AXIS + unsupervised LABEL** — every confirmed edge came from one (ker, trend_leg, imbalance axis); reach there first. **But ALSO explore the other modules** (features, reduce, sizer) — a win can come from any slot (IG reducer lifted GLD 3.47→4.02).
- **Never stop exploring** new axes/labels — literature-mined methods still win after "exhausted" calls (trend_leg, sadf, IG). On static data, though, don't grind deflation-dead names with config permutations (raises every co-tested name's bar for ~0 EV).
- **Leak-safe is non-negotiable** — bar thresholds TRAIN-only + OOS-invariant; run `tests/test_bar_threshold_leak.py` after any `bar_builder.py` change; verify every new method EMPIRICALLY (append-OOS-invariance), not just by code-audit (audits miss leaks).
- **Autonomous:** decide + run the next experiment; do not ask.

**The loop**
1. **Pick the weakest ticker** (lowest real OOS Calmar). Re-validate its stored best first — records go stale as the OOS window grows.
2. **Think** — read the provenance graph (`knowledge.json`) + findings; co-design one ticker's `axis × label × features × reduce × model × sizing`.
3. **Build a NEW method** (prefer bar axis or labeler) and A/B it vs the champion. Leak-safe: bar thresholds fit TRAIN-only; labels may look ahead (target only); the model sees past-only features.
4. **Race:** `scripts/run_autoresearch_round.py '<A>' '<B>'` (auto-updates the report). Hypothesis = `{ticker, axis, labeler, thresh, sizing[, reduce, n_components, rebal_band, max_depth]}`.
5. **KEEP** iff ALL hold, else DISCARD; record → commit:
   - deployable (**trades > 80**)
   - **Calmar > 0** AND **> re-validated prev best**
   - **val_auc > 0.52** (below = coin-flip model, no learnable structure)
   - **beats `always_long`** (baseline carries no selection bias)
   - **survives deflation** (Deflated Sharpe / best-of-N noise floor)
   - **survives the permuted-label control**

**Permuted-label control (honesty harness).** Re-run the KEEP with `"permute_labels":true` — shuffles ONLY TRAIN labels (leak-safe, writes a distinct `_perm` cell). A REAL edge COLLAPSES toward buy-hold; if it survives, the "edge" is drift/sizing/leak → discard. Validate every KEEP: UUP 1.30→−0.08 (pure label alpha), GLD 4.02→+0.73 excess-over-BH. The decisive real-vs-artifact test; orthogonal to deflation haircuts.

**Wang's backbone.** Resample off the clock → label **unsupervised** (label may look ahead; causality lives in the supervised model on past-only features) → **rich features then reduce** (fit on TRAIN) → **bet-size.** Detectors: trend-scan / change-point / clustering — **NOT HMM.** Aim Calmar > 3, reproducible, deployable.

Fixed downstream (`modules/trainer.py`, all seeded `random_state=42`):
```
StandardScaler → reduce_dims(corr-20 | reduce=infogain) → XGBoost(depth 3, lr .03, n 200) → isotonic calibrate (embargoed VAL)
```
`reduce=infogain` = Wang information-gain selection (top-K by mutual_info with TRAIN label vs corr-filter's variance); opt-in, `correlation` default byte-identical. Multi-file render: `bar_builder.py` is a SEPARATE QC file (imported by main.py) → the 64k char limit is PER FILE, so 3-way+ ensembles fit. Keep separate files QC-lint-clean (no `getattr`, no nested-quote f-strings).

**BACKTEST CONTRACT — never break (audited clean, `BACKTEST_AUDIT.md`).** Real OOS backtest is **online, leak-free, model-only-from-QC-ObjectStore.**
- `infer.py` holds NO model — pure replay of saved predictions + causal `_size`. Test enters only via predict.
- Every `.fit` is in `footer.py` on TRAIN(+embargoed VAL) only. Features past-only; thresholds TRAIN-only.
- Proven: `verify.py` bars ≤ 1e-9; `infer_online.py` p_live==p_saved ≤ 1e-6. **Don't trust a champion until its `infer_online` shows preds_match=1.** Ensembles deploy live (footer saves multi-member bundle; `live_trade.py.tmpl` averages calibrated+gated member probs online, rbuf-warmed from history).

**Leak guard (regression test).** After the logdollar/kyle bar-threshold leak (scaled TRAIN rate by full-series `np.sum(valid)` = OOS-inclusive count → inflated Calmars), `tests/test_bar_threshold_leak.py` (numpy-free, CI-runnable) asserts every bar-threshold scaling is TRAIN-masked / OOS-invariant and forbids the `np.sum(valid)` full-series signature. Negative-tested. **RUN IT after any `bar_builder.py` change.** Deepest lesson: in-sample + code audits MISS leaks — only re-running with the fix reveals impact (full leak history below).

## Confirmed edges & deployable book

Durable single-ticker alpha is scarce + asset-intrinsic. Leak-free, permute-confirmed, the book is **one strong gold edge + one softening dollar edge + a small-cap edge, on a diversified buy-hold core.** Conviction order: **GLD > UUP > IWM.**

### Confirmed model edges

| Ticker | Config | Calmar | Status |
|--------|--------|-------:|--------|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t0.40 n15 rebal_band=0.03 / reduce=infogain | **4.02** | **the one durable edge.** decay-HEALTHY (early→late Sharpe 1.84→2.30). re-validated bit-exact 3× (4.0218), online-proven, gold-specific (not SLV). |
| **UUP** | imbalance / `bgm+sadf_explosive+ker` / cdf_overlay | **1.85** | provisional. permute-real but decay-STALE (early→late 2.67→0.74; alpha front-loaded in 2014-15). Bonferroni-boundary (N=72). Earns its seat by decorrelation. |
| **IWM** | logdollar / `trend_leg` / cdf_overlay / reduce=infogain | **0.665** | provisional. beats buy-hold (0.55), permute-PASS, decay-HEALTHY (0.63→1.31), online-proven. fails strict deflation (DSR 0.845, N=64). |

GLD decomposition: ~2.0 gold buy-hold + ~1.16 real label-timing alpha. GLD load-bearing module = trend (trend_leg/ker); regime_gmm+dd_overlay = justified Calmar/drawdown machinery (+22% / +9%, ablation-confirmed). UUP load-bearing module = bgm regime (carries 1.11 of 1.30); sadf adds explosive-regime. Edge type + axis are asset-intrinsic: GLD/IWM trend on logdollar (info clock), UUP regime on imbalance (order-flow).

### Deployable book

**GLD / UUP / IWM / TIP / DBC / HYG**, Calmar²-weighted (gross ≤ 1). TIP/DBC/HYG are `always_long` buy-hold diversifiers (no timing edge — they earn seats by decorrelation, not alpha).

- Weekly grid: **Calmar 4.617 / MaxDD 2.46% / Sharpe 2.46.** Honest daily haircut ~1.15× → **~4.0 Calmar / ~2.8% MaxDD.**
- Net-of-cost: gross → **~3.4 @5bp** (GLD 4.02→3.43 on 602 orders; UUP 1.85→1.54; IWM 0.665→0.61) → ~2.8 @10bp. Buy-hold core barely trades, so cost erosion is GLD-driven; the IG+0.03-band GLD config is more cost-robust than the old one.
- GLD anchors (~63% Calmar²-weight). UUP earns its seat by decorrelation (negative-correlated with every other member; with-UUP cuts book MaxDD) despite standalone staleness. Drop-UUP and drop-IWM both LOWER book Calmar — keep the weak names for decorrelation.
- **SOXX DROPPED**: its edge was the bar-threshold leak (leak-free 0.81 ≈ buy-hold). IWM replaced it as the 6th decorrelation seat.

Live-deployable: ensembles deploy via the multi-member `model_{cell}` bundle (footer saves all members, `live_trade.py` averages calibrated+gated probs online); verified end-to-end on QC.

## Governing rules & failure modes (lessons)

**The governing rule — two edge mechanisms, asset-intrinsic.** Every result fits this. Each mechanism needs its own labeler; which one wins (and the axis) is a property of the asset, not a choice.
1. **TREND-MOMENTUM** (`ker` → beaten by `trend_leg`): wins ⟺ drawdowns are momentum-cyclical / trend-predictable AND trimming-cost < MaxDD-saved. Wins: GLD ✓ (logdollar/info clock). Fails: event/shock-driven (XLE, XBI, DBC → 0.05–0.3) and V-recovery up-drifters (QQQ/SPY → timing trims carry). SOXX was a trend crown pre-leak (3.02) → **leak-dead 0.81 ≈ buy-hold**; not recoverable by any method. XME, SLV near-miss → mechanism is sector-idiosyncratic, not a general "cyclical sector" property.
2. **REGIME** (`bgm`): wins on macro-regime oscillation → UUP ✓ (imbalance / order-flow clock). The trend core gets only ~0.55 on UUP; `ker` gets ~0.40. Each mechanism's labeler fails on the other's asset.

**Where wins come from (the meta-pattern). Three sources, proven across the session:**
- **(a) a BETTER core labeler IN the edge's own mechanism** — `trend_leg` > `ker` on GLD-trend (2.51→3.47).
- **(b) an ORTHOGONAL ADD** — `sadf_explosive` adds to UUP-regime (bgm+ker 1.30 → bgm+sadf+ker 1.85, sadf is orthogonal to bgm/ker; same sadf DISCARDS on GLD-trend → mechanism matters).
- **(c) a DIFFERENT module slot** — IG reducer (`reduce=infogain`) lifts GLD 3.47→4.02.
- **What always LOSES:** swapping the mechanism's IMPLEMENTATION (`jump_model`↔bgm, `sliced_wasserstein`↔bgm/regime_gmm, `calmar_scan`/`sortino_scan`↔trend_leg, `vpin`↔imbalance, `run`/`spectral`↔logdollar) — incumbent always wins. And a **3rd same-family label DILUTES** (GLD trend_leg+regime_gmm+{accel/cusum/sadf/sliced_w/Wang-ladder} all < 4.02; UUP any 3rd dilutes). GLD's trend core has now beaten 7 cousins (ker, calmar_scan, sortino_scan, accel, sharpe_scan, tleg-ladder, sadf).

**Label-relevance (val_auc / MI) ≠ profit-relevance (Calmar).** High val_auc + low Calmar = the "predictable-not-profitable" trap. Confirmed 5×: `jump_model` (val_auc 0.82, persistent regimes predictable not directional), `sliced_wasserstein` (val_auc 0.81–0.89, OT vol-regime learnable not tradeable), `features=rich` VR-persistence (val_auc HIGHER, Calmar collapses — IG SELECTS high-MI features that are drawdown-increasing → crowds out profitable trend_leg features), QQQ intraday-revert. Mirror trap = high Calmar + low val_auc (SOXX IG-revival 2.6 at val_auc 0.48 = drift/path-luck). The val_auc>0.52 gate + permute + deflation jointly catch both directions. **IG amplifies a real trend edge (val_auc>0.6) but cannot manufacture one** (SOXX 0.48, every drifter) and helps only the single-trend-label component, not pure-regime ensembles (UUP IG +1.65 < corr 1.85).

**Clean drifters are buy-hold-optimal.** Timing a low-drawdown drifter sacrifices the carry it needs; there's little drawdown to cut so any trimming loses → HYG always_long 1.83 (credit-spread signal can't beat carry-sacrifice), QQQ/SPY/TIP/DBC. These are val_auc≈0.5 names; ML finds nothing.

**New methods help only where there's structure (val_auc>0.6).** On val_auc≈0.5 drifters no method beats buy-hold — don't grind them (TLT 0.40–0.49, FXY 0.50, XME 0.500, VIXY 0.53). Vol products = uncrackable without external data: short-carry is Calmar-incompatible (crash fat-left-tail, DA 71), long-timing val_auc<0.5 (spikes exogenous), term-structure features lift val_auc 0.56→0.61 but Calmar stays ~0.

**Sample where the edge RESOLVES, not where the trend is.** A trend clock (`run` axis) is silent during chop / at turning points — exactly where `trend_leg` resolves its EXITS — so MaxDD balloons (GLD 4.55→0.80). `logdollar`/`dc` sample the drawdown-onsets the trend edge must time → win. Corollary: the resolution insight is a bar-AXIS property (where to sample), NOT a labeling target (`turn_scan` reversal-timing label lost on UUP). Bar AXIS is asset-intrinsic; the custom-axis lever is CLOSED (kyle/run/spectral/vpin all lose to champions already on the right clock).

**Per-crown Calmar ablation reveals the load-bearing module (asset-intrinsic):** GLD = trend-primary (ker/trend_leg core; regime_gmm = +22% drawdown machinery; dd_overlay GOLD-specific, hurts semis' V-recoveries). SOXX (pre-leak) / UUP = regime-primary (bgm carries it). **Ablate on the DEPLOYED objective (Calmar), not a proxy (Sharpe)** — PBO-on-Sharpe called GLD's additions overfit; Calmar ablation showed they earn +22%/+9% by managing drawdowns Sharpe under-weights. Trusting the Sharpe proxy would have cost ~22% Calmar.

**Leak history + the only reliable detection (2026-06-03).** `logdollar` (champion axis) and `kyle` bar-thresholds scaled the TRAIN-fit rate by `int(np.sum(valid))` = valid-minute count over the **FULL series incl. OOS** → finer OOS bars → inflated Calmar. Impact was non-uniform OOS-dependent variance: inflated GLD (4.71→2.76) and SOXX (3.02→0.81), suppressed XLE (0.86→1.35) → the crowned `logdollar` champions were the leak's lucky cases (selection-under-leak). Fix: extrapolate TRAIN valid-density to full length (`train_valid/train_total * len(c)`), OOS-invariant. **Lessons: (1) only RE-RUNNING with the fix reveals a leak's impact — in-sample/code audits miss it (the prior 13-agent audit AND this workflow's own agents did).** (2) Bar-coarseness fragility (a threshold change swinging 4.71→2.76) is itself an overfit-to-bar-realization signal. Guarded by `tests/test_bar_threshold_leak.py` (above).

**Other closed levers — don't re-grind:** model capacity (depth-5 overfits 3.20→1.71; depth-3 optimal) · model family (XGBoost only — sklearn ExtraTrees platform-blocked in QC) · meta-labeling (weaker than ker, over-filters UUP, decayed EEM) · VR + SPY cross-asset features (crowd correlation-select; high-MI-but-unprofitable even under IG) · universe siblings (SMH ≠ SOXX, SLV ≠ GLD — edges are fund/asset-idiosyncratic; a real cross-asset edge needs a 2-symbol PAIRS strategy, which violates single-ticker) · sample-uniqueness weights (HURT directional edges — overlap IS the signal) · 64k render limit (SOLVED by multi-file render).

**Honesty discipline:** Records go stale (OOS window grows → re-validate before trusting). In-sample Calmar HIDES temporal fragility — run early/late-Sharpe decay on real equity (caught UUP 1.85 = front-loaded 2.67→0.74 STALE vs GLD/IWM strengthening). Multiple-testing haircuts (DSR/Harvey-Liu) are NECESSARY NOT SUFFICIENT — permute + replication are orthogonal gates that catch fund/path-specificity (SLV, QQQ pass haircut but die to permute). Each new trial RAISES every co-tested name's deflation bar → on static data, manufacturing experiments inflates the bar for ~0 EV. Never crown on LLM judgment (idea-eval barely beats chance) — gate on real QC Calmar.

## Method inventory (axes, labels, levers)

Three module slots carry the search: **bar axis** (how you clock the price path) → **unsupervised label** (what you call "up") → **levers** (reduce / sizer / band / horizons). The edge lives in axis + label; reach for a new one of those before touching features/reducers/sizers. Champions are marked ★. Everything unmarked is built + leak-safe but lost or is dormant.

### Bar axes — `modules/bar_builder.py` AXES (18; registry order)

| axis | clock | status |
|---|---|---|
| `dollar` | equal traded notional | base (Wang) |
| `tick` | N transactions | base (Wang) |
| `vol` | realized-variance × √vol | base (Wang) |
| `range` | equal % price move | base |
| `logdollar` ★ | log1p(close·vol), tail-compressed info clock | **GLD/SOXX/IWM champion axis** |
| `entropy` | TRAIN-frozen surprise −log(p_bucket) | dormant |
| `imbalance` ★ | signed-dollar directional runs (de Prado) | **UUP champion axis** |
| `tickimb` | sign-only directional runs | dormant |
| `volumeimb` | signed share-volume runs | dormant |
| `fracdiff` | FFD memory-preserving increments (ADF-fit d) | dormant |
| `dc` | directional-change / intrinsic-time reversals | dormant |
| `zcusum` | standardized log-dollar CUSUM event clock | dormant |
| `kyle` | price-impact / illiquidity (\|Δlc\|/√vol) | LOST (no win) |
| `run` | within-run \|Δlc\| trend clock | LOST (GLD 4.55→0.80: silent at turns) |
| `spectral` | dominant-cycle zero-crossings | LOST (UUP 1.30→0.26) |
| `vpin` | BVC soft order-flow toxicity | dormant |
| `jump` | Lee-Mykland significant-jump clock | dormant |
| `volofvol` | bipower vol-of-vol repricings | dormant |

**Axis lever CLOSED.** kyle/run/spectral all built + A/B'd, all lose to the champion axes. The bar axis is asset-intrinsic: `logdollar` = info clock for trend (GLD/SOXX/IWM), `imbalance` = order-flow for UUP. Don't build more axes — sample where the edge *resolves*, not where the trend is.

### Unsupervised labels — `modules/labeler.py` LABELERS (41; FEATURED + 2 BASELINE)

**Champions / wins**
- `ker` ★ — Kaufman efficiency-ratio clean-trend. Load-bearing on GLD; the Sharpe edge.
- `trend_leg` ★ — Wang flagship connected-leg trend SEGMENTATION. GLD champion core (beat ker), IWM standalone win.
- `regime_gmm` ★ — causal-feature GMM regimes. GLD +regime machinery (asset-specific detector).
- `bgm` ★ — Bayesian-GMM regime. UUP + SOXX load-bearing (regime-primary, +52% on SOXX).
- `accel` — trend-acceleration. Worthless solo, won in a GLD 3-way ensemble (same-family).
- `sadf_explosive` — supremum-ADF explosive/bubble regime. Real ORTHOGONAL ADD on UUP regime side.
- `trend_scan` — AFML trend-scanning. SOXX champion component (pre-leak).

**Built, valid, no win (dormant)**
`sharpe_scan` (risk-adj trend) · `sortino_scan` (downside-adj) · `calmar_scan` (drawdown-adj) · `mfe_mae` (excursion asymmetry, gold-specific 2.5) · `revert` (mean-reversion turn; signal on TLT/IWM/EEM, never beats up-drift) · `turn_scan` (extremum-timing reversal) · `perment` (permutation-entropy; SOXX near-miss 1.98) · `ofsc` (order-flow serial-corr) · `bde_cusum` (recursive-CUSUM break) · `changepoint` (Wang mean-shift) · `hurst_persist` (DFA persistence) · `sliced_wasserstein` (optimal-transport tail regime) · `jump_model` (statistical jump-model regimes) · `crash_ahead` (tail target, pair `crashveto`) · `cusum_regime` · `tertile` · `carry` · `kmeans2stage` · `agglomerative` · `dc_trend` · `dc_reversal` · `multi_horizon`.

**Strength ladders** (Wang trend-strength ENSEMBLE path): `tleg_fast/mid/slow` (trend_leg @ H=20/60/150), `ker_fast/mid/slow` (ker @ H=20/60/150).

**Meta / triple-barrier family**: `triple_barrier`, `_tight`, `_meta` (secondary decision model + GATE), `_tight_meta`, `_ae` (autoencoder DR). Meta-labeling weaker than ker on GLD, over-filters UUP — dormant.

**Baselines** (gate comparators, never crowned): `hmm` (forbidden as a detector), `always_long` (buy-hold floor).

**Label frontier MAPPED.** Every distinct forward-window readout is built — slope, efficiency, curvature, magnitude, trail-sign, extremum-timing, risk-adj, regime, tail, ordinal-entropy. None beats the per-asset champion. Two mechanisms only: TREND-MOMENTUM (`ker`/`trend_leg`/`trend_scan`) and REGIME (`bgm`/`regime_gmm`); each labeler fails on the other's asset.

### Levers — `templates/header.py.tmpl` CONFIG (each writes a distinct cell-key suffix)

| lever | values (default) | suffix | verdict |
|---|---|---|---|
| `reduce` ★ | `correlation` (def) · `infogain` · `variance` · `autoencoder` | `_ig` | **infogain VALIDATED** — top-K by mutual-info w/ TRAIN label vs corr's variance. Lifts single trend-shape labels: TLT corr −0.10→+0.49, IWM win +0.665, **GLD 3.47→4.02 (+16%, crowned)**. NOT pure-regime ensembles (UUP no help) / val_auc≈0.5 names (SOXX). Leak-safe (MI TRAIN-only, frozen kept_idx, online-proven preds_match=1). `autoencoder` = plain sklearn-MLP bottleneck (NOT a VAE — torch unavailable in QC), LOST to corr+IG (TLT AE −0.01 < IG +0.49) → dormant. |
| `rebal_band` ★ | float (`0.01`) | `_b` | **GLD `0.03` CROWNED** (4.55→4.71 gross + net-of-cost). Benefit scales with trade freq: GLD(high) gross+net win, SOXX(mid ~0.02) net-only, UUP(low) negligible. Default 0.01 over-trades cost-sensitive crowns. |
| `n_components` | int (`20`) | `_n` | reducer width. GLD `15` was a real step in the arc; not universal. |
| `features` | `base` (def) · `rich` · `termstruct` · `fx` | `_fr`/`_ts`/`_fx` | LOSES everywhere. `rich` (VR Lo-MacKinlay trend-persistence) DISCARD on GLD+IWM (high-MI ≠ profit). `termstruct` = cross-asset log-ratio z-score (`CROSS_ASSET`: VIXY→VIXM vol-curve, HYG→LQD credit-spread) — VIXY DISCARD, HYG degenerate. Capabilities permanent, no edge in reachable universe. |
| `horizons` | None=daily (def) · bar-list | `_hz` | INTRADAY-holding lever. No new edge — GLD intraday 1.39 ≪ daily 3.47; QQQ/IWM intraday fail Bonferroni. Gold edge is daily. |
| `permute_labels` | `0` (def) · `1` | `_perm` | **honesty harness, not an edge** (see loop). Shuffles ONLY TRAIN labels; a real edge COLLAPSES (UUP 1.30→−0.08, GLD 3.22→1.27, IWM +0.665→+0.14). Run on every KEEP. |

### Sizers — `_size()` in infer/portfolio templates

`cdf_overlay` ★ (default; de Prado CDF bet × causal inverse-vol overlay) · `dd_overlay` ★ (drawdown-aware throttle — **GLD-specific**, suits slow gold drawdowns, hurts semis' V-recoveries) · `cdf_plain` · `binary` · `ramp` · `crashveto` (tail veto, pair `crash_ahead`) · `longshort` · `ls_cdf` · `ls_overlay` (up-drifters reject shorts).

## Honesty infra (all built)

Seven independent overfitting/robustness lenses. A KEEP must survive the per-round gates; the book is audited by all seven.

- **Deflated Sharpe / deflation** (`honest_audit.py`, `deflated_audit.py`) — the max of N trials is upward-biased; an edge must clear best-of-N noise. Run on all logged trials; each new trial RAISES every co-tested name's bar. `always_long` baselines carry no selection bias.
- **Permute control** — re-run the KEEP with `permute_labels:true` (TRAIN labels only, leak-safe). A real edge collapses toward buy-hold. Decisive real-vs-artifact test (caught SPY/SLV/QQQ fakes; UUP 1.30→−0.08 = pure alpha).
- **Decay** (`champion_series.py`) — early→late Sharpe + Page-Hinkley/CUSUM on the real OOS equity curve. Half-window read is reliable; Page-Hinkley over-sensitive at defaults. Flags front-loaded/softening edges (UUP STALE 2.67→0.74; GLD/IWM HOLDING).
- **E-value monitor** (`evalue_oos`, native in infer.py) — anytime-valid e-process (Ville), peeking-robust, re-validations MULTIPLY in. Frequency-invariant (daily≈weekly). Tests profitability (mean>0), bounded-bet-conservative → a liveness/DECAY monitor, NOT a significance/Calmar test. **The standing gate.** Supersedes peeking-invalidated DSR/p-value re-checks for ongoing monitoring.
- **PBO-via-CSCV** (`pbo_gld.py`) — gold-standard config-selection overfit probe (1000 partitions). GLD PBO 0.581 = labeler SELECTION is Sharpe-overfit (all configs positive OOS; ker is load-bearing). Resolved by Calmar ablation: regime_gmm/dd_overlay are Sharpe-neutral but Calmar-positive → validate ablations on the DEPLOYED objective (Calmar), not a proxy (Sharpe).
- **Cost-stress** (`cost_stress.py`) — pure-replay infer with explicit slippage (pipeline default = optimistic). Book holds at 5bp (GLD 4.02→3.43, UUP 1.85→1.54, IWM 0.665→0.61). Dead-band knob (`rebal_band`) cuts cost drag; benefit scales with trade frequency.
- **Harvey-Liu haircut** (`harvey_liu_haircut.py`) — independent multiple-testing method (Bonferroni/BH on each Sharpe t-stat). Cross-checks DSR; the two can disagree (disagreement IS the finding). Necessary-not-sufficient — permute + replication catch what they cannot.

Standing facts: nothing survives Holm-Bonferroni across the full ~500+-trial session burden; per-round Bonferroni does not capture cumulative burden. Weight conviction by DSR, not raw Calmar. Series cached in `results/series_cache.json` for cheap re-runs.

## Frontier — what is exhausted, what re-opens

The single-ticker × fixed-universe space is COMPREHENSIVELY explored across method, mechanism-class, and holding-horizon. Static OOS data + 5-min ticks ⇒ no new information per tick; further config/universe grinding only inflates the deflation bar.

- **Method space SATURATED.** Trend & regime readout spaces well-covered (segmentation/efficiency/slope/curvature/drawdown/vol-ratio/distributional/persistence/ordinal-entropy all built). Wins come ONLY from a better CORE labeler in the edge's own mechanism, an orthogonal ADD, or a different module slot — swapping a mechanism's implementation always loses, a same-family 3rd label always dilutes (see Governing rules). Custom bar-AXIS lever closed. Recurring negative: label-relevance (val_auc/MI) ≠ profit-relevance (Calmar).
- **Vol mechanism class REACHABLE but Calmar-INCOMPATIBLE.** VIXY probed from every angle (short-carry, long-timing, mechanism-matched sadf, term-structure features). Carry is REAL (positive expectancy) but the short-vol CRASH TAIL (fat left tail, DA 55–97) makes it fundamentally Calmar-incompatible; vol spikes are EXOGENOUS/shock-driven (val_auc ~0.5 from price/volume alone — needs VIX term structure / options flow).
- **Cross-asset-curve features (`features=termstruct`) — TWO failure modes spanning the reachable universe.** Cross-asset FEATURES (VIXY→VIXM, HYG→LQD), single-ticker-compliant (not a traded pair). (1) crash-prone names (vol) = Calmar-INCOMPATIBLE (fat tail; IG lifts val_auc 0.56→0.61 but Calmar stays ~0); (2) clean drifters (HYG credit spread) = buy-hold-OPTIMAL (carry>timing; little drawdown to cut). The sweet spot (predictable signal + benign tail + timing-beats-carry) doesn't exist in the reachable single-ticker universe. Capability is permanent.

**Re-openers (each needs a NEW INPUT):**
1. **New DATA modality** (highest leverage) — options-IV/skew, positioning (COT, ETF flows), macro-release surprises, credit spreads, VIX term structure. New information, not new tuning. *(Proprietary alt-data requires the user.)*
2. **Cross-asset PAIRS** — relaxes single-ticker; the one place a real relative-value edge can live. *(Requires user authorization.)*
3. **Intraday holding** — `CONFIG['horizons']` lever built; minute bars but ~daily holding probed (no intraday edge: GLD/QQQ/IWM all fail Bonferroni/deflation). A different regime, mostly unsearched.
4. **Regime change / real-time decay** — the only thing that changes on current inputs is the OOS window growing. `evalue_oos` is the standing monitor.

## Fallback behavior (only if genuinely out of new methods)

The honesty stack is complete; the bottleneck is self-deception (now armored), not throughput. The loop is CURRENTLY LIVE (see Active exploration below) — this is only the fallback if the new-method queue is truly empty AND no lead has positive EV:

- **Monitor** the deployed book's `evalue_oos` liveness/decay (re-validations multiply in). Re-validate + act when it flags.
- **Don't grind deflation-dead names with config permutations** — that inflates the deflation bar for ~0 edge. (Trying a genuinely-NEW axis/label on the weakest name is NOT this — it can find structure price-only methods missed.)
- A truly new EDGE-CLASS still needs a new input (alt-data / pairs). But new METHODS in the priority modules keep surfacing real edges — keep building them.
## Active exploration (2026-06-04 — LOOP IS LIVE, not at terminus)

Per user directive the loop is RUNNING, **weakest-ETF-each-round**, priority on custom axes + unsupervised labels, **also exploring other modules**, gating hard, leak-safe. The earlier "terminus" is superseded — literature-mined new methods keep surfacing real edges.

**Method pipeline.** A novel-method workflow (ideate→adversarial-vet→implement, 46 candidates) delivers leak-audited drop-in code. Built + leak-test PASS so far: `volofvol` (★2nd-order vol-of-vol bipower clock), `wavelet` (à-trous Haar multi-scale clock), `amihud` (Amihud illiquidity clock), `transfer_entropy_dir` (Schreiber nonlinear directed info-flow label). Queue: `ddonset` (drawdown-onset clock — literal "sample where the edge resolves"), `lzc` (Lempel-Ziv complexity clock), `visgraph` (HVG time-irreversibility label).

**Races so far (all DISCARD — new axes lose to champion clocks, as expected):** QQQ volofvol (buy-hold-optimal, val_auc 0.97/Calmar 0.69 = predictable-not-profitable); UUP volofvol (0.51 < imbalance 1.30); GLD wavelet (1.35 < logdollar 4.02). TLT (weakest) amihud + transfer_entropy_dir = in flight — the real question on these is **val_auc > 0.6** (does a genuinely-new method find structure where price-only/linear methods got ~0.5?).

**Deep leak hunt (running).** Adversarial 6-locus hunt, every finding verified EMPIRICALLY (append-OOS-invariance / leak test / render-grep), focused on the new axes — because code audits MISS leaks (the prior 13-agent audit missed the logdollar leak). Backstop before trusting any race.

**Infra fixed:** round counter (was frozen at 131 = KEEP-only → real per-round count via `_round_count()`); dashboard readability pass (WCAG contrast, status-first layout, plain-English copy).
