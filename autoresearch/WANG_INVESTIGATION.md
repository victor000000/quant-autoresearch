# Deep Investigation: What Wang (王一鸣 / "uni 的量化日记") Actually Said

Date: 2026-06-04. Method: 5 parallel agents mining 16 Chinese-language Wang video transcripts
(`uni_yt/*.large-v3.txt`, Whisper) + `qa_doc/wang_qa.tex` + the course PDF. Every claim grounded in
the literal transcript (BV/line cited in the agent reports). Wang = co-founder of an overseas fund
("DCP", >US$1.2B AUM), ex-Merrill/Knight, ~20yr quant.

## 1. Wang's core thesis (what he actually says)

- **The edge lives in the AXIS.** Asked point-blank "where should an individual focus — axis, label, or
  model?" he answers unambiguously: **the custom resampled axis (自定义轴)**. "你能降多少这个峰值，就代表你能
  减少多大的损失" — how much you pull the return-distribution *peak* (kurtosis) down is exactly how much loss
  you cut. Time bars have non-uniform information density + no statistical properties → capped ≈ Calmar 2
  ("国内标准水平", because 99% of institutions use time bars). Resampled bars (dollar/volume/price-action) →
  near-IID/Gaussian diffs → fewer chop bars → the fat tails (where trend money is) survive.
- **Push complexity to the FRONT** (sampling + labeling) so feature engineering stays simple/mathematical.
- **7-step pipeline:** data → custom axis/resample → trend label → feature engineering → dim-reduction →
  model train+backtest+**combine** → inference/live. The modules are *coupled* (协同); skip one and it craters.
- **Discriminative models only** (consistency + reproducibility); never LLM/generative for the decision.

## 2. Where WE are faithful (Wang validates our choices)

- ✅ Custom non-time axis as the core edge (we have dollar/tick/vol/range/logdollar/imbalance/fracdiff/dc/
  spectral/vpin/jump — a superset of his named axes).
- ✅ Unsupervised **look-ahead** trend labels as *target only* — he confirms the label legitimately uses
  future returns (log-returns); it's the label, not a feature. Not a leak.
- ✅ **Tree models (XGB/LGB/CatBoost), NOT raw LSTM/Transformer** on tabular — he is emphatic. Vindicates
  our XGB-only finding (and the ExtraTrees-platform-block is moot — he'd still pick boosting).
- ✅ **Single-ticker ETFs, not single stocks** ("个股不具备统计特性"). Models are **asset-intrinsic** ("专用性");
  don't force cross-asset transfer — validates our "edges are asset-intrinsic" + no-cross-ticker-ensembling.
- ✅ **A/B one module at a time, hold the pipeline fixed, look at the result; don't pre-judge** — his exact
  label-evaluation protocol == our A/B-vs-champion.
- ✅ **Fit-on-TRAIN-only, freeze the transform, replay online** — his exact leak discipline (== our audited
  infer/verify design). PCA fit per-walk-forward-step = leak; fit on train, store, reapply.
- ✅ **Rolling/causal standardization of the axis threshold is MANDATORY.** ⚠️ Our `material-logdollar-leak`
  (threshold scaled by full-series `np.sum(valid)`) was a *deviation* from his causal-rolling spec. Following
  Wang exactly would have prevented it.
- ✅ Diversify decorrelated models for scale (vs single-model perfection) — supports keeping weak names in the
  book for decorrelation.

**Our honesty gates (Deflated-Sharpe / Bonferroni / PBO / permuted-label / formal leak audits) are an
ENHANCEMENT BEYOND Wang** — he never operationalizes multiple-testing statistically. He relies on near-IID
bars + baseline-benchmarking + "change one thing, look at the result." So we are *more* rigorous on honesty,
not contradicted. (Consistent with our efficiency-review: DSR/PBO is the #1 gap — and it's a gap in Wang too.)

## 3. The GAPS — what Wang does that we DON'T (ranked by leverage)

### A. THE WALL-BREAKER — "arbitrage = trend on the SPREAD" (cross-sectional)
Wang's single most strategically important idea for us. He **turns arbitrage into a trend problem on a spread**,
then runs the spread through the *exact same single-asset pipeline*:
- Form a pair (economic prior OR unsupervised DBSCAN/hierarchical clustering; features = vol + autocorr;
  distance = DTW / correlation). Build spread `A − β·B` (β from physical prior, or **back-solved statistically**
  for financial assets — he explicitly sanctions this).
- **Validity gate = does the spread have positive autocorrelation?** (DW / ADF / ACF). He *rejects* relying on
  cointegration (linear, fragile tails). A spread can carry trend structure even when **both legs are
  val_auc≈0.5 individually** — exactly our no-structure wall.
- Resample the spread with **price-action custom bars** (no volume needed) → our existing trend labels →
  XGBoost → sizing. **Reuses ~90% of our code; needs NO new data** (we have 20 ETFs' minute OHLCV).
- Also: **two-basket index arbitrage** (index-ize two ETF baskets, trade the basket spread as trend), and a
  **CS-gate** (broad-ETF trend model = *when* to trade; cross-sectional rank model = *which* names).
- Claim: glass/soda-ash spread, 24mo, Calmar 4.58 (2 models). Result: **earns RELATIVE money — the input
  dimension our single-name pipeline structurally lacks.**
- ⚠️ Relaxes our self-imposed "single-ticker only" rule → needs user authorization.

### B. ENSEMBLE a trend-strength label SWEEP (his central regime-adaptation trick)
His entire RB lift (Calmar 3.3→5.63) is **ensembling ~5 models trained on different difference-order trend
labels (5阶…9阶)**, combined. "标签没有好坏之分" — stop hunting one holy-grail label; ensemble the trend-strength
sweep, which **implicitly adapts to regime**. **We crown ONE labeler per ticker; he deploys the combination.**
2–3 models is enough (tested up to 100 → overfits). Biggest *architectural* divergence.

### C. Feature-level levers (cheap, no new data, his live-used methods)
1. **Fractional-diff FEATURES (d≈0.7–0.8)** — he uses "大量" (heavily) in live; integer-diff "throws away
   history." We have a fracdiff *axis* but should add fracdiff *features* + sweep `d`. Plus **ratio (商) features**.
2. **Sample-entropy FEATURES** (no-distribution, rolling, range-scaled tolerance — NOT std-scaled). His A/B:
   swapping 30 int-diff → 30 sample-entropy features lifted **Calmar 3→4.5, lower MaxDD**. (Distinct from our
   `perment` permutation-entropy *labeler* — he doesn't use permutation entropy at all.)
3. **Information Gain (IG) feature selection targeting the LABEL** (not raw return, not ICIR) — rank features
   by IG vs the trend label, top-N dynamic. Our corr-filter(20) is correlation-based.
4. **Per-asset-class memory tuning** — longer fractional memory for commodities (GLD), shorter for equities.

### D. Method/module levers
5. **Change-point labeler** — his *preferred* labeler. He **rejects triple-barrier** (AFML book 1) and finds
   AFML book-2's OLS-t **trend_scan "not particularly good"** — "还不如直接用change point." (Our `trend_leg`
   already beat `trend_scan` on GLD 3.47 vs 2.51 — consistent. An explicit change-point labeler is untried.)
6. **VMD + NRBO frequency features** — he **rejects FFT** ("不要用FFT"); prefers **VMD** (variational mode
   decomposition) or CWT+SWT(synchrosqueezing), with **NRBO boundary-optimization** as the anti-look-ahead
   device. ⚠️ If our `spectral` axis is FFT/periodogram-based and computed globally, it is BOTH the method he
   warns against AND a latent **non-causal boundary leak** (he admits these transforms are non-causal). Leak-audit item.
7. **VAE / LSTM-autoencoder non-linear DR** — concat PCA-dims + AE-dims (his explicit trick) for ensemble
   diversity; pick bottleneck by reconstruction-MSE sweep. We have only linear corr-filter. (He's SILENT on
   frozen-encoder leak-safety — that discipline is on us: fit-on-train-only + freeze + causal apply.)
8. **Volatility forecaster (FIGARCH via `arch`, or GARCH+tree hybrid)** → target = next-bar 20-day realized
   vol, benchmarked vs unconditional vol by MSE/RMSE/MAE → feed sizing as a **rolling-threshold gate** ("外挂").
   We have NO vol-forecasting model. Vol-regime-conditioned sizing is under-explored.
9. **Regression / multi-class-binning labels + reg×clf ensemble** — we are binary-only.
10. **Peak-RAISING axis → mean-reversion framing** for no-edge up-drifters (axis design *chooses* the strategy
    class: near-Gaussian→trend, raised-peak→mean-reversion, bimodal→event-driven).

## 4. Honest caveats on Wang's claims

- All numbers are **self-reported, single-example, no OOS replication shown, vendor/course-recruiting context.**
- His headline results are on **Chinese commodities (RB rebar) + CSI300 + crypto** — markets with **stronger,
  more persistent trend autocorrelation** than US ETFs. Our own legacy verdict: "US equity/credit/treasury
  ETFs don't have this structure at minute granularity; Wang's results may be more achievable on Chinese
  commodities/HS300." His RB Calmar 5.63 ≠ portable to TLT.
- Trend vs chop **cannot** be separated at the current bar (boundary problem) — only chop *frequency* reduced.
- He has **no multiple-testing / deflation discipline** — our gates are the honest upgrade.
- Philosophy gap: he prefers **fully-local native Python**; we run on QuantConnect (not a correctness issue).

## 5. Recommended next experiments (ranked)

1. **[WALL-BREAKER, needs user OK — relaxes single-ticker] Spread-as-trend.** Compute DW/ADF/ACF autocorrelation
   on all C(20,2)=190 ETF spreads (β back-solved via OLS/Kalman), rank by autocorr strength (a few lines, fast
   falsifiable test of whether ANY spread structure exists in our universe). Feed survivors through our existing
   custom-bar → trend-label → XGBoost → sizing pipeline as synthetic single assets. Permute-control like UUP.
   Natural first pairs: GLD-SLV, GLD-UUP, UUP-TLT, XLE-SPY. **This directly attacks the no-structure wall by
   changing the INPUT, with no new data.**
2. **[in-rule] Trend-strength label ENSEMBLE** — sweep one labeler's strength param + combine 2–3 models per
   ticker (his core trick), A/B vs the single-label champion on GLD/UUP.
3. **[in-rule] Sample-entropy features + IG selection** — clean Calmar lift in his A/B, no new data.
4. **[in-rule] Fractional-diff features + `d` sweep (0.5–0.9)**; per-asset-class memory.
5. **[audit] Check whether our `spectral` axis is FFT-based + computed causally** — if FFT/global, it's the
   method Wang rejects + a latent boundary leak. Consider VMD features.
6. **[in-rule] Change-point labeler** (his preferred); **VAE non-linear DR** (frozen-encoder, leak-safe).
7. **[bigger build] Vol forecaster → rolling-threshold sizing gate.**

**Bottom line:** We faithfully implement Wang's steps 1–6 and his model choice, and we are MORE honest than he
is (deflation/permute/leak gates he lacks). The two things we are NOT doing that he'd consider essential are
**(a) ensembling a trend-strength label sweep per ticker** and **(b) the cross-sectional / spread path** — and
(b) is the concrete, no-new-data escape from our single-ticker no-structure convergence wall.
