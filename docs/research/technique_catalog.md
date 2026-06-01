# Technique Catalog — Ranked, De-duplicated, Module-Mapped

Mined from 90 raw findings (AFML, MLAM, Causal Factor Investing, Wang course slides/transcripts/QA,
plus 2018-2026 web/arXiv/SSRN sources). Objective: raise **REAL out-of-sample Calmar (CAGR/MaxDD)**
on the QuantConnect ETF pipeline. Robustness and drawdown control beat in-sample accuracy.

Module legend: ⓪ strategy-type router · ① custom axis · ② labeling · ③ feature engineering ·
④ dim-reduction · ⑤ training/validation · ⑥ calibration · ⑦ ensemble · ⑧ sizing/risk ·
`vol` volatility block · `freq` frequency-domain · `arb` arbitrage/spread.

**Library availability (all confirmed on QC Cloud):** numpy, pandas, scipy 1.13.1, scikit-learn 1.6.1,
statsmodels 0.14.6, xgboost 3.0, lightgbm 4.6, arch 8.0 (GARCH/EGARCH), hmmlearn, torch 2.8, numba 0.61.
Anything below that says "needs new lib" names which of these to wire into the QC template — none are blockers.

---

## Current harness baseline (what already exists — do NOT re-implement)

| Module | What the harness has today |
|---|---|
| ① axis | `bar_builder.py`: dollar, tick, vol, range, logdollar, entropy bars. Threshold = `total/target_bars` (**static**, swept generically, not per-asset-selected). |
| ② labels | `labeler.py`: kmeans2stage, carry, tertile, bgm, agglomerative, triple_barrier, multi_horizon + HMM (baseline only). All gated on TRAIN/VAL label balance (0.2,0.8). |
| ③ features | `features.py`: 80 feats (momentum 1-20, z-momentum, rolling std/mean/kurt, vol-ratio, price-vs-MA, sample entropy). |
| ④ dim reduce | `trainer.reduce_dims`: variance + |corr|>0.90 filter + top-K-by-variance (K=20). |
| ⑤ train | XGBoost depth-3, lr 0.03, L1/L2 reg, `scale_pos_weight`, early stop. Plain train/val/test split (**no purged CV**). |
| ⑥ calibrate | `calibrator.py`: isotonic / Platt / none. |
| ⑦ ensemble | `ensembler.py`: multi-seed mean (n=1 or 5). |
| ⑧ size/risk | `consensus.py` + `trainer.realistic_cstats`: fixed threshold 0.45, linear ramp `(p-0.45)*200`, consensus `min(p)>0.5 & avg(p)>0.55`. **No vol-targeting, no meta-label gate, no prob→size CDF map.** |
| ⓪ router | **Does not exist.** Every asset runs the identical trend/TS pipeline; axes+labelers swept generically. |

The harness's stated weak spots (per README): real per-asset Calmar ranges 0.25 (TLT) to 2.25 (XLE);
bonds/EM under-perform — consistent with running a trend pipeline on mean-reverting assets (no ⓪ router).

---

## TOP 12 TO IMPLEMENT FIRST

Ranked by (expected real-Calmar impact × implementability), with ⓪/①/② up-weighted per the user.
"New lib" column = what to add to the QC template; all confirmed available.

| # | Technique | Module | Pri | New lib | Why it lifts REAL Calmar (1-line) |
|---|---|---|---|---|---|
| 1 | **Strategy-type router (5-class) via TRAIN-only ACF/DW + Hurst + VR + ADF + ARCH-LM** | ⓪ | 5 | statsmodels | Stops fitting a trend model to a mean-reverting/random asset — the #1 cause of OOS blow-ups; gates axis+label+sizing per asset. |
| 2 | **Statistical Jump Model (penalized k-means) regime → risk-on/off routing** | ⓪/② | 5 | numpy/numba | Jump penalty enforces persistent regimes; OOS S&P500 MaxDD -55%→-27%, Calmar 0.16→0.33; kills regime-churn turnover the current KMeans labels cause. |
| 3 | **Meta-labeling secondary gate (trade / no-trade) on the primary signal** | ⑧ | 5 | xgboost (have) | Suppresses low-conviction losers → precision↑; documented MaxDD -31.7%→-7.4%, Sharpe 0.45→1.53. Direct Calmar-denominator lever. |
| 4 | **Conditional vol-targeting / inverse-vol position scaling overlay** | ⑧/vol | 5 | arch (EWMA needs none) | Cuts notional into vol spikes where drawdowns cluster; structural, low-parameter, generalizes OOS; +15-50% Calmar in cited studies. |
| 5 | **Triple-barrier labels with volatility-scaled barriers + |t|/conviction weighting** | ② | 5 | numpy/statsmodels | Label embeds the stop/PT/holding the live trade uses → bounded per-trade loss; balanced classes; risk-aligned target. |
| 6 | **Trend-leg labeling (label whole leg, ignore pullbacks; binary, no 3-class)** | ② | 5 | numpy | Denoised target → model holds through pullbacks instead of flipping; cuts churn/cost bleed (Wang's flagship). |
| 7 | **Probability→bet-size map (de Prado CDF) × vol-target, with active-bet averaging + discretization** | ⑧ | 5 | scipy (have) | Small on weak/uncertain signals, large only on conviction; shrinks equity-curve variance and turnover cost. |
| 8 | **Per-asset axis selection by kurtosis/peak (closest-to-Gaussian diff at fixed bar-count)** | ⓪/① | 4 | scipy | Lower peak = fewer whipsaw bars; near-IID bars make CV/calibration valid → smoother equity, less DD. |
| 9 | **Dynamic (rolling) dollar/vol-bar threshold (recomputed monthly from trailing turnover)** | ① | 4 | numpy | Keeps bar frequency constant as turnover drifts over 17 yrs; prevents silent under/over-sampling that breaks an old static threshold. |
| 10 | **Combinatorial Purged CV + embargo (offline) → distribution of OOS Calmar; select worst-path** | ⑤ | 4 | sklearn (have) | Removes label-overlap leakage that inflates the current train/val split; pick configs whose WORST plausible Calmar is acceptable. |
| 11 | **Deflated Sharpe Ratio gate (trials-count-aware acceptance)** | ⑤/arb | 4 | scipy (have) | Prevents deploying a multiple-testing fluke — dominant cause of backtest→live Calmar collapse. |
| 12 | **HAR-RV + downside-semivariance causal vol features (+ vol-of-vol)** | ③/vol | 4 | numpy/statsmodels | 3-6 low-DoF, gold-standard vol features; downside-vol leads drawdowns → feeds the vol-target/de-risk gate without overfitting. |

Notes on ordering: ⓪/① items (1, 2, 8, 9) are weighted up because routing+sampling sit upstream of
everything and protect every downstream gain. ⑧ items (3, 4, 7) are the most directly Calmar-denominator-
reducing and are cheap. ⑤ items (10, 11) are offline guards that make the other gains real, not illusory.

---

## ⓪ — Strategy-type router (NEW MODULE; highest leverage per user)

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **First-principles 5-class router** (trend / mean-reversion / cross-sectional / arbitrage / volatility) | Wang transcripts q922tUTWmCY, uVnOeOcoivw; LdP CFI §6.3 | Gate each asset to the class whose mechanism is actually present → in-sample edge is a real economic edge that survives OOS. | pandas/numpy + statsmodels; cross-sectional/arb/vol classes need a universe/pairs/options (not in 7-ETF minute set) → first version routes trend vs MR vs stand-aside. | No router today; pipeline hard-codes trend/TS for all assets. | 5 |
| **ACF / Durbin-Watson trend-tradability gate** (positive return autocorrelation ⇒ trend-tradable; also on a constructed spread) | Wang q922tUTWmCY l.128-142; AFML | Positive ACF is the necessary condition for a real trend edge; TRAIN-only gate culls range-bound names where XGBoost memorizes noise. | statsmodels acf + DW on TRAIN returns; ms. | Harness never tests autocorrelation to decide tradability. | 5 |
| **Statistical Jump Model regime → 0/1 risk-on/off** | Shu/Mulvey arXiv:2402.05272; repo jump-models | Persistence penalty λ → ~0.8 switches/yr vs HMM ~8.5; OOS MaxDD -55%→-27%, Calmar 0.16→0.33; robust to 5-10d delay. | ~80-line coordinate-descent (centroid update + DP over 2-state transition-penalty); numpy + numba; sklearn KMeans init. **Reimplement in-process** (pip `jumpmodels` may not be whitelisted). | KMeans labels have no temporal regularization → flip noisily; JM is the non-HMM persistent detector. | 5 |
| **Differenced-distribution + ACF profiling as FIRST step** | Wang slides §2 | Route axis+label by the asset's actual normality/memory signature; low-DoF, overfit-resistant; culls unmodelable assets. | scipy.stats.kurtosis/jarque_bera, statsmodels acf/pacf/adfuller on resampled minutes. | No upstream profiling today. | 5 |
| **Hurst / variance-ratio memory test → trend vs MR polarity** | Wang slides §2,§8; Lo-MacKinlay 1988 | Mislabeling an MR asset as trend is the classic in-sample-good / OOS-loss failure; correct routing removes it. | numpy R/S + DFA Hurst; ~20-line Lo-MacKinlay VR with het-robust z. | No memory-based label-polarity decision. | 5 |
| **Per-asset axis selection by kurtosis/peak diagnostic** | Wang uVnOeOcoivw l.460-775; QA Q1 | Pick the axis whose differenced bar-returns are closest to Gaussian at a fixed bar-count → fewer whipsaw bars, near-IID inputs. | numpy kurtosis/peak_frac sweep on TRAIN bars; pure numpy. | Harness sweeps axes generically; no per-asset selection rule. | 4 |
| **Hurst regime routing with neutral dead-zone (0.45-0.55 = stand aside)** | Macrosynergy 2023; Sanchez-Granero FI 2022 | Dead-zone forces flat in random-walk regime where trend+MR both bleed via whipsaw — biggest OOS Calmar destroyer; bias-corrected/ensembled H avoids false-trend on short windows. | numpy (polyfit log-var, adjusted R/S, DFA); ≥~100 pts ⇒ use daily windows. | Adds an explicit DON'T-TRADE regime the always-on pipeline lacks. | 4 |
| **Rolling/wild-bootstrap Automatic Variance Ratio (AMH) gate** | Choi 1999; Charles-Darne-Kim; JRFM 2019 | Keeps capital out of efficient (random) sub-periods that are negative-expectancy after costs; wild bootstrap kills small-sample false positives. | ~30-line numpy VR + automatic-q + few-hundred-draw wild bootstrap on rolling daily windows. | Harness treats strategy-type as static across the backtest. | 4 |
| **Permutation entropy + Jensen-Shannon complexity forecastability gate** | Bandt-Pompe; Zunino-Rosso; arXiv:2502.09079; arXiv:1404.6823 | Model-agnostic "is this even predictable?" filter; high PE ⇒ no model adds value ⇒ don't risk capital. | ~15-line numpy ordinal patterns + scipy.spatial.distance.jensenshannon; needs ~d!·30 pts. | New information-theoretic gate; no equivalent today. | 4 |
| **ARCH-LM volatility-clustering profiling → route to vol class / skip GARCH if absent** | Engle 1982; Engle-Ng 1993; statsmodels het_arch | Deploy vol machinery only where the clustering mechanism is real; avoids overfit/turnover on weak-ARCH assets. | statsmodels het_arch one call; arch persistence; per-asset one-time. | No vol-strategy branch nor pre-test gate. | 3 |
| **Treat routing as a falsifiable hypothesis (re-validate, distrust as permanent)** | LdP CFI §6.3, §3.3 | Distributional properties (stationarity/serial-indep/linearity) are unstable; re-checking keeps the strategy inside the regime where its assumptions hold. | ADF/VR/Hurst/ARCH-LM recomputed on rolling windows. | One-shot KMeans labeling doesn't re-validate the route. | 3 |
| **Causal-mechanism / regime-health live circuit-breaker** | LdP CFI §8.3, §6.4.2.1 | De-risk when the mechanism (cov(feature,vol-state), Hurst, ARCH) drifts out of the training envelope — leads the drawdown vs lagging structural-break tests. | rolling cov/Hurst/VR/ARCH per bar; gate size→0 outside envelope. | Turns ⓪ profiling into a forward-looking circuit-breaker feeding ⑧. | 4 |
| **Prevailing-mean OOS-R² admission gate** | Rapach-Zhou MLAM Ch.1; Goyal-Welch 2008 | Admit a signal/feature only if recursive OOS-R² beats the do-nothing mean — stops sizing into noise. | numpy recursive running mean + cumulative SE during warm-up. | No do-nothing benchmark gate today. | 3 |
| **Unsupervised cross-asset pair/cluster discovery (Agglom/DBSCAN + DTW), economic-prior-constrained** | Wang q922tUTWmCY l.596-659 | Add a pair only if it survives distance/cluster AND an economic-prior constraint → filters spurious in-sample correlations (DBSCAN-overfit failure mode). | sklearn AgglomerativeClustering/DBSCAN + DTW/corr distance on TRAIN; small for 7 ETFs. | Harness clusters WITHIN an asset (as labeler), never ACROSS assets. | 3 |

---

## ① — Custom axis / sampling

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Dynamic (rolling) dollar-bar threshold** = trailing-12mo avg daily $vol / target-bars-per-day, recomputed monthly | LdP AFML Ch.2; H&T study | Constant-information sampling as turnover drifts over 17 yrs; prevents the regime-drift under/over-sampling that silently breaks a static-threshold model. More-Gaussian, lower-autocorr inputs → steadier signal, lower DD. | numpy rolling threshold inside existing `DollarBarBuilder`; ~20 lines. | Harness uses a single static `total/target_bars` threshold. | 4 |
| **Volatility-clock axis** (emit after cumulative realized vol reaches threshold) | Wang slides §3; AFML | Equalizes risk per bar → trades more in calm, throttled in turbulence (structural DD brake); stabilizes feature scaling/label balance across regimes. | numpy cumulative-RV thresholding in Builder interface; threshold fit on TRAIN. | Has a `vol` bar (variance-weighted) but not a dedicated equal-realized-vol clock. | 4 |
| **Normality-optimized custom axis** (tune emission rule on TRAIN to minimize Jarque-Bera / excess-kurtosis of bar diffs) | Wang slides §3 | Gaussian-diff bars make XGBoost splits, calibration, vol-forecasting and sizing all valid OOS → honest probabilities → lower realized DD. | scipy.stats.jarque_bera/kurtosis as objective; sweep candidate rules on TRAIN. | Unifying axis-selection criterion absent today. | 5 |
| **CUSUM event filter** (sample only when symmetric CUSUM of returns exceeds h = rolling vol) feeding labels | Grądzki FI 2025; AFML Ch.2 §2.5.2.1 | Trades structural deviations not noise → fewer cost-driven losers, higher label precision; cited positive net-of-cost OOS outperformance vs time bars. | ~15-line stateful numpy; h = multiple of rolling vol (adaptive). | v5 had dollar+CUSUM; not generalized as an event sampler with adaptive h. | 5 |
| **Native VolumeRenko / ClassicRenko / ClassicRange consolidators** | QC docs; AFML | Same constant-information mechanism with zero custom-code risk (maintained LEAN consolidators) → lower bug/lookahead surface. | native LEAN `VolumeRenkoConsolidator` / `ClassicRenkoConsolidator`; reset N from trailing volume to avoid static drift. | Harness builds bars in custom numpy; native consolidators are an alternative low-risk path. | 4 |
| **Tick/volume/dollar imbalance bars (EWMA-adaptive threshold, tick rule)** | LdP AFML Ch.2; mlfinlab; Grądzki 2025 | Emits a decision point as one-sided flow builds → earlier regime-change detection; lets sizing cut exposure ahead of trend exhaustion. | numba accumulation; **caveat**: QC minute bars aren't signed prints, tick rule approximated on minute close-to-close → experimental, must beat plain dollar bars before keeping. | New order-flow axis; current set has none. | 3 |
| **Range/Renko (constant price-move) sampling, brick = k·ATR** | QC docs; Grądzki 2025 | Filters flat chop (no bar in quiet periods → fewer whipsaw losses); homogenizes return scale. | native consolidators or numpy; recompute brick from trailing ATR. | Harness has `range` bar (fixed %); ATR-adaptive brick is the refinement. | 4 |

---

## ② — Labeling

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Trend-leg labeling** (label whole long/short leg, ignore intra-trend pullbacks; binary only, refuse 3-class range) | Wang uVnOeOcoivw l.798-862 | Denoised target → model holds through pullbacks instead of flipping every wiggle → cuts cost bleed/false reversals; 3-class range is unlearnable (model would split on vol instead). | numpy trend-segmentation (PELT/CUSUM-style or sign-of-fracdiff-trend), density param tuned on TRAIN. | None of the 8 labelers implements connected-leg trend labels. | 5 |
| **Triple-barrier with vol-scaled barriers + |t|/conviction sample-weight + low-vol zero-mask** | LdP AFML Ch.3; arXiv:2504.02249; MQL5 Pt.3 | Label embeds stop/PT/holding the live trade uses (bounded loss); |t|-weight down-weights noisy near-zero trends; zero-mask suppresses no-edge regimes. OOS Sharpe 2.62 vs 0.98 for fixed TB; lowest DD duration. | numpy/statsmodels OLS t-stat; xgboost `sample_weight`. Harness has plain triple_barrier — add vol-scaled width + |t| weighting + zero-mask. | TB exists but uses bare label; conviction weighting+zero-mask are new. | 5 |
| **Statistical Jump Model labels (penalized k-means, EWM downside-dev + Sortino features)** | Shu/Mulvey arXiv:2402.05272 | Persistent, low-turnover regimes without HMM assumptions; turnover 44-72% vs HMM 141-290%; cost-aware DD reduction. | penalized-kmeans + DP assignment, numba; daily-aggregated EWM features. | KMeans/agglom labels lack a transition penalty → flicker. | 5 |
| **ML-generated trend labels with tunable label DENSITY** | Wang slides §4 | Density knob controls trade frequency/turnover; per-asset density tuned to trend persistence → class-balanced, cost-aware labels. | trend-segmentation labeler with explicit density param; numpy/sklearn. | New labeler family; density is a deliberate knob vs accident of a fixed barrier. | 5 |
| **Sliced-Wasserstein k-means regime labels (distribution/tail-aware, no HMM)** | Horvath arXiv:2110.11848; Luan-Hamp arXiv:2310.01285 | Compares whole return distributions (tails/skew) → separates calm vs crash regimes Gaussian clustering blurs → earlier de-risk → lower MaxDD. | scipy.stats.wasserstein_distance or sort-diff one-liner + Lloyd; recompute weekly on daily windows. | Replaces Euclidean KMeans geometry with full-distribution metric. | 4 |
| **Bayesian DP-GMM auto-K regime labels (soft membership)** | sklearn BayesianGaussianMixture; arXiv:1805.00306 | Data picks K instead of a brittle fixed K; soft probabilities feed prob-scaled sizing for smoother de-risking; Bayesian shrinkage resists spurious tiny regimes. | sklearn BayesianGaussianMixture (dirichlet_process); refit periodically, predict_proba live. | Harness BGM uses fixed K list {3,4,5}; DP auto-prunes. | 3 |
| **Observation-transform / K diagnostic for label model** (use r_t or [r,|r|] not log(P·V); pick K for label balance) | Wang QA Q2; transcripts | Wrong observation teaches the model VOLUME/VOL regimes instead of DIRECTION (silent objective mismatch); [r,|r|]+K=3 gives balanced direction-true labels that survive retrain. | expose obs transform {r,[r,|r|],[r,r²]} and K as config; select on TRAIN/VAL balance. | Harness fixes label inputs; doesn't sweep the observation transform. | 2 |

---

## ③ — Feature engineering

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Fixed-width fractional differentiation (FFD), minimal d* via ADF** | LdP AFML Ch.5; Wang slides §5 | Stationary AND memory-preserving (returns over-difference, killing signal; raw price is non-stationary) → model maps OOS points to comparable training points → fewer regime-driven DDs. | numpy FFD weights + statsmodels adfuller to pick d* on TRAIN; apply fixed weights forward. | Not in the current 80-feature set. | 5 |
| **HAR-RV causal vol features (RV_day/week/month + ratios) + 1-step RV forecast** | Corsi 2009 | Gold-standard parsimonious RV forecaster; day/week/month ratio is a low-variance vol-expansion regime descriptor; 3-6 low-DoF feats vs dozens of noisy ones → generalizes. | pandas aggregation; optional OLS forecast; strictly causal with lagging. | Current rolling-std features approximate vol clustering only crudely. | 5 |
| **Realized semivariance (good vs bad vol): RS⁻, RS⁺, signed jump, downside ratio** | Barndorff-Nielsen 2010; Patton-Sheppard 2015 | Downside semivariance drives future vol persistence and precedes crashes; rising RS⁻ is an early de-risk trigger that symmetric std misses. | numpy on intraday returns of completed days; negligible runtime. | New asymmetry features; harness treats up/down vol symmetrically. | 4 |
| **Volatility-of-volatility tail-risk feature** | FEDS 2013-54; Baltussen | Markets crash when vol is not just high but UNSTABLE; vol-of-vol spike pre-empts regime-transition DDs the vol level misses. | pure numpy rolling on own returns; no VIX/options data needed. | New feature + de-risk trigger. | 3 |
| **Entropy features (LZ/Kontoyiannis/plug-in + permutation entropy on encoded returns)** | LdP AFML Ch.18; Wang slides §5 | Model-free predictability gauge; low entropy = structured (lean in), high = noise (size down / abstain) → suppresses trading into noise. | numpy LZ match-length + permutation entropy; rolling, cheap. | Harness has sample_entropy only; add permutation + discretized Shannon + use as sizing gate. | 4 |
| **Backdoor/partial-correlation de-confounding of features vs vol/market state** | LdP CFI §4.3.2.2, §6.4.2.1 | A feature that is really a vol/market-regime proxy flips sign when the covariance shifts → time-varying betas → DDs; residualizing on the state variable makes the signal resilient. | OLS residualization (statsmodels/numpy); confounder = rolling RV or SPY minute bars (already subscribed). | Features are raw + corr-filtered only. | 4 |
| **Collider-aware selection: add a feature only if it improves a PURGED OOS metric (never in-sample R²)** | LdP CFI §6.4.2.2, §7.2-7.3 | Collider/spec-search features look great in-sample, reverse sign OOS → "systematic transition from profit to loss"; OOS-gated selection screens them out. | sklearn/numpy; replace any train-R²-improvement logic with purged-OOS-improvement. | corr-filter rewards in-sample fit; this adds an OOS-gated, parsimony-biased step. | 4 |
| **Non-time-axis factors as features (compute indicators on the alt-axis series)** | Wang slides §5 | Axis-native features are more stationary/regime-invariant → keep meaning OOS and add orthogonal information to the ensemble. | compute indicators on alt-axis bars already built; align to trading axis. | Single-axis feature set today. | 3 |
| **Reject/down-weight Granger / lead-lag features unless cross-regime sign-stable** | LdP CFI §5.3 | Lead-lag features driven by a shared latent driver reverse when it shifts → DDs; demote those favoring cross-regime-stable features. | regime-conditional coefficient sign-stability screen across ② regime labels. | corr-filter selects globally; this adds a stability screen. | 3 |
| **Leak-free per-window FFT band-power + spectral entropy** | VMDNet arXiv:2509.15394 | Spectral entropy separates trending (low-freq power, low entropy) vs choppy regimes → gate trend bets; per-trailing-window transform restores causality (whole-series decomposition leaks). | numpy.fft.rfft per trailing window, strided; no PyWavelets needed. | New `freq` regime feature; honest (causal) variant. | 4 |
| **Causal SWT/à-trous wavelet trend-noise SNR** | Shensa 1992; arXiv:2103.03505 | Causal trend-vs-noise SNR favors holding winners (lower turnover) in high SNR, de-size in chop; value is as much in AVOIDING the pervasive leaky-wavelet results. | scipy.signal/numpy convolution cascade (PyWavelets not confirmed); more engineering than FFT. | New; lower priority than FFT variant. | 2 |

---

## ④ — Dimensionality reduction

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Frozen non-linear autoencoder (live-inference-safe)** | Wang slides §6 | Frozen encoder gives identical latent in backtest and live (removes the train/live PCA-drift bug); compression reduces model variance → less overfit → smaller OOS DD. | torch 2.8 small MLP (bottleneck 8-16), encoder frozen, applied causally; few epochs for 5-min budget. | Harness DR is corr-filter only; genuinely new. | 5 |
| **PCA orthogonalization + MDA-under-purged-CV + Kendall-tau(PCA-rank, importance) anti-overfit check** | LdP AFML Ch.8 | Permutation MDA under purged CV is the only OOS-valid importance; tau concordance flags fluke features; orthogonalization removes substitution effects. | sklearn PCA + permutation_importance + scipy weightedtau; MDA at research time. | Stronger, leakage-aware selection than the corr-filter. | 3 |
| **Parsimony 1-SE rule on latent dim / n_clusters / feature count** | LdP CFI §3.4 | Fewer DoF → smaller backtest-vs-live gap → protects against deep DDs; pick smallest config not statistically worse OOS. | numpy 1-SE rule on purged-CV score-vs-complexity curve. | DR/labeling pick components by reconstruction/silhouette, not parsimony-vs-OOS. | 3 |

---

## ⑤ — Training / validation (offline guards that make gains REAL)

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Combinatorial Purged CV + embargo → OOS Calmar distribution; select by worst-path/lower percentile** | LdP AFML Ch.7,12; SSRN 3257497 | Purge label-overlap leakage that inflates the current train/val split; path distribution exposes config fragility → pick params whose WORST Calmar is acceptable. | numpy/itertools + sklearn BaseCrossValidator; **offline** research step, chosen config runs the 5-min backtest. | No purged/embargoed CV today; plain split. | 4 |
| **Purged K-Fold with embargo (minimum viable version of above)** | LdP AFML Ch.7 Snippets 7.1-7.4 | Without purging, CV accuracy is inflated → you select overfit hyperparams that collapse OOS. | PurgedKFold ~40 lines on sklearn KFold. | Corrects label-overlap leakage in any in-sample tuning. | 5 |
| **Sample-weight by uniqueness/concurrency + return attribution + time decay; sequential bootstrap; max_samples=avgU** | LdP AFML Ch.4 | Overlapping triple-barrier labels are non-IID; down-weighting redundancy makes the ensemble diverse, better-calibrated confidence → OOS stability → fewer DD clusters. | numpy concurrency from t1; xgboost/sklearn `sample_weight`; negligible. | Uniform weights today. | 3-4 |
| **AICc for elastic-net λ selection instead of arbitrary K-fold** | Rapach-Zhou MLAM Ch.1; Hurvich-Tsai 1989 | Deterministic, reproducible penalty removes a researcher-freedom overfitting channel; stable λ → fewer regime-driven model flips. | sklearn enet_path + numpy AICc argmin. | Removes time-blind CV splitter for any linear combiner. | 3 |
| **MDA/Shapley as associational SCREENING only, on separate folds from final selection** | LdP CFI §4.3.1, §6.1 | Separating screening from selection avoids re-introducing spec-search; purged permutation MDA gives honest importance (MDI/gain is biased). | sklearn permutation_importance with purged CV (SHAP not whitelisted). | Harness uses XGBoost gain (MDI) implicitly. | 3 |

---

## ⑥ — Calibration

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Isotonic calibrate-THEN-size coupling** (calibrated p is the SIZING input, not just a label threshold) | H&T meta-labeling repo; Wang §10 | Over-confident probs cause over-betting on weak signals → DD; monotone isotonic aligns size with true edge. Calibration is underused unless it feeds the prob→size map. | sklearn CalibratedClassifierCV(isotonic) on a **purged/embargoed** holdout; fit on a separate fold from the meta-model. | Harness has isotonic but uses p only for a 0.45 threshold; couple it to sizing. | 3 |

---

## ⑦ — Ensemble

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Cross-axis model combination** (one model per axis: dollar / vol-clock / price-action → consensus/inverse-var weight) | Wang slides §9 | Models on different axes make decorrelated errors → consensus has lower variance, fewer simultaneous wrong bets → smoother equity, smaller MaxDD even at equal per-model accuracy. Lowest-overfit-risk Calmar lift. | train existing pipeline on 2-3 axes; aggregate via `consensus.py` vote or inverse-variance. | Harness ensembles SEEDS on one axis, not models across axes. | 5 |
| **Combination Elastic-Net (C-ENet): select sub-forecasts, then equal-weight the survivors (θ≥0)** | Rapach-Zhou MLAM Ch.1 | Keeps equal-weight overfit-resistance while pruning noise/sign-flipped sub-models; consistent positive OOS R² across recessions. | sklearn ElasticNetCV + LinearRegression(positive=True); average selected survivors. | New combiner layer. | 4 |
| **Forecast combination as shrinkage** (J univariate forecasts averaged, shrunk toward prevailing mean) | Rapach-Zhou MLAM Ch.1 | Parameter-free regularizer; shrinking toward do-nothing caps aggressiveness when signal is weak → less whipsaw/tail DD. | numpy rolling univariate slopes + average + blend with trailing-mean. | New. | 4 |
| **TS + CS dual-book diversification** (run time-series AND cross-sectional models on same universe) | Wang transcripts | TS and CS earn uncorrelated money → portfolio MaxDD ≪ single-strategy MaxDD; implicit-regime-via-ensemble avoids lagged regime-boundary failure. | ensembler/consensus exist; CS path needs multi-name cross-section (7 ETFs small). | Harness ensembles models but no parallel CS book. | 3 |

---

## ⑧ — Sizing / risk management (most direct Calmar-denominator levers)

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Novelty vs harness | Pri |
|---|---|---|---|---|---|
| **Meta-labeling secondary gate** (primary calls side; secondary predicts P(correct), trade only above threshold, size by P) | LdP AFML Ch.3,10; H&T JFDS; Wang §10 "二级模型" | Raises precision by suppressing low-conviction trades → removes the loser tail that drives DD. Documented: MaxDD -31.7%→-7.4%, Sharpe 0.45→1.53; Alpaca -6.2%→-4.1% with Sharpe 1.18→3.08. Single most documented DD-reduction technique. | second xgboost/lightgbm on features+primary-prob; isotonic-calibrate, purged CV vs leakage. | Pipeline ends at xgboost+isotonic with no secondary trade gate. | 5 |
| **Conditional vol-targeting / inverse-vol overlay** (position = signal · target_vol/forecast_vol, capped, smoothed) | Harvey et al. JPM 2018; AlphaArchitect; arXiv:2212.07288 | Vol is persistent and DDs cluster in high-vol regimes; cut notional there to truncate the worst losses; structural low-parameter rule that generalizes OOS; +15-50% Calmar. | EWMA vol (no lib) or arch GARCH(1,1) on daily returns; trivial. | Sizing is xgboost+isotonic only; orthogonal robust overlay. | 5 |
| **Probability→bet-size map** (de Prado z=(p-0.5)/√(p(1-p)), size=2Φ(z)-1) × vol-target, + active-bet averaging + size discretization | LdP AFML Ch.10; H&T repo | Confidence-scaled sizing small on marginal/large on conviction → shrinks equity-curve variance & DD; averaging prevents stacking correlated bets; discretization cuts turnover cost. | numpy/scipy; uses the calibrated p already produced. | Has isotonic but fixed-threshold linear ramp; no CDF map / averaging / discretization. | 5 |
| **EGARCH/GJR 1-step conditional vol as causal feature AND top-decile de-risk gate** | Nelson 1991; arch docs; PLOS One 2024 | Equity vol is asymmetric (drops spike vol more); EGARCH anticipates the post-drop surge one step ahead → de-risk into the regime where DDs happen; strictly causal. | arch 8.0 EGARCH/GJR, forecast(h=1); **aggregate to daily, refit ≤ daily** (per-minute blows 5-min budget). | No GARCH gate today. | 5 |
| **Confidence- + vol-targeted sizing to scale capital up safely** (size up only where conviction AND vol-budget justify) | Wang slides §10 "提升资金管理规模" | Keeps risk roughly constant per bet: more capital on high-edge low-vol setups (CAGR↑), de-lever in high-vol/low-conviction (MaxDD↓) — literally maximizing CAGR/MaxDD. | calibrated p × target_vol/realized_vol, capped at max leverage; numpy. | Sizes off p but no vol-target or leverage scaling. | 4 |
| **ETF "indexification" cross-correlation features** (rolling corr/beta/spread vs SPY / sister ETFs) | Wang slides §8 | Cross-asset context supplies risk-off info a single-asset model can't see → stand aside in broad selloffs (DD brake); high-signal for ETFs. | harness already passes spy_lc/spy_lr; extend to rolling corr/beta/spread. | spy passed in but cross-asset features disabled; re-enable as corr/beta/spread. | 4 |
| **Drawdown / Time-under-Water + HHI runs-concentration risk gate** | LdP AFML Ch.14 | Realized DD/TuW as a live de-risk trigger caps MaxDD directly; low return-HHI confirms a broad (not lucky-outlier) edge that survives OOS. | pandas on running equity curve; throttle size when DD/negative-HHI rises. | No realized-DD throttle nor concentration filter. | 3 |
| **Deflated Sharpe / PBO acceptance gate (trials-count-aware)** | Bailey-LdP SSRN 2460551, 2326253; CFI §6.4.1 | Expected max Sharpe grows ~√(log N) on noise; deflating by #configs-tried means only genuine edge survives → deployed Calmar tracks backtest Calmar. | ~20-line numpy/scipy (skew, kurtosis, N, T); offline go/no-go on the final config. | No multiple-testing deflation; selection has no penalty for search size. | 4-5 |
| **Risky-forecast / worst-regime adversarial acceptance gate** | LdP CFI §3.3, §8.3 | MaxDD is set in crash/high-vol windows; require the config to survive held-out worst-regime slices before deployment → structurally smaller tail loss. | partition history by vol-quantile/known-stress windows; evaluate DD on worst slices. | Adds worst-regime gate on top of purged walk-forward. | 4 |
| **Economic non-negativity constraint on any combination weights (θ≥0)** | Rapach-Zhou MLAM Ch.1 | Sign-flipping weights are a classic overfit symptom; forcing θ≥0 stops loading on perverse-sign noise predictors → fewer regime-driven blowups. | sklearn LinearRegression(positive=True) / scipy.optimize.nnls. | New prior on any stacking/meta combiner. | 4 |

---

## vol — Volatility block (cross-cut; feeds ⑧ and ③)

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Pri |
|---|---|---|---|---|
| **GARCH/EGARCH/GJR conditional-vol forecasting** (daily refit, forecast h=1) | Nelson 1991; arch 8.0 | Anticipates clustered high-vol crash periods → de-risk ahead; EGARCH beats symmetric GARCH for equity tail risk. | arch 8.0; daily aggregate, refit ≤ daily. | 5 (covered in ⑧ EGARCH gate) |
| **ARCH-LM / Engle-Ng sign-bias pre-test** | Engle 1982; Engle-Ng 1993 | Gate which assets actually have clustering structure that vol-targeting exploits → deploy DD-control only where the mechanism is real. | statsmodels het_arch; one-time per asset. | 3 (covered in ⓪) |

---

## freq — Frequency-domain (cross-cut; feeds ③ and ⓪)

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Pri |
|---|---|---|---|---|
| **Leak-free per-window FFT band-power + spectral entropy** | VMDNet arXiv:2509.15394 | Routes trend bets to low-freq-power regimes, avoids whipsaw; causal per-window transform → real (not leaky) OOS gain. | numpy.fft only. | 4 (covered in ③) |
| **Causal SWT/à-trous trend-noise SNR** | Shensa 1992 | Causal SNR regime indicator; chiefly avoids the pervasive leaky-wavelet fake backtests. | scipy.signal/numpy. | 2 (covered in ③) |

---

## arb — Arbitrage / spread (portfolio-level; mostly outside single-asset 5-min budget)

| Technique | Source | Mechanism for REAL Calmar | QC impl / lib | Pri |
|---|---|---|---|---|
| **Spread-as-trend transform** (build spread A-k·B / ratio / standardized / log-spread, run the FULL trend pipeline on it as a synthetic 1-D asset) | Wang q922tUTWmCY | Removes classic stat-arb's fatal linear N-sigma band (don't-know-add-or-stop) → inherits trend-following's built-in stop logic → smaller tail DD. Demo: glass/soda-ash Calmar 4.58, MaxDD 8.99%. | reuse bar_builder/labeler/trainer on the spread series; needs 2 related instruments (7-ETF set offers QQQ/IWM, HYG/TLT candidates, no physical-ratio pairs). | 2 |
| **Tail-dependence-clustered Hierarchical Risk Parity** | Lohre/Rother/Schäfer (TOC only) | Cluster on co-crash probability not correlation → spread risk across independent crash buckets → direct MaxDD control. | cvxpy/scipy; **multi-asset only** — does not fit single-asset 5-min backtest; portfolio-level module. TOC-only, estimator must be sourced. | 2 |

---

## De-duplication note

Meta-labeling, prob→size, vol-targeting, triple-barrier, dollar bars, CUSUM, Jump Model,
sliced-Wasserstein, CPCV/Deflated-Sharpe, sample-uniqueness weighting, and entropy features each
appeared 2-3× across AFML / Wang / web sources; each is listed ONCE above with sources merged.
Findings #36, #50 (dollar/volume bars) and #66 (calibration) confirm existing harness components and are
recorded as "keep, low novelty" (Pri 2) rather than new work.
