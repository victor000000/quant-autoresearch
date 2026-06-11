# 311-ETF exploration plan (ML-for-quant research, Wang backbone)

_Workflow ml-quant-research-311-plan (w09z5l2jm), 2026-06-06. 10 research facets → distilled → plan._

## SOTA survey

The relevant ML-for-quant SOTA splits into five layers, all of which map onto Wang's backbone without replacing it.

(1) PREDICTABILITY SCREENING / mechanism triage. Before fitting, the literature scores how exploitable a series is and which mechanism it admits: Lo-MacKinlay variance-ratio VR(q) with heteroskedastic-robust z (Choi 1999 AVR, Kim 2009), averaged Hurst via R/S+DFA, weighted permutation entropy as a forecastability ceiling (Bandt-Pompe 2002; Garland-James-Bradley 2014), Campbell-Thompson 2008 OOS-R2 of cheap predictors, and Amihud 2002 illiquidity. We already use beta200 as a 1-D version; the SOTA move is to fuse these into a composite a-priori router (trend vs revert vs hold) and to cross-sectionally de-duplicate the universe by correlation clustering (Lopez de Prado ONC, MLAM 2020).

(2) MULTIPLE-TESTING / honesty under search. de Prado's DSR + PBO-via-CSCV + MinBTL (Bailey-LdP 2014; Bailey-Borwein-LdP-Zhu 2014), Harvey-Liu 2015 backtest haircuts, and crucially the dependence-aware and sequential tools we lack: Romano-Wolf StepM / White Reality Check / Hansen SPA (FWER exploiting cross-strategy correlation) and online-FDR (LORD++/SAFFRON/ADDIS; Javanmard-Montanari 2018, Ramdas 2018, Tian-Ramdas 2019) for an open-ended resumable screen. CPCV (AFML ch.7/12; Arian-Norouzi-Seco 2024) gives a purged combinatorial RETRAIN path-distribution, stronger than CSCV-on-returns.

(3) EXPLORATION POLICY. AutoML racing — Hyperband/successive-halving (Jamieson-Talwalkar 2016; Li 2018), BOHB (Falkner-Klein-Hutter 2018), TPE (Bergstra 2011) — and bandit/Thompson budget allocation (Russo 2018) replace uniform grids; this is the stated target architecture.

(4) FEATURES. Range-based vol estimators (Parkinson 1980, Garman-Klass 1980, Rogers-Satchell, Yang-Zhang 2000: 5-7x more efficient than close-to-close), fractional differentiation (AFML ch.5), microstructure (Amihud, Kyle 1985, Roll 1984), and offline automated discovery (tsfresh/FRESH-BH, catch22).

(5) SIZING + BOOK. Vol/CVaR-targeting (Harvey 2018; Rockafellar-Uryasev 2000), fractional Kelly, conformal selective abstention + ACI (Gibbs-Candes 2021), and risk-based allocation HRP/HERC/NCO with MP-denoising (LdP 2016/2019; Raffinot 2018), benchmarked against 1/N (DeMiguel-Garlappi-Uppal 2009) and Max-Diversification (Choueifaty-Coignard 2008). Meta-labeling caveats (AFML; "Meta-Labeling Is Not a Silver Bullet"; Thumm-Barucca-Joubert 2023) demand an end-to-end XGB-prob baseline before booking a stacked result.

## Where we already lead

Our research-honesty bar already exceeds most published practice and most shops. (1) Leak-safe ONLINE backtest is the standout: threshold fit on TRAIN and extrapolated OOS-invariant, infer = pure ObjectStore replay, proven by 13- and 47-agent adversarial audits, a bar-threshold leak unit test, and the recent per-labeler embargo bound (dc_reversal fix, commit 993edcf) — most backtests in the wild are subtly look-ahead-contaminated and never re-run with the fix that reveals it. (2) The honesty stack is already deep and wired into the loop: DSR + Holm-Bonferroni/Benjamini-Hochberg, permuted-label control (which empirically proved UUP/sadf_explosive real by collapsing to ~0), PBO-via-CSCV, decay + e-value monitors, and transaction-cost stress — practitioners typically ship none of these. (3) beta200 is a genuine a-priori mechanism router (buy-hold pre-filter), and the screen panel tests 5 distinct mechanisms + a buy-hold baseline per name rather than one recipe — already a triage discipline most screens lack. (4) Single-ticker discipline with strict purge/embargo and a multi-file QC build that permanently solved the render limit. In short, our confirmation rigor matches de Prado's prescriptions; the deficit is on the FRONT of the funnel (what/which-order to test) and the formal multiple-testing accounting, not on whether a booked edge is real.

## Gaps (front-of-funnel + multiple-testing accounting)

The real gaps are concentrated at the entry and accounting ends of the funnel, not the leak-safety end. (1) SCREEN ORDERING is hand-coded by asset-class PRIORITY + AUM_Rank in screen_etfs.py — not predictability-driven and not de-duplicated, so driver-hours are spent on redundant/0-EV names and the effective universe size is unknown. (2) DSR OVER-DEFLATES: assess_dsr.py uses raw n_trials Bonferroni (thr = 1 - 0.05/nt) which ignores the heavy correlation inside a 21x27 sweep, so real borderline fits can be killed; there is also no persistent GLOBAL true-trial counter across all ETFxaxisxlabelerxsizer runs. (3) MULTIPLE-TESTING ASSUMES INDEPENDENCE: stats_rigor.py has only Holm/BH/Bonferroni — no dependence-aware Romano-Wolf StepM/SPA to treat the correlated commodity/leveraged cluster jointly, and no online-FDR ledger for the genuinely sequential, resumable, open-ended 311 screen. (4) VALIDATION IS SINGLE-PATH: we have CSCV-on-returns (config overfit) but no CPCV purged-combinatorial RETRAIN path distribution, and no MinBTL a-priori budget gate to refuse over-searching short-history names. (5) FEATURES are close-to-close only: no range-vol (Parkinson/Yang-Zhang), no microstructure/liquidity, no FFD, no path-position — leaving efficiency and orthogonal-edge upside on the table. (6) SIZING levers are dormant/untuned: the inverse-vol overlay is OFF (VOL_FLOOR=1.0), and there is no conformal trade-count standardization, no fractional-Kelly, no calibration A/B. (7) BOOK is ad-hoc Calmar^2 with no HRP/HERC/NCO, no 1/N honesty baseline, and no marginal-IR orthogonality gate — so there is no objective stopping rule for how many of the 311 to deploy. (8) No automated exploration policy (Hyperband/bandit) — the grid is uniform. (9) The single-ticker wall itself (cross-sectional/pooled relaxations gated by user authorization).

## Plan (ordered)

### Step 1 — (high EV / easy)
**What:** Build a TRAIN-only predictability pre-screen + mechanism router (scripts/predictability_screen.py): composite exploitability score from VR(q) for q in {2,5,10,20,50,100,200} with HAC-robust z, averaged Hurst (R/S+DFA), weighted permutation entropy, Campbell-Thompson OOS-R2 of 2-3 cheap predictors, and Amihud. Auto-route VR<1/H<0.45->revert, VR>1/H>0.55->trend_leg, VR~1/H~0.5/R2<=0->buy-hold-skip.

**Why:** Replaces hand-coded class+AUM ordering in screen_etfs.py with a cheap a-priori triage that re-ranks the ~290 untested queue and avoids 0-EV grinding — the single biggest lever on FIT-yield-per-driver-hour.

**Method:** Pure numpy on TRAIN bars, leak-safe exactly like the beta200 footer injection; generalizes beta_router.py/mechanism_router.py. Calibrate cut-points by cross-tabbing the signature against realized FIT + winning mechanism on the 21 screened ETFs, then report the gate's hit-rate on the 8 known fits (SSO/IAU/USO/AGQ/GDX/DJP/GSG/UCO).

### Step 2 — (high EV / moderate) · depends: 1
**What:** Cross-sectional de-duplication: cluster the 311 on 2009-2016 TRAIN daily-return correlation (corr->distance->hierarchical/ONC) into ~40-60 groups; screen ONE representative (longest-history/highest-AUM) per cluster first, drill siblings only if the rep fits.

**Why:** Stops redundant grinding (e.g. USO/UCO, GLD/IAU collapse together) AND yields the effective universe size that prices the multiple-testing burden for N_eff and online-FDR.

**Method:** Offline numpy 311x311 correlation + agglomerative linkage (no cross-ticker info ever enters a model). Re-point the screen queue; verify the 8 known fits land in distinct clusters (shared cluster = flag for already-done redundant work).

### Step 3 — (high EV / moderate)
**What:** N_eff deflation for DSR + a persistent global true-trial counter. Replace raw n_trials in assess_dsr.py with N_eff from the trial OOS-PnL correlation structure (eigenvalue participation ratio (sum lambda)^2/sum lambda^2, or ONC), and instrument a durable counter over every ETFxaxisxlabelerxsizer run.

**Why:** Current Bonferroni-by-count over-deflates correlated 21x27 sweeps and can kill real borderline fits; N_eff should collapse from hundreds to single digits per ETF. Confirms GLD/UUP/USO/UCO still clear at corrected N and resurrects unfairly-killed names.

**Method:** np.linalg.eigh on the per-ETF axisxlabeler OOS daily-return matrix, feed N_eff (and V[SR]) into the existing expected_max_sharpe/DSR/Holm path in stats_rigor.py; surface SR0 on the dashboard next to each edge. Uses the 2413-row round_results.csv already on disk.

### Step 4 — (high EV / easy) · depends: 1,3
**What:** Online-FDR ledger (LORD++/ADDIS) over the streaming screen: maintain an alpha-wealth account keyed on each ETF's permute/DSR p-value, spend per test, replenish on each confirmed fit, stamp FIT only when p < the wealth-derived threshold.

**Why:** The statistically-correct replacement for batch Holm/BH on a genuinely sequential, resumable, open-ended 311 screen; self-throttles grinding when wealth depletes and gives a live 'exploration budget' on the dashboard.

**Method:** Pure scalar bookkeeping around screen_etfs.py at FDR=0.10; order remaining names by the step-1 composite (most-exploitable first), replay over the existing screen log to confirm the 8 known fits survive.

### Step 5 — (high EV / moderate) · depends: 1,2,3
**What:** Backtest-the-screen, then deploy 3-rung successive-halving/Hyperband racing (eta=3) over the (ETF x axis x labeler) grid: rung0 = step-1 predictability on all candidates -> keep top 1/eta; rung1 = one quick trend + one quick revert run; rung2 = full deep-sweep on survivors.

**Why:** Replaces the uniform hand-ordered 21x27-per-name screen with the stated BOHB-target policy; must demonstrably reach the 8 known fits in materially fewer driver-hours.

**Method:** Orchestration wrapper around screen_etfs.py/run_axis_label_parallel.py; tree fidelities = n_estimators/sub-years/reduced-features. Backtest offline against the completed screen log and report FIT-yield per driver-hour + rung counts (feeds effective-N). Optionally a Thompson/UCB scheduler over mechanism-class arms seeded by historical class fit-rate.

### Step 6 — (high EV / moderate) · depends: 3,4
**What:** Two-stage funnel hard-coded into the loop + CPCV confirmation validator. Stage-1 cheap (predictability order -> permuted-label p -> online-FDR gate) decides who earns the expensive sweep; Stage-2 strict: CPCV 5-path 5th-pct > buy-hold AND DSR>thr at N_eff AND survives StepM AND e-process alive gates deployable=true.

**Why:** Stops paying CPCV/sweep cost on 0-EV names and never books a name failing the strict battery; converts single-path fragility into a path distribution.

**Method:** numpy CPCV splitter (N=6,k=2 -> 5 paths) reusing the dc_reversal purge+embargo machinery, XGB retrain per split, single-ticker, online replay unchanged. Re-run the 8 STRONG fits + the SUSPECT sliced_wasserstein NO-BASELINE outliers (IAU 3.35, QLD 4.06) to separate real survivors from single-path artifacts.

### Step 7 — (high EV / moderate) · depends: 3,6
**What:** Romano-Wolf StepM joint reality-check vs buy-hold across the deployable set, plus MinBTL admission gate and a per-labeler max_forward_reach embargo guard.

**Why:** Holm/Bonferroni waste power on the correlated commodity/leveraged cluster; StepM exploits that correlation for a dependence-aware deployable SET. MinBTL refuses over-searching short-history names; the reach attribute systematizes the one-off dc_reversal leak fix across all 27 labelers.

**Method:** Stationary block-bootstrap (mean block 10-20d, 2000 reps) the T x N excess-vs-BH matrix from champion_series.py, max-t step-down at FWER 0.10; compute MinBTL ~ 2 ln(N)/SR*^2 as the per-ETF config cap; add max_forward_reach to each labeler + a unit test in tests/test_bar_threshold_leak.py that fails if reach exceeds embargo.

### Step 8 — (high EV / easy)
**What:** Range-based volatility + microstructure/liquidity feature blocks in modules/features.py: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang over {5,20,50,200} + short/long ratios; Amihud(20/100), rolling Kyle-lambda(100), Roll spread, signed-volume OFI(20/60), dollar-vol z(50/200).

**Why:** Current features are close-to-close std only (5-7x less efficient); range-vol sharpens every vol split all three mechanisms depend on, and single-ticker liquidity signals are orthogonal to the price-only bank — a likely source of NEW edges on names now screening buy-hold.

**Method:** Pure-numpy causal columns from bar OHLC (already retained) and the volume stats already computed for the logdollar/imbalance axes; A/B on the 42 fit-prone commodity/currency/leveraged names under IG/mRMR, gated val_auc>0.55 + beats-BH + permute collapses.

### Step 9 — (medium EV / moderate) · depends: 8
**What:** Path-position/entropy features + fixed-width fractional differentiation: rolling Hurst(100/200), perm-entropy(50/100), LZ-complexity, drawdown-from-high, Donchian-position(20/50/200), bars-since-high; FFD of log-price at the min d passing ADF on TRAIN.

**Why:** Encodes WHERE in a move the asset sits (the reversion/trend trigger raw returns miss) and gives the stationary-but-memory-preserving middle ground between non-stationary log-close and memory-destroying k-bar returns; expected largest lift on the mean-reversion oil names.

**Method:** numpy causal rolling windows + recursive FFD weight kernel (deterministic, OOS-invariant); per-ticker d grid on TRAIN-only for GLD/IWM (trend) and USO/UCO (revert); A/B under mRMR + permute, adopt only where real Calmar beats champion AND permute collapses.

### Step 10 — (high EV / moderate) · depends: 6
**What:** trend_scan_ens labeler (the open Wang trend-strength-ENSEMBLE gap) + heterogeneous NNLS stacking of mechanism-family base models. Ensemble de Prado trend-scanning across L in {20,50,100,200}, label = sign of max-|t| slope, magnitude = |t| as XGB sample_weight, plus a trend_leg-AND-trend_scan side-agreement gate; then stack trend_leg/ker/sadf_explosive/revert/bgm base XGBs via purged-OOF non-negative-weighted meta-blend.

**Why:** Closes the documented Wang gap and tests whether a calibrated NNLS blend of orthogonal weak families beats the single-best champion without drowning the weak-but-decorrelated ones.

**Method:** Rolling OLS slope/SE in numpy, magnitude-as-sample_weight native to XGB; base preds strictly walk-forward (purged-CV), isotonic-calibrate the blend, pure OOS replay. A/B on GLD/IWM/XME/SPXL (trend_scan_ens) and the 8 STRONG fits (stacking) with DSR + permute + decay; win = stacked > best-single AND permute->~0.

### Step 11 — (high EV / easy) · depends: 6,8
**What:** Re-activate and modernize sizing: conditional/quantile-gated + smoothed vol-targeting (re-enable the dormant VOL_FLOOR<1 overlay with a no-trade deadband), a Mondrian split-conformal selective-abstention trade gate (+ ACI online), and a fractional-Kelly map f=clip(lambda(2p-1),0,1).

**Why:** The inverse-vol overlay is OFF and there is no trade-count standardization; conditional vol-targeting is best-evidenced for our trend/commodity/leveraged family and kills momentum crashes, conformal fixes a deployable trade-count uniformly across all 311, and fractional-Kelly is robust to edge-estimation error.

**Method:** Module-8 overlays in modules/trainer.py scaling the model output (not features); causal EWMA/quantile + TRAIN-fit cutoffs, OOS-invariant; conformal quantile on a TRAIN fold extrapolated OOS; select lambda and quantile cutoffs by DSR, not raw Calmar. A/B on GLD/IWM/USO/UCO/SPXL/XME, cost-stressed Calmar AND turnover; re-run decaying EEM/TLT with ACI vs static threshold.

### Step 12 — (high EV / moderate) · depends: 7
**What:** Risk-based book construction + objective stopping rule: add a marginal-IR orthogonality admission gate (residual Sharpe after regressing a candidate's OOS returns on the current book; reject if |corr|>~0.7 unless residual Sharpe clears the deflated bar) and replace Calmar^2 weights with ERC/inverse-vol and HRP/HERC-CDaR, MP-denoised+detoned NCO, all gated to beat a 1/N + ERC honesty baseline OOS.

**Why:** Reframes screen success as marginal contribution to book IR (breadth over depth) and gives an objective stopping rule for how many of the 311 to deploy; ERC/HRP auto-collapse redundant exposures (USO de-weighted vs UCO endogenously) and the 1/N gate prevents over-fit weighting at N=6-42.

**Method:** Book/meta-layer on stored champion equity curves (champion_series.py/portfolio_rederive.py/portfolio_weights.py), pure numpy (distance matrix + linkage + inverse-variance recursion + np.linalg.eigh denoise), MCOS to rank {Calmar^2,1/N,HRP,HERC,NCO} on simulated paths matched to our edge count and OOS length. No model contamination.

## Quick wins (do first)

- Ship the TRAIN-only predictability pre-screen + mechanism router (plan step 1): VR(q)/Hurst/WPE/CT-R2/Amihud composite, leak-safe numpy like beta200, re-ranks the ~290 untested queue and auto-routes trend/revert/hold. Calibrate on the 21 screened and report hit-rate on the 8 known fits — biggest FIT-per-driver-hour lever, near-zero risk.
- Fix DSR over-deflation (step 3): swap raw n_trials in assess_dsr.py for N_eff = eigenvalue participation ratio of the trial-PnL correlation matrix and add a persistent global trial counter. Expect N to collapse to single digits and recheck that GLD/UUP/USO/UCO still clear — may resurrect borderline fits Bonferroni killed.
- Add range-based vol features (step 8): Parkinson + Yang-Zhang + short/long ratios from existing bar OHLC, A/B on the 42 fit-prone commodity/currency/leveraged names under IG/mRMR. 5-7x more efficient than close-to-close std, sharpens every mechanism's vol split, trivial numpy.
- Stand up the online-FDR alpha-wealth ledger (step 4): LORD++/ADDIS scalar bookkeeping around screen_etfs.py keyed on each ETF's permute p-value, replay over the existing screen log, surface remaining alpha-wealth on the dashboard as the live exploration budget — correct sequential control with ~30 lines.
- Re-enable the dormant conditional vol-targeting overlay (part of step 11): set VOL_FLOOR<1 with a quantile-gated + smoothed + deadband variant, A/B cost-stressed Calmar AND turnover on GLD/IWM/USO/UCO/SPXL — a known leverage point currently switched off, easy win on MaxDD.

## Big bets (higher upside; several need user authorization)

- CROSS-SECTIONAL / RELATIVE-VALUE PAIRS (needs user authorization — relaxes single-ticker): build spread/pairs strategies on the redundant clusters the de-dup surfaces (USO/UCO, GLD/IAU, sector spreads) and the already-scaffolded term-structure pair (VIXY short vs VIXM mid VIX futures). This is Wang's documented no-new-data escape from the single-ticker wall and the natural use of the correlation structure we otherwise only de-duplicate.
- POOLED / UNIVERSAL CLASS MODEL (needs user sign-off — relaxes 'no cross-ticker info in trained params'): fit ONE XGB on np.vstack of a whole asset class's TRAIN rows with a ticker-id-free feature set, infer per-ticker by pure replay. Pools richer scenarios to help data-poor short-history names that currently screen buy-hold; run as a fenced track benchmarked against per-ticker champions.
- NEW DATA MODALITIES (needs authorization — breaks minute-OHLCV-only): pull VIX/term-structure futures to harden the permute-confirmed SPY crash-veto edge (first life ever on SPY, currently not deployable for lack of exogenous vol), plus options-implied vol, COT positioning, and rates/macro for the drift-bound bond names where price-only features hit a wall.
- OFFLINE AUTOMATED FEATURE DISCOVERY: run tsfresh (~800 characteristics) + FRESH/Benjamini-Hochberg relevance and a curated catch22 block in the QC-blocked scratch harness on currently-buy-hold equity/bond ETFs; hand-port only BH-surviving winners as numpy columns and re-validate vs champion. Higher build cost, but the principled way to break the equity/bond drift wall.
- ACTIVE ALPHA-DECAY GOVERNANCE + learning-curve routing: ADWIN/CUSUM change-detection on each champion's OOS curve to trigger a TRAIN-only walk-forward refit at the break and emit a RETIRE flag (freeing FDR budget) when edge half-life < remaining horizon; plus a learning-curve diagnostic (val_auc curvature on 25/50/75/100% of TRAIN) to route bottleneck names to {pool-data | new-method | new-modality | abandon}. Backtest whether it would have pre-empted EEM/TLT decay.

## Sources

- Lo & MacKinlay 1988 (variance ratio); Choi 1999 / Kim 2009 (automatic & wild-bootstrap VR)
- Bandt-Pompe 2002; Garland-James-Bradley 2014 (permutation entropy / forecastability ceiling)
- Campbell & Thompson 2008 (OOS R^2 of return predictors); Amihud 2002 (illiquidity)
- Lopez de Prado, AFML 2018 (ch.4 clustering, ch.5 fractional differentiation, ch.7&12 CPCV/purge-embargo, ch.10 bet sizing)
- Lopez de Prado, ML for Asset Managers 2020 (ONC clustering, MP denoising/detoning, NCO)
- Bailey & Lopez de Prado 2014 (Deflated Sharpe / PSR); Bailey-Borwein-LdP-Zhu 2014 (MinBTL / Pseudo-Mathematics)
- Harvey & Liu 2015 (backtest haircuts / multiple testing); Lopez de Prado & Fabozzi (FDR identification failure)
- White 2000 (Reality Check); Hansen 2005 (SPA); Romano-Wolf 2005 (StepM); Politis-Romano 1994 (stationary bootstrap)
- Foster-Stine 2008; Javanmard-Montanari 2018 (LORD); Ramdas et al 2018 (SAFFRON); Tian-Ramdas 2019 (ADDIS) — online FDR
- Arian-Norouzi-Seco 2024 (CPCV best for confirmation, Knowledge-Based Systems)
- Jamieson-Talwalkar 2016; Li et al 2018 (Hyperband); Falkner-Klein-Hutter 2018 (BOHB); Bergstra 2011 (TPE); Russo et al 2018 (Thompson sampling)
- Parkinson 1980; Garman-Klass 1980; Rogers-Satchell; Yang-Zhang 2000 (range-based vol)
- Kyle 1985; Roll 1984; Chordia et al 2002 (microstructure/liquidity); Christ-Kempa-Liehr-Feindt 2018 (FRESH); Lubba et al 2019 (catch22)
- de Prado MLAM trend-scanning; Thumm-Barucca-Joubert JFDS 2023 (ensemble meta-labeling); 'Meta-Labeling Is Not a Silver Bullet' (QuantConnect/Baldisserri)
- Rockafellar-Uryasev 2000 (CVaR); Harvey et al 2018 (vol-targeting); MacLean-Thorp-Ziemba 2011 (Kelly); Gibbs-Candes 2021 (ACI); Calibrated Selective Classification 2022
- Lopez de Prado 2016 (HRP); Raffinot 2018 (HERC); Maillard-Roncalli-Teiletche 2010 (ERC); DeMiguel-Garlappi-Uppal 2009 (1/N); Choueifaty-Coignard 2008 (Max Diversification); Grinold 1989 (Fundamental Law)
- Digalakis-Perignon-Saurin-Sentenac 2025 (The Challenger, learning-curve routing); Sirignano & Cont 2018 (universal features of price formation, pooled training)
- Internal: autoresearch/WANG_INVESTIGATION.md, BACKTEST_AUDIT.md, RESEARCH_REVIEW.md; commit 993edcf (dc_reversal embargo bound); modules/features.py, modules/trainer.py, scripts/assess_dsr.py, scripts/stats_rigor.py, scripts/screen_etfs.py, scripts/beta_router.py, scripts/mechanism_router.py, results/round_results.csv