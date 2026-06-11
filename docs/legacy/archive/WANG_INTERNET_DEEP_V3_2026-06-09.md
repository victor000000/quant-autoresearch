# Deep Wang+internet v3 (2026-06-09)

All load-bearing claims verified against the live repo. Writing the report now.

---

# Deep Wang+internet v3 (2026-06-08) — anything genuinely NEW + QC-FEASIBLE after this session's exhaustive closure?

Repo state independently re-verified this pass (not taken on faith): `grep -i bootstrap` over all `*.py` = **0 hits**; `import arch` = **ModuleNotFoundError** (not installed); SPA/RealityCheck/StepM/MCS/Romano/selective/e-BH = **0 hits**; `stats_rigor.py` = {PSR, expected_max_sharpe, DSR, MinBTL, LOND, lo_sharpe_se, Holm, BH, pbo_cscv, effective_n_trials} only; `evalue_monitor.betting_eprocess` present; `semivar`/`signedjumpvar` exist **only** as `BarBuilder` sampling clocks in `_AXES_ORDER`, never as `build_feats` columns; `cost_stress.py` = flat `ConstantSlippageModel` only; `harvey_liu_haircut.py` = per-strategy Bonferroni/BH haircut (no π0/FDP/power); `portfolio_weights.py` = {equal, Calmar², Calmar²×DSR, inverse-variance}, book noted weighting-robust.

## 1. Bottom line

**No** — v3 found no new EDGE lever that is simultaneously novel + QC-feasible + untested + positive-prior. **The single-ticker edge frontier is structurally closed.** That is the honest, valid result, and it holds on all three axes the digest already established: empirical (universe screen + 15 hypothesis rounds), mechanistic (3 asset-intrinsic mechanisms, β200-explained), structural (platform walls + trades<80 cap).

Two honest qualifications keep this from being a flat "nothing":
- **One borderline EDGE shot survives as genuinely-untested**: signed realized-semivariance features (RS+/RS−/dJ). It accesses a minute-resolution, sign-conditioned variance channel that none of the (all-failed) bar-series feature-adds could see. But it carries a strong **negative prior** (same Patton-Sheppard signal is already dead as a *sampling axis*; this session's feature-add scoreboard is 0/N). Treat it as one decisive A/B that most likely **closes** the last open feature channel — not a positive-prior bet.
- **The open frontier is entirely TOOLING + new-data.** v3's real contribution is mapping the *honesty/validation* family that v2 only began (v2's A1 = Lo autocorr-adjusted SE, A3 = spec-curve were the first two of exactly this family). The biggest verified gap: the **stationary-bootstrap data-snooping family is wholly absent** (0 bootstrap hits repo-wide). These can only **tighten/retire** sleeves — they cannot create edge, so they cannot dishonestly reopen a dead frontier.

So: no new edge; yes, real new validation/portfolio tooling; one negative-prior edge-feature A/B left to formally close.

## 2. Genuinely-new actionable levers (NOVEL_ACTIONABLE only)

### EDGE levers (only 2 exist; both negative-prior)

**E1 — Signed jump variation / realized semivariance features (RS+, RS−, dJ).** conf 0.58 / value 0.55.
- *Source*: Patton & Sheppard, "Good Volatility, Bad Volatility," REStat 97(3) 2015.
- *QC-feasible*: the off-clock bar builders already consume the minute stream; accumulate per-bar `rs_plus += r²·1{r>0}`, `rs_minus += r²·1{r<0}` at emit (O(1)), reset on emission, thread through footer/infer_online/live_trade into `build_feats` as 3 standard XGB columns (RS+, RS−, dJ=RS+−RS−, plus a rolling z of dJ). No torch/custom-obj/OHLC/HMM.
- *Why not closed*: every closed feature-add (evt / dispersion-entropy / path-sig / termstruct / sig / FFD / VR / DC) was a **symmetric bar-series transform**; this is the first **minute-resolution, sign-conditioned** channel — verified absent (`semivar`/`signedjumpvar` are sampling axes, never features).
- *Experiment + gate*: single A/B on GLD (logdollar / trend_leg+regime_gmm / dd_overlay / **reduce=infogain** so the cols compete on MI) and USO (imbalance / revert). **GATE = beats champion Calmar AND permuted-label control collapses the RS-feature contribution to ~0 AND infer online==offline byte-exact.** Both fail → this closes the last open feature channel.

**E2 — Intrabar realized skewness & kurtosis.** conf 0.35 / value 0.25.
- *Source*: Amaya, Christoffersen, Jacobs & Vasquez, JFE 118(1) 2015.
- *QC-feasible*: same bar-emit accumulation (`√n·Σr³/RV^1.5`, `n·Σr⁴/RV²`) with a min-minute-count gate (n≥20) + winsorization; standard XGB columns.
- *Why not closed / caveat*: distinct from `modules/features.py:180-184` rolling-kurt-on-**bar**-returns. But 3rd/4th moments are outlier-dominated on short off-clock bars, the gate NaNs many bars, and ACJV is an equity **cross-section** result (weak single-ticker TS prior). Lower-prior sibling of E1; **likely DISCARD**.

### TOOLING levers (path-2: can only validate/tighten — never create edge)

Multiple-testing / data-snooping family (the verified 0-bootstrap gap — this is the genuinely-unbuilt cluster):

**T1 — Hansen SPA / White Reality Check.** conf 0.82 / value 0.7. *Strongest, runnable today.* Recentered studentized-max stationary-bootstrap test of "does the best sleeve beat its benchmark given the whole 3483-trial search." Distinct from DSR (EVT/Gaussian best-of-N), PBO/CSCV (combinatorial split overfit), Holm/BH/LOND (p-adjust) — none estimates the joint null of the max statistic under realized cross-trial correlation; the block bootstrap **preserves serial dependence**, directly attacking the iid hole the Lo-SE gap flagged. Build `scripts/hansen_spa.py`, **hand-rolled numpy** (correction below). Honesty self-test: noise universe → p≈0.5. Source: Hansen 2005 JBES 23(4); Hsu-Hsu-Kuan 2010 (stepwise-SPA for trading rules).

**T2 — Romano-Wolf StepM.** conf up to 0.82 / value up to 0.7. Studentized max-stat stepdown under arbitrary dependence; strictly tighter than the `effective_n_trials` *heuristic* count-shrink. Phase 1 (now): over cached `series_cache.json` book series. Phase 2: needs per-trial OOS return **panels** logged during sweeps (internal regenerable data, not external). Source: Romano & Wolf 2005 Econometrica; Clarke-Romano-Wolf 2020 Stata J.

**T3 — Hansen-Lunde-Nason Model Confidence Set.** conf 0.58-0.70 / value 0.3-0.4. Returns a confidence **set** of statistically-indistinguishable-from-best names — the exact question the book currently answers by decorrelation heuristic. **Report-only, no auto-prune** (book change needs sign-off). Source: HLN 2011 Econometrica 79(2).

**T4 — Stationary/circular block-bootstrap surrogate null.** conf 0.7 / value 0.45. The autocorrelation-preserving Monte-Carlo null for Calmar/Sharpe — complements (does not duplicate) the iid permute-control, which destroys serial structure. **Foundational**: it is the resampling engine T1/T2/T3 are built on. Source: Politis-Romano 1994 JASA; Patton-Politis-White 2009.

**T5 — Harvey-Liu FDP + power (False-and-Missed-Discoveries).** conf 0.68 / value 0.6. *Most decision-relevant.* Estimates π0 (true-null fraction), realized FDP among winners, and the **missed-discovery / power** rate — directly answers the live worry "are 15 rounds cutting real edges (IXP/AAXJ/EWL/DJP)?", which no current tool addresses. Reduced t-stat-cross-section form runnable **now** over `results/round_results.csv` (`real_sharpe`, `n_days` columns present); full time-resampling double-bootstrap gated behind per-trial panel logging. Source: Harvey-Liu "False (and Missed) Discoveries"; Lucky Factors JFE 2021.

**T6 — Selective / post-selection inference on max-Sharpe.** conf 0.7 / value 0.4. Exact truncated-normal conditional p-value/CI on the argmax-selected spec — valid where DSR's normality/effective-N asymptotics fail. Pure scipy/numpy on the cached 149-spec spec-curve. Source: arXiv:1906.00573; arXiv:2502.20917.

**T7 — e-BH FDR (Wang & Ramdas).** conf 0.85 / value 0.3. *Cleanest.* One numpy pass; inputs ready (`evalue_monitor.betting_eprocess` already mints per-champion e-values). Controls FDR under **arbitrary dependence** — strictly more honest than the PRDS-assuming p-value LOND/BH given known trial correlation. Self-test: 10×-duplicating a champion must not inflate the discovery count. Source: Wang & Ramdas 2022 JRSS-B 84(3), arXiv:2009.02824.

**T8 — Square-root-law (Toth/Bouchaud) market-impact cost stress.** conf 0.78 / value 0.5. Adds the **capacity/AUM ceiling** dimension flat-bp cannot express (`cost_stress.py` confirmed flat-only). impact_bps = Y·σ_cc·√(Q$/ADV$), Y∈[0.5,1] swept; uses logdollar dollar-volume + close-to-close vol (no true OHLC); re-charges fills only (never moves positions → does not reopen the closed sizer lever). Reports the AUM at which each name's net Calmar < buy-hold. Sanity gate: Q/ADV→0 converges to existing flat-bp. Source: Toth et al. 2011 PRX 1:021006; TSE universality arXiv:2411.13965.

**T9 — Conformal test martingale (Simple Jumper) drift gate.** conf 0.65 / value 0.3. Anytime-valid covariate/score-distribution-shift detector on the model's own nonconformity scores — fires *before* PnL decays. Distinct from `decay_monitor` (Page-Hinkley/CUSUM on realized **returns**) and `evalue_monitor` (mean>0). Reuses the embargoed isotonic calibration array. De-risk gate, not a position change.

**T10 — CONCH conformal changepoint localization.** conf 0.55 / value 0.35. Finite-sample-valid **CI for where** decay broke (current monitors return only a first-detection index). Validate on UUP (known stale) vs GLD (healthy). Source: arXiv:2505.00292 (Ramdas group, 2025).

**T11 — Stochastic-dominance e-test.** conf 0.65 / value 0.25. Whole-distribution / left-tail test vs buy-hold (`evalue_monitor` tests mean>0 only) — would expose a mean-only edge that is fragile in the tail. Reuses the Ville/supermartingale machinery. Source: Waudby-Smith & Ramdas 2024 JRSS-B 86(1).

**T12 — RADABOUND / reusable-holdout adaptive-data-analysis bound.** conf 0.55 / value 0.5. Bounds generalization error from **sequential adaptive reuse of one OOS window** across 15+ rounds — a leak class DSR/MinBTL/LOND/PBO do not cover (they count static best-of-N or stream-FDR, not adaptive holdout reuse). Source: Dwork et al. Science 2015; RADABOUND arXiv:1910.03493.

**T13 — E-detector (Shiryaev-Roberts mixture) flatten gate.** conf 0.5 / value 0.2. *Borderline.* ARL-calibrated anytime-valid restart-mixture changepoint as a de-risk overlay. **Structural risk**: flattening reduces trade count → threatens the trades≥80 floor on marginal names, and neighbors the closed sizer + closed crash-veto sleeves. Single cheap A/B that will most likely close. Source: NEJSDS 2024, arXiv:2203.03532.

Portfolio/book TOOLING (book is downstream/human, not the research target → flag-only, sign-off required):

**P1 — True ERC via cyclical coordinate descent.** conf 0.55 / value 0.3. Note: `portfolio_weights.py` already A/B's equal/Calmar²/DSR/inverse-variance and found the 6-name book **robust** to weighting; on 6 already-decorrelated sleeves (MaxDD 2.46%) ERC's marginal lift over inverse-variance is likely negligible. Source: Griveau-Billion et al. arXiv:1311.4057.

**P2 — Schur complementary allocation.** conf 0.5 / value 0.25. Single-knob HRP↔MVP interpolation. At N=6 the cross-sleeve covariance is noise-dominated and γ-tuning is overfit-prone; low value. Source: arXiv:2411.05807.

## 3. Confirmed closed / infeasible (v3 re-surfaced — shows the frontier is mapped)

- **All new edge mechanisms/labelers/axes** — CLOSED; β200 lens explains fit-ability; sticky-HMM (predictable-not-profitable); all change-point/regime-labeler family DISCARD.
- **All feature-adds** (evt / dispersion-entropy / path-signature / termstruct / sig / FFD / VR / DC) — 0/N this session; cross-asset features (GLD←UUP 3.29, USO←XOP) CLOSED.
- **Risk/sizing family from the verdict list** — CLOSED_THIS_SESSION: Wasserstein-Kelly, prospect-theory/RDU sizer, BOCPD inverse-hazard sizer, parameter-uncertainty Kelly haircut, friction-Kelly hysteresis, EVaR/CVaR/ES tail-budget, risk-constrained Kelly, EPO, HERC-CDaR. Garleanu-Pedersen (aim) Calmar-negative. **Sizer lever closed.**
- **More feature variants** — CLOSED_THIS_SESSION: jump-robust RV/bipower (MedRV/MinRV/BNS), Roll/serial-cov effective spread, realized-quarticity HAR-Q attenuation, intraday-MAX/lottery, rough-vol Hurst path. **Feature lever closed.**
- **Calibration** — Venn-Abers NEUTRAL (3.92<4.02; de Prado CDF already maps p→size).
- **Model objectives** — survival:aft + arctan-pinball/GMADL custom-obj **PLATFORM-BLOCKED** (crash QC inference). Only standard `XGBClassifier(binary:logistic)`.
- **EVT/DSPOT/Hawkes crash sleeve** — SPY/QQQ crash-veto hits the structural trades<80 wall (~44); continuous crash features cannot raise flatten-frequency. Dead.
- **Deep-sweep fits** — SPXL 2.82 STALE (re-runs 0.05); XME/IAU/QLD high-Calmars are vol-axis/no-baseline flukes. Trust trend_leg > sliced_wasserstein.
- **DEAD portfolio methods**: NCO+RMT, CRISP, Gerber-statistic covariance, geometric/Berry-phase spectral overlay.
- **ALREADY_HAVE** (don't rebuild): SAFFRON/ADDIS (have LOND), anytime-valid Sharpe/Calmar confidence sequence (have e-value/`betting_eprocess`), confidence-sequence-crossing detector, multi-stream change detection with online-FDR.

**Factual correction to the candidate set (flag):** the SPA entry claiming "arch verified present, pure numpy" is **wrong** — `import arch` raises ModuleNotFoundError in this env. All bootstrap tooling (T1-T4) must be **hand-rolled numpy** (still in-rule and feasible — a stationary bootstrap is ~15 lines — just more work than that entry implied). The competing entries that already say "arch NOT installed, hand-roll numpy" are the correct ones.

## 4. Needs new data (real but out-of-scope — verdict path-1, needs user sign-off)

- **Options IV / skew / implied-vol term-structure** — the canonical missing tail/sentiment modality.
- **COT / positioning / fund-flow** data.
- **VIX term-structure / VVIX** (VXX→SPY crash was negative R1232, but native VIX TS is unbuilt).
- **Credit spreads** (HY-IG, OAS) as a macro-risk gate.
- **True per-bar OHLC** — `bar_builder` proxies high/low via min/max of minute closes; without real OHLC the entire **range-vol family is infeasible** (Parkinson / Garman-Klass / Yang-Zhang / Rogers-Satchell). This is a data wall, not a method gap. Note E1/E2 deliberately avoid it (minute-close distribution, not true intrabar extremes).

## 5. Verdict

**EDGE: structurally closed.** There is no positive-prior new edge. The one remaining feasible *edge shot* is **E1 (signed-semivariance features)** — genuinely untested on a distinct minute-resolution channel — but with a strong negative prior; run it as **one decisive A/B**, expecting it to formally close the last open feature channel rather than open one. H₀ (no more independent single-ticker edges) holds empirically, mechanistically, and structurally.

**TOOLING: there IS real new feasible work — but it can only validate, never create edge.** The verified high-value gap is the entire **stationary-bootstrap data-snooping family** (0 hits repo-wide): **T1 Hansen SPA / Reality Check** is the strongest single build (runnable today on cached series, pure numpy, recentered max-statistic under realized correlation — the formal answer to "does GLD/USO survive the 3483-trial search"), and **T5 Harvey-Liu FDP+power** is the most decision-relevant (it can tell you whether the 15 rounds discarded *real* edges). v2's Lo-SE (A1) and spec-curve (A3) were the first two members of exactly this family; v3's job was to map the rest — done.

**One real new FEASIBLE shot, or structurally closed?** Honest answer: **structurally closed for edge; open only for honesty/validation tooling and new-data modalities.** If forced to name a single highest-leverage new build, it is **T1 (Hansen SPA / Reality Check)** as a tightening tool, with **E1** as the one — and last — edge A/B worth running to confirm the feature channel is shut. Neither is an edge-creator. The valuable, honest result is that the frontier is mapped and closed.

Key docs grounding this: `/home/ubuntu/lb/docs/research/WANG_INTERNET_DEEP_V2_2026-06-08.md`, `/home/ubuntu/lb/docs/research/NEW_METHODS_2026-06-08.md`, `/home/ubuntu/lb/docs/analysis/SESSION_SUMMARY_2026-06-08.md`, `/home/ubuntu/lb/program.md`. Verified-absent (build targets): `/home/ubuntu/lb/scripts/stats_rigor.py` (no bootstrap/SPA/MCS/StepM/selective/e-BH), `/home/ubuntu/lb/scripts/cost_stress.py` (flat slippage only), `/home/ubuntu/lb/scripts/harvey_liu_haircut.py` (no FDP/power), `/home/ubuntu/lb/modules/features.py` + `/home/ubuntu/lb/modules/bar_builder.py` (semivar/signedjumpvar are axes, never features).