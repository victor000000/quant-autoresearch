# Deep Wang+internet re-investigation v2 (2026-06-08)

Verified. Key claims hold: `harvey_liu_haircut.py:33` uses the iid t-stat `sr_ann*sqrt(n/PPY)` (no autocorrelation adjustment — the Lo gap is real); `features.py:build_feats` is close-to-close only (no Parkinson/Yang-Zhang/Garman-Klass); `evalue_monitor.py` tests only H0: mean-return≤0 (no whole-distribution dominance); the 65 "wasserstein" ledger hits are all the `sliced_wasserstein` *labeler*, not a Wasserstein-Kelly *sizer* (that sizer is genuinely unraced); and surviv/aft/hawkes/knockoff/venn/abers/kelly/pinball/parkinson/rocket/euler/bootstrap = 0 hits across the 3483-row ledger. Report follows.

---

# Deep Wang+internet re-investigation (2026-06-08 v2) — anything genuinely NEW after 15 hypothesis rounds?

## 1. Bottom line

**Mostly no — with one honest qualifier.** On the *edge side* (a new mechanism or input that could lift Calmar), the deep pass + newest-internet sweep found **nothing that is simultaneously novel, in-rule, untested-this-session, AND carries a positive prior.** Every genuinely-new edge candidate it surfaced lands in a family with a documented **0/N track record this session** (feature-adds, sizers, labelers all DISCARDED) or hits a **known structural wall** (the SPY/QQQ crash sleeve's trades<80 cap that just killed the EVT conversion). The three booked mechanisms (TREND/GLD, REGIME, MEAN-REVERSION/USO) remain asset-intrinsic; the universe screen already confirmed they are the only edges.

**The one qualifier:** the pass *did* find real, un-built **honesty-stack gaps** — most importantly a genuine **correctness bug** in the multiple-testing haircut (iid Sharpe t-stat ignoring autocorrelation of overlapping bars). These items can only *tighten or validate* the existing book, never create edge, but several are clearly worth building because they need **no sign-off and carry no fake-readmission risk.** That is a valuable, in-scope result even though it is not "new alpha."

So: **no new alpha lever clears the bar; ~1 genuine edge-side shot is worth a single disciplined race (Survival/AFT); the real remaining yield is bar-tightening tooling, not edge.**

---

## 2. Genuinely-new actionable levers (NOVEL_ACTIONABLE only), ranked

Ranked by *expected value × confidence-it-is-truly-new*, not by raw novelty. Tier A = build (tightens/validates the book, no sign-off, ~0 risk). Tier B = one gated race each (genuinely-new edge mechanism, real but low prior). Tier C = technically-novel-but-low-EV (compressed).

### Tier A — honesty tooling that is NOT built and can only raise the bar (build these)

**A1. Lo autocorrelation-adjusted Sharpe SE + block-bootstrap Sharpe CI**
- *Idea:* Replace the iid t-stat in the Harvey-Liu/DSR haircut with Lo's autocorrelation-corrected SE: `SE_adj = SE_iid · sqrt(1 + 2·Σ_{k=1..q}(1−k/q)·ρ_k)`; cross-check with a stationary/circular block-bootstrap Sharpe CI.
- *Source:* Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal* (2002). (Real, pre-cutoff, verified.)
- *Why not already tested:* **Confirmed code gap** — `scripts/harvey_liu_haircut.py:33` literally computes `sr_ann * math.sqrt(n/PPY)`, the iid statistic, on *overlapping weekly bars*. Every downstream Bonferroni/BH/DSR haircut is therefore anti-conservative exactly where bars overlap. Not MinBTL, not LOND — those are built; this is orthogonal.
- *Experiment:* Add `lo_se()` to `scripts/stats_rigor.py`; in `harvey_liu_haircut.py` divide each champion's t by the Lo factor (ρ_k from that champion's stored OOS per-bar returns, which already exist). Add a block-bootstrap CI column (block≈20, 10k resamples).
- *Honesty gate:* Verify every haircut moves in the **conservative** direction; report whether any borderline crown (IWM N≈64, UUP N≈72) now fails. No model/leak/data change. **Unique virtue: it can only ever raise the bar, so needs no FIT-rule sign-off.**

**A2. Anytime-valid stochastic-dominance e-test (strategy ≻ buy-hold)**
- *Idea:* Add a whole-distribution first-order-dominance e-process beside the existing mean-only e-value: `S(λ,z)=1+λ·[1(X≤z)−1(Y≤z)]` over a return-threshold grid; reject H0(no SD over buy-hold) at e≥1/α.
- *Source:* Arnold, Choe, Scarsini, Tsetlin, "Betting on Bets," arXiv:2604.21851 (2026). **Honest flag: this specific arXiv id is post my Jan-2026 cutoff — I cannot personally verify the paper.** The underlying method (betting-based SD e-process; PySDTest lineage, arXiv:2307.10694) is sound and buildable regardless.
- *Why not already tested:* Verified — `evalue_monitor.py:10` tests **only** H0: mean-return≤0. An edge can win on mean yet lose in the left tail undetected. Reuses the existing K_t supermartingale / Ville machinery; swap the e-variable.
- *Experiment:* `betting_eprocess_sd(strategy_rets, bh_rets)` → one "SD-dominance e-value" column for GLD/USO vs each name's own buy-hold.
- *Honesty gate:* Pass the existing zero-mean self-test (false-positive ≤ Ville bound) before trusting.

**A3. Non-Standard Errors + specification-curve honesty artifact**
- *Idea:* Render the *dispersion of Calmar/edge across defensible specs* (researcher-DoF / NSE) and the fraction of defensible specs that beat own buy-hold — a spec-robustness flag per champion.
- *Source:* Menkveld et al., "Nonstandard Errors," *J. Finance* 79(3) (2024); Simonsohn-Simmons-Nelson specification curve (2020). (Real, verified.)
- *Why not already tested:* DSR haircuts best-of-N via EVT but never *plots the distribution* across specs. Pure offline pandas over the existing 3483-row ledger.
- *Experiment:* For GLD/USO define the defensible-spec subset ({logdollar,imbalance}×{trend/regime/revert-family}×{cdf_overlay,dd_overlay}); compute NSE + beat-rate; render a spec-curve panel.
- *Honesty gate:* Diagnostic only, no deploy surface; flags whether GLD/USO are spec-robust or knife-edge.

**A4. Stationary block-bootstrap surrogate-data null (single champion)**
- *Idea:* A serial-structure-preserving null complementary to label-permutation: circular block-bootstrap of champion per-bar returns, recompute Calmar/Sharpe over B≈2000 surrogates → MC p-value.
- *Source:* Politis-Romano stationary bootstrap (standard, pre-cutoff). (No external citation needed.)
- *Why not already tested:* `grep bootstrap|surrogate = 0`. Permutation destroys serial structure, so it **cannot** detect a strategy profiting purely from autocorrelation/momentum-in-noise — a strictly harder, complementary null. Also a cheap stepping-stone to the queued-but-unbuilt Hansen-SPA (T4).
- *Honesty gate:* Sanity — buy-hold-of-noise → p≈0.5; real edge → small p with structure preserved. Add as a column beside permute.

*(Adjacent Tier-A-ish, lower priority: TSKI feature-FDR knockoffs, and the synthetic-controlled validator harness — both real, un-built, offline, no leak surface. They are genuine gaps but lower-confidence/heavier; build after A1–A4. The Breaking-the-Trend correlation-aware selection correction overlaps the same T4/T6 dependence-bootstrap hole and additionally needs per-trial return-vector logging + a sign-off to flip the gate — so treat as a future ADDED column, not a now-build.)*

### Tier B — one disciplined gated race each (genuinely-new edge mechanism, real but low prior)

**B1. Survival / AFT target — censored time-to-profit-barrier (native `survival:aft`)**  *(the single strongest "new edge" candidate)*
- *Idea:* Label = forward bars-until-upper-barrier-touch, right-censored at the vertical barrier as `[a,+∞)`; train `XGBoost(objective=survival:aft)` inside the existing pipeline; replace isotonic with a monotone time-rank→conviction map (short predicted time ⇒ high long conviction) feeding the unchanged de Prado CDF × inverse-vol sizer.
- *Source:* Barnwal, Cho, Hocking (2022); shipped in XGBoost since 1.2. (Real, pre-cutoff, verified.)
- *Why not already tested:* `surviv/aft/hazard = 0` in the ledger; all 41 registry labelers are hard classification. This is a **new mechanism (speed-of-payoff under right-censoring)** distinct from the four booked mechanisms, and it is a **native XGBoost objective** — so it honors the "XGBoost-only" closed lever (unlike SMARTboost, which was infeasible) and leaves the offline-mint→ObjectStore-replay deploy path unchanged.
- *Experiment:* A/B on GLD(logdollar) and USO(imbalance/revert); guard the `[a,+∞)` censoring window ⊆ embargo (extend `tests/test_bar_threshold_leak.py`).
- *Honesty gate:* Beats own buy-hold AND permute-collapse <40% AND decay-monitor holding AND cost@10bp survivable AND `infer.py` preds_match ≤1e-6. **Skeptic note:** label-add track record is 0/N this session — treat as NOVEL_ACTIONABLE, not "will win."

**B2. Range-based (OHLC) volatility features — Yang-Zhang / Parkinson / Garman-Klass / Rogers-Satchell**
- *Idea:* Rolling Parkinson + Yang-Zhang + Garman-Klass over {20,50,200} bars + a short/long YZ ratio into `build_feats`, reduce=infogain; optionally swap the inverse-vol overlay denominator to YZ.
- *Source:* Petnehazi (2019), *Intelligent Sys. in Acct./Finance/Mgmt* 28; 2023 G7 evidence (Finance Research Letters v55). (Real, verified.)
- *Why not already tested:* `features.py:build_feats` is **close-to-close only** (confirmed: sample_entropy/RSI/BB/momentum + the dead disp/sig/evt adds; zero parkinson/yang/garman/rogers hits). Intrabar high-low range is a genuinely different information source (path vs endpoint) — every dead feature this session (FFD, VR, dispersion-entropy, path-signature) was close-based. It is the team's own NEW_METHODS #1 unbuilt candidate.
- *Verified caveat:* `bar_builder.py:1063` proxies per-bar high/low by max/min of minute closes (no true OHLC) — estimators run on a proxy range, marginally less efficient but implementable.
- *Honesty gate:* KEEP iff cost-stressed Calmar > champion (GLD 4.02 / USO 3.85) AND permute-collapse <40% AND trades>80. **Skeptic note:** 0/N feature prior; this is the least-bad of the feature track only because it is a new *information source*, not a new transform of the same close series.

**B3. Composite quantile target via Arctan Pinball Loss (native XGBoost custom objective)**
- *Idea:* `labeler='quantile_pinball'`: fit XGBoost(depth3) with the arctan-pinball custom objective on vol-normalized fwd return at τ={.05,.25,.5,.75,.95}; conviction = sign(q50) scaled by asymmetry `(q95−q50)/(q50−q05)` → de Prado CDF bet.
- *Source:* Hatalis et al., "Composite Quantile Regression With XGBoost Using the Novel Arctan Pinball Loss," arXiv:2406.02293 (2024). (Real, verified.)
- *Why not already tested:* No quantile/pinball/custom-objective in the registry. The arctan-pinball has an analytic non-vanishing Hessian native to XGBoost's `obj=` callable — this **specifically sidesteps the torch blocker that killed GMADL.** A distributional (skew-as-conviction) target is a new *representation*, not a 42nd sign-labeler.
- *Honesty gate:* Calmar>champion AND permute-collapse<40% AND trades>80 AND **not** (high val_auc + low Calmar) — the sticky-HMM predictable-not-profitable trap.

**B4. Venn-Abers calibration (the one untouched module)**
- *Idea:* Replace/augment isotonic with two PAV fits (label-pinned to 0 and to 1) → interval `[p0,p1]`; midpoint → CDF bet, or abstain (size 0) when the interval straddles the threshold.
- *Source:* Vovk & Petej (2014); van der Laan & Alaa, arXiv:2502.05676 (2025) for the generalized variant. (Real, verified.)
- *Why not already tested:* `venn/abers = 0` in the ledger; calibration has never been A/B'd. Venn-Abers **is** two isotonic/PAV fits, so it inherits the existing embargoed-VAL leak-safe training and frozen-map replay — no new deploy surface.
- *Honesty gate:* net-Calmar@10bp ≥ 4.02 AND turnover/cost drop AND permute still collapses. **Skeptic note (strong):** the de Prado CDF bet already maps p→size and the standing lesson is "calibration overlays don't create edge" — expect at best a turnover/cost trim on GLD's 602-order erosion, not new Calmar.

### Tier C — technically NOVEL_ACTIONABLE but low-EV (race only if Tier A/B exhausted)

| Candidate | Honest one-line | 
|---|---|
| Transfer-entropy reduce method | Only genuine angle is ranking *lagged exogenous* channels (gold←UUP, oil←energy-eq) that static MI can't order; build the cheap pure-numpy version, **not** F-PCMCI/tigramite. reduce-EV prior is weak (infogain helped only val_auc>0.6 names; mrmr never worth racing). |
| Euler-characteristic / QUANT / MiniRocket / scattering / randomized-signatures / NVAR features | Each is a distinct *object*, but all sit in the 0/N feature-graveyard and the last three are mechanistically adjacent to the already-dead path/log-signature add. Race **at most one** (QUANT or Euler-char) after range-vol. |
| GROW e-process / coin-betting / W-Kelly / risk-constrained-Kelly / CPS sizers | All genuinely-unraced sizing *mechanisms*, but the sizer lever is closed and the closest principled analog (Garleanu-Pedersen "aim") was **Calmar-negative this session**. At best a MaxDD/turnover trim. One cheap race only, on a decay-stale name (UUP/IWM) where defensive de-sizing is the point. |
| ACI / CPTC online-conformal coverage gate | Honesty tooling (drift-robust coverage), leak-safe, but must first be shown to fire on calibration drift that the e-value/Page-Hinkley monitors miss, else redundant. |

---

## 3. Confirmed re-treads — the deep pass re-surfaced these, we already tested/killed them

This is the strongest signal that the frontier is mapped: most "new" ideas map cleanly onto an already-dead family.

- **Crash-sleeve features — Hawkes jump-intensity, BOCD-AR collective-anomaly, TDA persistence-entropy, Spectral-Residual saliency.** All four target the *same* SPY/QQQ crash-veto lead and all hit the **identical trades<80 structural wall that killed the EVT/DSPOT conversion this session** (evt_tail_score, ~44 trades). Adding a continuous feature does not raise the flatten-frequency that sets the trade count. Re-tread *by failure mode.*
- **Sizers — W-Kelly, risk-constrained Kelly, coin-betting/KT, CPS Bayes-Kelly, GROW-sizer.** Sizer lever is closed; the Garleanu-Pedersen "aim" sizer was raced this session and came back **Calmar-negative** (ledger 2026-06-08, aim/aim_dd 3.70<4.02). The de Prado CDF bet is already an uncertainty-aware shrinkage. Re-tread *by mechanism.*
- **Weak-supervision LabelModel / soft-label smoothing.** Label-blending was tested this session as the trend-strength **ensemble** and DIED ("blend dilutes," GLD 3.66<4.02, also lost TLT). Pooling 41 cross-mechanism labelers on a single trend asset dilutes worse. Re-tread *by mechanism.*
- **Robust-BOCPD / BOCD-AR / nearest-historical-analog regime.** The entire change-point/regime-labeler family is documented DISCARD; sticky-HMM died (predictable-not-profitable, GLD 0.40 @ val_auc 0.878). Robustifying the detector doesn't establish the CP signal carries edge here.
- **Tail-weighted entropy.** Sits at the intersection of two closed graveyards — the entropy refactor (commit 1d5916f, "tested + closed, not viable") and the EVT crash work. Doubly-discounted.
- **Randomized signatures / wavelet scattering / NVAR.** Nearest neighbour is the path/log-signature add — explicitly the session's "4th feature-add fail."
- **Directional-change features.** `dc` is already one of the 21 sampling axes (tested-dead: SPXL 0.75, USO −0.12), and dc_trend/dc_reversal labelers lost to champions. DC features crowd trend_leg+regime_gmm.
- **Flagged ALREADY_HAVE by the pass itself:** CPCV→multi-path PBO, conformal test-martingale regime feature, sequential testing-by-betting best-arm, score-driven AR-BOCPD, weighted-conformal monitoring (WATCH), repeated-FCS-Detector, asymptotically-optimal e-detector. The honesty stack already covers these.

---

## 4. Needs new data — real but out-of-scope (requires user sign-off)

These are genuinely real and potentially edge-bearing, but introduce a banned external modality:

- **Options-implied vol / skew / term-structure** (the natural crash-sleeve fix — a forward-looking risk input the price series cannot supply; would directly attack the trades<80 wall by widening the flatten signal).
- **COT / positioning / dealer-gamma / order-flow** (the documented H₀ escape hatch).
- **VIX term-structure and credit spreads** (regime conditioning beyond endogenous price).
- **Harvey et al. (2025) nearest-analog regime detection — the MACRO state-variable variant.** The *endogenous* variant (own price/vol + pre-cleared cross-asset) is in-rule and listed in Tier C; the macro variant is NEEDS_SIGNOFF.
- **Time-series foundation-model zero-shot embeddings** — DEAD_OR_INFEASIBLE under the current platform (torch banned in QC inference; cannot mint/replay leak-free). Out of scope until the deploy path changes.

Note: exogenous **cross-asset features on a single traded ticker** are *already pre-cleared* (gold←UUP, oil←energy-eq) and are NOT new data — they belong in Tier B/C, not here.

---

## 5. Verdict

**The edge frontier is effectively closed — H₀ holds — with exactly one honest exception worth a single race.**

- On *new alpha*: the deep pass + newest-internet sweep confirms the working hypothesis. The three booked mechanisms are asset-intrinsic; no surfaced candidate is simultaneously novel, in-rule, untested, and positive-prior. The crash sleeve is gated by a *structural* trades<80 wall that four different new crash features cannot move; the sizer and label/feature levers are 0/N this session and the deep pass mostly re-surfaced their graveyards.
- **The one genuine in-rule edge-side shot:** **Survival/AFT (`survival:aft`)** — a new target *mechanism* (speed-of-payoff under censoring), native to the allowed learner, with an unchanged deploy path. It deserves exactly one disciplined gated race. Its prior is poor (label-add 0/N), so the rational expectation is that it joins the graveyard — but it is the only candidate that is a *new mechanism* rather than a re-skin of a dead one. Range-vol features and the arctan-pinball target are credible distant seconds.
- **The real, certain yield is not edge — it is the honesty stack.** A1 (Lo autocorrelation-adjusted Sharpe SE) is a **confirmed correctness bug** in the haircut, and A1–A4 can only *tighten/validate* the book with no sign-off and no fake-readmission risk. Building them is the highest-confidence work the deep pass produced. They will not raise any Calmar; they may *lower* a borderline crown (IWM N≈64, UUP N≈72) — which is exactly the point at this stage of the project.

**Plain statement:** After 15 hypothesis rounds, the deep re-investigation found no new edge that beats the asset-intrinsic incumbents without a new data modality. That is a valid and valuable result. The correct next moves are (1) build the Lo SE correction + spec-curve + SD e-test to harden the existing book, and (2) spend one race on Survival/AFT as the last in-rule edge experiment — not to keep grinding the closed feature/sizer/label levers.