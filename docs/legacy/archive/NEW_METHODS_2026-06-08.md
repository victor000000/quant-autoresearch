# New methods to try — Wang transcripts + internet (2026-06-08)

# New methods to try — Wang transcripts + internet research

Audience: the autonomous single-ticker ETF research loop, choosing the next A/B. Scoring = fit × novelty × evidence from the verified verdicts (composite shown as `f·n·e`). "EDGE" = a driver A/B the loop actually races; "TOOL" = an honesty/validation-stack upgrade (no new edge, but changes what we trust). Sources are cited inline; none invented.

---

## 1. Executive summary

- **Honest bottom line: internet research surfaced clearly more in-rule, evidence-backed material than Wang did — but most of the *highest-scoring* hits are validation tooling, not new edges.** Wang's concrete external leads are exhausted: every buildable one is already adopted (`infogain`) or raced-and-dead (trend-strength ensemble = his "Lever 1"; sticky-HMM; β200-as-router). The genuinely-new candidates came from the literature search, not the transcripts.

- **The two top-composite items are both honesty tooling that fixes our #1 self-identified deficit** (multiple-testing accounting + staleness): **MinBTL sufficiency gate** (`5·4·5=100`) and **online-FDR / N_eff-deflation / Hansen-SPA** family. These don't create alpha; they stop the loop re-litigating borderline crowns (IWM N=64, UUP N=72) and grinding 0-EV rounds. Cheap, leak-trivial, immediately in-rule.

- **On the *edge* side, be skeptical: our priors are stacked against most of it.** Feature-block additions are 0/2 (VR and FFD both *crowded and hurt* GLD even under infogain); label swaps lose to the asset-intrinsic champion; sizing overlays "do not create edge." So a marginal fit-chase on GLD is low-EV regardless of how new the feature sounds.

- **The one edge play with a credible path to a *genuinely new deployable* result (not a marginal GLD tweak) is converting the SPY/QQQ crash-veto lead.** That lead is permute-confirmed real (Calmar 2.53, val_auc 0.547) but non-deployable because it is threshold-fragile and trades<80 (73). **DSPOT (self-calibrating EVT tail) and a Hawkes jump-intensity feature** replace the brittle fixed threshold with a continuous self-calibrating one — directly targeting the documented non-deployability, consistent with our own lesson "new mechanism (tail-avoidance) > new tuning."

- **The strongest *incremental* in-rule edge candidate is range-based volatility (Yang-Zhang / Parkinson / Garman-Klass) — `f5·n4·e4`, confirmed unbuilt, dual-payoff (sharper vol features *and* a lower-variance sizing denominator), pure-numpy OHLC, zero new deps.** Tempered only by the 0/2 feature-crowding prior, which is why it ships with a hard "keep iff beats champion + permute-collapse" gate. **Dispersion entropy** and the **Garleanu-Pedersen "partial-adjustment" sizer** are the next two (`80` each): the former fills the exact gap our entropy-refactor closure named (fast *and* amplitude-aware); the latter is a distinct turnover mechanism aimed at GLD's documented 602-order cost erosion.

- **Net: little genuinely-new *edge* is likely in-rule, but a few real ones are worth one race each, and the validation-tooling wins are real, cheap, and overdue.** Recommended next build is the crash-lead conversion (highest upside) plus the MinBTL gate (near-zero cost) — details in §6.

---

## 2. Top candidates to try (ranked)

### 2A. New-edge driver A/Bs (what the loop races)

**#1 — Range-based volatility estimators (Yang-Zhang / Parkinson / Garman-Klass / Rogers-Satchell)** · `f5·n4·e4`
- *What it is:* drift-independent OHLC volatility estimators ~5–8× more efficient than close-to-close.
- *Source:* Yang & Zhang, *J. Business* 73(3):477-491 (2000); Korkusuz, Kambouroudis & McMillan, *Finance Research Letters* 55 (2023), https://www.sciencedirect.com/science/article/pii/S1544612323003641 ; Portfolio Optimizer overview, https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/
- *Why novel vs have/dead:* `modules/features.py` is close-to-close only (momentum / MA / sample_entropy / RSI / BB); RS± exist only as bar-sampling *axes* (semivar/signedjumpvar), never as features or sizing-vol. Confirmed unbuilt; it is EXPLORATION_PLAN_311 step-8 quick-win.
- *Fit + leak:* pure past-only numpy on existing bar OHLC → StandardScaler → infogain → XGBoost; leak-safe, tree-compatible, no torch/sklearn. **Prereq: confirm `bar_builder` exposes per-bar open/high/low** (plan asserts it; not verified in the bar classes).
- *Proposed A/B:* {GLD, logdollar, `trend_leg+regime_gmm`} and {USO, imbalance/revert} — add rolling Parkinson + Yang-Zhang over {20,50,200} + a short/long YZ ratio, reduce=infogain. Second arm: swap the inverse-vol overlay denominator to YZ vs close-to-close. Score cost-stressed Calmar **and** turnover. Keep iff Calmar>champion AND permute<40% real.

**#2 — Dispersion entropy (NCDF mapping)** · `f5·n4·e4`
- *What it is:* O(N) complexity measure that is amplitude-sensitive (via NCDF symbolization), unlike permutation entropy.
- *Source:* Rostaghi & Azami, *IEEE Signal Processing Letters* 23(5):610-614 (2016); EntropyHub (PMC8568273).
- *Why novel vs have/dead:* fills the exact missing quadrant our entropy-refactor closure (commit `1d5916f`) named — sample_entropy is amplitude-aware but O(W²)-slow; permutation entropy is O(N) but ordinal-only and *degraded* edges (GLD 1.76→1.10). Dispersion entropy is fast **and** amplitude-aware. Already on our NEW_METHODS_BACKLOG.
- *Fit + leak:* ~20 lines numpy, causal trailing window, feeds StandardScaler→infogain→XGB unchanged; no torch/sklearn/HMM.
- *Proposed A/B:* {GLD, logdollar} (where sample_entropy is Calmar-load-bearing) and {IWM} — add causal dispersion-entropy (m=2–3, c=6) alongside sample_entropy, reduce=infogain. Verify byte-identical fallback when off. Same KEEP gate.

**#3 — Trade-partially-toward-aim sizer (Garleanu-Pedersen)** · `f5·n4·e4`
- *What it is:* convex partial position adjustment `x_t=(1-a)x_{t-1}+a·aim_t` — move a fraction toward target each bar instead of snapping.
- *Source:* Garleanu & Pedersen, *Journal of Finance* 68(6):2309-2340 (2013), SSRN 1364170; NBER w15205.
- *Why novel vs have/dead:* distinct from `rebal_band` (a no-trade *deadband*) — this is a smooth *partial* adjustment, a different mechanism. The single-asset reduction respects single-ticker (no pairs). Directly attacks the trim-cost term that gates trend names.
- *Fit + leak:* pure causal recursion on past positions wrapping the existing `cdf_bet·overlay` target; leak-safe, model untouched. Must mirror the recursion into infer/live templates and confirm `infer_online` preds_match=1.
- *Proposed A/B:* sizing='aim' on {GLD, IWM, logdollar/trend_leg}, scalar `a` from alpha-decay≈1/horizon + cost; A/B vs cdf_overlay and dd_overlay on **net** Calmar @5/10bp; gate trades>80, net-Calmar up, permute-stable.

**#4 — DSPOT streaming peaks-over-threshold (EVT tail detector)** · `f4·n4·e4`
- *What it is:* drift-adjusted streaming EVT/GPD that maintains a self-calibrating extreme-quantile threshold.
- *Source:* Siffer, Fouque, Termier, Largouet, "Anomaly Detection in Streams with Extreme Value Theory," KDD 2017; impl github.com/cbhua/peak-over-threshold.
- *Why novel vs have/dead:* no EVT/GPD model in the 41-labeler / 21-axis registry; our crash work is `crash_ahead` label + `crashveto`/`ddbreaker` sizers (fixed threshold), not EVT. Method-of-moments GPD is closed-form pure-numpy (sidesteps no-scipy-MLE), forward-only calibration = leak-safe.
- *Fit + leak:* causal tail-exceedance feature AND a breach-flag modulating crashveto.
- *Proposed A/B:* {SPY (then QQQ), jump axis + crash_ahead} — add DSPOT exceedance-prob feature and wire breach-flag into crashveto (flatten only on EVT breach); A/B vs the existing crash_ahead+crashveto. **Target = convert the lead: trades>80, val_auc>0.52, beats-BH, permute collapse.**

**#5 — Hawkes self-exciting jump-intensity feature** · `f4·n3·e4`
- *What it is:* continuous self-excitation intensity `λ(t)=μ+Σα·exp(-β(t-t_i))` over detected jumps — a smooth "jumps cluster" signal.
- *Source:* Bacry, Mastromatteo & Muzy, "Hawkes processes in finance," arXiv:1502.04592 (2015); Liu et al., *Finance Research Letters* 55 (2023), https://www.sciencedirect.com/science/article/abs/pii/S154461232300212X
- *Why novel vs have/dead:* we already detect jumps (Lee-Mykland for the jump axis); MLE `(μ,α,β)` on TRAIN then a *frozen-parameter* causal recursion online is allowed offline-mint + causal-deploy, NOT an online-HMM. A Hawkes-intensity *axis* is on the deferred backlog; intensity-as-*feature*/crash-sizer is unbuilt.
- *Fit + leak:* pure causal EWMA-over-jumps; no torch; tree-compatible. Endogenous, so **needs no external VIX data** (key vs the dead vol-product track).
- *Proposed A/B:* {SPY, jump axis} — frozen-Hawkes `λ(t)` as a causal feature (and a smooth crash-sizer multiplier) vs the fragile crash_ahead/crashveto; primary win = trades>80 + permute collapse + Calmar>BH. Pairs naturally with #4.

**#6 — Path / log-signature features** · `f4·n4·e4` (`e3` for the iisignature variant)
- *What it is:* truncated (better: log-) signature = order-aware path geometry; level-2 term is signed Lévy-area lead-lag.
- *Source:* Chevyrev & Kormilitzin, "A Primer on the Signature Method in ML," arXiv:1603.03788; Lyons & McLeod, arXiv:2506.01815 (2025); Reizenstein & Graham (iisignature), arXiv:1802.08252.
- *Why novel vs have/dead:* grep signature=0; flagged in NEW_METHODS_BACKLOG (#3 order-sensitive geometry that sliced_wasserstein/fracdiff/spectral are blind to). Clears the no-torch wall — depth-2/3 log-sig of ≤3 channels is ~30 lines of numpy iterated integrals (no torch `signatory`).
- *Fit + leak:* past-only window → infogain → XGB (universal approximation gives expressiveness without depth>3); multi-file render for the 64k limit. **The now-allowed exogenous channel** (single traded ticker) enables a gold←UUP level-2 lead-lag without forming a pair.
- *Proposed A/B:* {GLD, IWM, logdollar} features='sig' (channels: log-price + signed-imbalance, depth 3, trailing window), reduce=infogain; second arm adds an exogenous UUP log-return channel. May need the clustered-MDA reduce (#11) so the correlated block isn't crowded out.

**#7 — Monotonic constraints in XGBoost (mechanism priors)** · `f4.5·n3.5·e4`
- *What it is:* `monotone_constraints` forcing known-sign features to be monotone in the prediction.
- *Source:* XGBoost docs, https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html (secondary arXiv:2512.17945 not load-bearing).
- *Why novel vs have/dead:* grep finds "monotone" only in bar-threshold comments, never as a constraint. Sits on the model-config lever our component guide flags as the unswept n=1 grid. Encodes the known mechanism sign (trend/momentum +1 on trend_leg names; oversold −1 on oil revert) → a robustness/decay regularizer, not a fit-chase.
- *Fit + leak:* single training-time param; leak-irrelevant by construction; depth-3 rs42 unchanged. Implementation note: apply signs to *post-infogain selected columns* per fit.
- *Proposed A/B:* {GLD, trend_leg} +1 on persistence/efficiency features, 0 elsewhere; score real Calmar **and decay-monitor stability** vs unconstrained champion. Mirror with −1 on {USO, revert} oversold features.

**#8 — Oracle / Optimal Trend Labeling (DP-optimal) + train-free label-robustness metric** · `f4·n4·e4`
- *What it is:* cost-aware global DP-optimal trend label; plus a train-free metric ranking labelers by return-degradation-under-accuracy-loss.
- *Source:* Kovacevic, Mercep, Begusic & Kostanjcar, "Optimal Trend Labeling in Financial Time Series," *IEEE Access* 11:83822 (2023), https://ieeexplore.ieee.org/document/10210534/
- *Why novel vs have/dead:* the cost-aware *global* DP optimum is distinct from trend_leg (local-leg heuristic) and calmar_scan/sharpe_scan (objective-aware but not per-switch-cost-optimal). New labeler on the weakest name is explicitly allowed.
- *Risk (flag hard):* an oracle/perfect-hindsight label is the textbook predictable-not-profitable trap (cf. sticky-HMM 0.40@0.878, sliced_wasserstein) — the explicit per-switch cost is the only thing blunting it. The bigger prize is the **train-free robustness metric** — a label-QUALITY pre-screen our honesty stack lacks.
- *Proposed A/B:* {GLD, IWM, logdollar} labeler='oracle_trend' (DP segmentation maximizing return net of 5bp per-switch cost) vs trend_leg champion, **extra scrutiny**: require Calmar>champion AND permute<40% AND not (high val_auc + low Calmar). Separately prototype the robustness metric offline and confirm it ranks trend_leg (winner) above sticky_hmm (loser) on the existing screen log before any race.

**#9 — Venn-Abers calibration + selective (abstention) sizing** · `f4·n3.5·e4`
- *What it is:* distribution-free calibration giving a validity interval [p0,p1]; abstain (size 0) when it straddles the threshold.
- *Source:* Vovk & Petej, "Venn-Abers predictors," arXiv:1211.0025; OpenReview 2024/25 https://openreview.net/forum?id=kl2SA1N03E
- *Why novel vs have/dead:* drops into the exact slot our single isotonic occupies (fit on the same embargoed VAL, replayed by infer.py) — no new leak surface, no sklearn-in-QC issue. The interval-as-abstention gate is principled selective trading that cuts low-confidence churn → directly relevant to GLD's cost.
- *Proposed A/B:* {GLD, logdollar/trend_leg+regime_gmm/dd_overlay} — Venn-Abers (two PAV fits) → de-Prado CDF bet on the point estimate, abstain when [p0,p1] straddles t=0.40; keep iff trades/cost drop with Calmar≥4.02 (net@10bp improves) and permute still collapses. Then UUP.

**#10 — HAR-RV forward-variance (Corsi 2009)** · `f4·n3·e5`
- *What it is:* OLS-trivial long-memory realized-vol model; here a *forward* variance forecast vs our *trailing* vol.
- *Source:* Corsi (2009), *J. Financial Econometrics*; extension arXiv:1907.08522.
- *Why novel vs have/dead:* our overlay sizes on trailing std_slow/std_fast; HAR is a forward forecast — a different, defensible signal. OLS coeffs fit on TRAIN, applied as causal numpy recursion = leak-safe. Lower novelty (lives in the vol-sizing lane). Strictly dominated as a *vol play* by #1 (range-vol), so run after it.
- *Proposed A/B:* {GLD, IWM} two arms — (a) sizing=har_overlay replacing the inverse-vol mult; (b) har_vol_fcst as an infogain feature. Gate net-Calmar@5bp up, trades>80, permute-stable.

**Tail edge candidates (one race each, lower EV — table):**

| Method | f·n·e | Source | One-line / why-bounded |
|---|---|---|---|
| Clustered MDA / MDI reduce | 48 | LdP SSRN 3517595; mlfinlab | New `reduce` fixing infogain's substitution effect; **enabler** for correlated blocks (signatures). sklearn OK offline in footer.py. |
| catch22 features | 49 | Lubba et al., arXiv:1901.10200 | Canonical 22-feature basis; pure-C/numpy. Tempered by 0/2 feature-crowding prior; only on val_auc>0.6 names. |
| Matrix-profile LEFT/discord features | 49 | Yeh ICDM 2016; STUMPY stumpi | Distance-to-nearest-*past*-pattern; **must use LEFT/streaming MP** (ordinary stump leaks future). numba or numpy fallback. |
| Matrix-Profile FLUSS labeler | 48 | STUMPY; Gharghabi FLUSS/FLOSS | New change-point class — but our entire CP-labeler family is DISCARD; one UUP/GLD race only. |
| Growth-optimal fractional Kelly | 36 | Thorp 2008; AFML ch.10 | No Kelly sizer exists, but b must be ATR/leg-proxy (champions aren't triple-barrier) → likely collapses to a re-thresholded CDF bet. |
| Causal rolling-window VMD IMF features | 36 | Dragomiretskiy-Zosso 2014; arXiv:2509.15394 | Adaptive band-limited IMFs; **must be trailing-window causal**; heaviest per-bar compute = real QC-inference risk. Overlaps wavelet/spectral. |
| EVT-CVaR / CDaR tail sizing | 31.5 | Strub SSRN 2063848 | Inverse-CVaR overlay on fat-tailed USO/SPY; sizing rarely creates edge, at best trims MaxDD. |
| NPMM labeling | 32 | Han, Kim & Enke, ESWA 211:118581 (2023) | Rolling-window min=BUY/max=SELL; high redundancy with turn_scan/tlb_reversal/trend_leg. |
| BOCPD run-length posterior *as features* | 30 | Adams-MacKay arXiv:0710.3742; arXiv:2307.02375 | Reuses existing `bocpd_label` forward filter; expose P_reset/E[run-length] as features. Regime is most-mined → uncertain headroom. |
| Parameter-uncertainty Kelly shrinkage | 27 | Baker & McHale, *Decision Analysis* 10(3) (2013) | Shrink by Var(p̂) not p(1−p); modest, sizing-lane. |
| Vol-parity (freeze-at-entry) sizing | 24 | Concretum 2024 (blog, e=2) | Turnover-cut for GLD cost; **pyramiding is a REJECT** (raises MaxDD, hurts Calmar). |
| Conformal test-martingale feature | 24 | Vovk et al., arXiv:2102.10439 | Likely redundant with changepoint/bocpd/newma; cheap try only. |
| N-period vol-label + **instance selection** | 18 | Song et al., *Complexity* 2024:5036389 | Vol-label = HAVE; the *ambiguous-row pruning* is the new, orthogonal-to-uniqueness half. |
| RQA features | 18 | *Nonlinear Dynamics* 100 (2020) | O(W²)/bar — same wall that timed out stride-1 sample-entropy; only after catch22 proves the lever. |
| FIGARCH vol | 18 | BBM 1996; Sheppard `arch` | Dominated by HAR-RV; feeds the OFF (VOL_FLOOR=1.0), label-independent overlay. Lowest priority. |

### 2B. Validation / honesty-stack tooling (top composites, but NOT ticker A/Bs)

These score highest but produce no edge — they change what we *trust* and give a principled stop rule. Each needs only persisting per-trial OOS return vectors (small harness add); none touches the deployed model or leak surface.

**T1 — Minimum Backtest Length (MinBTL) sufficiency gate** · `f5·n4·e5` (top composite)
- *Source:* Bailey, Borwein, Lopez de Prado & Zhu, "Pseudo-Mathematics and Financial Charlatanism," *Notices of the AMS* 61(5):458-471 (2014), https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf
- *Why novel:* we have only POST-HOC haircuts (DSR, PBO/CSCV, deflated-Sharpe, Harvey-Liu) — no a-priori length-sufficiency PRE-check. `MinBTL≈2·ln(N)/SR*²` tells us when the OOS window is simply too short for the search burden — operationalizing "leaderboard goes stale" / "val_auc≈0.5 window artifact."
- *Build:* add `minbtl()` to `scripts/stats_rigor.py`; flag borderline crowns (IWM N=64, UUP N=72) whose realized OOS length < MinBTL; cross-check DSR/Harvey-Liu.

**T2 — Online-FDR / alpha-investing ledger (LORD++ / SAFFRON / ADDIS)** · `f4·n4·e4`
- *Source:* Ramdas et al. SAFFRON, ICML 2018 (arXiv:1802.09098); Tian-Ramdas ADDIS, NeurIPS 2019 (arXiv:1908.10597); Fisher (2024), arXiv:2110.08161.
- *Why novel:* our batch Holm/BH re-inflates the count on "one more round" — exactly the pathology of an open-ended resumable 311-screen. Alpha-investing maintains a wealth budget over streaming trials, spends per test, refunds on discovery, controls FDR at every stopping point → a principled stop rule that self-throttles 0-EV grinding.
- *Build:* LORD++/ADDIS bookkeeping around `scripts/screen_etfs.py` keyed on the permute/DSR p-values already in `results/round_results.csv` (3091 rows); replay-validate that the 8 STRONG fits + GLD/UUP/IWM/USO survive; surface remaining alpha-wealth on the dashboard.

**T3 — N_eff-deflated Sharpe via ONC clustering** · `f4·n3·e4`
- *Source:* Bailey & Lopez de Prado, "The Deflated Sharpe Ratio," JPM 2014; LdP, "A Robust Estimator of the Effective Number of Tests" (ONC), *MLAM* ch.8.
- *Why novel:* `honest_audit.py`/`deflated_audit.py` deflate by RAW N, over-penalizing a correlated 21×27 sweep. `N_eff=(Σλ)²/Σλ²` (eigenvalue participation ratio / ONC) corrects it — directly addresses our over-deflation-kills-correlated-but-real-edges tension. Re-score offline; check whether any Bonferroni-killed borderline fits resurrect.

**T4 — Hansen SPA + StepM + Model Confidence Set** · `f4·n3·e5`
- *Source:* Hansen, JBES 2005 (SSRN 264569); White, *Econometrica* 2000; Hansen-Lunde-Nason MCS, *Econometrica* 2011; ships in Sheppard's `arch`.
- *Why novel:* permute tests "is there signal" on TRAIN labels; DSR/Harvey-Liu are *parametric* Sharpe haircuts; SPA/MCS *stationary-bootstrap the joint OOS P&L across all trials* (pricing cross-trial correlation) and return a confidence SET for principled book pruning — a capability PBO/CSCV lacks. Offline `arch` harness only, never deployed.

**T5 — Ledoit-Wolf robust Sharpe-difference test** · `f3.5·n3·e4.5`
- *Source:* Ledoit & Wolf, *J. Empirical Finance* 15(5):850-859 (2008), https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-00007214c2bc/jef_2008pdf.pdf
- *Why novel:* our FIT gate is a deterministic "beats own BH by ~0.15 Calmar"; we never run a distribution-robust test on the strategy-minus-BH Sharpe *gap*, and our overlapping bars are exactly the autocorrelated/fat-tailed case where naive SR comparison is liberal. Studentized circular-block bootstrap; feeds the multiplicity layer as the per-trial input statistic.

**T6 — Romano-Wolf stepdown (bootstrap FWER under dependence)** · `f3·n2.5·e4`
- *Source:* Romano & Wolf, *Econometrica* 2005; Clarke-Romano-Wolf, *Stata Journal* 2020.
- *Why novel:* "nothing survives Holm across ~500 trials" because trials are correlated and Holm assumes independence; RW bootstraps that dependence and recovers power. **Must ADD a column, never REPLACE Holm** — validate it does not un-kill any permute-confirmed fake (SPY/SLV/QQQ) before trusting it.

---

## 3. Wang's external leads — followed up

The WANG EXTERNAL LEADS input was empty this batch, so this is the standing follow-up state from `docs/analysis/WANG_INVESTIGATION.md` and the closed-levers list:

- **Trend-strength / diff-order ensemble (Wang "Lever 1").** Followed up and **mined + DEAD** — `tleg_fast+mid+slow` raced and LOST GLD (3.66 < 4.02; strength blend *dilutes*) and lost TLT. The modern multi-scale-trend literature adds nothing here; the asset-intrinsic single-leg champion wins.
- **Sticky-HMM labeler (our impl omitted the sticky floor).** Built with the floor, raced, **DEAD** — GLD Calmar 0.40 @ val_auc 0.878 (the predictable-not-profitable trap). HMM family closed for good; reinforces that high val_auc ≠ profit. This is also the cautionary precedent for the Oracle-trend labeler (#8).
- **InfoGain feature selection.** Followed up and **ADOPTED** — Wang's one working lever; `reduce='infogain'` lifted GLD 3.47→4.02 and TLT −0.10→+0.49. Now a HAVE; the modern refinement is the **Clustered-MDA reduce** (#11) which fixes its substitution effect.
- **β200 fwd-return-positivity lens (routes asset→{revert|trend|buyhold}).** Followed up and **closed** — it is only a buy-hold *pre-filter*, not a trend/reversion *router*. Retrodicts the book but adds no new edge.
- **Cross-sectional spread path.** **Declined** (pairs/spreads, 2026-06-04). The later-allowed *exogenous cross-asset features* (single traded ticker) is the in-rule descendant — it powers the signature lead-lag arm (#6) and is already user-cleared.
- **"Read medicine" / analogical transfer (meta-advice).** Not a concrete method; the internet search is the operationalization of it, and it is what surfaced DSPOT/Hawkes/signatures/MinBTL — none of which are in the transcripts.

**Verdict: Wang's buildable external leads are exhausted (one adopted, the rest dead/closed). The new in-rule material in this report is from the literature search, not Wang.**

---

## 4. Rejected / already-have (coverage)

Sounds-new-but-maps-to-existing (`ALREADY_HAVE`): **FFD fractional-diff feature block** (HAVE, and it *hurt* GLD 3.20→2.60); **Selective-GA labeling**; **realized-variance "business-time" clock** and **DC event-clock / Renko / CUSUM-event / Hurst-adaptive bars** (we already have 21 axes incl. these clock families); **multi-class %-change labeling** (= tertile); **Continuous Trend Labeling / DTB / Kernel-PELT labeler** (covered by trend_leg/ker/scan family); **volatility-managed & regime-conditional-leverage & DD-modulation sizing** (= inverse-vol overlay + dd_overlay/ddbreaker); **causal MODWT/à-trous wavelet** (HAVE wavelet axis); **NEWMA, RuLSIF, NP-FOCuS, Online-Kernel-CUSUM, E-detectors, e-BH** (all = our change-point detectors + e-value liveness gate).

Real-but-infeasible/dead (`DEAD_OR_INFEASIBLE`): **permutation entropy / Bandt-Pompe complexity-entropy plane / ordinal-pattern family** (entropy refactor closed `1d5916f`: ordinal-only degrades edges); **entropy/information bars** (subordination dead); **GMADL custom directional loss** (no torch custom-objective path that survives QC + our overf_t prior); **AEDL multi-scale event labeling**; **VCRB volume-centred range bars**; **EMD/MODWT energy cousins** (whole-series leak + compute). Do not re-grind any of these.

---

## 5. Needs sign-off

- **Adopting a *less-conservative* multiple-testing FIT rule (online-FDR T2 / Romano-Wolf T6 / SPA T4).** These recover power vs Holm, but a looser gate *risks re-admitting fakes*. Changing what counts as a FIT is a methodology decision — recommend they ADD columns alongside Holm/BH, with the permuted-label control staying the decisive falsifier, and the user confirming before any newly-cleared edge enters the book.
- **MCS/SPA-driven book pruning or re-weighting (T4).** If the Model Confidence Set drops a deployed name (e.g. UUP, decay-stale) or SPA fails one, that is a book-composition change — flag for sign-off before re-weighting.
- **Harness change to persist per-trial OOS per-bar return vectors.** Required by T4/T5/T6 and N_eff (T3); currently we store only summary Calmars. Small but a real logging add — confirm before wiring.
- **Pre-cleared, no sign-off needed (note for completeness):** the exogenous cross-asset feature channel for the signature lead-lag arm (#6) is already user-approved (2026-06-06, single traded ticker, not a pair). The crash-lead picks (#4/#5) use *endogenous* jumps and explicitly need **no external VIX data** — they do not hit the dead vol-product / external-data wall.

---

## 6. Recommended next 1-2 experiments

**Primary (highest upside, in-rule): convert the SPY/QQQ crash-veto lead with DSPOT + Hawkes.**
This is the single result we have that is permute-confirmed real (Calmar 2.53, val_auc 0.547) but **non-deployable solely because it is threshold-fragile and trades<80 (73)** — the most addressable gap in the whole book, and the only life ever seen on equities. Both fixes replace the brittle fixed `crash_ahead` threshold with a self-calibrating/continuous one; consistent with our own lesson "new mechanism (tail-avoidance) > new tuning."
- **A/B:** {SPY, jump axis, crash_ahead label}. Arm 1: add DSPOT streaming tail-exceedance prob as a causal feature **and** wire its breach-flag into `crashveto` (flatten only on EVT breach). Arm 2: add the frozen-parameter Hawkes intensity `λ(t)` (MLE on TRAIN Lee-Mykland jump times) as a causal feature / smooth crash-sizer. Baseline = the existing crash_ahead+crashveto config. Replicate the winner on QQQ.
- **Win condition = deployable:** trades>80 AND val_auc>0.52 AND Calmar>buy-hold.
- **Honesty gate:** permuted-label control must collapse it (<40% of real, the decisive falsifier — same harness that proved the original crash signal real); session-burden DSR (Holm + Harvey-Liu, ideally N_eff once T3 lands); explicit buy-hold-of-SPY baseline; decay monitor (Page-Hinkley); cost-stress @5/10bp; leak: `tests/test_bar_threshold_leak.py` green + `infer_online` preds_match=1 after mirroring the recursion into the live template.

**Secondary (cheapest, parallel, near-zero cost): MinBTL sufficiency gate (T1).**
A closed-form scalar in `scripts/stats_rigor.py`; flags the borderline crowns (IWM N=64, UUP N=72) as length-insufficient and stops the loop re-litigating stale Calmars each round — directly fixing our #1 documented pain. No model/leak/data dependency; ship it alongside the SPY race.

**If a fresh incremental edge A/B is preferred over the harder SPY name:** run **#1 range-based volatility (Yang-Zhang)** on {GLD, USO} — highest-scored edge candidate, dual-payoff, cheapest build — but ship it with the hard "keep iff beats champion AND permute-collapse" gate because our feature-block track record is 0/2 (the honest reason this is the *secondary* edge pick, not the primary). **Step 0 before racing: confirm `bar_builder` exposes per-bar OHLC.**

---
_Generated 2026-06-08 by the wang-plus-internet-newmethods workflow · 63 unique candidates, 34 novel-actionable._
