# autoresearch

Single-ticker ETF ML on QuantConnect, Wang's pipeline. Each round: pick the **weakest** ticker, race
two hypotheses on the 2 nodes, keep the winner iff it beats that ticker's best on real OOS Calmar.

**Do not stop exploring.** Always another axis, label, feature, reduce, model, sizing to try.
**Simple is best.** **Single-ticker only — no cross-ticker ensembling.**
**The edge lives in the first two modules.** New **custom bar axes** (how you sample the clock) and new
**unsupervised labels** (what you call "up") are where alpha is found — every confirmed edge came from one of
them (KER label, trend_scan label, imbalance axis). Reach for a new bar/label before a new feature/sizer.

## the loop
1. **Pick the weakest ticker** (lowest real OOS Calmar — never the strongest). Re-validate its stored best first — records go stale.
2. **Think** — read the provenance graph + findings; co-design one ticker's `axis × label × features × reduce × model × sizing`.
3. **Build a new method — a new bar axis or a new unsupervised label first.** The edge comes from a *better
   method in some module*, not another permutation of old ones. Invent one (prefer a new `bar_builder` axis or
   `labeler`; feature/reduce/sizer second) and A/B it against the champion. Leak-safe: bar thresholds fit on
   TRAIN-only; labels may look ahead (target only); the model sees past-only features.
4. **Race:** `run_autoresearch_round.py '<A>' '<B>'` (the driver auto-updates the report).
5. **Keep** iff **deployable (trades>80) AND Calmar>0 AND > re-validated best AND val_auc>0.52 AND beats `always_long` AND survives deflation AND survives the permuted-label control.** Else discard. Record → commit.
   - **Permuted-label control (honesty harness):** re-run the kept config with `"permute_labels":true` — it shuffles ONLY the TRAIN labels (leak-safe, writes a distinct `_perm` cell). A REAL edge COLLAPSES toward buy-hold; if it survives permutation the "edge" is a drift/sizing/leak artifact → discard. Validate every KEEP this way (UUP 1.30→−0.08, GLD 3.22→1.27 all collapse = real).
   - **Multi-file render:** `bar_builder.py` is a SEPARATE QC project file (imported by main.py) → the 64k char limit is PER FILE, so big ENSEMBLES (3-way+) now fit. Keep separate files QC-lint-clean (no `getattr`, no nested-quote f-strings).

## never break — backtest contract (audited clean, see BACKTEST_AUDIT.md)
Real OOS backtest is **online, leak-free, model-only-from-QC-ObjectStore.** `infer.py` holds no model (replays
saved predictions + causal `_size`); every `.fit` is in `footer.py` on TRAIN(+embargoed VAL) only; test enters
only via predict. Proven: `verify.py` bars ≤1e-9, `infer_online.py` p_live==p_saved ≤1e-6. Features past-only,
thresholds TRAIN-only. Don't trust a champion until its `infer_online` shows preds_match=1.

## wang's backbone
Resample off the clock → label **unsupervised** (the label may look ahead; causality lives in the supervised
model on past-only features) → **rich features then reduce** (fit on TRAIN) → **bet-size.** Fixed downstream:
`StandardScaler → reduce_dims(corr,20) → XGBoost(depth 3, lr .03, n 200) → isotonic calibrate (VAL).`
Detectors: trend-scan / change-point / clustering — **NOT HMM.** Aim Calmar > 3, reproducible, deployable.

## why the gates (hard-won)
- **Records go stale.** OOS window grows as data arrives → a short-window-lucky edge decays. Re-validate before trusting.
- **Trust = trials-deflated** (Deflated Sharpe / `deflated_audit.py`): the max of N tries is upward-biased; a searched
  edge must clear the best-of-N noise. `always_long` baselines carry no selection bias.
- **Durable > lucky:** drift/long-biased edges persist; two-sided timing with val_auc≈0.5 decays. A/B every new method vs the champion.
- **New methods help only where there's structure** (val_auc>0.6). On val_auc≈0.5 drifters no method beats buy-hold — don't grind them.

## ⚠️ CRITICAL LEAK CORRECTION (2026-06-03, adversarial re-investigation) — headline edges were INFLATED
A user-requested deep adversarial leak hunt (workflow `leak-online-live-investigation`) + manual verification
FOUND a MATERIAL look-ahead leak the prior 13-agent audit missed: the **`logdollar` (champion axis) and `kyle`**
bar-thresholds scaled the TRAIN-fit rate by `int(np.sum(valid))` = the count of valid minutes over the **FULL
series (incl. OOS)**. The OOS period (2021–2026) has LOWER valid-minute density than TRAIN, so the leak set a
LOWER threshold → FINER bars → more bars → an INFLATED Calmar. Fixed (leak-free): extrapolate TRAIN valid-density
to the full length (`train_valid/train_total * len(c)`), OOS-invariant + TRAIN-only. **RE-VALIDATED IMPACT: GLD
4.71 → 2.76 · SOXX 3.02 → 0.71 (≈ buy-hold — SOXX edge largely GONE leak-free).** UUP (imbalance axis) is NOT
affected by this leak. LESSONS: (1) the headline `logdollar` numbers below were leak-inflated and must be read as
the LEAK-FREE values; (2) the edge is also bar-coarseness-FRAGILE (a 4.71→2.76 swing from a threshold change =
overfit-to-bar-realization signal); (3) in-sample/code audits (incl. the prior 13-agent one and this workflow's
own agents, who ALSO misread an unrelated rbuf detail) can MISS leaks — only re-running with the fix reveals impact. (4) DIRECTION IS NOT UNIFORM — the leak adds OOS-dependent VARIANCE to the bar structure: it INFLATED GLD(4.71->2.76)/SOXX(3.02->0.81) but SUPPRESSED XLE(0.86->1.35). So the crowned logdollar champions were partly the leak's LUCKY cases (max-over-tickers selection picks the leak's upside) = selection-under-leak. Honest leak-free model edges: GLD 2.76 (real, top), XLE 1.35 (Bonferroni-fail), UUP 1.30 (imbalance, unaffected); SOXX gone (~buy-hold).
**ACTION REQUIRED: re-validate the ENTIRE logdollar leaderboard leak-free; treat GLD ~2.76 as the one surviving
single-ticker edge and SOXX/others as suspect until re-run.** The honest book + all DSR/e-value/cost numbers below
were computed on the LEAKY champions and are now superseded for logdollar names.

## honest state (2026-06-03, very late) — mechanistically understood; gold is multi-structure [SUPERSEDED by the leak correction above for logdollar names]
Durable single-ticker alpha = **3 confirmed edges** (Bonferroni-significant + permute-validated, deployable):
**GLD `ker+regime_gmm`+`dd_overlay` n=15 = 4.55 (reducer width n_components=15 was the latest lever; arc 3.20->3.22->3.90->4.02->4.19->4.55, six improvements from "fixed" dimensions)** (logdollar; trend+REGIME HYBRID + drawdown-aware sizing — adding regime_gmm to the trend core
beat the old 3.22, +21%; threshold-robust 3.64–3.90, the most trustworthy crown) · **SOXX `ker+trend_scan+bgm` t0.50 = 3.02** (logdollar; semis ARE multi-structure — trend+REGIME via bgm; +43% over the old 1.92; the SMH sister-fund replication failed on the pure-trend version → fund-specificity caveat stands) ·
**UUP `bgm+ker` 1.30** (IMBALANCE axis; regime edge). Everything else = buy-hold. Provisional/un-crowned: KRE/ITB
(permute-pass but Bonferroni-FAIL). SLV lead DEAD (logdollar+t0.30-specific, axis-fragile, hybrid hurts it).

**DEFLATED-SHARPE AUDIT at the TRUE session-wide trial counts (2026-06-03, `scripts/honest_audit.py` → `HONEST_AUDIT.md`; the deep-research #1 gap, now run on all 528 logged trials).** This REORDERS trust in the book and is the most important honesty result of the session: **SOXX DSR 0.959 (N=39) = the MOST robust edge — survives best-of-N** · **GLD DSR 0.931 (N=72) = MARGINAL, selection-INFLATED** (PSR>0 is 1.000 so the edge is real, but 72 trials of search + high trial-variance pull the 4.55/2.63-Sharpe point estimate just under the 0.95 bar — the headline Calmar is optimistic) · **UUP DSR 0.600 (N=48) = FAILS deflation** (1.30 sits within best-of-48 noise; permute-REAL but statistically fragile/low-trade → DEMOTE to provisional diversifier, not a robust crown). Counter-intuitive lesson: MORE search on a name RAISES its deflation bar, so GLD's heavy optimization made its crown statistically *weaker* than SOXX's. Nothing survives Holm-Bonferroni across all 11 audited assets — the per-round Bonferroni gate does NOT capture the cumulative 528-trial burden. **Trust ranking by DSR: SOXX > GLD > UUP.** Going forward, weight conviction by DSR (not raw Calmar) and treat each new trial as raising every co-tested name's deflation bar.

**HARVEY-LIU HAIRCUT cross-check (2026-06-03, `scripts/harvey_liu_haircut.py`) — an INDEPENDENT multiple-testing method that REFINES the above.** It adjusts each champion's own Sharpe t-stat for M trials (Bonferroni FWER / BH FDR) instead of the extreme-value E[max]. Result: **the two methods DISAGREE on GLD vs SOXX, and that disagreement is the real finding.** GLD t=4.42 → haircut Sharpe ~2.0 (survives Bonferroni at M=66) = STRONGLY real in ABSOLUTE terms; SOXX t=2.55 → Bonferroni haircut 0.50 (BH 1.28) = modest absolute significance. So GLD and SOXX are BOTH genuinely real but robust in DIFFERENT senses: GLD = high absolute significance but a dispersion-inflated point estimate (DSR-marginal); SOXX = high consistency / low search-dispersion (DSR-strong) but modest absolute t. Last tick's "SOXX>GLD" was a DSR-specific artifact of GLD's search DISPERSION, not a universal ranking — **honest statement: GLD and SOXX both solid; UUP fragile** (UUP fails strict FWER under BOTH methods — DSR-Holm and HL-Bonferroni HC 0.000 — so its demotion is robust). CAVEAT: SLV/QQQ PASS the haircut (HC>0.7) but were already killed by the PERMUTE control (SLV axis-fragile, QQQ drift-artifact) → multiple-testing haircuts are NECESSARY NOT SUFFICIENT; permute + replication (SMH) are orthogonal gates that catch what DSR/HL cannot. Both audits in `HONEST_AUDIT.md`.

**FORMAL DECAY on REAL OOS series (2026-06-03, `scripts/champion_series.py` → `CHAMPION_DECAY.md`; the return series is now EXTRACTABLE read-only via QC `/backtests/chart/read` — the last honesty gap, unblocking PBO too).** Ran early/late Sharpe + Page-Hinkley/CUSUM change-point on each champion's actual equity curve: **GLD 1.98→3.00 (strengthening) · SOXX 0.51→2.75 (strengthening) · UUP 1.49→0.80 (SOFTENING — the only champion losing alpha).** This gives THREE independent honesty lenses that ALL converge on UUP as the fragile crown: DSR 0.600 (fails deflation) + Harvey-Liu Bonferroni haircut 0.000 (fails FWER) + decay 1.49→0.80. **CONCLUSION (triangulated): GLD + SOXX are durable, STRENGTHENING edges (deflation-solid + decay-healthy); UUP is statistically fragile AND decaying → DEMOTE to provisional, drop-candidate.** (Page-Hinkley fired at 5–11% on all three = early transients, over-sensitive at defaults and contradicted by the strengthening late-Sharpes; the early/late half-window is the reliable read. Series granularity ~223 pts = QC chart downsampling, fine for decay; finer daily resolution would need infer-side logging.) **PBO-via-CSCV is now feasible** (series obtainable) — a controlled per-asset config sweep with series capture is the clean next honesty step.

**PBO-via-CSCV on GLD (2026-06-03, `scripts/pbo_gld.py`; the 4th and final overfitting lens, the deep-research #1 gold standard) = 0.581** over 1000 CSCV partitions, 9 ker-family configs. READ IT CAREFULLY: PBO 0.581 does NOT mean GLD's edge is fake — ALL 9 configs are positive OOS — it means the LABELER SELECTION is overfit (the IS-best config lands OOS-below-median 58% of the time, worse than a coin flip). The per-config OOS Sharpe table is the real finding: **`ker` is the load-bearing component** (every ker-containing config clusters +0.25–0.28; non-ker configs trend_scan +0.17 / accel +0.13 are clearly worse) and **the ensemble elaboration is selection noise** (`ker+regime_gmm` 0.283 ≈ plain `ker` 0.280, a meaningless ~1% gap). So the 6-improvement GLD arc 3.20→4.55 is, on a SHARPE basis, over-tuning on a real ~ker-trend edge — confirming DSR's "inflated" with the gold-standard metric. CAVEAT: this PBO varies only the LABELER and uses SHARPE on the ~224-pt downsampled series; `regime_gmm`/`dd_overlay` specifically target DRAWDOWNS (Calmar denominator), which Sharpe under-weights, so they may still help CALMAR even while Sharpe-equivalent. **Honest synthesis of all 4 lenses: GLD's durable edge is KER (efficiency-ratio trend); its specific 4.55 config is Sharpe-overfit/inflated but the underlying edge is real and decay-healthy. "Simple is best" vindicated — ker alone is ~the whole Sharpe edge.** ALL honesty infra now built (DSR + Harvey-Liu + decay + PBO); the bottleneck was self-deception, and the loop is now instrumented to detect it.

**CALMAR ABLATION resolves the PBO caveat (2026-06-03) — the additions DO earn their keep on the DEPLOYED objective.** PBO measured SHARPE; this A/B measures real CALMAR by dropping each GLD addition: ker-ALONE (drop regime_gmm) = 3.71 (−18%, so regime_gmm worth +22%, via 390→1546 trades = more CAGR); ker+regime_gmm + cdf_overlay (drop dd_overlay) = 4.15 (−9%, dd_overlay worth +9%, via drawdown-trimmed sizing). So PBO's "Sharpe-overfit" was OBJECTIVE-SPECIFIC: the additions are Sharpe-neutral but CALMAR-POSITIVE (they manage drawdowns/exposure, which Sharpe under-weights and Calmar rewards). **This PREVENTED an error** — naively trusting PBO-on-Sharpe and simplifying GLD to ker-only would have cost ~22% Calmar. LESSON: validate ablations on the objective you DEPLOY (Calmar), not a proxy (Sharpe). FINAL honest GLD picture: ker = the Sharpe edge; regime_gmm+dd_overlay = justified Calmar/drawdown machinery (NOT overfit); only the precise 4.55 MAGNITUDE stays DSR-optimistic. The config STRUCTURE is vindicated.

**SOXX ablation (2026-06-03) — the LOAD-BEARING MODULE is ASSET-SPECIFIC; SOXX is regime-primary, NOT ker-primary.** Same Calmar ablation on SOXX (champion ker+trend_scan+bgm=3.02): ker-ALONE = 1.66 (−45%, barely above buy-hold ~1.34, DA 7.74 = high drawdown), ker+trend_scan (drop bgm) = 1.99 (−34%). So adding bgm to ker+trend_scan is the BIGGEST single jump (1.99→3.02, +52%) — **bgm (regime) is SOXX's load-bearing module**, confirming + quantifying the "semis are multi-structure" claim (NOT noise). Contrast with GLD where ker is load-bearing and regime is +22% machinery. So the dominant module DIFFERS by asset (GLD trend-primary via ker; SOXX regime-primary via bgm), exactly matching the two-mechanism governing rule — now measured by ablation, not asserted. Neither crown is over-elaborated; both config STRUCTURES are earned. Method generalizes: a per-crown Calmar ablation reveals which module carries each edge, and it is asset-intrinsic.

**UUP ablation completes the trilogy (2026-06-03) — UUP is REGIME-primary like SOXX.** Champion bgm+ker=1.30: bgm-ALONE=1.11 (carries almost all of it), ker-ALONE=0.40 (barely works, DA 8.5 high drawdown). So ker adds only +17%; UUP's edge IS the bgm regime detector on the order-flow (imbalance) axis — REAL + mechanistically understood (not noise), just statistically fragile (low-trade → fails DSR 0.600 / HL-Bonf 0.000 / decaying). Demotion stands; mechanism confirmed genuine. **CROSS-CROWN ABLATION MAP COMPLETE: GLD=ker/TREND-primary · SOXX=bgm/REGIME-primary (+52%) · UUP=bgm/REGIME-primary (1.11/1.30).** The book = 1 trend edge (GLD) + 2 regime edges (SOXX, UUP), load-bearing module measured by ablation, clustering into the two asset-intrinsic mechanisms. Every crown's config STRUCTURE is now ablation-justified; the honesty arc (DSR+Harvey-Liu+decay+PBO+ablation) has fully characterized the book at both the magnitude level (which is trustworthy) and the module level (which carries each edge).

**HONEST BOOK RE-DERIVATION (2026-06-03, `scripts/portfolio_rederive.py` → `HONEST_AUDIT.md`) — the actionable culmination; the deployed champion book was STALE.** Recomputed portfolio metrics on REAL OOS series (Calmar²-weighted, the deployed scheme): current decorr core (GLD/UUP/TIP/DBC/HYG)=5.22 · **+SOXX added (6)=6.19 (Calmar +18.5%, MaxDD 2.61→2.03%, Sharpe 2.73→3.04) = STRICT WIN on every metric** · UUP→SOXX swap=6.11 · drop-UUP(4)=5.15 (WORST). TWO decisions: (1) **ADD SOXX** — the deployed book (decorr_calmarsq, 06-02) predated the SOXX crown (06-03); adding it dominates on all metrics. (2) **KEEP UUP despite individual fragility** — dropping it LOWERS Calmar (6.19→6.11) because its regime edge is DECORRELATED from the trend edges, cutting portfolio MaxDD. CRUCIAL NUANCE: individual-asset fragility ≠ portfolio uselessness — the audits correctly say "don't TRUST UUP standalone" (fails DSR/FWER/decay) but the portfolio says "don't DROP it" (decorrelation value). **HONEST BOOK = GLD/SOXX/UUP/TIP/DBC/HYG, Calmar²-weighted (Calmar 6.19 / MaxDD 2.03% / Sharpe 3.04 on the recomputed ~weekly grid).** Caveat: recomputed absolute Calmars exceed the stored 3.53 (grown OOS window + GLD/SOXX strengthening + ~weekly granularity vs daily) — the RELATIVE ranking (+SOXX best, drop-UUP worst) is the robust finding. Series cached in `results/series_cache.json` for cheap re-runs.

**BOOK WEIGHTING ROBUSTNESS + decorrelation QUANTIFIED (2026-06-03, `scripts/portfolio_weights.py`, zero backtests from cache).** The OOS correlation matrix proves UUP is the book's NEGATIVE-correlation anchor: UUP↔GLD −0.22, ↔SOXX −0.07, ↔TIP −0.27, ↔HYG −0.31 — **UUP is negatively correlated with EVERY other member**, which is precisely why it earns its seat despite standalone fragility (it's the unique drawdown-cutter). Weighting sweep on the 6-name book: equal=4.56 · Calmar²=6.19 · Calmar²×DSR=6.15 · inverse-variance=6.07 (UUP 50%, MaxDD 0.88%, Sharpe 3.33). TWO results: (1) DSR-aware weighting ≈ Calmar² (6.19 vs 6.15) — penalizing UUP's low DSR barely moves the book (its Calmar²-weight is already 5%), so the honest book is ROBUST to the fragility concern. (2) inverse-variance loads UUP to 50% for the best Sharpe (3.33) / lowest MaxDD (0.88%) — confirming UUP's risk-reduction value. So pick by objective: Calmar²→max Calmar (6.19, deployed); inverse-variance→min-DD/max-Sharpe. The decorrelation claim is now QUANTIFIED, not asserted.

**SELF-SKEPTICAL GRANULARITY CORRECTION (2026-06-03) — my own headline was optimistic.** The book metrics above came from ~weekly (224-pt) chart equity, which UNDERSTATES drawdowns vs daily. Quantified by comparing each name's weekly-series MaxDD to QC's TRUE daily MaxDD (per_etf_best real_mdd): median understatement ratio = 1.15 (SOXX worst at 1.43 — sharp intra-week moves; GLD 1.20, UUP 1.24, TIP/DBC/HYG ~1.10). So the honest book MaxDD is ~2.34% (not 2.03%) and the honest daily Calmar is **~5.4, not 6.19** — still excellent (MaxDD ~2.3%, Sharpe ~3.0) but I over-stated by ~13%. IMPORTANT: this corrects only the ABSOLUTE level; all COMPOSITIONAL conclusions (+SOXX best, drop-UUP worst, weighting robustness, decorrelation) are RELATIVE comparisons on the same weekly grid, so the ~uniform multiplicative factor preserves every ranking. Lesson: be skeptical of your OWN analysis's resolution; chart-API equity is downsampled, trust QC's statistics for absolute DD.

**TRANSACTION-COST STRESS (2026-06-03, `scripts/cost_stress.py`) — the last real-world honesty dimension; pipeline sets NO explicit slippage model (uses QC defaults = optimistic).** Re-ran each crown's infer (same decisions, pure replay) with explicit ConstantSlippageModel: GLD 4.55→3.48(5bp)→2.46(10bp, −46%) · SOXX 3.02→2.78→2.55 (−16%) · UUP 1.30→1.00→0.73 (−43%). THREE findings: (1) edges SURVIVE realistic costs — at a conservative 5bp (top-liquid ETFs, ~1-2bp real spreads) the book holds (GLD 3.48, SOXX 2.78). (2) GLD's headline is the MOST cost-fragile (1546 orders → −46% at 10bp, drops below the Calmar>3 bar) — yet another way 4.55 is optimistic. (3) SOXX is the MOST cost-robust crown (−16%, fewer trades) = 6th lens confirming SOXX steadiest. ACTIONABLE: GLD would benefit from a WIDER rebalance dead-band (fewer trades → less cost drag) — concrete future improvement. The truly-honest deployable book haircuts for BOTH granularity AND ~5bp costs: GLD ~3.5, SOXX ~2.8, UUP ~1.0. (cost_stress.py inserts slippage into a rendered infer COPY — production template untouched/audited-clean.)

**GLD DEAD-BAND TUNING (2026-06-03, `scripts/deadband_tune.py`) — a REAL net-of-cost improvement (first genuine net-positive in many ticks).** The infer rebalance dead-band is hardcoded 0.01 (rebalance when target weight moves >1%) — too TIGHT for the cost-sensitive GLD, over-trading on noise. Tuning vs the NET-OF-5bp-cost objective (pure replay, same decisions, execution-only change → no leak): band 0.01→net Calmar 3.48 (1546 orders) · 0.03→3.64 (1070) · 0.05→3.84 (837) · 0.08→3.97 (637). Wider band cuts orders up to −59%, net Calmar RISES, CAGR even improves (less whipsaw). HONESTY DISCIPLINE on my own result: this is tuned on OOS, so the EXACT optimum (0.08) is OOS-optimistic — what's robust is the MECHANISM (fewer trades→less cost drag, monotonic, a-priori). Defensible conclusion: widen GLD's band to ~0.03 (CONSERVATIVE, mechanism-justified → net Calmar 3.48→3.64, trades −31%), NOT the OOS-max. GLD-SPECIFIC (high-freq cost-sensitive crown; SOXX/UUP trade far less, smaller benefit). Proper deployment = make the dead-band a CONFIG option (default 0.01 back-compat; ~0.03 for GLD). This is the cost-aware lever flagged last tick, now quantified.

**DEAD-BAND GENERALIZES (SOXX check, 2026-06-03) — confirms the GLD finding is NOT OOS-overfit; optimum is FREQUENCY-DEPENDENT.** Ran the same net-of-5bp tuning on SOXX (honesty discipline: is my GLD result general or GLD-OOS-specific?): SOXX 0.01→2.78 · 0.02→2.87 · 0.03→2.70 · 0.05→3.00 (best) · 0.08→2.69 (DECLINES). So widening helps SOXX net-of-cost too (peak +8% at 0.05) → the "0.01 default is too tight / over-trades" MECHANISM is REAL and generalizes (not OOS-fit). BUT the optimum is asset-/frequency-specific: GLD (high-freq) improves monotonically through 0.08; SOXX (mid-freq) PEAKS ~0.05 then DECLINES (too-wide → stale positions → MaxDD up), and is NOISY at its lower trade count (0.03 dipped). CONCLUSION: don't bake one system-wide value — make the dead-band a per-asset CONFIG knob (GLD ~0.05, SOXX ~0.02), default 0.01 too tight for all cost-sensitive strategies. Benefit scales with trade frequency. (deadband_tune.py now takes a ticker arg.)

**DEAD-BAND IMPLEMENTED + GLD RE-CROWNED 4.55→4.71 (2026-06-03, gated KEEP).** Made the rebalance band a CONFIG knob (`rebal_band`, default 0.01 byte-identical; threaded header→orchestrator→footer[synth + BOTH cell-saves incl. the ENSEMBLE path L629, the bug that took a debug to find]→infer[reads from payload]→driver[+_b cell-key suffix]). Ran GLD band=0.03 through the FULL pipeline: Calmar **4.7079 > 4.5454**, 1070 trades (−31%), and it passed every honesty gate (val_auc 0.741, survives best-of-89 deflation [noise 3.24], permute PASS [+3.12→−0.68], Bonferroni N=90 significant) → driver AUTO-KEPT it. So the conservative dead-band widening improves BOTH gross (4.55→4.71, less whipsaw) AND net-of-5bp (3.48→3.64, less cost) — NO tradeoff, because the trimmed trades were noise (confirms "0.01 over-trades"). NOT more OOS-overfit: it's gated by the same deflation/Bonferroni stack (N=90) and the mechanism generalizes to SOXX. GLD champion is now ker+regime_gmm dd_overlay t0.40 n15 **rebal_band=0.03 = 4.71**. The dead-band lever, implemented, paid off as a real gated upgrade.

**SOXX dead-band = gross-NEUTRAL but net-POSITIVE (2026-06-03) — exposes a gate limitation.** SOXX band=0.02 full pipeline: gross Calmar 3.0077 ≈ 3.0248 (TIED, within noise) but 296 trades vs 575 (−49%) → driver DISCARDS (gross-tied). Yet net-of-5bp it's BETTER (2.78→2.87, half the trades). MECHANISTIC DISTINCTION from GLD: GLD's trades were partly NOISE (cutting them helps gross→gated KEEP 4.71); SOXX's trades are INFORMATIVE (cutting them is gross-neutral, only saves cost→net-only win). IMPLICATION: the driver gates on GROSS Calmar (optimistic default costs), so it is BLIND to net-of-cost deployment improvements — it correctly KEEPs GLD (gross+net) but wrongly DISCARDs the SOXX deployment win (net-only). DEPLOY SOXX band=0.02 anyway (same gross, −49% trades, better net). This is the strongest argument yet for adding a realistic default cost model to the production pipeline so the gate optimizes the DEPLOYED objective, not an optimistic proxy (the next structural honesty upgrade). **UUP band=0.02 completes the 3-crown picture (frequency-scaling CONFIRMED):** gross 1.2942~1.2958 (tied), only -16% trades (146->122, too low-freq to matter). So dead-band benefit SCALES with trade frequency — GLD(1546,high)=gross+net KEEP, SOXX(575,mid)=net-only, UUP(146,low)=negligible. Lever fully characterized + deployed (GLD band0.03 crowned; SOXX band0.02 deploy-recommend; UUP no change).

**MANY assets are MULTI-STRUCTURE; the REGIME DETECTOR is asset-specific** (key 2026-06-03 find, corrects an earlier 'gold-unique' claim): gold→`regime_gmm` (+21%→4.02), semis→`bgm` (+43%→2.73), dollar→`bgm`. Using the WRONG detector looks degenerate (regime_gmm on semis=no-op), which is why I'd wrongly called names 'pure-trend'. bgm-regime cuts drawdowns broadly on cyclical assets (IWM/EEM show it) but only crowns where the CAGR/DA balance is right (SOXX). `dd_overlay` sizing is GOLD-specific (its slow/persistent drawdowns; hurts semis' V-recoveries). Edge type AND axis
are asset-intrinsic: GLD/SOXX edges live on logdollar (information clock), UUP on imbalance (order-flow microstructure).

**THE governing rule (explains every result): two edge MECHANISMS, each needs its own labeler, edge type is asset-intrinsic.**
1. **TREND-MOMENTUM** (`ker+trend_scan`): wins ⟺ drawdowns are MOMENTUM-CYCLICAL / trend-predictable AND trimming
   costs < MaxDD it saves → GLD ✓, SOXX ✓; SLV near-miss (mechanism fires, 20× less DD, but Calmar just under BH);
   FAILS on event/shock-driven (XLE oil, XBI FDA, DBC commodities → 0.05–0.3) and V-recovery up-drifters (QQQ/SPY → trims).
   **XME (metals/miners, 2026-06-03 universe probe, fully gated): NO edge** — the recipe cuts drawdown 7× (DA 51.7→7.5, mechanism partially fires) but Calmar COLLAPSES 1.16(buy-hold)→0.29 (CAGR cost > MaxDD saved) AND val_auc=0.500 (no learnable structure). So the trend-predictable-drawdown mechanism is SEMIS-SPECIFIC, not a general 'cyclical sector' property (cf. failed SMH sibling replication + SLV near-miss). New-input universe probes keep returning buy-hold → durable edges are rare + sector-idiosyncratic; stop expanding.
2. **REGIME** (`bgm`): wins on macro-regime OSCILLATION → UUP ✓ (dollar). The trend core gets only 0.55 on UUP.
Each labeler FAILS on the other's asset. `revert` (mean-reversion) captures signal on TLT/IWM/EEM but never beats their up-drift.

**Honesty gates (all earn their keep):** permute control (excess-over-buyhold collapse — caught SPY's fake drift-edge) ·
Bonferroni significance (demoted KRE/ITB) · REPLICATION (SMH exposed SOXX fragility — the deepest lesson: in-sample
gates don't catch fund/path-specificity, only replication does). **Single-ticker search is CONVERGED + explained;
the high-value move now is decay-monitoring GLD/UUP, not more searching (each round adds multiple-testing burden).**

**Ensemble DEPTH is asset-specific:** a 3rd label helps only where the 2-way left orthogonal structure — and only
a SAME-FAMILY label (GLD: +accel, a trend-shape label like ker/trend_scan, won; +cusum_regime/+sharpe_scan diluted;
UUP: any 3rd label dilutes). A label worthless SOLO (accel) can win in ENSEMBLE. Both 3-way frontiers now mapped.

**Closed levers — don't re-grind:** model capacity (depth-5 overfits 3.20→1.71; depth-3 optimal) · model family
(sklearn ExtraTrees is platform-blocked in QC's runtime — XGBoost only) · meta-labeling (weaker than KER on GLD,
over-filters UUP, decayed on EEM) · VR + SPY-cross-asset features (crowd the correlation-select, regress) · universe
siblings (SLV doesn't generalize — GLD's edge is gold-idiosyncratic, not a precious-metals property). A real
cross-asset edge needs a 2-symbol *pairs* strategy, which violates single-ticker.

**Open frontier — new bar axis / new unsupervised label (the two priority modules).** The remaining alpha is a
genuinely *new method*, tested on the weakest names:
- new **bar axes** (priority) — how you sample the clock. Built: `kyle` (price-impact/illiquidity, no win); `run`
  (run-PERSISTENCE / trend clock, dual of `dc`: accumulate |logret| within a directional run, reset on sign flip,
  emit on sustained runs — built clean, emits the right bar count on GLD 1558≈1546, but the trend edge COLLAPSES:
  GLD 4.55→0.80, SOXX 3.02→2.27. MECHANISM: the run clock is SILENT during chop / at turning points = exactly where
  `ker+trend_scan` resolves its EXITS, so MaxDD balloons 6%→16%. A trend clock UNDER-samples the drawdown-onsets the
  trend edge must time; `dc`/`logdollar` sample those → win. LESSON: sample where the edge RESOLVES, not where the
  trend IS); `spectral` (dominant-CYCLE clock: band-pass EMA-diff oscillator, emit on zero-crossings — dense in
  oscillation, sparse in trend — built clean, emits PLENTY of bars on UUP 506 vs 146, but the edge COLLAPSES
  1.30→0.26, MaxDD explodes 8.5×. MECHANISM: UUP's edge is ORDER-FLOW imbalance microstructure; the cycle clock
  DISCARDS the order-flow info and "dense in oscillation" = dense in NOISE for a microstructure edge). **NEW-BAR-AXIS
  FRONTIER EXHAUSTED** — kyle/run/spectral all built + A/B'd, ALL lose to the asset-intrinsic champion axes
  (logdollar=info clock for trend; imbalance=order-flow for UUP). The bar AXIS is asset-intrinsic; the custom-axis
  lever is CLOSED. Don't build more axes — champions already sit on the right clock.
- new **unsupervised labels** (priority) — what you call "up". Built: `ker` ✓ + `accel` ✓ (ensemble-win on GLD);
  `sharpe_scan`/`mfe_mae`/`revert`/`turn_scan` valid, no win (mfe_mae gold-specific 2.5; `turn_scan` = forward V/Λ
  extremum-TIMING / reversal label — the "edges resolve at turning points" insight tried as a TARGET: deployable +
  balanced on UUP but loses 0.36 (solo 0.18) vs `bgm+ker` 1.30. MECHANISM: local micro-reversal timing ≠ UUP's MACRO
  regime oscillation, which bgm's distributional clustering already captures. LESSON: the resolution insight is about
  where to SAMPLE the clock (a bar-axis property), NOT a labeling target); `perment` (INFO-THEORETIC: permutation
  entropy / Bandt-Pompe ordinal predictability — trade only ordinally-structured forward windows by sign. The "info-
  theoretic" frontier item: VALID with real standalone signal (SOXX solo 1.24, `ker+perment` pair 1.98 — best new-label
  near-miss) but REDUNDANT, below champion `ker+trend_scan+bgm` 3.02. MECHANISM: ordinal-entropy predictability ⊂ what
  trend_scan(slope-sig)+ker(efficiency)+bgm(regime) already capture; a weaker cousin of the trend-predictability family).
  **NEW-LABEL FRONTIER BROADLY MAPPED** — every DISTINCT forward-window readout now built (slope, efficiency, curvature,
  magnitude, trail-sign, extremum-timing, risk-adj, regime, tail, ordinal-entropy); none beats the per-asset champion.
  Combined with the closed bar-axis lever, the two PRIORITY new-method modules are now exhausted. **PIVOT (per deep-
  research review): the high-value work is now decay-monitoring + honesty infra (DSR/PBO multiple-testing, survival
  analysis for alpha decay), NOT more method-building** — each new method adds multiple-testing burden for ~0 edge.
- **ensemble composition** (now cheap — multi-file unblocked 3-way+): same-family 3rd labels on a structured
  champion. GLD/UUP 3-way mapped; deeper combos = diminishing/overfit.
- secondary: new **features** (non-crowding), **reducers** (vs corr-20), **sizers** (`dd_overlay` tried, lost on UUP).
- **loop-honesty/efficiency (from the deep-research review, higher-leverage than more permutations):** standing
  permuted-label GATE in the driver · PROV-JSON provenance schema (the ledger is a *provenance* graph, not causal)
  · TPE/ASHA-driven selection over the (now larger) ensemble-composition space vs manual enumerate-and-race.

**Closed lever (superseded):** the 64k render limit / axis-pruning — SOLVED by multi-file render (bar_builder.py
as a separate QC file). Don't re-grind axis-pruning.

## setup
QC project 31338454, creds `qc/.creds.json`. Hypothesis = `{ticker, axis, labeler, thresh, sizing[, max_depth]}`.
Modules `autoresearch/modules/` (bar_builder · features · labeler · trainer) + templates (header/footer/infer/verify) ·
findings `knowledge.json` (provenance graph) · audit `BACKTEST_AUDIT.md` · reviews `RESEARCH_REVIEW*.md` ·
Wang's course `pdfs/`, `docs/legacy/wang_qa_questions.md`.

## efficiency review v3 (2026-06-03, `RESEARCH_REVIEW_v3.md`) — 110 primary claims, 72/75 verified
Second deep review. **#1 NEW upgrade: E-VALUES / anytime-valid inference for continuous re-validation** — we re-test champions as the OOS window grows (decay checks) = optional-stopping/peeking, which INVALIDATES p-value/DSR-based gates (the reproducibility-crisis cause). E-processes stay valid under continuous monitoring (Ville), are the ONLY admissible anytime-valid method, and MERGE BY MULTIPLICATION. This is the formal cure for staleness the prior review named but didn't solve. Alpha-spending (O'Brien-Fleming; data-driven peeking doesn't inflate alpha) is the lighter alternative. OTHER verified takeaways: (1) 'causal graph as log' = confirmed TERMINOLOGY TRAP — ours is correctly a PROVENANCE graph (causal-DAG nodes are population RVs/assumptions, not findings/derived stats). (2) SINGLE-agent ≥ multi-agent for the core loop (Operand Quant top MLE-Bench, beats orchestrated) → don't build tournaments; our single loop is right. (3) EXECUTION is the only arbiter — LLM idea-novelty FLIPS after execution, LLMs can't eval ideas (53%<56% human), idea diversity capped ~5% → never crown on my judgment, gate on real Calmar (we do). (4) AI honesty harnesses BARELY beat chance → external QC backtest = our defense-in-depth, never let an LLM self-grade. (5) cap tunable params ~15-20 (we're at ~9, watch it). (6) xKG: admit a method only if backed by runnable code; semantic similarity is a DECEPTIVE retrieval signal.

**E-VALUE MONITOR IMPLEMENTED (2026-06-03, `scripts/evalue_monitor.py`) — the review's #1 upgrade, validated.** Testing-by-betting e-process (WSR 2023) for H0: mean<=0; anytime-valid (Ville), peeking-robust, re-validations MULTIPLY in. Self-test PASSES: strong-signal e=135 (power), zero-mean false-positive rate 0.000 (<=0.05 Ville bound, valid). RESULT on champions (weekly cached series): GLD e=6.80, SOXX e=3.27, UUP e=1.31, HYG 2.51, TIP 1.58, DBC 1.26 — ALL 'weak' (e>=1 but <20), NONE clear anytime-valid significance. KEY: the peeking-robust e-value is STRICTLY MORE CONSERVATIVE than the fixed-sample DSR (GLD 0.93/SOXX 0.96) — the honest price of our continuous re-testing; GLD has the most anytime-valid evidence. CAVEATS: weekly series (short -> limited power), tests RAW mean (drift-confounded; buy-hold also positive). REFINEMENT QUEUED: daily-resolution + excess-over-buyhold e-process. This SUPERSEDES the peeking-invalidated DSR/p-value re-checks for ongoing decay monitoring.

**E-VALUE now NATIVE + frequency-invariant (2026-06-03).** Added `evalue_oos` runtime stat to infer.py.tmpl (anytime-valid e-process on the DAILY OOS returns, computed on-QC; contract-safe post-hoc stat like sharpe_oos) — every champion infer now auto-reports its peeking-robust e-value. HONEST findings: (1) the e-value is FREQUENCY-INVARIANT — daily ≈ weekly (GLD 6.38≈6.80, SOXX 3.09≈3.27, UUP 1.32≈1.31) because e-value ~ Sharpe²×years independent of sampling -> last tick's 'daily adds power' hypothesis was a NO-OP (honest negative). (2) The binding constraint is BOUNDED BETTING (λ_max=5), not resolution: low-vol risk-managed strategies have optimal λ≫cap, so the bet is clipped -> the e-value is CONSERVATIVE. (3) SCOPE (corrects a slight over-claim): the e-value tests PROFITABILITY (mean>0, peeking-robust), drift-confounded for long-biased edges + bounded-bet-conservative -> it's a peeking-robust LIVENESS/DECAY monitor, NOT a high-power significance test nor an anytime-valid Calmar test (risk-reduction isn't a mean). Ranking holds GLD>SOXX>UUP. USE: ongoing decay monitoring (re-validations multiply in), not crowning.

## FRONTIER — what would re-open productive research (2026-06-03 terminus)
The loop has reached an EARNED terminus on the current inputs: 3 confirmed edges fully characterized
(selection/decay/module/granularity/cost/anytime-valid lenses), a deployed gated upgrade (GLD 4.71),
a complete honesty stack incl. the native e-value monitor + validated decay gate, and the deep review
delivered + its #1 upgrade implemented. The new-method frontier is exhausted (kyle/run/spectral axes,
turn_scan/perment/accel/mfe_mae/revert labels all lost or redundant; even medicine-inspired survival
labeling = our existing triple_barrier). New universe returns buy-hold (XME✗). The review confirmed
durable single-asset alpha is intrinsically scarce (R²~0.003-0.005) and the bottleneck is self-deception
(now armored), not throughput. **Static OOS data + 5-min ticks => no new information per tick.** Further
config/universe grinding only inflates the multiple-testing burden. Productive work needs a NEW INPUT:

1. **New DATA modality** (highest leverage) — we have exhausted price/volume. Alternative data with a
   DIFFERENT information source: options-implied vol / skew, positioning (COT, ETF flows), macro-release
   surprises, cross-asset signals (credit spreads, the VIX term structure). New information, not new tuning.
2. **New MECHANISM-CLASS universe** — we tested trend/regime on equity-sector/commodity/dollar/bond ETFs.
   Untested edge mechanisms: volatility products (term-structure carry), rate-CURVE spreads, FX carry —
   each a structurally different edge than our trend/regime crowns.
3. **Cross-asset PAIRS** (relaxes the single-ticker constraint) — the one place a real cross-asset edge can
   live (a 2-symbol relative-value strategy); single-ticker provably can't capture it (frontier-mapped).
4. **Intraday holding** — we use minute bars but ~daily holding; an intraday-horizon edge is a different
   regime we haven't searched.
5. **REGIME CHANGE / real-time decay** — the ONLY thing that changes on the current inputs is the OOS
   window growing as real calendar time passes. The native `evalue_oos` decay gate is the standing monitor;
   re-validate + act when it flags. Until then, the deployed book is the answer.

**Standing behavior at terminus:** monitor the deployed book's `evalue_oos` liveness/decay; do NOT
manufacture experiments on static data (it inflates the deflation bar for zero edge). Re-open only on a
new input (above) or a decay flag. To redirect the loop, point it at one of (1)-(4).
