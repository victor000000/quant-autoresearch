# Innovation backlog — every module of Wang's pipeline (2026-06-11)

Four parallel literature scans (2020–2026: arXiv q-fin/stat.ML, practitioner sources) over
ALL pipeline modules, each armed with the full current inventory + closed-list so nothing
below re-proposes a raced or closed idea. Every candidate is in-rule: OHLCV-only,
single-ticker, causal/TRAIN-only-fit, append-OOS-invariant, numpy+sklearn, 64k QC budget
(new reduces → `ml_ext.py`, axes → `bar_ext.py`, sizing → `sizing_ext.py`).

**Standing evidence the ranking respects:** label/feature innovations historically beat
model innovations here; model CAPACITY is closed (depth-3 hard); reduce is MECHANISM-PAIRED
(trend→projection, reversion→raw selection); panel additions dilute (orthogonal features
only); sizers reshape risk but add no signal; we select on Calmar but only deflate Sharpe.

## Cross-module TOP PICKS (EV-ranked, build in this order)

1. **`reduce='pls'` — Partial Least Squares (supervised projection)** [reduce, HIGH]
   The single missing idea: the *supervised twin* of the winning `pca` mechanism. PCA won
   GLD because trend lives in high-variance directions — PLS projects onto max
   *label-covariance* directions instead. ~30 LOC NIPALS in `ml_ext.py`, pca-identical
   leak contract. A/B: GLD vs pca 3.84 (+ a 1-component ablation). (arXiv 2409.05713)
2. **`reduce='spca'` — Bair screen-then-project** [reduce, HIGH]
   Selection THEN projection in one operator — the literal bridge of the mechanism-paired
   insight; the only candidate worth racing on BOTH engines. ~20 LOC. A/B: GLD vs 3.84 AND
   USO vs 2.72. (Bair JASA; arXiv 1710.06229)
3. **`labeler='dd_excursion'` — forward drawdown-shape label** [labeler, HIGH]
   New *path-quality* mechanism: y = "forward path reaches +k·σ while staying low-Ulcer"
   — predicts *tradeable smoothness*, not endpoint sign; optimizes the deployed objective
   (Calmar/Martin); reuses `lb.metrics`. ~40 LOC. A/B: GLD + USO refinement, TLT probe.
4. **`label_clean='cl_prune'` — confident-learning label pruning** [model, MED-HIGH]
   Label-axis (the winning axis): drop the ≤8% of TRAIN bars the purged-CV model itself
   confidently disputes; refit unchanged depth-3. Reuses the existing meta-block purged-CV
   loop (footer ~350–371). Capacity-neutral. A/B: GLD (val_auc 0.72 = noise to clean).
   (Northcutt 1911.00068)
5. **Deflated CALMAR** [honesty, HIGH]
   The cleanest hole in the stack: we crown/weight by Calmar but deflate only Sharpe; maxDD
   is the most selection-inflated order statistic. Magdon-Ismail/Atiya E[MDD] (or MC null)
   + expected-max-of-N → deflated Calmar gate. Host-side ~120 LOC in `stats_rigor`.
6. **Regime-split validation** [honesty, HIGH]
   Require the edge to survive independently in BOTH trailing-vol halves (past-only
   regime). Catches one-regime fragility every aggregate gate misses. ~50 LOC.
7. **`calibration='beta'`** [calibration, MED-HIGH]
   Isotonic provably overfits at our calibration-set sizes (<~1000); beta matches it at
   1/20th the variance. ~15 LOC. A/B: GLD vs isotonic. (Kull-Filho-Flach)
8. **Calendar/seasonality feature block** [features, MED-HIGH]
   Turn-of-month/day-of-week sin-cos — the one feature axis with ZERO price content
   (immune to panel-dilution), zero leak. Cheapest shot at the closed equity indices
   (turn-of-month is the robust S&P anomaly). ~10 LOC. Probe: SPY/IWM.
9. **Drawdown-state + trend-age features** [features, MED]
   Underwater depth/duration/recovery-slope + run-length/regime-age as causal INPUTS
   (we ship them as metrics; never as features). Path-dependent, not monotone-of-momentum.
   ~18 LOC. A/B: USO (reversion timing), GLD (trend exhaustion).
10. **`axis='perment'` — permutation-entropy ordinal clock** [axis, MED]
    Ordinal-pattern complexity transitions; invariant to monotone price transforms —
    a genuinely different state variable than every existing clock. ~40 LOC, leak-trivial.
    Probe: TLT/UUP/FXY/natgas (the edge-free set). (Bandt-Pompe)
11. **Robust-focal/GCE custom objective + seed-bagging** [model, MED]
    Loss-shape swap (bounded gradients on suspect bars; arXiv 2310.05067) ~40 LOC custom
    (grad,hess); seed-bagging (K=5 depth-3 average) ~15 LOC as variance/decay hardening.
12. **IAAFT spectrum-preserving surrogate null + e-BH** [honesty, MED]
    The orthogonal null: preserves linear autocorrelation, destroys phase — the right
    stress test for the USO reversion edge specifically. e-BH = FDR under arbitrary
    dependence (our variants are dependent), ~30 LOC on existing e-values.
13. **Risk-constrained Kelly (drawdown-capped)** [sizing, MED-LOW]
    The ONLY theoretically-motivated sizing challenger: provably growth-vs-drawdown
    frontier-optimal (Busseti-Boyd), competing on the exact deployed objective vs the
    heuristic dd_overlay champion. ~40 LOC. Expectations capped by "sizers add no signal."
14. **Matrix-profile novelty clock + analog label** [axis+labeler, SPECULATIVE]
    The asset-class-unlock swing: left-MP (strictly-past) discord clock + label-by-what-
    followed-the-nearest-past-motif. Highest ceiling (non-parametric analog mechanism),
    highest build+leak care. Probe: DBA/natgas/EEM.

## Explicitly rejected by the scans (with reasons)
Volatility-managed sizing (lit. negative OOS + cdf_overlay has it) · conformal-width
sizing (blocked by the (p,thresh,rbuf) interface; width ≈ constant) · shrunken Kelly
(collapses to a fixed fraction) · meta-sizing on strategy P&L (decay-adjacent) ·
monotone/interaction constraints on GLD (meaningless after PCA scrambling; USO-only,
needs sign map) · DART (fights early-stopping) · sklearn ensemble families at inference
(ExtraTrees platform-blocked in QC — RuleFit/oblique carry the same risk) · HSIC/dCor
(dominated by Chatterjee ξ) · knockoffs/Boruta (≈ the already-built null-importance) ·
label smoothing (subsumed by robust-focal) · temperature scaling (dominated by beta) ·
GAS regime label (regime rehash) · MF-DFA/Kleinberg clocks (noisy/near-duplicates) ·
co-teaching (DL-era complexity).

## Module coverage map
axis: #10, #14 · labeler: #3, #14 · features: #8, #9 · reduce: #1, #2 (+ξ for USO) ·
model: #4, #11 · calibration: #7 · sizing: #13 · honesty: #5, #6, #12.

**Execution note:** the 311 screen owns QC until it completes; race these after (or pause/
resume per the DIA protocol). Each build follows the standard discipline: leak gates
(append-invariance / render-smoke) → A/B vs champion → permute → deflation → decay.
Full agent reports preserved in session transcripts; sources cited inline above.
