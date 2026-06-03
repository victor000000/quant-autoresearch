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

## honest state (2026-06-03, very late) — mechanistically understood; gold is multi-structure
Durable single-ticker alpha = **3 confirmed edges** (Bonferroni-significant + permute-validated, deployable):
**GLD `ker+regime_gmm`+`dd_overlay` n=15 = 4.55 (reducer width n_components=15 was the latest lever; arc 3.20->3.22->3.90->4.02->4.19->4.55, six improvements from "fixed" dimensions)** (logdollar; trend+REGIME HYBRID + drawdown-aware sizing — adding regime_gmm to the trend core
beat the old 3.22, +21%; threshold-robust 3.64–3.90, the most trustworthy crown) · **SOXX `ker+trend_scan+bgm` t0.50 = 3.02** (logdollar; semis ARE multi-structure — trend+REGIME via bgm; +43% over the old 1.92; the SMH sister-fund replication failed on the pure-trend version → fund-specificity caveat stands) ·
**UUP `bgm+ker` 1.30** (IMBALANCE axis; regime edge). Everything else = buy-hold. Provisional/un-crowned: KRE/ITB
(permute-pass but Bonferroni-FAIL). SLV lead DEAD (logdollar+t0.30-specific, axis-fragile, hybrid hurts it).

**MANY assets are MULTI-STRUCTURE; the REGIME DETECTOR is asset-specific** (key 2026-06-03 find, corrects an earlier 'gold-unique' claim): gold→`regime_gmm` (+21%→4.02), semis→`bgm` (+43%→2.73), dollar→`bgm`. Using the WRONG detector looks degenerate (regime_gmm on semis=no-op), which is why I'd wrongly called names 'pure-trend'. bgm-regime cuts drawdowns broadly on cyclical assets (IWM/EEM show it) but only crowns where the CAGR/DA balance is right (SOXX). `dd_overlay` sizing is GOLD-specific (its slow/persistent drawdowns; hurts semis' V-recoveries). Edge type AND axis
are asset-intrinsic: GLD/SOXX edges live on logdollar (information clock), UUP on imbalance (order-flow microstructure).

**THE governing rule (explains every result): two edge MECHANISMS, each needs its own labeler, edge type is asset-intrinsic.**
1. **TREND-MOMENTUM** (`ker+trend_scan`): wins ⟺ drawdowns are MOMENTUM-CYCLICAL / trend-predictable AND trimming
   costs < MaxDD it saves → GLD ✓, SOXX ✓; SLV near-miss (mechanism fires, 20× less DD, but Calmar just under BH);
   FAILS on event/shock-driven (XLE oil, XBI FDA, DBC commodities → 0.05–0.3) and V-recovery up-drifters (QQQ/SPY → trims).
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
