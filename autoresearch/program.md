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

## honest state (2026-06-03, late)
Durable single-ticker alpha is **scarce** — two edges survive re-validation + deflation + the permuted-label control:
**GLD `ker+trend_scan+accel` 3.22** (gold's tradeable trend; the `accel` curvature label was added in a 3-WAY
ensemble — unlocked by multi-file render — and beat the old 2-way 3.20) and **UUP `bgm+ker` 1.30** (dollar regime,
optimal as 2-way). Everything else is buy-hold drift (QQQ/EEM/EFA/HYG/TIP/DBC/XLE/IWM/SLV) or no-edge (TLT). Both
edges are PERMUTE-VALIDATED real (collapse to ~buy-hold under label shuffle). The deployable book is a downstream
combination of these champions, not the research target.

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
- new **bar axes** (priority) — how you sample the clock. Built: `kyle` (price-impact/illiquidity, no win). Untried:
  run-length, spectral.
- new **unsupervised labels** (priority) — what you call "up". Built: `ker` ✓ + `accel` ✓ (ensemble-win on GLD);
  `sharpe_scan`/`mfe_mae` valid, no win (mfe_mae gold-specific 2.5). Untried: reversal-timing, info-theoretic.
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
