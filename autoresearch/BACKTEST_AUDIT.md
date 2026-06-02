# OOS Real-Backtest Correctness Audit (2026-06-02)

13-agent adversarial audit (6 dimensions × audit+verify + synthesis). **Question:** is the OOS real backtest bug-free, look-ahead-leak-free, fully ONLINE (custom-axis bar-gen, feature-gen, predict, set_holdings), and using ONLY the QC-Cloud-trained model from ObjectStore?

## Verdict: CLEAN (MINOR_ISSUES) — all four requirements CONFIRMED

| Requirement | Verdict |
|---|---|
| No bugs / no exploitable look-ahead leaks | ✅ CONFIRMED |
| Model only from QC ObjectStore (never re-fit on test) | ✅ CONFIRMED |
| Custom-axis bar generation online | ✅ CONFIRMED |
| Feature-gen / predict / set_holdings online | ✅ CONFIRMED |

## Evidence (data-flow proof)

**Headline metric = pure prediction replay.** The reported OOS Calmar comes from `infer.py.tmpl`, which instantiates **no model** (no `.fit`/StandardScaler/Isotonic/XGBClassifier/Booster/`build_feats`/`predict` anywhere). `initialize()` loads only `predictions`+`thresh`+`sizing` from the ObjectStore cell; `on_data` walks the frozen saved predictions and calls `_size`, whose only state is a **causal** trailing-return buffer (decide-then-append). `portfolio.py.tmpl` (the deployed book) is the identical pattern. Nothing in the OOS path can be re-fit on test data.

**Model provenance.** Every `.fit`/`fit_transform` in the entire `templates/` dir lives in `footer.py.tmpl` (training): model on `tx = fv & ly & tr_m` (TRAIN); StandardScaler + `reduce_dims` kept-indices from TRAIN only; calibrator + XGB early-stopping eval_set on `vx_lab = fv & ly & va_m & emb` (VAL, embargoed 200 bars off the val→test boundary); meta secondary on purged K-fold within TRAIN. Test enters **only** as `ex = fv & te_m` via `predict_proba`/`cal.transform` — predict-only, no future-label selection of which bars trade (G3 invariant; `n_pred == n_test_bars` leak sentinel). `infer_online.py.tmpl` loads every predictive parameter from the bundle `model_{CELL}.json` (scaler, kept_idx, isotonic knots, best_iter, booster, meta booster+gate, bar_thresh) — loads, never computes.

**Online-equivalence (empirically proven, not just asserted).** Every axis is an online `update()` class; `build_bars` iterates `update()` over minutes, so batch==online by construction. Proven two ways: `verify.py.tmpl` rebuilds OOS bars online from the frozen threshold → byte-identical (`bars_match`, `max_lc_diff ≤ 1e-9`); `infer_online.py.tmpl` rebuilds bars+features+booster+calibration+meta-gate online → `p_live == p_saved`, `max_pred_diff ≤ 1e-6`. Features are causal (rolling/min_periods, no forward window; entropy stride-grid anchored by `abs_start` so a trailing window reproduces the full-build grid exactly).

**Prior leak fixed.** Bar thresholds previously used full-series stats incl. OOS; now TRAIN-only via `_train_minute_mask` (fail-loud — raises rather than silently using all minutes). Every distributional statistic feeding a threshold (mean/std/quantile/sigma/bincount) is strictly TRAIN-masked.

## Two LOW items (neither is a leak)

1. **Cosmetic future-count.** The bar *granularity* knob scales the threshold by a full-series minute **count** (`len(close)`/`sum(valid)`/`n_all_min`) to target ~15k bars. It is a count, never a function of OOS prices/returns/volume, so it conveys no tradeable future info; and both batch and online paths consume the **same frozen** `bar_thresh`, so online-equivalence is unaffected. Fix (optional, deployment-realism only): replace with a TRAIN-extrapolated count `n_train_minutes / train_fraction`. Not worth the re-validation churn for the already-validated book.
2. **Verification-coverage gap.** `entropy`/`fracdiff` are outside `BUILDER_CLASSES`, so they skip the `verify`/`infer_online` cross-checks. Causal by construction (TRAIN-only fitted edges/probs/weights, trailing convolution) but lacking the automated proof. Simple guard: don't promote a non-`BUILDER_CLASSES` axis to a headline champion without the online proof. (Neither is currently a champion.)

**Bottom line:** the OOS real backtest uses only the QC-ObjectStore-trained model, never re-fits on test, never selects trades by future labels, and runs bar-gen/features/predict/set_holdings causally bar-by-bar — proven by the `verify` and `infer_online` assertions. The two LOW items are cosmetic/coverage, not look-ahead leaks.
