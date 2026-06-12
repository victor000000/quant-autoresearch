# autoresearch

An autonomous loop that hunts single-ticker trading edges on ETFs. Each round: read this file + `knowledge.json`, design ONE recipe, race it A/B against the champion on real out-of-sample Calmar in QuantConnect (project 31338454), keep iff every honesty gate passes. Single ticker, no cross-ticker ensembling, simple over complex.

This is an LLM pointed at a number it can game, so **most of this file is gates, not method**. A backtest happily reports a great Calmar that is a leak, a multiple-testing fluke, or plain drift. Telling a real edge from those three impostors is the whole job.

## The loop

1. **Re-validate, then target.** Stored Calmars go stale as the OOS window grows (UUP 1.85→0.60 in 8 days). Re-race champion-vs-`always_long` before trusting any stored number.
2. **Design one recipe** — `axis × label × features × reduce × model × sizing`. Every confirmed edge came from a NEW axis, labeler, or mechanism — never knob-tuning. Knob-permutation on a structureless name only raises everyone's deflation bar.
3. **Race** — `scripts/run_round.py '<A>' '<B>'`. A hypothesis is `{ticker, axis, labeler, thresh, sizing}` + optional `{reduce, n_components, features, model, calibration, horizons, permute_labels}`; labelers ensemble with `+`. Writes results/report; never commits.
4. **Keep iff ALL gates pass; log either way** (a discard raises every co-tested name's deflation bar). A human/Opus commits and crowns. Save each round: `round_results.csv` + `knowledge.json` + `ETF_SCREEN_TABLE.md` (exactly these 3).

### Keep gates — all must hold

| Gate | Threshold |
|---|---|
| Deployable | trades > 80, train+infer completed, DA present |
| Beats champion | winner Calmar > re-validated previous best |
| Learnable structure | val_auc > 0.52 (below = coin-flip artifact) |
| Survives deflation | > expected-max of N noise trials (best-of-N; `always_long` exempt) |
| PSR significance | Bonferroni-deflated `psr > 1 − 0.05/N_trials` |
| Permuted-label control | real edge over buy-hold > 0.15 AND permuted < 40% of real |

**The permute control is the decisive test** — shuffle only the TRAIN labels (`"permute_labels": true`, separate `_perm` cell); a real edge collapses toward buy-hold (GLD: 4.36→0.76). **It must FAIL LOUD**: a 0-trade control leg refuses the KEEP. Cell keys come only from `lb.harness.psuf.cell_suffix` (inline copies diverged three times; suffix-collision bugs made the gate silently vacuous once).

## Pipeline + leak contract (fixed, `random_state=42`)

```
event bars → StandardScaler → reduce → model(depth 3, lr .03, n 200) → calibrate (embargoed VAL)
           → de Prado CDF bet × causal overlay
```

- **Causality:** bar thresholds fit on TRAIN only, extrapolated identically OOS; labels may look ahead (they're targets) but never beyond `_EMBARGO = max(200, horizons, declared reach)`; full-sequence smoothers block-decode at the OOS boundary; features past-only; sizing identical in backtest and live.
- **The OOS backtest is online, leak-free, ObjectStore-replay-only** (`BACKTEST_AUDIT.md`). `infer.py` holds no model. Proofs: `verify.py` (bars ≤1e-9), `infer_online.py` (preds ≤1e-6).
- **Leaks are found by re-running, not reading.** After any bar-builder change run `tests/test_bar_threshold_leak.py`; when a feed reports "no data", probe the channel (`set_runtime_statistic`).
- **QC limits:** 64,000 UTF-8 bytes/file, code only (comments minified away). main.py is near budget — new axes → `bar_ext.py`, features/reduces/models/calibration → `ml_ext.py`, sizing → `sizing_ext.py` (each gets its own budget). Lint-clean QC files: no `getattr`, no nested-quote f-strings. **Lean CANNOT catch lookup errors** (AttributeError/NameError escape try/except and crash the run — proven twice): guard with `hasattr`, never rely on except.
- **Code layout:** installable `src/lb/` package. Pipeline: `lb.modules.*` (bar_builder+bar_ext, labeler, features, trainer, ml_ext, sizing_ext), render `lb.harness.orchestrator`, keys `lb.harness.psuf`, metrics `lb.metrics`. Drivers: `scripts/run_round.py`, `scripts/research/*` (screen/sweep/report). Honesty: `scripts/audit/*` + `tests/`. Results = the 3 files above; website deleted 2026-06-11.

## The honesty stack

Never crown on LLM judgment — only on real QC Calmar, then: deflated Sharpe (`audit/deflated_audit.py`) · session DSR Holm-Bonferroni (`audit/honest_audit.py`) · permute control · decay monitor (`research/champion_series.py`) · e-values · PBO/CSCV · cost stress (book holds at 5bp) · drawdown-shape (Ulcer/Pain/Martin via `lb.metrics`) · regime-split (`audit/regime_split.py`: edge must survive BOTH past-only trailing-vol halves — GLD 4.64/3.32 PASS, USO 3.83/1.87 PASS) · deflated Calmar (`audit/deflated_calmar.py`, MC max-of-N null on the OOS curve: bootstrap GLD p=0.003 PASS, USO p=0.043 PASS-narrow; `--null iaaft` variant exists but is structurally weak for Calmar — amplitude-exact surrogates pin CAGR≈0, so bootstrap is the binding null; asset-side IAAFT would need a QC custom-data harness). Standing facts: only GLD survives the full ~2500-trial Holm-Bonferroni burden; every other edge is provisional until the calendar grows (MinBTL). **Weight conviction by DSR and recovery-pain, not raw Calmar** (Calmar flatters slow-bleeders: DBC underwater 62% of days, Martin-last; USO recovers fastest, Martin-top).

## The book (common-grid 2026-06-12, all legs same window)

| Ticker | Config | Calmar | Status |
|---|---|---:|---|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t.40 / n16 **vae lgbm_bag** | **4.70** | Crown 2026-06-12: 5-seed-bagged lgbm 4.704 > lgbm 4.357 > xgb-vae 3.954 > pca 3.843 (backlog #11 — variance hardening at fixed capacity, sequential seeds, no shopping); permute 4.70→0.76 (84% collapse); bit-exact. lgbm family is **GLD-specific** (loses on USO/IAU; catboost loses on GLD). **LIVE-DEPLOYABLE 2026-06-12: preds_match=1, max_diff 0.0** — bundles now carry the frozen reduce transform (pca/vae) + family-dispatched boosters (xgb/lgbm/bags); before this fix every projection-reduce cell was silently non-deployable (kept_idx placeholder). |
| **USO** | logdollar / `revert` / cdf_plain / t.45 / correlation | **2.82** | Oil mean-reversion engine. Permute-confirmed, decay-strengthening, regime-split PASS, deflated-Calmar PASS-narrow, **online-proven 2026-06-12 (preds_match=1, max diff 3e-08)**. |
| DIA | imbalance / `bgm+ker` / cdf_overlay / t.50 | 1.66 | Screen find, fully gated + HARDENED 2026-06-12 (regime-split 3.04/1.32 PASS, deflated-Calmar p=0.018 PASS). The candidate seat. |
| FEZ | logdollar / `sadf_explosive` / cdf_overlay / t.50 | 1.34 | ⚠️ DEMOTED 2026-06-12: FAILS regime-split (low-vol Sharpe −0.72, one-regime fragility) AND deflated-Calmar (p=0.133). Watchlist, not a seat. |
| HYG/TIP/DBC/IWM/UUP | `always_long` (×cdf) | 1.97/0.90/0.69/0.50/0.37 | Decorrelation seats; timing retired on all. |
| PRFZ | sadf_explosive | 2.22 | ⚠️ RISKY — sibling replication FAILED (PRF/IJR); probation only. |

**Proposal A FINAL (Calmar² weights, awaiting human decision; FEZ removed after audit demotion):** GLD 57.7 / USO 20.8 / HYG 10.1 / DIA 7.1 / TIP 2.1 / DBC 1.2 / IWM 0.6 / UUP 0.4. Proposal B adds PRFZ (~11%) — not recommended (replication failure). Overlap CORRECTED 2026-06-12 (the first check accidentally used permute-leg curves): true champion-curve corr is 0.27–0.51 with co-activity modestly above independence — a moderate common component (sadf family + equity beta), so candidate seats add PARTIAL decorrelation, not full. Proposal A should seat DIA only; FEZ demoted (failed regime-split + deflated-Calmar). Weights always from **common-grid** numbers, gross ≤ 1; dropping weak names lowers book Calmar.

**Two replication-confirmed engines, asset-intrinsic:** gold-trend (GLD→IAU) and oil-reversion (USO→UCO, val_auc 0.97). The `sadf_explosive` regime family is alive on 3 independent funds (PRFZ/FEZ/IAT). Wang's β200 lens routes assets (≈0.5→trend, ≫0.5→buy-hold, <0.45→reversion) but is clock-dependent and admission-only.

## Laws (every one bought with races — they change decisions)

- **Predictability ≠ profit.** The strongest results in the project: indices carry oil-grade reversion auc (0.94–0.97) that never monetizes; calendar features set auc records on SPY/IWM (0.64/0.63) with no conversion; on GLD itself, pls (auc 0.756) and ddstate (auc 0.786) beat the champion's auc and LOSE on Calmar. Val_auc admits, only Calmar crowns.
- **Equity-index timing (QQQ/SPY/IWM) is closed at every door** — 4-pass sweep + monotonic action bracket (QQQ: gate 0.25 < overlay 0.33 < tilt 0.71 < always_long 1.25), VIX/VXX features, crash-veto/crashsoft, dd_excursion×tilt_up (the no-forfeit lean adds noise-timed DD), calendar. The less you act, the better. Their optimal config IS always_long. Reopen only with new data (options IV).
- **Mechanism-paired components.** Trend (bullion) wants unsupervised projection (vae > pca ≫ infogain) and tolerates lgbm; reversion (oil) wants raw selection (correlation ≫ any projection — pls/spca/pca all crater it) and stays xgb. Component choice is asset physics, not tuning; nothing transfers without a replication race.
- **Panel additions dilute.** Every feature block beyond base 80 lost on the engines (rich, fracdiff, regime, vix, ddstate, cross-asset, real-yield) under every reducer. Base panel is the optimum for both engines; only zero-price-content features (calendar) are dilution-immune — and still didn't convert.
- **Drifters (val_auc≈0.5) are buy-hold-optimal**; new methods only help where structure exists (val_auc>0.6). `trend_leg+regime_gmm` degenerates on up-drifters (0 trades) — use `tertile` there.
- **One source of truth for keys/configs** — psuf token collisions (`_fx`, `_va`, `_perm` cell-key class) made gates vacuous three times; fixed with fail-loud + single builder.

## Closed — don't re-grind

Depth-3 capacity (one model-family win exists: lgbm, GLD-only) · meta-labeling · uniqueness weights · sticky-HMM · cross-asset price features · exogenous GLD features in every form (proxies/yields: auc up, Calmar down = the overfit signature) · VIXY carry even with purpose-built `short_carry` · universe siblings (SLV≠GLD, GDX/XME projection-fragile) · oil-reversion beyond oil · innovation backlog #1 pls, #2 spca, #3 dd_excursion, #7 beta-cal, #8 calendar, #9 ddstate, #10 permclock axis, #13 kelly_dd (all raced 2026-06-12, all lost; #11 seed-bagging WON on GLD only — xgb_bag loses on USO 2.07<2.82) · Wang-transcript levers (cond_es, regime-KDE feature, spearman — corroboration, not extension) · 311 screen COMPLETE (5 permute-confirmed candidates above; full table `docs/research/ETF_SCREEN_TABLE.md`).

Retained-but-idle primitives live where they belong: sizers in `sizing_ext` (tilt, tilt_up, crashsoft, short_carry, cond_es), reduces in `ml_ext` (minor_pca, whiten, vae_rl, pls, spca), features in `build_feats`/`ml_ext` (regime, vix, calendar, ddstate).

## The frontier

- **Innovation backlog COMPLETE (2026-06-12, all 14 items raced):** two wins, both model-module and both GLD-only (lgbm, lgbm_bag = the crown); everything else closed. New axes retained for screens: permclock (ordinal), mpnov (novelty/discord — finds structure on EEM/DBA auc 0.68-0.73, no conversion). Chronos door CLOSED-BY-SANDBOX (2026-06-12 probe: QC backtests have NO internet — HF weights can't download; package present, weights absent). Only reopening path = ObjectStore weight side-load (user decision; low EV given the index structural barrier).
- **Human decisions pending:** book proposal A vs B vs status-quo · PRFZ probation · PR #1 merge (needs the lb.paths script fix applied to that branch) · lgbm live-deserialization before deploying GLD-lgbm live · any new data modality (options IV / COT / flows) — the only true reopener for the indices.

Autonomous — decide and run the next experiment, don't ask.
