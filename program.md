# autoresearch

An autonomous loop that hunts single-ticker trading edges on ETFs. Each round: read this file + `knowledge.json`, design ONE recipe, race it A/B against the champion on real out-of-sample Calmar in QuantConnect (project 31338454), keep iff every honesty gate passes. Single ticker, no cross-ticker ensembling, simple over complex.

This is an LLM pointed at a number it can game, so **most of this file is gates, not method**. A backtest happily reports a great Calmar that is a leak, a multiple-testing fluke, or plain drift. Telling a real edge from those three impostors is the whole job.

## The loop

1. **Re-validate, then target.** Stored Calmars go stale as the OOS window grows (UUP 1.85→0.60 in 8 days). Re-race champion-vs-`always_long` before trusting any stored number.
2. **Design one recipe** — `axis × label × features × reduce × model × sizing`. Every confirmed edge came from a NEW axis, labeler, or mechanism — never knob-tuning. Knob-permutation on a structureless name only raises everyone's deflation bar.
3. **Race** — `scripts/run_round.py '<A>' '<B>'`. A hypothesis is `{ticker, axis, labeler, thresh, sizing}` + optional `{reduce, n_components, features, horizons, permute_labels, ...}`; labelers ensemble with `+`. Writes results/report; never commits.
4. **Keep iff ALL gates pass; log either way** (a discard raises every co-tested name's deflation bar). A human/Opus commits and crowns.

### Keep gates — all must hold

| Gate | Threshold |
|---|---|
| Deployable | trades > 80, train+infer completed, DA present |
| Beats champion | winner Calmar > re-validated previous best |
| Learnable structure | val_auc > 0.52 (below = coin-flip artifact) |
| Survives deflation | > expected-max of N noise trials (best-of-N; `always_long` exempt) |
| PSR significance | Bonferroni-deflated `psr > 1 − 0.05/N_trials` |
| Permuted-label control | real edge over buy-hold > 0.15 AND permuted < 40% of real |

**The permute control is the decisive test** — shuffle only the TRAIN labels (`"permute_labels": true`, separate `_perm` cell); a real edge collapses toward buy-hold. GLD's genuine run: real edge +2.44 → permuted +0.09 (96% collapse). **The control must FAIL LOUD**: a 0-trade/failed control leg is indistinguishable from reading a nonexistent cell and refuses the KEEP (2026-06-10: a `_PSUF` cell-key mismatch made the gate silently vacuous for every suffix-keyed config — the third bug of that class; cell keys now come from the single `_cell_key()` builder, never inline).

## Pipeline + leak contract (fixed, `random_state=42`)

```
event bars → StandardScaler → reduce(correlation|infogain) → XGBoost(depth 3, lr .03, n 200)
           → isotonic calibrate (embargoed VAL) → de Prado CDF bet × causal overlay
```

- **Causality:** bar thresholds fit on TRAIN only, extrapolated identically OOS; labels may look ahead (they're targets) but never beyond `_EMBARGO = max(200, horizons, declared reach)`; full-sequence smoothers (Viterbi/backward passes) must **block-decode at the OOS boundary**; features past-only; sizing identical in backtest and live.
- **The OOS backtest is online, leak-free, ObjectStore-replay-only** (`BACKTEST_AUDIT.md`, re-audited 2026-06-10: 28/28 axes byte-invariant under post-TRAIN randomization). `infer.py` holds no model. Two proofs, run them: `verify.py` (bars ≤1e-9), `infer_online.py` (preds ≤1e-6).
- **Leaks are found by re-running, not reading.** The logdollar threshold leak survived a 13-agent review; the "external data wall" was an `on_data` ordering bug; the vacuous permute gate survived months. After any `bar_builder.py`/`bar_ext.py` change run `tests/test_bar_threshold_leak.py`; when a feed reports "no data", probe the channel directly (`set_runtime_statistic`, not `debug`).
- **QC limits:** 64,000 UTF-8 **bytes**/file (comments are minified away — only code counts). Extension files each get their own budget: new axes → `bar_ext.py`, features/reduces → `ml_ext.py`, the sizing engine lives in `sizing_ext.py` (moved 2026-06-10, bit-exact-verified; main.py now has ~4.4k headroom). Lint-clean QC files: no `getattr`, no nested-quote f-strings.
- **Code layout (2026-06-11 restructure, PR #1):** an installable `src/lb/` package (`pip install -e .`, relocatable — paths derive from `lb.paths.ROOT`, no hardcoded literals). Pipeline modules `lb.modules.*`; render engine `lb.harness.orchestrator`; cell key from the single `lb.harness.psuf.cell_suffix`. Driver `scripts/run_round.py` (`lb-round`); report server `lb-report` (`scripts/app.py` = shim → `lb.console.app`); runners under `scripts/{research,audit,diag}/`. The report service runs as root — re-`pip install -e .` for **both** user and root after any move.

## The honesty stack

Never crown on LLM judgment — only on real QC Calmar, then: deflated Sharpe (`scripts/audit/deflated_audit.py`) · session DSR Holm-Bonferroni (`scripts/audit/honest_audit.py`) · the permute control · decay monitor (`scripts/research/champion_series.py`, early-vs-late Sharpe) · e-values · PBO/CSCV · cost stress (book holds at 5bp) · **drawdown-shape** (Ulcer / Pain / time-under-water / Martin via `lb.metrics`, added 2026-06-11). Standing facts: nothing survives Holm-Bonferroni at the full ~2500-trial burden except GLD (DSR 0.96); MinBTL says only GLD is window-confirmed — every other edge is provisional until the calendar grows. **Weight conviction by DSR and recovery-pain, not raw Calmar** — Calmar sees only the single deepest trough, so it flatters slow-bleeders: on the real book curves DBC sits underwater 62% of the time (Martin 7.1, last) despite a mid-pack Calmar, while USO recovers fastest (Martin 48.8, top). Rank book members by Martin/Pain alongside Calmar².

## The book (re-validated 2026-06-10; **extended-window 2026-06-11**: TEST_END advanced to 06-11 — GLD **2.12** (BH 1.33, edge +0.80), USO **2.69** (BH 0.86, edge +1.83; now the stronger engine). Stored 4.02/3.85 are the old-window numbers; bar recalibration on the grown series restates everything — the phase-cert predicted this: honest GLD ≈ its family median, not its lucky top.)

| Ticker | Config | Calmar | Status |
|---|---|---:|---|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t.40 / infogain | **4.02** | The anchor. Bit-exact ×3, decay-strengthening (5.0→5.9), genuine permute-pass, Hansen-SPA p=.017. |
| **USO** | logdollar / `revert` / cdf_plain / t.45 | **3.85** | 3rd mechanism. Bit-exact, +2.93 over BH, decay-strengthening (4.5→7.7). **Crown proposed: +USO at natural Calmar² weight → book 5.03 / Sharpe 2.74.** |
| UUP | imbalance / `bgm+sadf_explosive+ker` / cdf_overlay | 0.60 | Decayed (was 1.85). **06-11 extended window: timing 0.29 < own BH 0.47 — retire to `always_long`.** |
| IWM | `always_long` | 0.56 | Timing retired (trend_leg 0.48 < BH 0.56). |

Deployed: GLD/UUP/IWM/TIP/DBC/HYG (+USO awaiting crown), weights ∝ **common-grid** Calmar² (never own-window numbers — mixing scales mis-weights), gross ≤ 1. Current book 4.65; +USO → 5.03. Dropping weak names lowers book Calmar — decorrelation pays.

**Three mechanisms, asset-intrinsic** (each labeler fails on the others' assets): trend-momentum (GLD), macro-regime (UUP, decayed), oil mean-reversion (USO — oil-specific; silver/gold/agri/natgas/FX all refuted). Wang's β-lens routes assets (β200≈0.5 symmetric→trend, ≫0.5→buy-hold, <0.45→reversion) and retrodicts the whole book — but it is **clock-dependent** (USO: 0.55 on bar clock, 0.44 on calendar days) and admission-only: β-symmetric XME was predictable-not-profitable (+0.04 over BH).

## Lessons that change decisions

- **Label-relevance ≠ profit-relevance.** High val_auc + no Calmar edge = predictable-not-profitable (sticky-HMM 0.40@auc.88, XME +.04@auc.72). The val_auc gate + permute + deflation jointly catch both failure modes.
- **Drifters (val_auc≈0.5, β200≫0.5) are buy-hold-optimal:** SPY/QQQ/HYG/TIP/DBC/IWM. New methods only help where structure exists (val_auc>0.6).
- **Exogenous features on GLD are closed in every form:** ETF-price proxies (UUP/TIP 4.02→3.18), nominal yield (→3.38), true DFII10 real yield (→3.12, full data, val_auc up Calmar down = the overfit signature).
- **Closed, don't re-grind:** depth-3 XGBoost only · meta-labeling · universe siblings (SLV≠GLD) · uniqueness weights · sticky-HMM · cross-asset price features · reduce on 80-feat panel (infogain won) · FXE/metals reversion.
- **One source of truth for keys/configs** — inline copies of `_PSUF` diverged three times; helpers, not duplication.

## The frontier (2026-06-11)

The 2026-06-10 "every loop-runnable lever raced; reopening needs new data" terminus was **reopened 2026-06-11** by a deep read of two new Wang transcripts (`docs/research/WANG_TRANSCRIPTS_2026-06-11.md`). They mostly **corroborate** the pipeline (info-density clock, `trend_leg`, β200 long-only routing, HMM real-time weakness, linear-reduce, strict OOS) — **no new mechanism** — but surfaced three in-rule, loop-runnable levers needing **no new data**. The read is corroboration-heavy so EV is modest, but all three are unraced and free.

**Picked experiments — run in this order** (single-ticker, past-only, leak-gated, fit the 64k budget; one variable changed per A/B):

1. **Conditional-ES regime sizer** → `sizing_ext.py`. Bucket bars by a TRAIN-fit vol regime; size so per-regime expected-shortfall is constant (vs the raw vol-overlay). A/B vs the champion sizer on **GLD** (`logdollar / trend_leg+regime_gmm / t.40 / infogain`, dd_overlay → cond_es) and **USO** (`logdollar / revert / t.45`, cdf_plain → cond_es). Most novel — a regime-aware sizer can lift Calmar/Martin by trimming bets in dangerous regimes without touching the signal. After the build, run `tests/test_bar_threshold_leak.py` + render-smoke before racing.
2. **Joint-KDE quadrant regime as a categorical FEATURE** → `ml_ext.py` / `features.py`. TRAIN-fit Gaussian KDE on (return, vol); emit the discretised cell id (+ the conditional return-quantile given the vol bucket) as past-only features; race under `reduce=infogain` on GLD. Distinct from the parametric `regime_gmm` gate — feed the state as a *feature*, not a hard gate (Wang: hand-wired rules are brittle).
3. **Rank (Spearman) correlation reduce** → `ml_ext.py`. Wang: "Pearson assumes normality; use rank-based." Cheap A/B vs `infogain`. Low EV (infogain already won the 80-feat panel) but unbuilt.

**Still open, not loop-runnable (needs the user):** a new authorized data modality — options IV / COT / flows. IV would also unlock the *forward* conditioning variable a regime needs (Wang: historical vol is backward, so a vol-cell is contemporaneous-only); without it the joint-dist work stays past-only/structural.

**Already raced to a verdict (don't re-grind):** de-scaled axis (1.79<3.85) · multi-axis netting (3.57<4.02) · phase cert (PASS, GLD honest ≈3.2) · SPY session momentum (auc .556 but 0.51<BH) · rich-features×compressor (2.97/2.14<4.02) · plus the standing closed list (fracdiff crowd-out, HMM family, long-short on up-drifters, cross-asset price/copula, meta-labeling, uniqueness weights).

Autonomous — decide and run the next experiment, don't ask.
