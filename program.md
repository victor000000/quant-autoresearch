# autoresearch

An autonomous loop that hunts single-ticker trading edges on ETFs. Each round: read this file + `knowledge.json`, design ONE recipe, race it A/B against the champion on real out-of-sample Calmar in QuantConnect (project 31338454), keep iff every honesty gate passes. Single ticker, no cross-ticker ensembling, simple over complex.

This is an LLM pointed at a number it can game, so **most of this file is gates, not method**. A backtest happily reports a great Calmar that is a leak, a multiple-testing fluke, or plain drift. Telling a real edge from those three impostors is the whole job.

## The loop

1. **Re-validate, then target.** Stored Calmars go stale as the OOS window grows. Re-race champion-vs-`always_long` before trusting any stored number.
2. **Design one recipe** — `axis × label × features × reduce × model × sizing`. Every confirmed edge came from a NEW mechanism, never knob-tuning; knob-grinding a structureless name only raises everyone's deflation bar.
3. **Race** — `scripts/run_round.py '<A>' '<B>'`. A hypothesis is `{ticker, axis, labeler, thresh, sizing}` + optional `{reduce, n_components, features, model, calibration, horizons, permute_labels}`; labelers ensemble with `+`.
4. **Keep iff ALL gates pass; log either way.** A human/Opus commits and crowns. Save each round exactly 3 files: `round_results.csv` + `knowledge.json` + `ETF_SCREEN_TABLE.md`.

### Keep gates — all must hold

| Gate | Threshold |
|---|---|
| Deployable | trades > 80, train+infer completed, DA present |
| Beats champion | winner Calmar > re-validated previous best |
| Learnable structure | val_auc > 0.52 |
| Survives deflation | > expected-max of N noise trials (`always_long` exempt) |
| PSR significance | Bonferroni-deflated `psr > 1 − 0.05/N_trials` |
| Permuted-label control | real edge over buy-hold > 0.15 AND permuted < 40% of real |

**The permute control is the decisive test** (shuffle TRAIN labels only, separate `_perm` cell; GLD: 4.70→0.76). It must FAIL LOUD: a 0-trade control leg refuses the KEEP. Cell keys come only from `lb.harness.psuf.cell_suffix` — collisions made gates vacuous three times; `tests/test_psuf_tokens.py` guards all knob combinations. When reading old backtests, derive champion-vs-permute legs from the LEDGER row (ledger time = QC+8h), never assume A/B orientation.

## Pipeline + leak contract (fixed, `random_state=42`)

```
event bars → StandardScaler → reduce → model(depth 3, lr .03, n 200) → calibrate (embargoed VAL)
           → de Prado CDF bet × causal overlay
```

- **Causality:** bar thresholds TRAIN-fit, extrapolated identically OOS; labels may look ahead but never beyond `_EMBARGO`; smoothers block-decode at the OOS boundary; features past-only; sizing identical in backtest and live.
- **OOS backtest is online, leak-free, ObjectStore-replay-only** (`BACKTEST_AUDIT.md`). Proofs: `verify.py` (bars ≤1e-9), `infer_online.py` (preds ≤1e-6). **Leaks are found by re-running, not reading** — after bar-builder changes run `tests/test_bar_threshold_leak.py`; probe silent feeds via `set_runtime_statistic`.
- **QC limits:** 64,000 UTF-8 bytes/file (code only; `tests/test_render_budget.py` guards champion renders). New axes → `bar_ext.py`, features/reduces/models/calibration → `ml_ext.py`, sizing → `sizing_ext.py`. Lint-clean QC files: no `getattr`, no nested-quote f-strings. **Lean CANNOT catch lookup errors** (AttributeError/NameError escape try/except — proven twice): guard with `hasattr`. **QC sandbox has no internet** (pretrained weights unloadable without ObjectStore side-load).
- **Code layout:** installable `src/lb/` package — pipeline `lb.modules.*`, render `lb.harness.orchestrator`, keys `lb.harness.psuf`, metrics `lb.metrics`; drivers `scripts/run_round.py` + `scripts/research/*`; honesty `scripts/audit/*` + `tests/`.

## The honesty stack

Never crown on LLM judgment — only on real QC Calmar, then: deflated Sharpe · session DSR Holm-Bonferroni · permute control · decay monitor · e-values · PBO/CSCV · cost stress (book holds at 5bp) · drawdown-shape (Ulcer/Pain/Martin, `lb.metrics`) · regime-split (`audit/regime_split.py` — both past-only trailing-vol halves) · deflated Calmar (`audit/deflated_calmar.py` — MC max-of-N bootstrap null; the IAAFT variant is structurally weak for Calmar, bootstrap is binding). Standing facts: only GLD survives the full Holm-Bonferroni burden; every other edge is provisional (MinBTL). **Weight conviction by DSR and recovery-pain (Martin), not raw Calmar** — Calmar flatters slow-bleeders.

## The book (common-grid 2026-06-12; engines + candidate all online-proven `preds_match=1`)

| Ticker | Config | Calmar | Status |
|---|---|---:|---|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t.40 / n16 vae **lgbm_bag** | **4.70** | Crown: 5-seed-bagged lgbm > lgbm 4.36 > xgb-vae 3.95 > pca 3.84; permute-confirmed, bit-exact, Bonferroni PASS, regime-split PASS, deflated-Calmar p=0.003, live-deployable (diff 0.0). lgbm family is GLD-specific. |
| **USO** | logdollar / `revert` / cdf_plain / t.45 / correlation | **2.82** | Oil mean-reversion engine. Permute-confirmed, decay-strengthening, regime-split PASS, deflated-Calmar p=0.043, online-proven (3e-08). |
| DIA | imbalance / `bgm+ker` / cdf_overlay / t.50 | 1.66 | Screen find, fully gated + hardened (regime-split PASS, deflated p=0.018, online-proven 8e-09). **The candidate seat.** |
| HYG/TIP/DBC/IWM/UUP | `always_long` (×cdf) | 1.97/0.90/0.69/0.50/0.37 | Decorrelation seats; timing retired on all. |
| IJR | imbalance / `bgm+ker` / cdf_overlay / t.50 | 2.51 | ⚠️ AUDITS-PASS, ONLINE-FAIL 2026-06-12: +2.09 edge, permute total-collapse, regime-split 3.03/1.92, deflated p=0.003 — AND it cross-fund-replicates the DIA mechanism. But preds_match=0 (diff 0.136 on 21% of bars; DIA same recipe = 8e-09) — NOT deployable until the online feature divergence is root-caused. |
| DXJ | logdollar / `ker` / cdf_overlay / t.45 | 1.85 | NEW 2026-06-12: sole survivor of Tier-3 gating (EWT/EWL refuted as stale). Permute 90% collapse, regime-split 2.95/2.94 (most uniform edge), deflated p=0.037, decay strengthening 2.11→3.62, online-proven (preds_match=1, 1.5e-08). Japan decorrelation candidate — seat = human decision. |
| FEZ | sadf_explosive | 1.34 | DEMOTED: fails regime-split (low-vol Sharpe −0.72) + deflated-Calmar (p=0.133). Watchlist. |
| PRFZ | sadf_explosive | 2.22 | RISKY: sibling replication failed (PRF/IJR). Probation only. |

**Proposal A FINAL (Calmar² weights, awaiting human decision):** GLD 57.7 / USO 20.8 / HYG 10.1 / DIA 7.1 / TIP 2.1 / DBC 1.2 / IWM 0.6 / UUP 0.4. Candidate overlap (champion curves): corr 0.27–0.51 — partial decorrelation, not full. Weights always common-grid, gross ≤ 1; dropping weak names lowers book Calmar.

**Two replication-confirmed engines, asset-intrinsic:** gold-trend (GLD→IAU) and oil-reversion (USO→UCO). The `sadf_explosive` family shows on 3 funds but monetizes robustly nowhere yet. Wang's β200 lens routes assets (≈0.5→trend, ≫0.5→buy-hold, <0.45→reversion); clock-dependent, admission-only.

## Laws (every one bought with races)

- **Predictability ≠ profit.** Indices carry auc 0.94–0.97 reversion that never monetizes; calendar features set auc records (SPY 0.64) with zero conversion; on GLD, pls (auc 0.756) and ddstate (0.786) beat the champion's auc and lose on Calmar. Val_auc admits, only Calmar crowns.
- **Equity-index timing (QQQ/SPY/IWM) closed at every door** — monotonic action bracket (the less you act, the better), VIX/VXX, crash sizers, no-forfeit lean, calendar. Their optimal config IS `always_long`. Reopen only with new data (options IV).
- **Mechanism-paired components.** Trend/bullion: unsupervised projection + lgbm. Reversion/oil: raw selection + xgb (any projection craters it; bagging hurts too). Component choice is asset physics; nothing transfers without a replication race.
- **Panel additions dilute.** Every feature block beyond base 80 lost on the engines under every reducer.
- **Drifters (val_auc≈0.5) are buy-hold-optimal**; new methods only help where structure exists (val_auc>0.6). `trend_leg+regime_gmm` degenerates on up-drifters — use `tertile` there.

## Closed — don't re-grind

Depth-3 capacity · meta-labeling · uniqueness weights · sticky-HMM · cross-asset/exogenous features in every form (auc up, Calmar down = the overfit signature) · VIXY carry · universe siblings (SLV/GDX/XME) · oil-reversion beyond oil · the FULL innovation backlog (14/14 raced 2026-06-12; only wins: lgbm + lgbm_bag, GLD-only) · Wang-transcript levers · 311 screen COMPLETE + **Tier-3 fully swept 2026-06-12** (all 13 ungated high-edge names permute+re-raced: 10 refuted as stale-window mirages — EWT/EWL/IDV/AAXJ/VT/EWG/IXG/EWY/ILF + EPP overlay-artifact; EWZ/UGL permute-pass but regime-fragile watchlist; **DXJ the sole verified discovery**) · options-IV channel (SPY auc 0.615, no conversion — harvester `precompute_iv.py` + `features='iv'` retained) · Chronos (sandbox has no internet). Retained-idle primitives: sizers in `sizing_ext`, reduces in `ml_ext`, feature blocks in `build_feats`/`ml_ext`, axes permclock/mpnov in `bar_ext`.

## The frontier

Research exhausted within current data. **Human decisions pending:** adopt proposal A · PRFZ probation · PR #1 (superseded — close it and merge the research branch instead) · new data modality (options IV / COT / flows — the only true reopener).

Autonomous — decide and run the next experiment, don't ask.
