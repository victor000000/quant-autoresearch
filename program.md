# autoresearch

An autonomous loop that hunts single-ticker trading edges on ETFs. Each round the researcher — Claude — reads this file + the provenance graph (`knowledge.json`) + past findings, designs ONE recipe for the weakest ticker, races it A/B against the champion on real out-of-sample (OOS) Calmar in QuantConnect (project 31338454), and keeps it only if every honesty gate passes. One ticker at a time, no cross-ticker ensembling, simple over complex.

It's an LLM pointed at a number it can game, so **most of this file is gates, not method.** A bad edge fails silently: a backtest will happily report a great Calmar that is a leak, a multiple-testing fluke, or plain buy-hold drift. Telling a real edge from those three impostors is the whole job.

## The loop

1. **Pick the weakest ticker** — lowest re-validated OOS Calmar in the book (a name failing the trade-count gate sorts weaker). Stored Calmars go stale as the OOS window grows; re-validate first.
2. **Design the whole recipe** — `axis × label × features × reduce × model × sizing`, reasoning over the graph. Don't swap one knob; reach for a new axis or labeler first (every confirmed edge came from an axis or labeler change, not knob-tuning), but a win can land anywhere — infogain lifted GLD 3.47→4.02.
3. **Race A/B vs champion** — `scripts/run_autoresearch_round.py '<A>' '<B>'`. A hypothesis is `{ticker, axis, labeler, thresh, sizing}` + optional `{reduce, n_components, features, horizons, permute_labels, ...}`; labelers can ensemble (`"bgm+ker"`). Both legs train+infer on the 2 QC nodes in parallel; the script writes results/report/log, never commits.
4. **Keep iff every gate passes**, else discard. Log either way — a discard raises every co-tested name's deflation bar, and the graph is the audit trail. A human/Opus commits.

### Keep gates — all must hold

| Gate | Threshold |
|---|---|
| Deployable | trades > 80, train+infer both completed, directional accuracy present |
| Beats champion | winner Calmar > re-validated previous best |
| Positive | Calmar > 0 |
| Learnable structure | val_auc > 0.52 (below = coin-flip, a window artifact) |
| Survives deflation | winner Calmar > expected-max of N noise trials (Bailey–López de Prado best-of-N; `always_long` and <3-trial cases exempt) |
| PSR significance | Bonferroni-deflated probabilistic Sharpe `psr > 1 − 0.05/N_trials` |
| Permuted-label control | the decisive real-vs-artifact test (below): real edge over buy-hold > 0.15 AND permuted < 40% of real |

### The permuted-label control — the decisive test

Re-run a gate-passing winner with `"permute_labels": true` — shuffles **only the TRAIN labels** (leak-safe, separate `_perm` cell). A real edge collapses toward buy-hold when its labels are scrambled; a survivor was drift, sizing, or leak — not signal. Pass iff real edge over buy-hold > 0.15 AND permuted < 40% of real. Caught SPY/SLV/QQQ fakes; UUP collapses 1.30 → −0.08. Orthogonal to the deflation haircuts — keep both.

## The pipeline (Wang backbone, fixed, `random_state=42`)

```
bars → StandardScaler → reduce(correlation | infogain) → XGBoost(depth 3, lr .03, n 200)
     → isotonic calibrate (embargoed VAL) → de Prado CDF bet × causal inverse-vol overlay
```

- **Causality.** Bar thresholds fit on TRAIN only and extrapolated identically in OOS; the label may look ahead (it's the target); the model sees past-only features; sizing is identical in backtest and live. Detectors are trend-scan / change-point / clustering — **never HMM** (can't run online).
- **Levers are closed.** Only `logdollar` (trend) and `imbalance` (regime) axes ever carry edges — axis choice is asset physics, not tuning. 28 axes (23 deployable) / 54 labelers / all sizers are built + leak-safe; everything else is dormant. `kyle/run/spectral/vpin/dp_oracle/vratio` all lose to the asset-intrinsic champions.
- **Multi-file render.** `bar_builder.py` is a separate QC file imported by `main.py`, so the 64k char/file limit is per file and 3-way ensembles fit. Keep QC files lint-clean (no `getattr`, no nested-quote f-strings).

## Leak contract (non-negotiable)

The OOS backtest is **online, leak-free, model-only-from-ObjectStore** (audited: `BACKTEST_AUDIT.md`).

- `infer.py` holds no model — pure replay of saved predictions + the causal `_size()`. Every `.fit` is on TRAIN (+ embargoed VAL) only; features past-only; `_EMBARGO = max(200, max(horizons))` covers the full forward-label horizon. Ensembles deploy live (footer saves a multi-member bundle; `live_trade.py` averages calibrated+gated member probs online).
- **Two proofs, run them.** `verify.py`: online-rebuilt bars match batch ≤1e-9. `infer_online.py`: live preds match saved ≤1e-6. Don't trust a champion until its `infer_online` reports preds_match=1.
- **Leaks are found by re-running, not by reading code.** The logdollar bar-threshold leak — thresholds scaled by a full-series count that included OOS — inflated GLD 4.71→2.76 (SOXX 3.02→0.81) and survived a 13-agent review; only the re-run with the fix revealed it. `tests/test_bar_threshold_leak.py` forbids that signature — **run it after any `bar_builder.py` change.**

## The honesty stack

Seven lenses, three roles. **Never crown on LLM judgment — only on real QC Calmar.**

- **Per-round gates** — deflated Sharpe (`deflated_audit.py`, beat the best-of-N noise floor) · session-wide DSR (`honest_audit.py`, Holm-Bonferroni + Benjamini-Hochberg, ≥0.95 = real) · the **permuted-label control (decisive)**.
- **Standing monitors** — decay (`champion_series.py`, early- vs late-half OOS Sharpe + Page-Hinkley; caught UUP front-loaded 2.67→0.74 while GLD/IWM strengthen) · e-value (`evalue_oos`, native in `infer.py` — anytime-valid liveness).
- **Post-crown** — PBO via CSCV (`pbo_gld.py`) · cost-stress + Harvey-Liu haircut (`cost_stress.py`; book holds at 5bp).

Standing fact: nothing survives Holm-Bonferroni across the full ~2400-trial session burden (only GLD comes close, DSR 0.96). Weight conviction by DSR, not raw Calmar.

## The book

Durable single-ticker alpha is scarce and asset-intrinsic. Leak-free, permute-confirmed. Conviction **GLD > UUP > IWM**.

| Ticker | Config | Calmar | Notes |
|---|---|---:|---|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t0.40 reduce=infogain | **4.02** | The one durable edge. Decay-healthy, bit-exact (4.0218), gold-specific (not SLV). Hansen-SPA data-snooping-robust (p=0.017). |
| **UUP** | imbalance / `bgm+sadf_explosive+ker` / cdf_overlay | **1.85** | Permute-real but decay-stale (alpha front-loaded 2014–15), Bonferroni-boundary. Earns its seat by decorrelation. |
| **IWM** | logdollar / `trend_leg` / cdf_overlay / reduce=infogain | **0.67** | Beats buy-hold (0.55), permute-pass, decay-healthy. Fails strict deflation. Now mostly a buy-hold diversifier. |

Deployed: **GLD / UUP / IWM / TIP / DBC / HYG**, weight ∝ Calmar², gross ≤ 1. TIP/DBC/HYG are `always_long` diversifiers — no timing edge, they earn seats by decorrelation. Weekly grid Calmar **4.62** / MaxDD **2.46%** / Sharpe **2.46**; net ~3.4 @5bp. Dropping a weak name lowers book Calmar — keep them. **USO oil-reversion (3.85, the 3rd mechanism) is proposed (+USO → 5.16), awaiting crown.**

### Three mechanisms — edges are asset-intrinsic

- **Trend-momentum** (`trend_leg` > `ker`) on trend-predictable-drawdown names where trim-cost < MaxDD-saved — GLD/IWM on the logdollar clock.
- **Regime** (`bgm`) on macro oscillation — UUP on the imbalance clock.
- **Mean-reversion** (`revert`) on oil — USO/UCO/XOP, fully validated (permute/decay/cost/DSR ≈ GLD). Oil-specific; fails on every other commodity.

Each mechanism's labeler fails on the others' assets; swapping a mechanism's implementation always loses. Which mechanism + axis wins is a property of the asset, not a choice.

## Lessons that change decisions

- **Label-relevance ≠ profit-relevance.** High val_auc + low Calmar = predictable-not-profitable; high Calmar + low val_auc = drift/luck. The val_auc>0.52 gate + permute + deflation jointly catch both. Infogain amplifies a real edge, never manufactures one.
- **Clean drifters (val_auc≈0.5) are buy-hold-optimal** — timing just sacrifices carry (HYG/QQQ/SPY/TIP/DBC). New methods help only where val_auc>0.6.
- **Don't grind deflation-dead names.** On static data, permuting knobs on a structureless name only raises every co-tested name's deflation bar for ~0 EV. A genuinely new axis/label is allowed; permuting knobs is not.
- **Closed, don't re-grind:** depth 3 · XGBoost only (sklearn/torch blocked at QC inference) · meta-labeling (< ker) · universe siblings (SLV≠GLD, SMH≠SOXX) · sample-uniqueness weights (overlap IS the signal) · cross-asset ETF-**price** features (R1242).

## The frontier

The **price+volume frontier is mapped and closed**: every axis/labeler/feature/sizer lever is exhausted, three asset-intrinsic mechanisms survive, and the outside literature agrees (single-asset timing edges mostly vanish OOS after a multiple-testing haircut). Static OOS = no new info per tick, so config grinding only inflates the deflation bar. Reopening needs a new INPUT:

1. **The one untested feature channel — exogenous fundamental macro.** NEXT A/B: add the 10y TIPS real yield (FRED `DFII10`, z-score + N-bar Δ) + the 2s10s slope (`DGS10−DGS2`) to GLD's trend model and try to beat Calmar 4.02. Free, QC-native, leak-clean at a 1-day lag; distinct from the closed UUP-price proxy. Gold is a real-rate duration asset — its cleanest upstream driver. Honest prior ~1-in-4 it lifts GLD; else it closes the last feature channel. (`docs/research/DIRECTION_REALYIELD_GLD_2026-06-09.md`)
2. **A new data modality (needs the user)** — options IV/skew, positioning (COT, ETF flows), credit spreads, VIX term-structure. New information, not new tuning.
3. **Honesty tooling (tightens, never creates edge)** — block-bootstrap Calmar CIs, Lo-SE DSR fix, Hansen SPA / Romano-Wolf, online-FDR ledger. Run in parallel.

Autonomous — decide and run the next experiment, don't ask.
