# autoresearch

An autonomous RSI loop that hunts single-ticker trading edges on ETFs. Each round the researcher — an LLM (Claude) — reads this manual + the provenance graph (`knowledge.json`) + past findings, designs ONE full recipe for the weakest ticker, races it A/B vs the champion on real out-of-sample (OOS) Calmar in QuantConnect (project 31338454), and keeps it only if every honesty gate passes. Findings update the graph, which seeds the next hypothesis, and confirmed new methods join the library — the loop self-improves its own direction and tooling. One ticker at a time, no cross-ticker ensembling, simple over complex.

It is an LLM pointed at a number it can game, so **most of this file is gates, not method.** A trading edge fails silently: a backtest will happily report a great Calmar that is really a leak, a multiple-testing fluke, or plain buy-hold drift. Telling a real edge from those three impostors is the entire job.

## The loop

1. **Pick the weakest ticker.** Lowest re-validated OOS Calmar in the book (a name failing the trade-count gate sorts weaker). Stored Calmars go stale as the OOS window grows — re-validate first.
2. **Design the whole recipe.** From the graph + past findings, co-design `axis × label × features × reduce × model × sizing`. Don't swap one knob; reach for a new axis or labeler first — every confirmed edge came from there — but a win can land anywhere (infogain lifted GLD 3.47→4.02).
3. **Race A/B vs champion.** `scripts/run_autoresearch_round.py '<A>' '<B>'`. A hypothesis is `{ticker, axis, labeler, thresh, sizing}` + optional `{reduce, n_components, rebal_band, max_depth, features, horizons, permute_labels}`; labelers can ensemble (`"bgm+ker"`). Both legs train+infer on the 2 QC nodes in parallel; the script writes results/report/log, never commits.
4. **KEEP iff every gate passes**, else DISCARD — a human/Opus commits. Log it either way: a DISCARD raises every co-tested name's deflation bar, and the graph is the audit trail.

### KEEP gates (all must hold)

| Gate | Threshold |
|---|---|
| Deployable | trades > 80, train+infer both completed, directional accuracy present |
| Beats champion | winner Calmar > re-validated previous best |
| Positive | Calmar > 0 |
| Learnable structure | val_auc > 0.52 (below = coin-flip model, a window artifact) |
| Survives deflation | winner Calmar > expected-max of N noise trials (Bailey-López de Prado best-of-N; `always_long` and <3-trial cases exempt) |
| PSR significance | Bonferroni-deflated probabilistic Sharpe `psr > 1 − 0.05/N_trials` |
| Permuted-label control | the decisive real-vs-artifact test, below |

### Permuted-label control — the decisive falsifier

Re-run a gate-passing winner with `"permute_labels": true` — shuffles **only the TRAIN labels** (leak-safe, separate `_perm` cell). A real edge collapses toward buy-hold when its labels are scrambled; a survivor was drift, sizing, or leak — not signal. Pass iff real edge over buy-hold > 0.15 AND permuted edge < 40% of real (no baseline: permuted Calmar < 60% of real). Caught SPY/SLV/QQQ fakes; UUP collapses 1.30 → −0.08. Orthogonal to the deflation haircuts — keep both.

## The pipeline (Wang backbone, fixed, `random_state=42`)

Resample off the clock → label unsupervised → many features then reduce → model → bet-size:

```
bars → StandardScaler → reduce_dims(correlation | infogain) → XGBoost(depth 3, lr .03, n 200) → isotonic calibrate (embargoed VAL) → de Prado CDF bet × causal inverse-vol overlay
```

Causality: bar thresholds fit on TRAIN only and extrapolated OOS-invariantly; the label may look ahead (it's the target); the model sees past-only features; sizing is identical in backtest and live. Detectors are trend-scan / change-point / clustering — **never HMM** (can't run online). Aim Calmar > 3, reproducible, deployable.

- **Reduce.** Default `correlation` (drop pairs >0.90, then top-K by variance) | `infogain` (Wang's MI selection, top-K by MI with the TRAIN label); byte-identical when infogain is off.
- **Sizing.** De Prado CDF bet (0 at p≤thresh, rising to 1) × a causal inverse-vol overlay that de-leverages on vol spikes; position = CDF(p) × overlay, per bar.
- **Axes / labels.** Champion axes `logdollar` (GLD/IWM trend), `imbalance` (UUP regime); labels `trend_leg`/`ker` (trend-momentum), `bgm`/`regime_gmm`/`sadf_explosive` (regime), `revert` (oil mean-reversion). Full registry — 21 axes, 41 labelers, all sizers — in `modules/bar_builder.py`, `modules/labeler.py`, the infer templates; everything else is built + leak-safe but dormant. Axis + label levers are CLOSED (kyle/run/spectral/vpin/new-axes all lose to the asset-intrinsic champions).
- **Multi-file render.** `bar_builder.py` is a separate QC file imported by main.py, so the 64k char/file limit is per file and 3-way ensembles fit. Keep QC files lint-clean (no `getattr`, no nested-quote f-strings).

## Backtest + leak contract (non-negotiable)

The OOS backtest is **online, leak-free, model-only-from-ObjectStore** (audited clean: `BACKTEST_AUDIT.md`).

- `infer.py` holds no model — pure replay of saved predictions + the causal `_size()`; test data enters only through prediction, never a fit.
- Every `.fit` lives in `footer.py`, on TRAIN (+ embargoed VAL) only; features past-only; thresholds TRAIN-only; `_EMBARGO = max(200, max(horizons))` covers the full forward-label horizon.
- Ensembles deploy live: footer saves a multi-member bundle; `live_trade.py` averages the calibrated+gated member probs online, warm-started from history.
- **Two proofs, run them.** `verify.py`: online-rebuilt bars match batch ≤1e-9. `infer_online.py`: live preds match saved ≤1e-6. **Don't trust a champion until its `infer_online` reports preds_match=1.**

**Leak detection is empirical, not by audit.** The logdollar/kyle leak — bar thresholds scaled by `int(np.sum(valid))`, a full-series count that includes OOS — inflated crowns (GLD 4.71→2.76, SOXX 3.02→0.81) and survived a 13-agent review; only re-running with the fix revealed it. `tests/test_bar_threshold_leak.py` (numpy-free, CI-runnable) forbids that signature and asserts every threshold scaling is TRAIN-masked or OOS-invariant. **Run it after any `bar_builder.py` change**, and verify new methods by append-OOS-invariance, not by reading code.

## The honesty stack

Seven lenses in three roles. **Never crown on LLM judgment — only on real QC Calmar.**

- **Per-round gates** — deflated Sharpe (`deflated_audit.py`, beat the best-of-N noise floor; every trial raises every co-tested name's bar) · DSR (`honest_audit.py`, session-wide at the true trial count, Holm-Bonferroni + Benjamini-Hochberg, ≥0.95 = real) · the **permuted-label control (decisive)**.
- **Standing monitors** (watch the book) — decay (`champion_series.py`, early- vs late-half OOS Sharpe + Page-Hinkley/CUSUM; caught UUP front-loaded 2.67→0.74 while GLD/IWM strengthen) · e-value (`evalue_oos`, native in infer.py — anytime-valid, peeking-robust liveness gate, mean>0 not significance; supersedes p-value/DSR re-peeks).
- **Post-crown checks** — PBO via CSCV (`pbo_gld.py`, prob the IS-best config is OOS-below-median; GLD 0.581 on Sharpe, resolved by ablating on Calmar) · cost-stress + Harvey-Liu haircut (`cost_stress.py`, `harvey_liu_haircut.py`, an independent multiple-testing haircut cross-checking DSR; book holds at 5bp).

Standing facts: nothing survives Holm-Bonferroni across the full ~500-trial session burden; weight conviction by DSR, not raw Calmar; haircuts are necessary but not sufficient — permute + replication catch path/fund-specificity they miss.

## The confirmed book

Durable single-ticker alpha is scarce and asset-intrinsic. Leak-free, permute-confirmed. Conviction **GLD > UUP > IWM**. (Stored Calmars go stale as OOS grows — re-validate before trusting.)

| Ticker | Config | Calmar | Notes |
|---|---|---:|---|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t0.40 n15 band0.03 / reduce=infogain | **4.02** | The one durable edge. Decay-HEALTHY, bit-exact 3× (4.0218), gold-specific (not SLV). ~2.0 buy-hold + ~1.16 timing alpha; regime_gmm+dd_overlay add +22% Calmar (ablation-confirmed). |
| **UUP** | imbalance / `bgm+sadf_explosive+ker` / cdf_overlay | **1.85** | Permute-real but decay-STALE (alpha front-loaded 2014–15), Bonferroni-boundary (N=72); bgm carries it. Earns its seat by decorrelation. |
| **IWM** | logdollar / `trend_leg` / cdf_overlay / reduce=infogain | **0.665** | Beats buy-hold (0.55), permute-PASS, decay-HEALTHY. Fails strict deflation (DSR 0.845, N=64). |

Deployed: **GLD / UUP / IWM / TIP / DBC / HYG**, Calmar²-weighted (gross ≤ 1). TIP/DBC/HYG are `always_long` buy-hold diversifiers — no timing edge, they earn seats by decorrelation. Weekly grid Calmar 4.62 / MaxDD 2.46% / Sharpe 2.46; net ~3.4 @5bp (GLD the cost driver at 602 orders), ~2.8 @10bp — the buy-hold core barely trades. Dropping UUP or IWM lowers book Calmar — keep the weak names. SOXX dropped (its edge was the bar-threshold leak; leak-free 0.81 ≈ buy-hold, IWM took the seat).

## Governing lessons

The few rules that actually change decisions:

- **Edges are asset-intrinsic — three mechanisms.** TREND-MOMENTUM (`trend_leg` > `ker`) on trend-predictable-drawdown names where trim-cost < MaxDD-saved — GLD/IWM on the logdollar clock. REGIME (`bgm`) on macro oscillation — UUP on the imbalance clock. MEAN-REVERSION (`revert`) on oil — USO/UCO/XOP, fully validated (permute/decay/cost/DSR 0.94 ≈ GLD); USO(1x) proposed to join the book (+12% Calmar/Sharpe), awaiting crown. Oil-specific: `revert` fails on every other commodity (silver/gold drift, agriculture/natgas structureless). Each mechanism's labeler fails on the others' assets; swapping a mechanism's implementation always loses. Which mechanism + axis wins is a property of the asset, not a choice.
- **Label-relevance ≠ profit-relevance.** High val_auc + low Calmar = predictable-not-profitable (jump_model 0.82/0.34, sliced_wasserstein); the mirror, high Calmar + low val_auc = drift/path-luck. The val_auc>0.52 gate + permute + deflation jointly catch both. Infogain amplifies a real edge, never manufactures one.
- **Clean drifters (val_auc≈0.5) are buy-hold-optimal** — nothing to cut, so timing just sacrifices carry (HYG/QQQ/SPY/TIP/DBC). New methods help only where val_auc>0.6; vol products are uncrackable without external data (short-carry has a fat crash tail; long-timing val_auc<0.5).
- **Don't grind deflation-dead names** — on static data, permuting knobs on a structureless name only raises every co-tested name's deflation bar for ~0 EV. A genuinely new axis/label on the weakest name is allowed; permuting knobs is not.
- **Closed levers, don't re-grind:** depth 3 (5 overfits) · XGBoost only (sklearn blocked in QC) · meta-labeling (< ker) · universe siblings (SLV≠GLD, SMH≠SOXX — edges idiosyncratic) · sample-uniqueness weights (hurt directional edges; overlap is the signal).

## The universe screen (current campaign)

The fixed-12 space saturated, so the search widened (user, 2026-06-04: *"explore the 311 ETFs, find which fit; don't forget buy-and-hold"*) from grinding 0-EV permutations to finding names that **have** structure. For every QC-confirmed pre-2009 ETF (~311), race the GLD-trend methodology vs `always_long` through the leak-safe driver, so each ETF gets its own buy-hold baseline. **A FIT = real OOS Calmar beats buy-hold AND val_auc > 0.55 AND deployable (>80 trades)** — matching buy-hold is NO-FIT (timing gold is worthless if holding gold already pays the same). Then **deep-sweep** the fit classes: all 21 axes × 27 labelers per ETF. Still pure single-ticker — each ETF screened independently; the universe just widened from 12 to 311.

8 STRONG fits so far — **SSO, IAU, USO, AGQ, GDX, DJP, GSG, UCO** — concentrated in commodity (gold/oil/silver) and leveraged-equity classes; most other names are NO-FIT (buy-hold wins). Live progress in `status.json`. Screen-strong ≠ book seat: a fit must still clear the full per-round gates + permute + decay. `screen_etfs.py` and `deep_sweep_etfs.py` are the single resumable coordinators (one driver call at a time) — don't launch a competing experiment while they run.

## Frontier + state

Autonomous — decide and run the next experiment, don't ask. The fixed-12 frontier is converged (every forward readout built, axis/label levers closed, the full module sweep all DISCARD, GLD reproducing 4.0218 bit-exact); the deployed book is healthy (all members holding, no decay); the screen + deep-sweep is the live candidate pipeline, and the per-round A/B driver is its engine. Static OOS = no new info per tick, so config grinding only inflates the deflation bar — re-opening *further* needs a genuinely new INPUT:

1. **New data modality** (highest leverage) — options IV/skew, positioning (COT, ETF flows), macro surprises, credit spreads, VIX term structure. New information, not new tuning. Needs the user.
2. **Cross-asset pairs** — trend on a constructed spread A−βB gated on positive autocorr. Offered 2026-06-04 and **declined** — stay single-ticker.
3. **Intraday holding** — the `horizons` lever is built but found no edge (fails Bonferroni); mostly unsearched, a different regime.
4. **Regime change / decay** — the only thing that moves on current inputs is the OOS window growing; `evalue_oos` is the standing monitor.
