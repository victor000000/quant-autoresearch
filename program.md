# autoresearch

This is an autonomous research loop that hunts for single-ticker trading edges on ETFs, using QuantConnect to backtest (project 31338454) and Wang's ML pipeline as the backbone. Each round it picks the weakest ticker, designs a config for it, races that config against the current champion on real out-of-sample (OOS) Calmar, and keeps the new config only if it clears every honesty gate. One ticker at a time, no cross-ticker ensembling, simple over complex. This file is the whole operating manual.

## The loop

One round:

1. **Pick the weakest ticker** — lowest real OOS Calmar across the book (ties broken so a name that fails the trade-count gate sorts weaker). Re-validate its stored best first; records go stale as the OOS window grows.
2. **Design a config for it** — read the provenance graph (`knowledge.json`) and past findings, then co-design the whole recipe `axis × label × features × reduce × model × sizing` for that ticker. Don't just swap one knob. Reach for a new bar axis or labeler first — that's where every confirmed edge has come from — but a win can land in any slot (the infogain reducer lifted GLD 3.47→4.02).
3. **Race A/B vs champion** — `scripts/run_autoresearch_round.py '<A>' '<B>'`. A hypothesis is a dict `{ticker, axis, labeler, thresh, sizing}` plus optional `{reduce, n_components, rebal_band, max_depth, features, horizons, permute_labels}`. Labelers can be ensembles (`"bgm+ker"`). Both legs train + infer on the 2 QC nodes in parallel. The script writes results, updates the report, and logs the round.
4. **KEEP iff every gate passes**, else DISCARD. (The script does not commit; a human/Opus does.)

### The KEEP gates (all must hold)

| Gate | Threshold |
|---|---|
| Deployable | trades > 80, train+infer both completed, directional accuracy present |
| Beats champion | winner Calmar > re-validated previous best |
| Positive | Calmar > 0 |
| Learnable structure | val_auc > 0.52 (below = coin-flip model, a window artifact) |
| Survives deflation | winner Calmar > expected-max of N noise trials (Bailey-López de Prado best-of-N floor; `always_long` and <3-trial cases exempt) |
| PSR significance | Bonferroni-deflated probabilistic Sharpe `psr > 1 − 0.05/N_trials` |
| Permuted-label control | see below — the decisive real-vs-artifact test |

### Permuted-label control

If a winner clears the gates above, re-run it with `"permute_labels": true`. This shuffles **only the TRAIN labels** (leak-safe, writes a separate `_perm` cell). A real edge collapses toward buy-hold when its labels are scrambled; an "edge" that survives the shuffle was drift/sizing/leak, not signal.

Pass iff: real edge over buy-hold > 0.15 AND permuted edge < 40% of real edge. Fallback when no buy-hold baseline exists: permuted Calmar < 60% of real Calmar.

This caught SPY/SLV/QQQ fakes; UUP collapses 1.30 → −0.08 (pure label alpha). It is orthogonal to the deflation haircuts — keep both.

## The pipeline (Wang backbone)

Resample off the clock → label unsupervised → rich features then reduce → model → bet-size. The downstream is fixed and seeded (`random_state=42`):

```
bars → StandardScaler → reduce_dims(correlation default | infogain) → XGBoost(depth 3, lr .03, n 200) → isotonic calibrate (embargoed VAL)
```

Causality contract: bar thresholds are fit on TRAIN only and extrapolated OOS-invariantly; the label may look ahead (it's the target); the supervised model sees past-only features. Detectors are trend-scan / change-point / clustering — never HMM (can't run online). Aim Calmar > 3, reproducible, deployable.

- **Reduce.** Default `correlation` (drop high-corr pairs >0.90, then top-K by variance). `infogain` = Wang's mutual-information selection (top-K by MI with the TRAIN label). The two are byte-identical when infogain is off.
- **Sizing.** De Prado CDF bet (0 at p≤thresh, rising to 1) times a causal inverse-vol overlay that de-leverages on vol spikes. Final position = CDF(p) × overlay, per bar, identical in backtest and live.
- **Axes / labels.** Champion axes: `logdollar` (tail-compressed info clock — GLD/IWM trend) and `imbalance` (signed-dollar order-flow runs — UUP regime). Champion labels: `trend_leg` and `ker` (trend-momentum), `bgm` / `regime_gmm` / `sadf_explosive` (regime). The full registry — 21 axes, 41 labelers, all sizers — lives in `modules/bar_builder.py`, `modules/labeler.py`, and the infer templates. Everything not named here is built and leak-safe but dormant or lost; the axis and label levers are CLOSED (kyle/run/spectral/vpin/new-axes all lose to the asset-intrinsic champions).
- **Multi-file render.** `bar_builder.py` is a separate QC file imported by main.py, so the 64k char-per-file limit is per file and 3-way ensembles fit. Keep these files QC-lint-clean (no `getattr`, no nested-quote f-strings).

## Backtest contract + leak safety (non-negotiable)

The real OOS backtest is **online, leak-free, and model-only-from-ObjectStore**. Audited clean (`BACKTEST_AUDIT.md`).

- `infer.py` holds NO model — it's pure replay of saved predictions plus the causal `_size()`. Test data enters only through prediction, never a fit.
- Every `.fit` lives in `footer.py`, on TRAIN (+ embargoed VAL) only. Features are past-only; thresholds are TRAIN-only; embargo covers the full forward-label horizon (`_EMBARGO = max(200, max(horizons))`).
- Ensembles deploy live: footer saves a multi-member bundle, `live_trade.py` averages the calibrated+gated member probs online, warm-started from history.

Two proofs, run them: `verify.py` shows online-rebuilt bars match batch bars to ≤1e-9; `infer_online.py` shows live preds match saved preds to ≤1e-6. **Don't trust a champion until its `infer_online` reports preds_match=1.**

**Leak detection is empirical, not by audit.** The logdollar/kyle leak — bar thresholds scaled by `int(np.sum(valid))`, a full-series count that includes OOS — inflated crowns (GLD 4.71→2.76, SOXX 3.02→0.81) and was missed by a 13-agent code review and by this loop's own agents. Only re-running with the fix revealed it. The fix extrapolates TRAIN valid-density to full length. `tests/test_bar_threshold_leak.py` (numpy-free, CI-runnable) now forbids the `np.sum(valid)` signature and asserts every threshold scaling is TRAIN-masked or OOS-invariant. **Run it after any `bar_builder.py` change**, and verify every new method empirically (append-OOS-invariance), not by reading the code.

## The honesty stack

Seven lenses. A KEEP must pass the per-round gates; the standing book is watched by the monitors.

1. **Deflated Sharpe** (`deflated_audit.py`) — per-round gate. The max of N trials is upward-biased; an edge must beat the best-of-N noise floor. Each new trial raises every co-tested name's bar.
2. **Honest audit / DSR** (`honest_audit.py`) — per-round gate, session-wide at the true trial count, with Holm-Bonferroni and Benjamini-Hochberg. DSR ≥ 0.95 = real.
3. **Permuted-label control** — per-round gate, **the decisive one**. Real edge collapses under label-shuffle; survival = artifact. The only test that falsifies spurious correlation.
4. **Decay monitor** (`champion_series.py`) — standing monitor. Early- vs late-half OOS Sharpe + Page-Hinkley/CUSUM on the real equity curve. Caught UUP as front-loaded (2.67→0.74 STALE) while GLD/IWM strengthen.
5. **E-value monitor** (`evalue_oos`, native in infer.py) — standing monitor, **the ongoing gate**. Anytime-valid, peeking-robust, re-validations multiply in. A liveness/decay test (mean>0), not a significance test. Supersedes p-value/DSR re-peeks for continuous monitoring.
6. **PBO via CSCV** (`pbo_gld.py`) — post-crown deep-dive. Probability the IS-best config is OOS-below-median. GLD PBO 0.581 on Sharpe was resolved by ablating on Calmar.
7. **Cost stress + Harvey-Liu haircut** (`cost_stress.py`, `harvey_liu_haircut.py`) — post-crown / cross-check. Explicit slippage (book holds at 5bp) and an independent multiple-testing haircut that cross-checks DSR.

Standing facts: nothing survives Holm-Bonferroni across the full ~500-trial session burden; weight conviction by DSR, not raw Calmar; haircuts are necessary but not sufficient — permute + replication catch fund/path-specificity they miss; never crown on LLM judgment, only on real QC Calmar.

## The confirmed book

Durable single-ticker alpha is scarce and asset-intrinsic. Leak-free, permute-confirmed. Conviction order: **GLD > UUP > IWM.**

| Ticker | Config | Calmar | Notes |
|---|---|---:|---|
| **GLD** | logdollar / `trend_leg+regime_gmm` / dd_overlay / t0.40 n15 band0.03 / reduce=infogain | **4.02** | The one durable edge. Decay-HEALTHY (1.84→2.30), bit-exact 3× (4.0218), gold-specific (not SLV). ~2.0 buy-hold + ~1.16 real timing alpha; trend core load-bearing, regime_gmm+dd_overlay add +22% Calmar / +9% MaxDD (ablation-confirmed). |
| **UUP** | imbalance / `bgm+sadf_explosive+ker` / cdf_overlay | **1.85** | Permute-real but decay-STALE (2.67→0.74, alpha front-loaded 2014–15), Bonferroni-boundary (N=72). bgm carries it; sadf adds explosive-regime. Earns its seat by decorrelation. |
| **IWM** | logdollar / `trend_leg` / cdf_overlay / reduce=infogain | **0.665** | Beats buy-hold (0.55), permute-PASS, decay-HEALTHY (0.63→1.31). Fails strict deflation (DSR 0.845, N=64). |

Deployed book: **GLD / UUP / IWM / TIP / DBC / HYG**, Calmar²-weighted (gross ≤ 1). TIP/DBC/HYG are `always_long` buy-hold diversifiers — no timing edge, they earn seats by decorrelation.

- Weekly grid: Calmar 4.617 / MaxDD 2.46% / Sharpe 2.46. Daily haircut ≈ 1.15× → ~4.0 Calmar.
- Net of cost: ~3.4 @5bp (GLD 4.02→3.43, the cost driver at 602 orders), ~2.8 @10bp. The buy-hold core barely trades.
- Dropping UUP or IWM both lower book Calmar — keep the weak names for decorrelation.
- **SOXX dropped**: its edge was the bar-threshold leak (leak-free 0.81 ≈ buy-hold). IWM took the seat.

## Governing lessons

The few rules that actually change decisions:

- **Two edge mechanisms, both asset-intrinsic.** TREND-MOMENTUM (`ker`, beaten by `trend_leg`) wins when drawdowns are trend-predictable and trimming-cost < MaxDD-saved — GLD/IWM on the logdollar info clock. REGIME (`bgm`) wins on macro-regime oscillation — UUP on the imbalance order-flow clock. Each mechanism's labeler fails on the other's asset, and swapping a mechanism's implementation always loses. Which one wins, and the axis, is a property of the asset, not a choice.
- **Label-relevance ≠ profit-relevance.** High val_auc + low Calmar is the "predictable-not-profitable" trap (jump_model 0.82 val_auc / 0.34 Calmar; sliced_wasserstein; rich-VR features). The mirror is high Calmar + low val_auc = drift/path-luck. The val_auc>0.52 gate + permute + deflation jointly catch both. Infogain amplifies a real trend edge but cannot manufacture one.
- **Clean drifters are buy-hold-optimal.** A low-drawdown drifter (val_auc≈0.5) has little to cut, so any timing just sacrifices carry — HYG, QQQ, SPY, TIP, DBC. ML finds nothing here.
- **New methods only help where there's structure (val_auc>0.6).** On val_auc≈0.5 names no method beats buy-hold. Vol products are uncrackable without external data (short-carry has a fat crash tail; long-timing val_auc<0.5).
- **Don't grind deflation-dead names.** On static data, manufacturing config permutations on a name with no structure just raises every co-tested name's deflation bar for ~0 EV. Trying a genuinely new axis/label on the weakest name is allowed; permuting knobs is not.

Other closed levers, don't re-grind: model depth (3 optimal, 5 overfits), model family (XGBoost only — sklearn blocked in QC), meta-labeling (weaker than ker), universe siblings (SLV≠GLD, SMH≠SOXX — edges are idiosyncratic), sample-uniqueness weights (hurt directional edges; overlap is the signal).

## The universe screen (current campaign)

The active workflow (user directive, 2026-06-04: *"explore the 311 ETFs, find which fit; don't forget buy-and-hold"*). The fixed-12-ticker space was saturated, so rather than grind 0-EV permutations on names with no structure, the search widened to find names that **do** have structure.

For every QC-confirmed pre-2009 ETF (~311), race the GLD-trend methodology against `always_long` buy-and-hold through the normal leak-safe driver, so each ETF gets its own buy-hold baseline. **A FIT = the method's real OOS Calmar beats buy-hold AND val_auc > 0.55 AND deployable (>80 trades)** — a high Calmar that only matches buy-hold is NO-FIT (timing gold is worthless if simply holding gold already returns the same). Then **deep-sweep** the fit-relevant classes: try all 21 axes × 27 labelers per ETF to find its best config. Still pure single-ticker — each ETF screened independently, no cross-ticker ensembling; the universe just widened from the fixed 12 to all 311.

Results so far (~100/311 screened, 2026-06-05): **8 STRONG fits — SSO, IAU, USO, AGQ, GDX, DJP, GSG, UCO** — concentrated in commodity (gold/oil/silver) and leveraged-equity classes; ~68 NO-FIT (buy-hold wins), the rest marginal or excluded (cash-artifact / degenerate baseline). New fits are screen-strong but must still clear the full per-round gates + permute + decay before they earn a book seat. `screen_etfs.py` and `deep_sweep_etfs.py` are the single coordinators (one driver call at a time, resumable) — don't launch a competing experiment while they run.

## Frontier status

The fixed-12-ticker space is comprehensively explored: every distinct forward readout built, axis and label levers closed, the full module sweep (2026-06-04) all DISCARD, GLD reproducing 4.0218 bit-exact through all churn. Static OOS data means no new information per tick, so config grinding on no-structure names only inflates the deflation bar — which is exactly why the search widened to the universe screen above. Re-opening the frontier *further* needs a genuinely new input:

1. **New data modality** (highest leverage) — options IV/skew, positioning (COT, ETF flows), macro surprises, credit spreads, VIX term structure. New information, not new tuning. Needs the user.
2. **Cross-asset pairs** — trend on a constructed spread A−βB gated on positive autocorrelation. The one place a real relative-value edge can live. Offered 2026-06-04 and **declined** — stay pure single-ticker.
3. **Intraday holding** — the `horizons` lever is built but found no edge (GLD/QQQ/IWM all fail Bonferroni); mostly unsearched, a different regime.
4. **Regime change / decay** — the only thing that moves on current inputs is the OOS window growing. `evalue_oos` is the standing monitor.

## Current state

Autonomous — decide and run the next experiment, don't ask. The loop is **actively running the universe screen + deep-sweep** (above); the per-round A/B driver is its engine. The deployed book (GLD/UUP/IWM + diversifiers) is healthy — all members holding, no decay — and the screen is the candidate pipeline that may add to it. Don't grind 0-EV permutations on no-structure names; do screen for new structure and deep-sweep the names that show it.
