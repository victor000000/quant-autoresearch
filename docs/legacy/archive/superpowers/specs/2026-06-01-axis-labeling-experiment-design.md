# Axis × Labeling Factorial Experiment — Design Spec

**Date:** 2026-06-01
**Scope:** Module ① Custom Axis and Module ② Unsupervised Labeling ONLY. Downstream (features ③, dim-reduce ④, model ⑤, calibration ⑥, ensemble ⑦, consensus ⑧) is HELD FIXED.
**Status:** Clean-slate design. No inherited dead-ends are assumed; prior negatives are re-tested under this controlled factorial, not taken as foreclosed.

---

## 0. Grounding in Wang's primary sources (why this design)

Read and used as the primary source:

- `qa_doc/wang_qa.tex` Q1 (custom-axis kurt / peak_frac / N_bars tradeoffs) and Q2 (label-balance config).
- `uni_yt/uVnOeOcoivw.large-v3.txt` (full pipeline: *"自定义轴…它的信息密度是不均匀的…我们要改变…变成自定义的base line"* — **the axis is the kingpin**; *标签的标注* via unsupervised clustering; *"我们没有办法分辨当下…是在趋势中还是在震荡中…但是我们有办法可以减少震荡"* — the axis reduces oscillation frequency, not regime classification).
- `uni_yt/Ca1n2jgKjrs.large-v3.txt` (entropy → information density; *"商它本身…可以作为特征…也可以作为处理特征的工具"*; sample-entropy preferred over Shannon for continuous series because it needs no distribution assumption; **explicit HMM stance**, see below).
- `uni_yt/tGzbK8c3R_A.large-v3.txt` (volatility axis;重采样自定义轴 strictly beats time axis even for vol forecasting; FI-GARCH long-memory).
- `autoresearch/program.md` and the locked harness (`templates/footer.py.tmpl`, `templates/infer.py.tmpl`, `harness/{constants,evaluator,orchestrator}.py`, `modules/{bar_builder,labeler,features,trainer}.py`).

**HMM is NOT Wang's method.** In `Ca1n2jgKjrs` Wang is explicit: HMM "其实是引状态…大部分讲HMM模型的论文…都是在讲用这个方法如何去交易，我个人是比较讨厌这种说法" — he *dislikes* HMM-as-trading-model. He states HMM is **global-optimum only and cannot do online (causal) computation** ("它没有办法做到在线计算…它永远是在计算全局最优…今天推理是做多，明天再推理那个多可能就变成空了"). His only sanctioned uses of HMM are (a) as a **label generator** (hidden state → {long/short} label) or (b) as a **feature** (hidden-state value, then take its information-entropy). Therefore in this study **HMM is a BASELINE comparator only** — a method the featured clustering/regime labelers are expected to beat. The Q2 doc's HMM-config exploration is treated as baseline exploration, not a recommendation.

The **featured** (Wang-taught) labelers are the clustering/regime methods: `kmeans2stage` (vol→direction), `bgm` (Bayesian GMM), `agglomerative` (Ward), `carry`, `tertile`, `triple_barrier`, `multi_horizon`.

---

## 1. Research question & hypotheses (falsifiable)

**RQ:** Within a fixed downstream pipeline, which *information-driven axis* and which *unsupervised labeler* maximize REAL out-of-sample Calmar, and do Wang's featured clustering/regime labelers beat the HMM baseline and a naive always-long baseline?

Each hypothesis states a measurable rejection rule on REAL OOS metrics from the QC infer backtest (2023-08-01 → 2026-06-01), replicated per §4.

- **H1 (axis reduces tails → better OOS Calmar).** Information-driven axes (`vol`, `dollar`, `logdollar`, `entropy`) produce lower TRAIN bar-return excess kurtosis than count/price axes (`tick`, `range`) AND yield higher median REAL OOS Calmar across the panel.
  - *Reject H1 if:* the four information-driven axes do not have lower median TRAIN kurtosis than {tick, range}, OR their median REAL OOS Calmar is ≤ that of {tick, range} (Δ ≤ 0, no replicated lift).

- **H1b (kurtosis is the mechanism, not just a correlate).** Across all 48 cells/ticker, lower TRAIN bar-return kurtosis is associated with higher REAL OOS Calmar (rank correlation ρ between −kurt and Calmar > 0, pooled and within-ticker).
  - *Reject H1b if:* ρ ≤ 0 or not directionally consistent across ≥2 tickers. (Guards against "more bars always wins regardless of distribution.")

- **H2 (featured labelers beat baselines).** The 6 featured labelers, at their best axis, beat BOTH the HMM baseline and the naive always-long baseline on REAL OOS Calmar, replicated across ≥2 tickers.
  - *Reject H2 if:* no featured labeler exceeds `max(HMM, always_long)` REAL Calmar on ≥2 tickers. **Featured-vs-baseline lift** is reported explicitly per ticker as `Calmar(best featured) − Calmar(best baseline)`.

- **H3 (axis × labeler interaction is real, not separable).** The best labeler depends on the axis (a non-trivial interaction), consistent with program.md's "Ensemble unlocks vol bars but destroys dollar bars."
  - *Reject H3 if:* the same labeler is optimal on every axis for every ticker (pure main effects, no interaction).

- **H4 (no-overfit / no-leak constraint holds for winners).** Any cell promoted to "confirmed" passes G3 (lookahead audit = 0 violations) and G4 (|train_auc − val_auc| < 0.05).
  - *Reject promotion if:* G3 or G4 fails. (This is a gate, not a scientific hypothesis, but it is a hard filter on every claim.)

- **H5 (structural fat tails on HYG/XLE survive any axis).** Per Q1, HYG (kurt ≥ 28 every axis) and XLE (kurt ≥ 12) have *structural* tails. Prediction: no axis brings HYG/XLE TRAIN kurtosis near-Gaussian, and their REAL OOS Calmar stays below the panel median regardless of axis.
  - *Reject H5 if:* some axis drives HYG or XLE TRAIN kurt below ~5 AND lifts its Calmar to panel-median or above.

---

## 2. CANONICAL DEFINITIONS — the build contract

These definitions are the exact, causal, computable rules to implement in `modules/bar_builder.py` (axes) and `modules/labeler.py` (labelers). They MUST be leak-free: every fit uses TRAIN-only data; everything applied at VAL/TEST is the frozen TRAIN-fit object replayed forward. Bar log-return is `r_bar[t] = log_close[t] − log_close[t−1]` over consecutive bars (NOT minutes).

Per-axis threshold is tuned to hit `target_bars = 15000` (per `constants.TARGET_BARS`) over the full minute history, exactly as the current `build_bars()` does (total-accumulator / target_bars). For axes that need a rolling z-score, threshold is fit on TRAIN minutes only (binary search to bar-count target), matching Q1's "threshold tuned per cell on TRAIN minutes only."

### 2.1 Axes (factor A — 6 levels)

For each axis: accumulator `acc`, emit rule, and reset. All accumulators iterate minute bars in time order; on emit, record `{ts_close, log_close=log(close)}`, reset accumulator, continue. `build_bars()` then derives `lr` as consecutive diffs of `lc`.

| Axis | Accumulator per minute t | Emit when | Notes / source |
|---|---|---|---|
| **dollar** | `acc += close_t · vol_t` | `acc ≥ T` | Existing `DollarBarBuilder`. `T = Σ(close·vol)/target_bars`. Wang's best for liquid equities. |
| **tick** | `count += 1` | `count ≥ N` | Existing `TickBarBuilder`. `N = len(minutes)/target_bars`. **Count axis — comparator, not information-driven.** |
| **vol** | `acc += (Δlog close_t)² · √vol_t` | `acc ≥ T` | Existing `VolBarBuilder`. `T = Σ contrib / target_bars`. Wang's "3rd axis," gold/metals. |
| **range** | track `last_sample_close`; `move = |close_t − last_sample_close| / last_sample_close` | `move ≥ θ` | Existing `RangeBarBuilder`. Price-move axis. peak_frac→0 by construction (Q1 PriceMove). **Comparator.** |
| **logdollar** | `acc += log(1 + close_t · vol_t)` | `acc ≥ T` | **NEW.** `T = Σ log(1+close·vol)/target_bars`. Causal, monotone. Compresses dollar-volume dynamic range so a few mega-volume minutes don't dominate one bar (the failure mode of raw dollar bars on spike days). Distinct from Q1's z-cumsum LogDollar; chosen as a *purely cumulative, parameter-free* variant that fits the existing total-accumulator/target_bars threshold machinery with zero rolling-window state. |
| **entropy** | Shannon surprise of the *return-sign / return-bucket* distribution since last bar (see §2.1.1) | `acc ≥ T` | **NEW.** Information-driven bar (Wang `Ca1n2jgKjrs`: 商 → information density → information-driven bars). |

#### 2.1.1 `entropy` axis — exact causal rule

Goal: emit a new bar once the *minute-return stream since the last bar* has accumulated `T` nats of Shannon surprise, so each bar carries ~equal information. Fully causal (uses only minutes ≤ t) and parameter-light.

Fit on TRAIN minutes only:
1. Compute TRAIN minute log-returns `m_t = log(close_t) − log(close_{t−1})`.
2. Define **fixed bucket edges** from TRAIN-minute return quantiles: 5 buckets B = {strong-down, down, flat, up, strong-up} at TRAIN percentiles {10, 35, 65, 90}. Store edges (these are frozen; replayed unchanged at VAL/TEST — no leakage).
3. Estimate **bucket base-rate probabilities** `p_b` = empirical frequency of each bucket on TRAIN minutes (Laplace-smoothed, `+1` per bucket).

Online accumulation (TRAIN, VAL, TEST identically), starting `acc = 0` after each emit:
4. For each new minute t, classify `m_t` into bucket `b(t)` using the frozen edges.
5. Add **self-information (surprise)** `acc += −log(p_{b(t)})` (natural log, per Wang's note that finance uses natural-log entropy, not base-2).
6. **Emit a bar when `acc ≥ T`**, reset `acc = 0`, continue.

Threshold `T` fit on TRAIN minutes only by binary search so the TRAIN bar count hits `target_bars · (TRAIN minutes / total minutes)`; reuse the same `T` for the whole stream (matches Q1's "tuned on TRAIN minutes only"). Rationale: rare large moves (low `p_b`) contribute more surprise → bars close faster around informative bursts and slowly in calm-but-traded stretches, directly targeting Wang's "calm-but-active" problem (Q1 of `wang_qa_questions.md`) and "信息密度尽可能均匀."

> Causal safeguard: bucket edges and base rates are TRAIN-frozen; the accumulator only ever reads the current and past minute. No forward path is consulted. Passes G3.

### 2.2 Labelers (factor B — 8 levels: 6 FEATURED + 2 BASELINE)

All labelers operate on bar arrays `(lc, lr)` with masks `tr_m / va_m / te_m` and forward metrics from `compute_forward_metrics(lc, lr, horizons=[50,100,200])`. Output is a per-bar array in {−1 (no-trade / abstain), 0 (short-or-flat), 1 (long)}; downstream consumes only labeled bars (`y >= 0`). What is fit on TRAIN, the state→{−1,0,1} mapping, and the leak safeguard are stated for each.

The forward metrics are the **label target** (Wang: *"标签是训练目标，允许 look-ahead"* — labels may peek forward because the real signal is the downstream model's causal output). Forward look-ahead is *inside the label only*; features and the model are causal, and the audit (G3) targets feature/model leaks, not the label's forward window.

#### FEATURED (Wang's clustering/regime methods)

1. **kmeans2stage** (vol→direction) — *existing.*
   - **Fit on TRAIN:** Stage-1 `KMeans(n_clusters=2, n_init=5, random_state=42)` on TRAIN forward-vol → low-vol cluster = the one with smaller center. Stage-2 `KMeans(n_clusters∈{2,3}, n_init=5)` on standardized `[fwd_ret, |fwd_ret|]` within the low-vol subset; up-cluster = max center on the `fwd_ret` axis.
   - **Map:** within low-vol, up-cluster → 1, else 0; all high-vol bars → −1 (abstain). Swept over horizons {50,100,200} and nc {2,3}; pick config with VAL balance ∈ (0.2, 0.8).
   - **Safeguard:** cluster centers fit on TRAIN; VAL/TEST bars `predict`-assigned with frozen centers + frozen StandardScaler.

2. **carry** (low forward-vol ⇒ long) — *existing.*
   - **Fit on TRAIN:** median forward-vol `med_v` on TRAIN labeled bars.
   - **Map:** `fwd_vol ≤ med_v → 1` (carry/long), `> med_v → 0`. Swept over horizons; VAL balance gate (0.2, 0.8).
   - **Safeguard:** `med_v` is a TRAIN scalar applied forward.

3. **tertile** (extreme-move purity) — *existing.*
   - **Fit on TRAIN:** 33rd/67th percentiles of TRAIN forward return.
   - **Map:** `fwd_ret ≥ top → 1`, `≤ bottom → 0`, middle third → −1 (abstain). TRAIN balance gate (0.3, 0.7).
   - **Safeguard:** percentile cuts are TRAIN scalars.

4. **bgm** (Bayesian GMM, sparse Dirichlet) — *existing.*
   - **Fit on TRAIN:** KMeans-2 vol filter (as above), then `BayesianGaussianMixture(n_components∈{3,4,5}, covariance_type='full', weight_concentration_prior=0.1, n_init=3, max_iter=300, random_state=42)` on low-vol `[fwd_ret, |fwd_ret|]`.
   - **Map:** up-component = max mean on `fwd_ret`; label 1 iff `predict==up_c AND posterior(up_c) > POST_THRESH(=0.40)`, else 0; high-vol → −1. Sparse Dirichlet self-prunes K (Wang's anti-fragment lever, the GMM analogue of his "3-state absorbs flat" point).
   - **Safeguard:** GMM + scaler fit on TRAIN; `predict_proba` on VAL/TEST with frozen model.

5. **agglomerative** (Ward) — *existing.*
   - **Fit on TRAIN:** vol filter, then `AgglomerativeClustering(n_clusters∈{2,3,4}, linkage='ward')` on TRAIN low-vol `[fwd_ret,|fwd_ret|]`.
   - **Map:** up-cluster = max mean `fwd_ret`; up→1 else 0; high-vol→−1.
   - **Safeguard / caveat:** Agglomerative has **no native `predict`**. To stay leak-free, fit on TRAIN low-vol points and assign VAL/TEST bars by **nearest TRAIN cluster centroid** (compute centroids from TRAIN cluster membership, then 1-NN assign forward). This makes it causal and OOS-applicable; do NOT refit on TRAIN+VAL+TEST jointly.

6. **multi_horizon** (agreement gate) — **NEW.**
   - **Definition:** run a base labeler (default `kmeans2stage`) independently at each horizon h ∈ {50, 100, 200}. Trade only where all three horizons agree.
   - **Map:** `1` iff label_h==1 for all h; `0` iff label_h==0 for all h; otherwise `−1` (abstain — horizons disagree). This is Wang's multi-scale-consensus idea ("不同尺度…组合…变相感受市场regime") applied at the *label* level (distinct from the trainer's seed-consensus).
   - **Safeguard:** each per-horizon labeler is TRAIN-fit and frozen; agreement is computed per bar from the three frozen labelers.

#### BASELINE (must be beaten)

7. **hmm** (GaussianHMM, 3-state, `[r, |r|]`) — **BASELINE COMPARATOR ONLY. Wang does NOT use this as a method.**
   - **Fit on TRAIN:** `from hmmlearn.hmm import GaussianHMM; GaussianHMM(n_components=3, covariance_type='diag', n_iter=100, random_state=42)` on TRAIN bar observations `o_t = [r_bar_t, |r_bar_t|]` (the Q2 "winning" 3-state `[r,|r|]` config — used here so the baseline is the *strongest* HMM, making the featured-vs-baseline test conservative).
   - **Map:** sort states by mean of dimension 0 (`r_bar`); `up` = highest-mean state. Label uses the **causal forward filter** posterior `φ_t^up = α_t(up)/Σ_k α_t(k)` (NOT the smoothed posterior — smoothing peeks at future bars and is exactly the lookahead Wang warned about). `y_t = 1{φ_t^up > 0.5}`, else 0. Never abstains (always emits 0/1) — by design this is the dense, less-discriminating baseline.
   - **Safeguard:** HMM params fit on TRAIN; forward filter (`_do_forward_pass` / online `score_samples` per-step) applied causally at VAL/TEST. No `predict` (Viterbi/smoothed) on the test segment. This honors Wang's "no online global-optimum" warning while keeping it a fair, leak-free baseline.

8. **always_long** (naive) — **BASELINE.**
   - **Definition:** `y_t = 1` for every labeled bar (no model selection at the label stage). Under drift-heavy ETFs (Q2 of `wang_qa_questions.md`: QQQ β₂₀₀ = 0.74) this is a hard-to-beat majority baseline. Implemented as a constant labeler; the downstream model still trains on features but the label carries no direction information, so its REAL Calmar ≈ the buy-and-hold-while-above-threshold floor.
   - **Map:** all labeled bars → 1.
   - **Purpose:** quantifies how much of any cell's Calmar is just drift capture vs genuine regime signal. **Featured-vs-baseline lift is reported against `max(hmm, always_long)`.**

> Note on `forest` (the existing 7th labeler in `labeler.py`): NOT in the 8-level factor. It is a label-ensemble of carry+tertile+km and is redundant with `multi_horizon` for this study; excluded to keep the factorial clean. It may be revisited as a follow-up.

---

## 3. FACTORIAL DESIGN

**Factor A (axis):** {dollar, tick, vol, range, logdollar, entropy} — 6 levels.
**Factor B (labeler):** {kmeans2stage, carry, tertile, bgm, agglomerative, multi_horizon} = 6 FEATURED + {hmm, always_long} = 2 BASELINE → 8 levels.
**Cells per ticker:** 6 × 8 = **48**.

**Downstream HELD FIXED** (exactly as locked in `trainer.py` / `features.py`):
- Features = existing 80-set (`build_feats`): 20 momentum + 20 z-momentum + 16 rolling std/mean + 8 kurtosis + 4 vol-ratio + 4 price-vs-MA + 8 sample-entropy. Cross-asset SPY features stay DISABLED.
- Dim-reduce = correlation filter → top-20 by variance (`reduce_dims(method="correlation", n_components=20)`).
- Model = `XGBClassifier(max_depth=3, learning_rate=0.03, n_estimators=200, reg_alpha=1, reg_lambda=2, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=n_neg/n_pos, eval_metric="auc", early_stopping_rounds=30)`.
- Calibration = isotonic on VAL (`IsotonicRegression(out_of_bounds='clip')`).
- Ensemble/consensus = **n_seeds = 1** (fix; do NOT sweep {1,5} — the seed sweep is a confound for a clean A×B factorial and program.md shows 1-seed usually wins). MA-period and inversion sub-sweeps inside `trainer.py` are also pinned (ma=100, inv=False) so the only moving parts are A and B.
- Threshold = per-ETF fixed from `infer.py.tmpl` (XLE 0.10, QQQ 0.15, HYG/TLT 0.55, GLD 0.35, else 0.45). REAL execution uses these; synthetic selection's flat 0.45 is a known mismatch (program.md) and is NOT used for any confirmed claim.

> **Harness implication / required change.** Today `footer.py.tmpl:20` hard-codes axis (`tick` for QQQ/EEM/XLE, else `dollar`) and `:44–47` hard-codes per-ticker labeler routing. To run this factorial the footer must be made to **sweep all 6 axes × 8 labelers and save per-cell test predictions** under distinct ObjectStore keys (`autoresearch/{TICKER}/{axis}_{labeler}/test_preds_*.json`). This touches a LOCKED template and needs sign-off (it is exactly "Required Upgrade #1: expose volatility bars" generalized to a full sweep). `vol`, `logdollar`, `entropy` are otherwise UNREACHABLE.

### Blocks — ticker panel (justification)

A small representative panel spanning the three regime archetypes named in `ASSET_AFFINITY` and validated by both Q-docs:

| Role | Ticker | Why (regime archetype) | Q-doc evidence |
|---|---|---|---|
| **Pilot** | **GLD** | Gold/metals = the volatility_regime archetype; the only Track-A asset with a stable >1.5 REAL Calmar (1.59–1.61), and where vol bars are the proven best axis. Lowest-variance cell → cleanest signal to debug the harness. | program.md GLD vol/Carry 1.61; Q2 balanced labels. |
| **Expansion 1** | **QQQ** | Liquid equity / trend_following archetype; **drift-heavy** (β₂₀₀=0.74) so it stresses the always_long baseline hardest and is where balancing historically *hurt* (Q2). Dollar/tick axis natural home. | `wang_qa_questions.md` Q2: QQQ alpha −0.55 despite AUC 0.839. |
| **Expansion 2** | **TLT** | Long bond / mean_reversion archetype; **does not trend at minute scale** (Track-A Cal −0.15). A genuine *negative control*: if any axis/labeler "confirms" on TLT, suspect overfit. | program.md TLT dollar/Carry −0.15. |

This 3-ticker panel spans {gold, equity, bond} = {regime, trend, mean-reversion/negative-control}, giving cross-regime generalization evidence (program.md Tertiary goal: "techniques that generalize across ≥2 assets") while keeping the pilot budget small. HYG/XLE (structural-fat-tail, H5) are reserved for the full study, not the pilot.

---

## 4. METRIC + INFERENCE

**Primary metric:** REAL OOS Calmar = `Compounding Annual Return / Drawdown` from the **QC infer backtest** (`SetHoldings`), parsed exactly as `orchestrator.py:79–82` / `evaluator.py:40–53`, over TEST 2023-08-01 → 2026-06-01.

**Why synthetic Calmar is invalid for cross-axis comparison.** `realistic_cstats` (`trainer.py:18`) computes Calmar on `positions · forward-log-ret` per bar with a flat 0.45 threshold and an `ann = mean(strat_rets)·880` annualization that assumes a fixed bar-per-year count. **Different axes produce different numbers of bars per calendar year** (15k target is over 17 years, but vol/entropy bars cluster in volatile periods, so bars-per-year varies by axis), so the `·880` factor and the per-bar drawdown are **not comparable across axes** — synthetic Calmar is inflated 2–100× and even sign-reversed vs REAL (program.md table: GDX synth 8.32 → REAL −0.37; EEM synth −0.01 → REAL +1.52). It also uses the wrong (flat 0.45) threshold vs real per-ETF execution. Synthetic Calmar is therefore used ONLY for within-cell internal model selection inside the train backtest (which dim/ma/inv config to save), never to rank cells or confirm a hypothesis.

**Secondary metrics:** REAL OOS trade count (`Total Orders`, gate G2 > 80); AUC divergence `|train_auc − val_auc|` (gate G4 < 0.05); TRAIN bar-return excess kurtosis and peak_frac (Q1 distributional diagnostics, for H1/H1b/H5); featured-vs-baseline Calmar lift.

**Multiplicity / false-discovery control.** 48 cells/ticker is a large search; raw "best cell" is an order statistic that overstates performance. Controls:
1. **Replication requirement for "confirmed":** a cell (axis,labeler) is *confirmed* only if it improves REAL Calmar over the relevant baseline on **≥2 of 3 panel tickers** OR survives **≥2 seeds** (re-run with `random_state ∈ {42, 7}` for the stochastic labelers/model; note label clustering is seeded and the trainer is seeded, so seed variance is the model+KMeans/BGM init variance). Single-ticker single-seed wins are logged as *candidate*, never *confirmed*.
2. **Baseline-relative reporting:** every cell's Calmar is reported as raw and as lift over `max(hmm, always_long)` for that ticker — controls for the drift floor.
3. **Seed-variance caveat:** program.md records seed variance can swamp small "wins" (community ≈0.002 bpb); treat any Δ Calmar below the within-cell seed spread as noise. Report the seed spread alongside the point estimate.
4. **Negative control:** TLT confirmations are treated as red flags for overfit/leak, not successes.
5. **Family-wise sanity:** with 48 cells, expect ~2–3 "Calmar > panel-median" by chance; require the *pattern* (H1/H2/H3 direction) to hold, not just a single lucky cell.

---

## 5. EXECUTION PLAN (matched to QC limits: 2 nodes, ~5-min backtests)

Two-phase per program.md: ONE TRAIN backtest per (ticker, axis) that internally sweeps all 8 labelers and saves per-cell TEST predictions to ObjectStore; then ONE lightweight INFER backtest per (axis, labeler) cell that replays those predictions through `SetHoldings`.

**Why TRAIN is per-(ticker,axis), not per-cell:** building bars + 80 features dominates train runtime and is shared across all 8 labelers for a fixed axis. So one train backtest emits 8 prediction sets (one per labeler) under keys `autoresearch/{TICKER}/{axis}_{labeler}/...`. Train timeout 480 s (`orchestrator.py:57`); 8 labelers × (label + XGBoost fit on ~20-dim, ~9–12k TRAIN bars) fits in budget — XGBoost depth-3/200-trees on 20 features is sub-second; HMM `n_iter=100` and BGM `n_init=3` are the heaviest, both well under a minute each. If a per-axis train risks the 480 s wall, split into two train backtests of 4 labelers each (still ≪ per-cell).

**Why INFER is per-cell:** infer just reads one prediction list and runs `SetHoldings` over 2.8 years of minutes; ~1–3 min each, timeout 180 s (`orchestrator.py:75`).

### Budget

Let **T** = number of TRAIN backtests, **I** = number of INFER backtests.

**Per ticker, full factorial:**
- TRAIN: 6 axes × 1 = **6** train backtests (each sweeps 8 labelers).
- INFER: 6 × 8 = **48** infer backtests.
- = 54 backtests/ticker.

**PILOT (GLD only, full A×B):**
- TRAIN = 6, INFER = 48 → **54 backtests**.
- With 2 nodes and ~5 min/backtest: 54 / 2 × 5 min ≈ **135 min ≈ 2.25 h** wall (train+infer pipelined). Single seed.
- Deliverable: the 48-cell GLD Calmar matrix + featured-vs-baseline lift + kurtosis/peak_frac per axis (H1, H1b, H2, H3, H5 first read).

**FULL STUDY (3-ticker panel GLD/QQQ/TLT, full A×B, +replication seed on candidates):**
- Base: 3 × 54 = **162 backtests** (T=18, I=144).
- Replication: re-run only *candidate* cells (those beating baseline on the pilot) with seed 7 — budget ~2 axes × 8 × 3 tickers for train re-emit + their infers ≈ **+30–60 backtests**.
- Total ≈ **~200–220 backtests**. At 2 nodes × 5 min ≈ **8–9 h** wall.

**Reduced pilot (if harness sweep-footer not yet signed off):** keep the current per-ticker single-route footer but add the 3 new axes and `multi_horizon` only on GLD: axes {dollar, vol, logdollar, entropy} × labelers {agglomerative, carry, kmeans2stage, multi_horizon, hmm, always_long} = 4 × 6 = 24 cells → T=4, I=24 = **28 backtests ≈ 70 min**. This is the minimum viable run to test H1/H2 on the pilot before committing to the full sweep.

### Pilot cell list (GLD, 48 cells)

Axes A = [dollar, tick, vol, range, logdollar, entropy]; Labelers B = [kmeans2stage, carry, tertile, bgm, agglomerative, multi_horizon, hmm*, always_long*] (`*` = baseline). The 48 ordered cells:

```
GLD/dollar/kmeans2stage      GLD/tick/kmeans2stage      GLD/vol/kmeans2stage
GLD/dollar/carry             GLD/tick/carry             GLD/vol/carry
GLD/dollar/tertile           GLD/tick/tertile           GLD/vol/tertile
GLD/dollar/bgm               GLD/tick/bgm               GLD/vol/bgm
GLD/dollar/agglomerative     GLD/tick/agglomerative     GLD/vol/agglomerative
GLD/dollar/multi_horizon     GLD/tick/multi_horizon     GLD/vol/multi_horizon
GLD/dollar/hmm*              GLD/tick/hmm*              GLD/vol/hmm*
GLD/dollar/always_long*      GLD/tick/always_long*      GLD/vol/always_long*

GLD/range/kmeans2stage       GLD/logdollar/kmeans2stage GLD/entropy/kmeans2stage
GLD/range/carry              GLD/logdollar/carry        GLD/entropy/carry
GLD/range/tertile            GLD/logdollar/tertile      GLD/entropy/tertile
GLD/range/bgm                GLD/logdollar/bgm          GLD/entropy/bgm
GLD/range/agglomerative      GLD/logdollar/agglomerative GLD/entropy/agglomerative
GLD/range/multi_horizon      GLD/logdollar/multi_horizon GLD/entropy/multi_horizon
GLD/range/hmm*               GLD/logdollar/hmm*         GLD/entropy/hmm*
GLD/range/always_long*       GLD/logdollar/always_long* GLD/entropy/always_long*
```

Train backtests for the pilot (6): `GLD/{dollar,tick,vol,range,logdollar,entropy}` (each emits its column of 8 labeler prediction sets). Infer backtests: the 48 cells above.

---

## 6. Build contract summary (what implementers must add)

- `modules/bar_builder.py`: add `LogDollarBarBuilder` (cumulative `log(1+close·vol)`) and `EntropyBarBuilder` (TRAIN-frozen 5-bucket surprise accumulator per §2.1.1); extend `build_bars(bar_type∈{"dollar","tick","vol","range","logdollar","entropy"})`.
- `modules/labeler.py`: add `generate_labels_multi_horizon(...)` (per-horizon kmeans agreement → {−1,0,1}), `generate_labels_hmm(...)` (GaussianHMM-3 on `[r,|r|]`, **causal forward-filter** posterior, no smoothing/Viterbi on TEST), `generate_labels_always_long(...)` (constant 1 on labeled bars). Keep existing kmeans2stage/carry/tertile/bgm/agglomerative; for agglomerative add TRAIN-centroid 1-NN forward assignment so it is OOS-applicable and leak-free.
- `templates/footer.py.tmpl` (LOCKED — needs sign-off): replace hard-coded axis/labeler routing with the 6×8 sweep, saving per-cell predictions under `autoresearch/{TICKER}/{axis}_{labeler}/test_preds_*.json` + a per-cell `latest_key`.
- `templates/infer.py.tmpl` (LOCKED): parameterize the ObjectStore key by `{axis}_{labeler}` so each infer backtest loads its own cell; keep per-ETF thresholds.
- Everything in `features.py` and `trainer.py` stays fixed (pin n_seeds=1, ma=100, inv=False for the factorial).

**Runtime conformance (QC LEAN Foundation):** Python 3.11.11 / numpy 1.26.4 / pandas 2.3.3 / scikit-learn 1.6.1 / xgboost 3.0.5 / hmmlearn 0.3.3. Required code fixes:
- `KMeans`/`BGM` already pass `n_init` explicitly — keep (sklearn 1.6.1 requires it).
- **pandas 2.3.3:** `features.py:97,103` currently call `.fillna(method='ffill')` which is REMOVED — must change to `.ffill().fillna(0.0)`. (This is a pre-existing latent bug that will crash on the QC runtime; fix before any run.)
- xgboost 3.0.5: `early_stopping_rounds`/`eval_metric` in constructor + `eval_set` in `.fit()` — already used, valid.
- hmmlearn 0.3.3: `from hmmlearn.hmm import GaussianHMM` — available for the baseline.
