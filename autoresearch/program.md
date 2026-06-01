# Autoresearch — ML Quant Trading Pipeline

> **Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (~84K stars, 12K forks).**
> Karpathy's loop: lock `prepare.py`, let an agent edit `train.py`, train 5 min, keep-if-`val_bpb`-improves,
> `git reset` otherwise, never stop. We port that design to quantitative finance:
> **single locked metric, locked evaluation, free exploration, structured memory, never stop.**
>
> Two experiment tracks feed this file:
> - **Track A — this modular harness** (`harness/` + `modules/`): logged in `results.tsv` (t004–t061) and `knowledge.json`.
> - **Track B — predecessor standalone pipelines** (`/experiment_summary/`, v236–v392): 49 versions, 105 ETFs,
>   330 pairs. Its findings are PROVEN PRIOR KNOWLEDGE to exploit, but must be RE-VALIDATED in Track A
>   before being trusted here (different code path).

---

## Goals (staged — the 3.0 gate is aspirational, not yet met)
- **Metric**: REAL QC backtest **Calmar = CAGR / MaxDrawdown** on the OOS period via real `SetHoldings`.
  Synthetic Calmar is for internal model selection ONLY and is KNOWN UNRELIABLE (see §"Synthetic is a Lie").
- **Keep-gate (locked, `constants.py`)**: REAL Calmar > **3.0** AND REAL trades > **80** on OOS.
- **Reality check**: best REAL single-ETF Calmar ever achieved ≈ **1.61** (GLD, vol bars); best REAL **pair** ≈
  **2.91** (SHY-IWM). **No single ETF has cleared 3.0.** Treat the gate as the north star; track progress against
  these intermediate ceilings and report honestly. Closing the gap requires the upgrades in §"Required Upgrades".
- **Secondary**: trades > 80 OOS, AUC divergence < 0.05 (no overfit), zero lookahead.
- **Tertiary**: techniques that generalize across ≥2 assets.

---

## The Editable Surface (READ THIS — it differs from intuition)

**You may edit ONLY these 4 files** (they are the only ones `render_script` concatenates, `orchestrator.py:27`):

| File | Module | What lives here |
|---|---|---|
| `modules/bar_builder.py` | ① Custom axis | `DollarBarBuilder`, `TickBarBuilder`, `VolBarBuilder`, `RangeBarBuilder`, `build_bars()` |
| `modules/labeler.py` | ② Labels | `compute_forward_metrics()` + 6 unsupervised labelers (kmeans2stage, carry, tertile, bgm, agglomerative, forest) |
| `modules/features.py` | ③ Features | `build_feats()` — 80 features (momentum, z-mom, rolling vol/mean, kurtosis, vol-ratio, price-vs-MA, sample entropy) |
| `modules/trainer.py` | ④⑤⑥⑦⑧ | `reduce_dims()`, `train_and_evaluate()`, `realistic_cstats()` — dim-reduce → XGBoost → **isotonic cal (inline)** → ensemble → **consensus (inline)** |

**DEAD modules — editing them does NOTHING** (never imported by the renderer or templates):
`modules/calibrator.py`, `modules/consensus.py`, `modules/ensembler.py`. Their logic is duplicated inline in `trainer.py`.
If you want to change calibration/consensus/ensembling, edit `trainer.py`.

**LOCKED — do NOT edit** (this is our `prepare.py`): `harness/` and `templates/`. In particular note what the lock implies:
- `templates/footer.py.tmpl:20` chooses the axis: **`tick` for QQQ/EEM/XLE, `dollar` for everything else.**
  ⚠️ **Volatility bars are therefore UNREACHABLE** in this harness even though `VolBarBuilder` exists and Track B
  proved vol bars are the single best axis for the gold/metals complex. See §"Required Upgrades".
- `templates/footer.py.tmpl:44–47` hard-codes per-ticker labeler routing (GLD→Agglomerative, else Carry→KMeans).
- `templates/infer.py.tmpl:34` hard-codes REAL per-ETF thresholds: XLE 0.10, QQQ 0.15, HYG/TLT 0.55, GLD 0.35, else 0.45.
  ⚠️ Note `realistic_cstats` (synthetic selection, `trainer.py:31`) uses a FLAT 0.45 — so synthetic model
  selection and real execution use *different* thresholds. This is a known selection/execution mismatch.

---

## Evaluation Gates (from `harness/evaluator.py` — locked)

| Gate | Check | Threshold | Source |
|---|---|---|---|
| **G0** Completed | backtest finished, no timeout/crash | status startswith "Completed" | infer statistics |
| **G1** Calmar | REAL OOS Calmar | **> 3.0** | `CAGR / Drawdown` from infer |
| **G2** Trades | REAL OOS order count | **> 80** | `Total Orders` from infer |
| **G3** Lookahead | static scan for leaks | 0 violations | scans `.shift(-N)`, `arr[::-1]`, `tr_m|te_m`, `bfill` on price |
| **G4** Overfit | `|train_auc − val_auc|` | **< 0.05** | train runtime stats |

Verdict precedence: not-completed → `timeout`/`crash`; else all-pass → `keep`; else G3 fail → `leak`;
else G4 fail → `overfit`; else → `discard`. `keep` requires ALL of G0–G4.

---

## Universe & Splits

- **Harness CORE-7** (`constants.py`): QQQ, IWM, EEM, XLE, HYG, TLT, GLD.
  ⚠️ These are mostly trend/macro ETFs. Track B showed the **proven alpha is NOT here** — it is in the
  **gold/metals complex** (GDX, GDXJ, IAU, SLV) on vol bars, and in **cross-asset pairs**. The harness is
  currently pointed at the least fertile slice of the universe.
- **Extended universe** (Track B, 105 ETFs): `/experiment_summary/results/per_ticker_optimal.csv`.
- **Splits** (`constants.py`): TRAIN → 2021-08-01, VAL → 2023-08-01, TEST → 2026-06-01.
  Per-ETF exception found useful: IWM=2018 split (+30% REAL Cal).
- **Data**: QuantConnect minute bars, 2009-08 → 2026-06. **Target ≈ 15,000 information bars** per history.
- **Infra**: QC Cloud, project 31338454. Train timeout 480 s, infer 180 s, poll 30 s. (`constants.py` has
  TIME_BUDGET=300.) ⚠️ `QC_CREDS_PATH`/`TEMPLATES_DIR`/etc. in `constants.py` point at
  `/Users/liyuanjun/ai_work/lb/...` — these are stale on this machine (`/home/ubuntu/lb`). Fix paths before running.

---

## Karpathy Design Principles, Applied

1. **Lock evaluation, free exploration.** `harness/` + `templates/` are read-only; you change only `modules/`.
   This prevents metric-gaming — you can't improve the score except by genuinely improving the model.
2. **One metric, verified by real trading.** Karpathy: `val_bpb`. Us: REAL `SetHoldings` Calmar. Synthetic is a lie.
3. **Statistical validation is mandatory.** The community found seed variance alone ≈ 0.002 bpb — many "wins" were
   noise. Run multiple seeds (the trainer sweeps n_seeds∈{1,5}); trust only what survives ≥2 runs. Log failures.
4. **Shared memory accelerates discovery.** `knowledge.json` + `techniques.json` are our shared brain. Update after
   EVERY experiment with REAL results, including negatives.
5. **Many "improvements" are noise.** Community: weight tying (3.216 bpb), label smoothing, z-loss all failed.
   Us: PCA destroys signal; continuous sizing hurts QQQ; forced 5-seed ensemble = 1-seed. Negative results count.
6. **More data > bigger models.** Community: more steps beats bigger batch. Us: more bars > more features. The
   correlation filter (80→20) improves generalization by cutting the feature:sample ratio. Add bars before features.
7. **Simple > complex.** A 0.001 gain that adds 20 lines of hacky code is not worth it. Single-seed XGBoost beats
   the ensemble; binary threshold beats continuous sizing.

---

## Wang's 8-Module Workflow (the intellectual core)

Source: "uni 的量化日记" course transcripts (`/uni_yt/`, `/uni/transcripts/`) + `pdfs/wang_course_2026-06.pdf` + AFML.

### ① Custom Axis — *"the axis is the kingpin"* (`bar_builder.py`)
Resample minute OHLCV into information-driven bars, NOT time bars. Time bars have huge kurtosis (200–470 here);
custom bars push the return distribution toward Gaussian, cleaning every downstream module.
- **Dollar bars** (`close*volume` threshold): best for liquid equities/semis (SMH, SOXX, QQQ).
- **Tick bars** (count threshold): best for some commodities/defensive (DIA, GDXJ-tick).
- **Volatility bars** (`Σ (logret)²·√vol`): **Wang's 3rd axis — the breakthrough for gold/metals** (Track B). ⚠️ unreachable in Track A today.
- **Range bars** (price-move): never selected.
- **Imbalance / Run bars**: require tick data — NOT available at minute granularity.

### ② Unsupervised Labeling (`labeler.py`)
Label regimes causally (forward returns/vols at horizons [50,100,200]). Ranked by REAL Calmar:
- **Carry** (low forward-vol ⇒ long): best overall; wins on gold/commodities. Fails on tech/equity.
- **KMeans two-stage** (vol-cluster → direction-cluster): universal baseline; Cal 1.32–3.43 across ETFs.
- **BGM (Bayesian GMM, K=3–5, sparse Dirichlet)**: strong on vol bars (GDXJ 1.60, MTUM 1.35); isotonic-calibrated.
- **Ensemble (KM+Carry+BGM, averaged)**: **axis-specific** — unlocks vol bars (SLV 0→1.10, GDXJ→1.60) but
  *destroys* dollar bars (SMH +485%→+2.6%). Use only on vol bars.
- **Agglomerative Ward**: GLD-specific (1.59). **Triple-Barrier** (AFML standard): GLD/GDXJ ≈1.59.
- **Tertile / Forest of Opinions**: too conservative (Forest gave QQQ 1 trade).

### ③ Feature Engineering (`features.py`) — 80 features
Momentum (20) + z-scored momentum (20) + rolling vol/mean (16) + kurtosis (8) + vol-ratio (4) + price-vs-MA (4)
+ **sample entropy (8)**. Correlation filter 80→20 is optimal. Cross-asset SPY features are plumbed but DISABLED
(they crowded out own-asset features: QQQ trades 521→1).
Wang's richer feature ideas not yet implemented: **fractional differentiation (d≈0.3–0.81)**, **frequency-domain
(FFT / VMD+NRBO)**, **multi-scale entropy**, **Information-Gain feature selection**.

### ④ Dimensionality Reduction (`trainer.py reduce_dims`)
Correlation filter (drop r>0.90, keep top-20 by variance) is BEST. Variance-threshold ≈ slightly worse.
**PCA destroys signal** (confirmed v362 + Track A). Wang advocates **non-linear VAE (K=8–16)** as the next step — untested here.

### ⑤ Model Training (`trainer.py`)
XGBoost: `depth=3, n_est=200, lr=0.03, reg_alpha=1, reg_lambda=2, subsample=0.85, colsample=0.85`,
`scale_pos_weight=n_neg/n_pos`, early-stop 30. Depth sweeps [2,4] overfit. `binary:logistic` only viable objective.

### ⑥ Calibration (`trainer.py`, inline isotonic)
`IsotonicRegression` on val probs — consistently selected over none. Platt untested.

### ⑦ Ensemble (`trainer.py`, n_seeds∈{1,5})
1-seed wins in most cases (simpler, less overfit). Averaged multi-seed rarely helps.

### ⑧ Consensus (`trainer.py`, active only n_seeds≥3)
Filter `min(p)>0.5 AND avg(p)>0.55`. Too strict for most ETFs → near-zero trades. Single-seed usually wins.

---

## Per-Asset Optimal Configs (REAL Calmar verified)

**Track A (this harness — `knowledge.json`, `results.tsv`):**

| ETF | Bars | Label | REAL Cal | Trades | Note |
|-----|------|-------|----------|--------|------|
| GLD | dollar | Agglom | 1.59 | 26 | most stable in Track A |
| EEM | tick | Carry | 1.52 | 77 | synth −0.01 but REAL 1.52 |
| HYG | dollar | Carry | 1.25 | 143 | thresh 0.55 fixes flooding |
| QQQ | tick | Carry | 1.11 | 521 | thresh 0.15; scale_pos_weight helps |
| XLE | tick | Carry | 0.60 | 353 | thresh 0.10 |
| IWM | dollar | Carry | 0.57 | 177 | 2018 split (+30%) |
| TLT | dollar | Carry | −0.15 | 22 | bonds don't trend at minute scale |

**Track B (predecessor pipelines — `/experiment_summary/results/`; RE-VALIDATE before trusting):**

| Asset | Axis | Label | REAL Cal | Pipeline | Note |
|-------|------|-------|----------|----------|------|
| SHY | dollar | Ridge+MR | **1.90** | v274 | best single (rates, mean-reversion) |
| SMH | dollar | KMeans-2 | 1.78 | v246 | +485% return |
| IAU | vol | Carry | 1.62 | v384 | gold trust |
| **GLD** | **vol** | **Carry** | **1.61** | **v384** | best gold; vol-bar record |
| GDXJ | vol | BGM | 1.60 | v372 | gold juniors |
| GDX | vol | Carry / KMeans+Ridge | 1.54 | v384 | gold miners |
| MTUM | vol | BGM | 1.35 | v372 | momentum |
| SOXX | dollar | KMeans-2 | 1.34 | v246 | semis #2 |
| QQQ | dollar | KMeans-2 | 1.29 | v246 | (note: Track A got QQQ via tick) |
| EMB | vol | Carry | 1.25 | v372 | EM bonds |
| SLV | vol | Ensemble | 1.10 | v372 | silver (0→1.10 via ensemble) |

**Cross-asset pairs (Track B — the only route to ≥2.0 found so far):**

| Pair | REAL Cal | Trades | Hub | Type |
|------|----------|--------|-----|------|
| **SHY-IWM** | **2.91** | 108 | SHY | rates + small-cap |
| SIL-IWM | 2.80 | 281 | SIL | silver miners + small-cap |
| SHY-XLI | 2.38 | 184 | SHY | rates + industrials |
| SMH-XLF | 2.19 | 290 | SMH | semis + financials |
| SOXX-XLP | 2.03 | 312 | SOXX | semis + staples |
| GDX-GLD | 1.70 | 191 | GDX | gold miners + gold |

**Hub theory** (`hub_ranking.csv`): GDX (12 pairs >1.0), SOXX (10), SMH (10), SHY (7), GLD (6) are universal hubs.
**Direction is asymmetric**: SMH→XLY 1.76 vs XLY→SMH 0.06 (29×). Wang's arbitrage video frames this exactly:
**treat the spread as its own time series and trend-forecast it** rather than mean-revert it.

---

## Synthetic Calmar is a Lie (the central reliability lesson)

Synthetic Calmar (`realistic_cstats` on `pos*ret`) is fine for **within-axis labeling comparison** but is
**10–100× inflated and not comparable across axes** — so any auto-axis selector that trusts it picks the wrong axis.
Concrete:

| Case | Synthetic | REAL | Lesson |
|------|-----------|------|--------|
| EEM tick/Carry | −0.01 | **+1.52** | negative synth, strongly positive real |
| QQQ | 4.30 | 1.11 | 3.9× gap |
| GLD vol | 3.82 | 1.61 | 2.4× inflation |
| GDX dollar+Ridge | 8.32 | **−0.37** | sign reversed — Ridge overfits dollar bars |

⇒ **Only the infer-phase REAL Calmar may be trusted for keep/discard.** v366 and v373 (auto-axis selectors) both
failed for this reason.

---

## Dead Ends (do NOT retry — `knowledge.json` `dead_ends_global`)
HMM labels · CUSUM · Trend-Scanning labels · fractional-diff *as implemented* (hurt OOS) · PCA/KernelPCA ·
autoencoder bottleneck (no lift so far) · DBSCAN (noisy except GDX) · spectral clustering · imbalance/run bars
(need tick data) · daily/hourly data · Renko · GMM soft labels · continuous labels+Ridge (flooded, Cal 0.16) ·
continuous position sizing (QQQ 1.11→0.33) · long-short (XGB low-p ≠ short) · trailing stops · z-score labels ·
depth sweep [2,4] · AUC-div selection (better AUC ≠ better Calmar) · forced 5-seed ensemble · cross-asset SPY
features for non-GLD · ensemble on dollar bars (vol-specific only).

---

## Workflow (NEVER STOP)

1. Read all `modules/`, `knowledge.json`, `techniques.json`, `results.tsv`.
2. Pick the highest-priority idea from `techniques.json`; cross-check `dead_ends` to avoid retries.
3. Edit one or more of the 4 EFFECTIVE modules (`bar_builder`, `labeler`, `features`, `trainer`).
4. `git add modules/ && git commit -m "<technique>: <description>"`.
5. Run two-phase: train → ObjectStore → infer → REAL Calmar from QC `statistics`.
6. Evaluate against G0–G4.
7. **KEEP** if a target asset improves REAL Calmar (and ideally clears the gate); else **DISCARD** (`git reset --hard HEAD~1`).
8. Append to `results.tsv`; update `knowledge.json` (incl. negatives) and `techniques.json`.
9. If idea queue < 5: mine new ideas (AFML, Wang transcripts, `pdfs/`, arXiv).
10. **Do NOT pause to ask the human. The loop runs until interrupted, period.**

---

## Idea Queue — highest-leverage next experiments

Ordered by expected payoff, given everything above:

1. **Point the harness at the proven-fertile universe.** Track A grinds CORE-7 (trend ETFs) where the ceiling is
   ~1.5; Track B's ≥1.5 results are gold/metals on vol bars. Re-validate GLD/GDX/GDXJ/IAU/SLV here.
2. **Triple-Barrier labeling** (AFML) in `labeler.py` — Track B reached GLD 1.59 with it; never tested in Track A.
3. **Multi-threshold τ-ensemble** (Wang's actual ensemble: 5 models at τ∈{0.5..0.9}, average) — different from the
   current seed-ensemble; implement in `trainer.py`.
4. **Fractional differentiation done right** (d swept 0.3–0.81, ADF-stationary) + **frequency features (VMD+NRBO)**
   in `features.py` — Wang's modules 2 & 4; prior frac-diff attempt failed, but as *additional* (not replacement) features it may add lift.
5. **VAE non-linear dim reduction (K=8–16)** in `trainer.py` as an *alternative* to the correlation filter — Wang's module 5; PCA failed but non-linear may not.
6. **Meta-labeling** (AFML): primary model picks direction, secondary model sizes/vetoes — addresses the
   low-confidence problem behind binary-vs-continuous sizing.

---

## Required Upgrades (harness changes — need human sign-off, they touch LOCKED files)

These are structural blockers the agent cannot fix from `modules/` alone:

1. **Expose volatility bars.** `footer.py.tmpl:20` must sweep/route `"vol"` (per-ticker), or the whole gold/metals
   thesis stays untestable here. Highest-impact single change.
2. **Support pairs/portfolios.** Every ≥2.0 result is a cross-asset pair, but `footer`/`infer` trade ONE symbol.
   To chase the 3.0 gate, the harness needs a 2-symbol spread/portfolio path (Wang's spread-as-time-series).
3. **Unify thresholds.** Make synthetic selection (`trainer.py:31`, flat 0.45) and real execution
   (`infer.py.tmpl:34`, per-ETF) use the SAME per-ETF threshold, or selection optimizes the wrong objective.
4. **Fix stale paths** in `constants.py`/`orchestrator.py` (`/Users/liyuanjun/...` → `/home/ubuntu/lb`).
5. **Reconsider the 3.0 keep-gate** vs the demonstrated single-ETF ceiling (~1.6): either lower it to a level the
   harness can actually reach, or accept that "keep" means only the pairs/upgrade path can ever satisfy it.
6. **Delete or wire up** the dead `calibrator.py`/`consensus.py`/`ensembler.py` to remove the false affordance.

## Next Frontier (needs new infrastructure / data)
Tick data (imbalance/run bars, order-flow) · deep temporal features (LSTM/Transformer embeddings) · adaptive online
retraining (close the train→test gap) · multi-agent research swarm (the community ran 81+ agents self-organizing
into researcher/reviewer/statistician roles — our single loop could scale to one).
