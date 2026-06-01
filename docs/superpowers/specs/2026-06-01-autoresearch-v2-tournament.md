# Autoresearch v2 — Per-ETF-Worst, 2-Hypothesis-Parallel Tournament

**Date:** 2026-06-01
**Supersedes the *loop* of:** `2026-06-01-axis-labeling-experiment-design.md` (that factorial is the v1 full sweep; v2 replaces the round loop with a per-ETF tournament). The build contract (axes, labelers, causal rules, runtime conformance) from that spec is unchanged and assumed.
**Motivation (user):** the 7 ETFs are naturally different, so a single global sweep wastes budget. Six rounds proved single-asset long-only ML ties buy-hold because the *labeler is washed out* — overlay-ON (R5) makes `w≈g` independent of the label; overlay-OFF (R6) reverts to always-long because calibrated `p>tau` nearly everywhere on trending assets. The fix is to make the model **go FLAT sometimes** (binary / higher-threshold sizing, regime labels) so the label actually drives exposure and can add alpha, and to **exploit each asset's nature** (TLT bond = mean-reverting; XLE energy; IWM small-cap). Each round attacks the single weakest ETF with two competing, fully render-time-specified CONFIGs run in parallel on the 2 QC nodes.

---

## 0. The CONFIG contract (no code edits between hypotheses)

A **hypothesis is a CONFIG dict**, injected via the header (alongside `TICKER`/`AXIS`); **no module edits between hypotheses.** The footer builds ONLY `CONFIG['axis']` and runs ONLY `CONFIG['labeler']` as ONE cell (no full sweep) → fast and reproducible.

```python
CONFIG = {"ticker": str, "axis": str, "labeler": str, "thresh": float, "sizing": str}
```

- `axis`    ∈ `bar_builder.AXES` = {dollar, tick, vol, range, logdollar, entropy}
- `labeler` ∈ `labeler.LABELERS` = {kmeans2stage, carry, tertile, bgm, agglomerative, triple_barrier, multi_horizon, hmm, always_long}
- `thresh`  per-ETF entry threshold, float, typically 0.10–0.55
- `sizing`  ∈ {ramp, binary, cdf_plain, cdf_overlay}:

| sizing | rule (`p`=calibrated prob, `tau`=thresh) | character |
|---|---|---|
| `ramp` | `min(1,(p-tau)*200) if p>tau else 0` | legacy; saturates to full-long fast (R1 failure mode) |
| `binary` | `1.0 if p>tau else 0.0` | **most label-responsive: FLAT whenever model says not-up** |
| `cdf_plain` | `clip(2·Φ((p-tau)/√(p(1-p)))-1,0,1) if p>tau else 0` | conviction-scaled, label-driven, no vol overlay |
| `cdf_overlay` | `cdf_plain · clip(std_slow/std_fast, 0.6, 1.0)` | active vol-targeted (R5 family; label can be washed out) |

**Parity invariant (LOCKED behaviour, must hold):** the SAME `sizing`+`thresh` is read from the saved cell payload and applied IDENTICALLY in `trainer.realistic_cstats` (VAL synthetic selection) and in `infer.py.tmpl` (OOS execution). Today both hard-code `_cdf_bet × _invvol_mult` with `VOL_FLOOR=1.0` (= `cdf_plain`); v2 must read `sizing` from the payload and dispatch the four rules in both places via one shared `_size(p, tau, sizing, rbuf)` helper so VAL and OOS never diverge. `std_fast=std(rbuf[-10:])`, `std_slow=std(rbuf[-60:])`, causal (decide-then-append, the existing off-by-one fix). Causality otherwise unchanged: fit on TRAIN only; labels may use the future; features may not.

**Rendering:** footer reads `CONFIG`, builds the single `CONFIG['axis']` series, runs the single `LABELERS[CONFIG['labeler']]`, runs the one fixed downstream cell (StandardScaler → corr-reduce-20 → XGBoost d3 → isotonic), scores VAL via `realistic_cstats(..., thresh=CONFIG['thresh'], sizing=CONFIG['sizing'])`, and saves the cell payload `{ticker,axis,label,thresh,sizing,predictions,synth_cal,val_da,train_auc,val_auc,n_trades}` to `autoresearch/{TICKER}/cell_{axis}_{label}_{sizing}_t{thresh}.json`. Infer loads that exact key and replays with the SAME thresh+sizing. Rendered script stays < 64,000 chars (orchestrator AST-minifies). QC runtime: py3.11, pandas 2.3.3 (`.ffill()`), sklearn 1.6.1, xgboost 3.0.5, hmmlearn 0.3.3.

---

## 1. Per-ETF-best selection rule (rank the 7 ETFs from knowledge.json)

The 7 ETFs are **{QQQ, IWM, EEM, XLE, HYG, TLT, GLD}**. The score is **REAL OOS Calmar of the best DEPLOYABLE config**, where deployable means it actually trades (G2). Buy-hold (1 trade) is the *reference ceiling*, never the per-ETF-best.

### 1.1 What counts as a candidate cell
From `knowledge.json.cells`, a cell is a per-ETF-best candidate iff:
1. it has a numeric `real_calmar`;
2. **G2: `trades > 80`** (active). Cells with `trades ≤ 80` (buy-hold ties at 1, the R3 breaker at 2, EEM_vol_R4detuned at 23) are **excluded from the per-ETF best** — they are recorded as the asset's buy-hold *ceiling reference* only;
3. preferred but not required: G2-passing **and** non-overfit (the cell's saved `aucdiv = |train_auc−val_auc| < 0.05`, gate G4). When two cells tie on Calmar, prefer the G4-passing one, then the lower `real_da`, then more trades.

### 1.2 Ranking
`per_etf_best[etf]` = the candidate cell (per §1.1) with the **highest `real_calmar`**. Rank the 7 ETFs ascending by `per_etf_best[etf].real_calmar`. The **lowest is the round's target (the weakest link).**

### 1.3 Seeding the per_etf_best map from existing cells (as of 2026-06-01)

Computed from `knowledge.json.cells` (active, G2 `trades>80`):

| Rank | ETF | best active Calmar | best cell (axis/sizing) | trades | DA | buy-hold ceiling |
|---|---|---|---|---|---|---|
| 1 | EEM | **1.3279** | EEM_dollar_R5 (cdf_overlay) | 113 | 4.36 | 1.25 |
| 2 | HYG | **1.2625** | HYG_dollar_R5 (cdf_overlay) | 1310 | 4.00 | 1.65 |
| 3 | QQQ | **1.0965** | QQQ_dollar_R5 (cdf_overlay) | 1619 | 20.84 | 1.11 |
| 4 | GLD | **0.7792** | GLD_dollar_R5 (cdf_overlay) | 541 | 10.42 | 1.62 |
| 5 | XLE | **0.7185** | XLE_dollar_R5 (cdf_overlay) | 866 | 39.27 | 0.61 |
| 6 | IWM | **0.6542** | IWM_vol_R4detuned (cdf_overlay) | 544 | 22.62 | 0.56 |
| **7** | **TLT** | **−0.1543** | TLT_dollar_R5 (cdf_overlay) | 1215 | 59.89 | −0.06 |

> Only EEM and XLE currently *beat their own buy-hold ceiling* while trading (EEM 1.33>1.25, XLE 0.72>0.61). Everyone else is still below ceiling — that is the alpha gap each round chases. **TLT is the unique negative; it is the round-1 v2 target.**

Seeding procedure (re-runnable each round): iterate `cells`, bucket by `name.split('_')[0]`, keep only `trades>80`, take argmax `real_calmar` per ETF, store `{calmar, da, trades, cell_name, round}`. Persist as `knowledge.json.per_etf_best`.

---

## 2. Tournament round algorithm

```
ROUND N:
 1. SEED/REFRESH per_etf_best from knowledge.json.cells (§1).
 2. TARGET = argmin_etf per_etf_best[etf].real_calmar       # the weakest link
    cur_best = per_etf_best[TARGET].real_calmar             # bar to beat
    ceiling  = buy_hold_calmar[TARGET]                      # exempt reference
 3. PROPOSE two competing hypotheses H1, H2 for TARGET (full CONFIGs, §0).
    They MUST DIFFER on >=1 structural axis (axis OR labeler OR sizing family),
    be low-DoF (each CONFIG is 5 scalars/names; no new code), and each have a
    written MECHANISM ("why this makes TARGET trade with edge, not churn").
 4. RUN IN PARALLEL on the 2 QC nodes (driver dispatches node0<-H1, node1<-H2):
      per hypothesis: render header(CONFIG) -> ONE train backtest (build axis,
      run labeler, fixed downstream, save cell payload w/ thresh+sizing) -> ONE
      infer backtest (replay w/ same thresh+sizing) -> parse REAL Calmar, DA,
      trades, train_auc, val_auc.  5-min cap each; on timeout cancel+delete via
      QC API, record "timeout".
 5. SCORE each hypothesis on REAL OOS (TEST 2023-08 -> 2026-06):
      valid  = (trades > 80)            # G2; deployable
            and (no-lookahead audit ok) # G3
            and (|train_auc-val_auc|<0.05 OR labeler in {always_long})  # G4
      rank valid hypotheses by Calmar desc, tiebreak lower DA.
    winner = best valid hypothesis (or "none" if both invalid).
 6. KEEP if winner.Calmar > cur_best  (strictly beats TARGET's current best,
      Calmar primary; require DA not materially worse).  Then:
        per_etf_best[TARGET] = winner cell;  write its CONFIG into knowledge.json.
    ELSE keep per_etf_best[TARGET] unchanged (log both as candidates/dead-ends).
 7. LOG BOTH hypotheses (winner and loser) to knowledge.json.cells under keys
    {TARGET}_R{N}_h1 / _h2 with full CONFIG + real metrics + note.
 8. WRITE reports/round_N.html DIRECTLY from reports/TEMPLATE.html (the two
    CONFIGs, mechanisms, the head-to-head metric table, KEEP/REVERT verdict,
    updated per-ETF leaderboard). Add link in reports/index.html. Append results.tsv.
 9. UPDATE knowledge.json: per_etf_best[TARGET], cells[...h1/h2], confirmed/dead_ends.
10. git commit ("autoresearch v2 round N: TARGET H1 vs H2 -> winner").
11. Next round: re-seed -> the NEW weakest link becomes the target (could be the
    same ETF if it is still last, or a different one once TARGET is lifted).
```

**Why this loop, given the 6-round null:** prior rounds varied ONE global sizing on ONE forced axis across all 7 ETFs, so the labeler never mattered and no asset's nature was exploited. v2 (a) targets the asset that most drags the panel, (b) makes the two hypotheses differ in the *label-expression* dimension (binary/high-thresh sizing → the model can be FLAT, breaking the "always-long → buy-hold" trap), and (c) uses both nodes for a clean A/B instead of a broad sweep.

---

## 3. Round 1 target and the two hypotheses

### 3.1 Target: **TLT**, current best active Calmar **−0.1543** (TLT_dollar_R5)

TLT (20+yr Treasury) is the panel's only negative and the textbook **mean-reverting / no-minute-trend** asset (program.md negative control). Over 2023-08→2026-06 rates rose, so even **buy-hold is negative (−0.06)** and the R5 cdf_overlay config churned 1215 trades while staying long the whole down-drift → −0.15 (worse than buy-hold: it paid costs to track a falling asset). The two failure modes to fix: (i) the model is **long almost always** (calibrated `p>tau` everywhere on the dollar axis), so it never harvests the down-legs; (ii) the **vol overlay washes the label** so regime information can't express itself. Both hypotheses therefore use **label-driven `binary` sizing (flat when the model says not-up)** so TLT can SIT OUT or be uninvested through the bear legs — the only way an active long-only config beats a negative buy-hold is to be **flat during the declines and long only the mean-reversion bounces**.

The two hypotheses differ on **axis + labeler** (the two upstream levers never properly tested on TLT), holding the label-responsive `binary` sizing common as the controlled variable that makes the label finally matter.

### 3.2 H1 — "carry the mean-reversion bounce on a vol clock"

```json
{"axis":"vol","labeler":"carry","thresh":0.55,"sizing":"binary"}
```
- **Mechanism:** TLT's tradable edge at the bar scale is mean-reversion: low-forward-vol regimes are when the bond *carries* / bounces, high-vol regimes are the rate-shock down-legs. The **`carry` labeler** explicitly labels long only when forward vol is below the TRAIN median → it teaches the model to be long the calm carry windows and 0 (→ flat under `binary`) in turbulent selloffs. The **`vol` axis** (equal-realized-vol clock) emits bars fast in turbulence and slow in calm, so the model gets dense, well-scaled samples exactly where the regime flips — the right sampling for a vol-driven mean-reverter. **High `thresh=0.55`** + `binary` forces the model to be FLAT unless it is genuinely confident of an up-bar, so TLT spends the rate-rise down-legs in cash (Calmar denominator shrinks) instead of churning long. Beats buy-hold by *not holding* the declines.
- **Why it should add edge, not churn:** `binary` at a high threshold trades only on regime *transitions* (long↔flat), not every wiggle; `carry` ties those transitions to the vol regime that actually drives bond bounces.

### 3.3 H2 — "regime-flat via two-stage kmeans on dollar bars"

```json
{"axis":"dollar","labeler":"kmeans2stage","thresh":0.45,"sizing":"binary"}
```
- **Mechanism:** Same goal (be flat in the down-legs) by a **different, harder upstream route** — a direct *regime* labeler instead of a vol-proxy. **`kmeans2stage`** first clusters bars into low-/high-forward-vol, then within low-vol clusters direction; only the low-vol up-cluster → 1, everything else → 0 (→ flat under `binary`). On TLT this should isolate the brief mean-reversion-up regimes and abstain from the trending-down rate-shock regime. The **`dollar` axis** is TLT's most-liquid native sampling (and the R5 axis that produced the current best), so this is a clean apples-to-apples test of "does a *regime label + flat sizing* beat the *label-independent overlay* on the SAME axis?" **`thresh=0.45`** (neutral) + `binary` lets the kmeans regime label, not a hand-tuned threshold, decide flatness.
- **Why it should add edge, not churn:** the two-stage regime label is sticky (cluster membership persists), so `binary` flips long↔flat only at regime boundaries → bounded turnover, and the flat periods are precisely the down-legs that sink buy-hold.

### 3.4 Why H1 and H2 genuinely differ (controlled A/B)
- **Common (controlled):** `sizing=binary` (label-driven, can be flat — the v2 thesis), single-asset TLT, fixed downstream.
- **Differ (the test):** axis (`vol` clock vs `dollar` liquidity) **and** labeler (`carry` vol-proxy vs `kmeans2stage` explicit two-stage regime) **and** thresh (0.55 conviction-gate vs 0.45 neutral). So the round answers two questions at once: *which axis suits TLT* and *which label family expresses the mean-reversion-flat regime* — while both share the mechanism (flat through the rate-rise down-legs) that is the only way to beat a negative buy-hold. If both beat −0.15 we learn `binary`+flat is the lever; if one wins we learn the axis/labeler that fits a bond.

---

## 4. Risks / guards
- **Multiplicity:** two hypotheses/round on one ETF is low multiplicity, but a single OOS run is still an order statistic — only KEEP on a strict Calmar beat and confirm a kept winner with seed 7 before promoting from *candidate* to *confirmed* (program.md replication rule).
- **TLT is the negative control:** a *large* unexplained jump on TLT is a red flag for leak/overfit, not a triumph. Beating a −0.06 buy-hold to a small positive by being flat in the declines is the credible, mechanism-backed win; a > +1 Calmar would warrant a lookahead re-audit.
- **G2 under `binary` at high thresh:** if `thresh=0.55` makes the model flat so often that trades ≤ 80, H1 fails G2; the fallback within the round is to lower H1 thresh to 0.50 (still conviction-gated) — recorded, not a silent change.
- **Parity:** the single `_size()` helper must back both `realistic_cstats` and `infer`; a divergence reintroduces the historical train/infer mismatch and invalidates the round.
