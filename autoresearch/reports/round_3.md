# Autoresearch Round 3 — ⑧ stay-long drawdown circuit-breaker on QQQ (vol axis)

| field | value |
|---|---|
| round | 3 |
| date | 2026-06-01 |
| git commit | `dfe1dea` (module edit) → reverted this round |
| ticker(s) | QQQ (frontier from R1/R2) |
| module(s) under test | ⑧ Sizing/Risk — binary realized-drawdown stop (`trainer.realistic_cstats` + `infer.py.tmpl`) |
| verdict | **discard** (worse on BOTH Calmar and DA; also fails G2) — reverted |

> Run live in the main session. R2 showed a continuous inverse-vol overlay over-de-levers a drift asset
> (CAGR fell faster than MaxDD). R3 tests the opposite shape: hold full exposure by default, step flat
> **only** in a genuine bear leg.

---

## 1. Hypothesis
On QQQ (vol axis), a binary **stay-long-by-default** drawdown circuit-breaker — hold the model's long, step
**flat** once the strategy's own realized drawdown exceeds `DD_CUT`, re-enter once back within `DD_RESUME`
of the running peak (hysteresis) — will truncate the deep drawdowns that set MaxDD/DA **while leaving
drift-period exposure (CAGR) intact**, raising Calmar above the buy-and-hold ceiling and cutting DA.

Falsifiable failure: on a V-shaped-recovery asset the stop sells the dip and the hysteresis re-enters only
near new highs → flat through the rebound → CAGR *and* DA both worsen. **Outcome: this failure.**

## 2. Method
Per bar, model weight $w^{\text{raw}}_t=\min\!\big(1,(p_t-\tau)\cdot 200\big)$ if $p_t>\tau$ else $0$.
Overlay a causal drawdown state on the strategy's own equity $E_t$ (peak $P_t=\max_{s\le t}E_s$,
underwater $d_t=1-E_t/P_t$):

$$\text{derisk}_t=\begin{cases}\text{false} & d_t\le \text{DD\_RESUME}\ (\text{0.04})\\ \text{true} & d_t\ge \text{DD\_CUT}\ (\text{0.12})\\ \text{derisk}_{t-1}&\text{otherwise}\end{cases}\qquad w_t=\begin{cases}0 & \text{derisk}_t\\ w^{\text{raw}}_t&\text{else}\end{cases}$$

Thresholds **fixed a priori** (0.12 / 0.04), **identical** in `realistic_cstats` (VAL) and `infer.py.tmpl`
(OOS). Causal: uses only realized equity up to $t$. ①② held fixed (vol axis, carry + always_long).
Splits: train ≤ 2021-08, val ≤ 2023-08, test ≤ 2026-06.

## 3. Configuration
```
QQQ × vol × {carry, always_long} | thresh=0.15 | DD_CUT=0.12 DD_RESUME=0.04 | seeds=1 | n_cells=9
```

## 4. Results — REAL OOS (QC SetHoldings)

$$\text{Calmar}=\frac{\text{CAGR}}{\text{MaxDD}},\qquad \text{DA}=\sum_t\Big(1-\frac{E_t}{\max_{s\le t}E_s}\Big)\ (\text{lower better}).$$

VAL diagnostic: synth_cal **−0.1566** (the consistent VAL metric again pre-warned of OOS underperformance).

| ticker | axis | label | REAL Calmar | DA (OOS) | trades | G2 (>80)? |
|---|---|---|---|---|---|---|
| QQQ | vol | carry + breaker | **0.2605** | **76.54** | 2 | ✗ FAIL |
| QQQ | vol | always_long + breaker | 0.2605 | 76.54 | 2 | ✗ FAIL |
| QQQ | vol | *buy-and-hold (R1, baseline/ceiling, G2-exempt)* | 1.1099 | 22.83 | 1 | — |
| QQQ | vol | *R2 CDF×inv-vol (reference)* | 0.5387 | 7.07 | 431 | ✓ |

Gates: **G1 FAIL** (0.26). **G2 FAIL** (2 trades — the breaker barely acts, then sits flat). G3 pass (causal,
audited). G4 n/a (selected cell is constant-long, AUC 0.5).

## 5. Verdict & interpretation — **discard** (decisive negative result)
The breaker made **everything worse**: Calmar 1.11 → **0.26**, DA 22.83 → **76.54 (3× worse)**. Mechanism:
QQQ drifts up with V-shaped recoveries; stepping flat at −12% **sells the dip**, and re-entering only within
4% of the peak keeps the strategy **flat through the entire rebound**. Equity therefore stays underwater far
longer → DA explodes; and the missed recovery collapses CAGR → Calmar craters. **Binary drawdown stops are
pro-cyclical on mean-reverting-up assets.** `carry ≡ always_long` once more — the overlay dominates, the
labeler is irrelevant (third independent confirmation that ⑧, not ②, is the lever here).

**Reframe (G2 = trades > 80, buy-hold exempt):** buy-and-hold (1 trade) and this breaker (2 trades) are
**not deployable** regardless of Calmar — only an *actively* trading config can pass. So the only family that
clears G2 so far is **R2's de-saturated active sizing (431 trades)**; the task is to keep that activity and
CAGR while trimming turbulence.

- Multiplicity: 2 cells, 1 run; the carry≡always_long identity is a mechanism replication, not a Calmar win.
- The two de-risking shapes tried (R2 continuous, R3 binary stop) **both** fail on drift assets → mark the
  whole "reduce exposure to beat buy-hold on a drift asset" family a dead end.

## 6. Next (Round 4)
Return to **R2's active CDF×vol sizing** (it clears G2 with 431 trades and cuts DA −69%) and **de-tune it to
preserve CAGR**, exactly as R2 prescribed: raise `VOL_FLOOR` 0.25 → 0.6 and trigger only on large spikes
(`clip(σ_slow/σ_fast, 0.6, 1.0)`, fast=10/slow=60); recenter the CDF on $\tau$ (conviction
$c=(p-\tau)/(1-\tau)$) so the low per-ETF $\tau$ keeps drift assets long. Target QQQ vol: **trades > 80 AND
Calmar > 1.11 AND DA < 22.83**. Pre-screen with the now-informative VAL synth_cal before spending a backtest.
