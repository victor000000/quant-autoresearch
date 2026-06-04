# Autoresearch Round 5 — ① dollar axis × ② {kmeans2stage, carry, always_long}, 7 ETFs (2-node parallel)

| field | value |
|---|---|
| round | 5 |
| date | 2026-06-01 |
| ticker(s) | QQQ, IWM, EEM, XLE, HYG, TLT, GLD |
| under test | history hypothesis: dollar axis + kmeans2stage de-saturates equities and **beats carry** |
| verdict | **history hypothesis NOT confirmed** — labeler still washed out; but found WHY (the sizing overlay), + best active configs yet |

## 1. Hypothesis (from archived history, to *confirm* not accept)
Archive: dollar axis suits equities; `carry` fails on QQQ/IWM while `kmeans2stage` is the universal *active*
winner. Predictions: (a) kmeans2stage on dollar trades > 80 where vol+carry gave 1; (b) **kmeans2stage beats
carry on QQQ/IWM**; (c) carry fine on commodity-like ETFs.

## 2. Method
Dollar axis (held constant) × {kmeans2stage, carry, always_long}, all 7 ETFs, current de-tuned sizing
`w = _cdf_bet(p,τ)·_invvol(g)` (floor 0.6). 2-node parallel driver. Splits as standard.

## 3. Results — REAL OOS

| ETF | Calmar | DA | trades |
|---|---|---|---|
| EEM | **1.328** | 4.36 | 113 |
| HYG | **1.263** | 4.00 | 1310 |
| QQQ | 1.097 | 20.84 | 1619 |
| GLD | 0.779 | 10.42 | 541 |
| XLE | 0.719 | 39.27 | 866 |
| IWM | 0.132 | 22.42 | 854 |
| TLT | −0.154 | 59.89 | 1215 |

**All seven ETFs: kmeans2stage = carry = always_long, identical to 4 dp** (Calmar, DA, trades).

## 4. Verdict & the real finding — **it's the overlay, not the axis**
(a) is confirmed (dollar de-saturates: trades 113–1619, all clear G2). **(b) is FALSE** — kmeans2stage does
**not** beat carry; they're *identical*, and identical to constant-long. The mechanism, now clear:

> The **inverse-vol overlay `g = clip(σ_slow/σ_fast, 0.6, 1)` is label-independent** and drives the entire
> rebalance path. On these (mostly up-trending) ETFs the calibrated model probability is usually high, so
> `_cdf_bet(p,τ) ≈ 1` for *every* labeler — even an AUC-0.88 model (IWM) maps to ≈ full long. Thus
> `w ≈ g`, the same path regardless of the label. **The overlay washes the labeler out.**

So "labeler is irrelevant" (rounds 1–5) is an **artifact of the sizing**, not a property of labels or axes.
The history hypothesis was never actually testable while the overlay dominates.

**Silver lining:** EEM (1.33, DA 4.4, 113 trades) and HYG (1.26, DA 4.0, 1310 trades) are the **best *active*
(G2-passing) configs in the study so far** — dollar axis genuinely helps them vs vol. Still far from G1 (>3.0).
TLT −0.15 (negative control, as expected).

## 5. Next — Round 6 (running): turn the overlay OFF
To let the label finally express itself, **disable the inverse-vol overlay** (`VOL_FLOOR = 1.0` ⇒ `g ≡ 1`) so
`w = _cdf_bet(p,τ)` — i.e. the position responds to the *model's probability* and goes flat when `p ≤ τ`.
Then `always_long` reverts to ~buy-hold (1 trade), and a real labeler (kmeans2stage) that sometimes predicts
down should **diverge** from it. Same dollar axis (isolates the overlay on→off change). If kmeans2stage now
≠ always_long and trades > 80 with Calmar > buy-hold, the label matters and the history hypothesis is finally
testable. (Identical sizing in trainer VAL + infer OOS, per the audit-fixed parity.)
