# Autoresearch Round 4 — ⑧ de-tuned active sizing, vol axis, 7 ETFs

| field | value |
|---|---|
| round | 4 |
| date | 2026-06-01 |
| git commit | `ffa1ff5` (sizing) |
| ticker(s) | QQQ, IWM, EEM, XLE, HYG (5/7 completed before pivot to R5) |
| module(s) | ⑧ sizing — de Prado CDF (recentered on τ) × inverse-vol overlay, floor 0.6, trigger 10/60 |
| verdict | **discard the vol-for-all design** — active but all sub-buy-hold; pivot to history-informed R5 |

## 1. Hypothesis
De-tuning R2's over-aggressive sizing (raise `VOL_FLOOR` 0.25→0.6, slower trigger, CDF recentered on τ)
keeps it active (>80 trades) while recovering CAGR, beating buy-hold on QQQ/vol.

## 2. Method
Same `_cdf_bet(p,τ)·_invvol(g)` in `trainer` (VAL) and `infer` (OOS); vol axis for all 7 ETFs; carry +
always_long. (The R2 over-de-lever fix.) Splits train≤2021-08, val≤2023-08, test≤2026-06.

## 3. Results — REAL OOS (vol axis)

| ETF | Calmar | DA | trades | G2 |
|---|---|---|---|---|
| QQQ | 0.90 | 14.9 | 629 | ✓ |
| IWM | 0.65 | 22.6 | 544 | ✓ |
| EEM | 0.57 | 1.57 | 23 | ✗ |
| XLE | 0.11 | 70.3 | 398 | ✓ |
| HYG | (pivoted before recording) | | | |

$$\text{Calmar}=\text{CAGR}/\text{MaxDD},\quad \text{DA}=\textstyle\sum_t(1-E_t/\max_{s\le t}E_s).$$

The de-tune *did* recover QQQ Calmar (R2 0.54 → R4 0.90, 629 trades, DA 14.9 < buy-hold 22.8) — best
active config so far — but **no ETF beats its buy-hold Calmar**, XLE collapses (0.11), EEM under-trades
(τ=0.45 → only 23, fails G2). `carry ≡ always_long` on every ETF (4th confirmation the overlay dominates).

## 4. Verdict & the history-informed pivot — **discard vol-for-all**
Three rounds of *sizing* tweaks (R2 continuous, R3 binary stop, R4 de-tuned) on the **vol axis** never beat
buy-hold. Referencing the archived project history (`_archive/`) explains why — and gives a falsifiable
**hypothesis to confirm, not accept**:

- **Per-ETF axis matters.** Archive `axis_types`: dollar→QQQ/SMH/SOXX/DIA; tick→GLD/GDXJ/XLE; vol→gold
  complex. **Vol is the wrong axis for the equities** (QQQ/IWM) — we forced it on all 7.
- **`carry` FAILS on tech/equity.** Archive: carry QQQ/IWM = Calmar 0, **0 trades** ("works on commodities,
  fails on tech/equity"). **KMeans-2-stage is the universal winner** (archive: QQQ 1.29, HYG 1.61, GLD 1.42)
  and *actively trades*. So "labeler is irrelevant" (R1–R4) was **conditional on the vol axis** — on the
  right axis the labeler should matter.

So the bottleneck may **not** be ⑧ sizing at all; it was ① axis × ② labeler being wrong for the asset.

## 5. Audit fix applied
The sizing audit found the VAL `realistic_cstats` inverse-vol buffer was 1 bar fresher than `infer` (a mild
VAL-only lookahead; REAL OOS was always causal/correct). Fixed (`append log_rets[i-1]` → exact train/infer
parity) + unified the cost dead-band to 0.01. REAL round-4 numbers above are unaffected (they come from infer).

## 6. Next — Round 5 (running, 2-node parallel)
**Confirm the history hypothesis experimentally:** dollar axis × {kmeans2stage, carry, always_long} on all 7
ETFs. Predictions: (a) kmeans2stage on dollar de-saturates equities (active, trades>80) where vol+carry
collapsed; (b) kmeans2stage **beats** carry on QQQ/IWM (labeler matters on the right axis); (c) carry still
fine on commodity-like ETFs. If confirmed → next test tick/vol per-ETF optimal axes. Run on both QC nodes.
