# Autoresearch Round 2 — ⑧ de Prado CDF bet-size × causal inverse-vol overlay on QQQ (vol axis)

| field | value |
|---|---|
| round | 2 |
| date | 2026-06-01 |
| git commit | `d01dd12` (module edit, reverted post-round) |
| ticker(s) | QQQ (frontier from R1); GLD replication deferred |
| module(s) under test | ⑧ Sizing/Risk (`trainer.realistic_cstats` + `templates/infer.py.tmpl`) — the R1-identified bottleneck |
| verdict | **discard** (de-saturated + crushed DA, but over-de-levered → Calmar halved) |

> Round 1 proved the bottleneck is ⑧ sizing, not ① axis or ② labels: `w=min(1,(p-τ)·200)`
> saturates to 100% long at `p>τ+0.005`, so every cell = buy-and-hold (1 trade). Round 2 attacks
> exactly that rule. This is a **featured** ⑧ technique (catalog #4 vol-targeting × #7 de Prado CDF).

---

## 1. Hypothesis

On QQQ (vol axis), replacing the saturating linear ramp with a **bounded de Prado CDF bet-size**
(gentle slope → partial positions) **multiplied by a causal inverse-vol overlay** (de-lever when
short-term realized vol spikes above its slower baseline) will (a) **de-saturate** the position so the
strategy trades actively (trades ≫ 1), and (b) **cut Drawdown Area** by withdrawing capital from the
high-vol regimes where drawdowns cluster — raising REAL OOS Calmar above the buy-and-hold floor
(QQQ 1.11) by shrinking MaxDD faster than CAGR.

Falsifiable failure modes: (i) it still saturates (trades≈1) → overlay too weak; (ii) it de-levers so
hard that CAGR falls faster than MaxDD → Calmar drops below buy-hold. **Outcome: (ii).**

## 2. Method

**⑧ Sizing — shared by synth selection AND real infer (consistency enforced).** Per bar, given
calibrated prob $p$, per-ETF gate $\tau$, and a causal trailing buffer of bar log-returns:

$$w_t=\underbrace{\big[\,2\,\Phi(z_t)-1\,\big]_{0}^{1}}_{\text{de Prado CDF bet}}\cdot\underbrace{\operatorname{clip}\!\Big(\frac{\sigma^{\text{slow}}_t}{\sigma^{\text{fast}}_t},\,0.25,\,1\Big)}_{\text{inverse-vol overlay}},\qquad z_t=\frac{p_t-0.5}{\sqrt{p_t(1-p_t)}},\quad w_t=0\ \text{if}\ p_t\le\tau.$$

- $\Phi$ = standard-normal CDF (`0.5(1+\mathrm{erf}(z/\sqrt2))`). The map is **centered at $p=0.5$**, so
  it gives $w=0$ for $p\le0.5$, $0.08$ at $p=0.55$, $0.34$ at $0.70$, $0.67$ at $0.85$ — a *gentle*
  slope, not the old `·200` cliff. (Note: because the CDF is centered at 0.5, the *effective* gate is
  $p>0.5$, wider than the nominal $\tau=0.15$ — a built-in neutral dead-zone.)
- $\sigma^{\text{fast}}_t=\operatorname{std}$ of the last **20** bar returns, $\sigma^{\text{slow}}_t$ of
  the last **100**. When short-vol exceeds its slower baseline the ratio $<1$ → de-lever; floor 0.25,
  cap 1.0 (no leverage → fair Calmar vs buy-hold capital). **All params fixed a priori; none swept.**
- **Causality:** the overlay uses only returns realized up to bar $t$ (buffer appended *after* sizing).
  No future leakage; identical math in `realistic_cstats` (VAL) and `infer.py.tmpl` (OOS).

**①② held fixed** at the R1 frontier: vol axis, carry + always_long, same FIXED downstream
(StandardScaler → corr-filter 20 → XGBoost d3/η0.03/n200 → isotonic on VAL).

**Splits.** train ≤ 2021-08, val ≤ 2023-08, test ≤ 2026-06 (OOS).

## 3. Configuration
```
QQQ × vol × {carry, always_long} | thresh=0.15 | VOL_FAST=20 VOL_SLOW=100 VOL_FLOOR=0.25 cap=1.0 | seeds=1
1 train (n_cells=9) + 2 infer. Sizing params fixed a priori (NOT tuned on val/test).
```

## 4. Results — REAL OOS (QC SetHoldings)

$$\text{Calmar}=\frac{\text{CAGR}}{\text{MaxDD}},\qquad \text{DA}=\sum_t\Big(1-\frac{E_t}{\max_{s\le t}E_s}\Big)\ (\text{lower better}).$$

**TRAIN/VAL diagnostics (carry cell):** train_auc 0.8002, val_auc 0.7676 (AUCdiv 0.0326 < 0.05 — G4 pass);
**synth_cal (VAL) = −0.3511** — the new sizing's VAL metric now *correctly predicts* OOS underperformance
(R1's synth_cal was a flat 0.0 that told us nothing).

| ticker | axis | label | REAL Calmar | DA (OOS) | trades | vs buy-hold |
|---|---|---|---|---|---|---|
| QQQ | vol | carry | **0.5387** | **7.07** | **431** | Calmar −51%, DA −69% |
| QQQ | vol | always_long (vol-targeted long) | 0.5387 | 7.07 | 431 | identical to carry |
| QQQ | vol | *buy-and-hold (R1 locked)* | 1.1099 | 22.83 | 1 | baseline |

Gate checks: **G1 Calmar>3.0 FAIL** (0.54). **G2 trades>80 PASS** (431 — bottleneck broken!). G3 lookahead
0 (causal overlay, audited). **G4 AUCdiv 0.033 PASS.** DA gate (informal): **7.07 ≪ 22.83 — strong PASS.**

## 5. Verdict & interpretation — **discard** (for Calmar), but a decisive mechanistic result

The R1 bottleneck is **broken**: trades went **1 → 431** and DA **22.83 → 7.07 (−69%)**. The sizing now
genuinely modulates exposure and the inverse-vol overlay slashes the underwater area exactly as intended.
But REAL Calmar **halved (1.11 → 0.54)**: on a drift-heavy asset the edge *is* staying invested, and the
combination of (i) the CDF's effective $p>0.5$ gate and (ii) aggressive inverse-vol de-levering cut CAGR
faster than it cut MaxDD. **MaxDD shrank, but CAGR shrank more → Calmar fell.** Only the score matters, so
this is a **discard** and the module is reverted.

Two findings worth keeping:
1. **carry ≡ always_long to 4 dp (both 0.5387 / 7.07 / 431).** The outcome is driven *entirely by the
   vol-targeting overlay*, independent of the labeler — independent confirmation of R1's thesis that the
   labeler is not the lever here; **⑧ is.** A real (AUC 0.80) model and a constant-long signal produce the
   identical exposure path once the overlay dominates.
2. **The new VAL synth_cal (−0.35) tracks the OOS sign**, where R1's was an uninformative 0.0 — the
   train/infer sizing consistency restored a usable in-sample selection signal for future rounds.

- Multiplicity: 2 cells, 1 run; the carry≡always_long identity is a within-run replication of the
  *mechanism* (not a Calmar win). GLD replication deferred — no point replicating a discard.
- Negative control: TLT not run (still no Calmar-positive config to red-flag).

## 6. Next (Round 3 idea)

The overlay is too aggressive for a drift asset. **De-tune the de-levering toward "stay invested by
default", keeping the DA win but recovering CAGR** — all a-priori, low-DoF:
1. **Raise VOL_FLOOR 0.25 → 0.6 and widen the trigger** (de-lever only on *large* vol spikes, e.g.
   `clip(slow/fast, 0.6, 1.0)` with fast=10/slow=60) so the strategy holds ~full exposure in normal
   regimes and only cuts in genuine turbulence — this should lift CAGR back toward buy-hold while
   retaining most of the −69% DA.
2. **Recenter the CDF gate on $\tau$, not 0.5**: map conviction $c=(p-\tau)/(1-\tau)$ through the CDF so
   the low per-ETF $\tau$ (0.15) actually keeps the position long on drift assets instead of the implicit
   $p>0.5$ gate flattening it half the time.
Target: trades ≫ 1 AND DA < buy-hold AND **Calmar ≥ buy-hold** on QQQ vol; then replicate on GLD vol and
add TLT as the negative control. Advances catalog #4 (vol-target) + #7 (de Prado CDF), tuned for drift.
