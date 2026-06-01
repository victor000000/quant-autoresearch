# Autoresearch Round 1 — LANDSCAPE + BASELINES (vol/dollar × carry/kmeans2stage/always_long/hmm on GLD & QQQ)

| field | value |
|---|---|
| round | 1 |
| date | 2026-06-01 |
| git commit | `807a412` (pre-commit) |
| ticker(s) | GLD, QQQ |
| module(s) under test | ① Custom Axis · ② Unsupervised Labeling (diagnostic only; no module edited) |
| verdict | **baseline_tie** (diagnostic: every active labeler collapsed to buy-and-hold) |

> Round 1 is a landscape map, not a feature change. The "idea" is the diagnostic
> itself: find where there is ACTIVE tradable signal (trades >> 1) and establish
> the always_long / hmm baselines per asset.

---

## 1. Hypothesis

Mapping hypothesis: across the `vol` and `dollar` axes, at least one featured
labeler (carry, kmeans2stage) or the hmm baseline will produce an *actively
trading* cell (trades >> 1) on GLD and/or QQQ that beats the always_long
buy-and-hold floor on REAL OOS Calmar/DA. Where everything collapses to 1 trade,
locate the stage responsible so rounds 2+ attack the right module.

## 2. Method

**① Axes tested — `vol` and `dollar`.**
- vol: $\sum_i (\Delta\log p_i)^2\sqrt{v_i}\ge\theta$ (realized-variance clock, Wang's workhorse).
- dollar: $\sum_i p_i v_i \ge \theta$ (equal-notional clock).
- $\theta$ auto-calibrated to $\approx$ TARGET_BARS via $\theta=\text{total}/\text{target\_bars}$.

**② Labelers — carry, kmeans2stage (featured); always_long, hmm (baselines).** Each
trains the FIXED downstream (StandardScaler → corr-filter K=20 → XGBoost d3/η0.03/n200,
`scale_pos_weight` → isotonic on VAL), saves TEST predictions, then a separate infer
backtest replays them with real `SetHoldings`.

**⑧ Sizing (held fixed, the suspect).** Infer rule per bar:
$$w_t = \min\!\big(1,\,(p_t-\tau)\cdot 200\big)\ \text{if}\ p_t>\tau,\ \text{else liquidate},\qquad \tau_{\text{GLD}}=0.35,\ \tau_{\text{QQQ}}=0.15.$$

**Splits.** train ≤ 2021-08, val ≤ 2023-08, test ≤ 2026-06 (OOS).

## 3. Configuration
```
axes={vol,dollar} × labelers={carry,kmeans2stage,always_long,hmm} × tickers={GLD,QQQ}
= 16 infer cells, 4 train runs. seeds=1. thresholds A PRIORI from _cell_threshold().
n_cells per train run = 7 (GLD vol), 9 (GLD dollar), 9 (QQQ vol), 5 (QQQ dollar).
```

## 4. Results — REAL OOS (QC SetHoldings)

$$\text{Calmar}=\frac{\text{CAGR}}{\text{MaxDD}},\qquad \text{DA}=\sum_t\Big(1-\frac{E_t}{\max_{s\le t}E_s}\Big)\ (\text{lower better}).$$

| ticker | axis | label | REAL Calmar | DA (OOS) | trades | status |
|---|---|---|---|---|---|---|
| GLD | vol | carry | 1.6165 | 24.021 | 1 | completed |
| GLD | vol | kmeans2stage | 1.6165 | 24.021 | 1 | completed |
| GLD | vol | **always_long** | 1.6165 | 24.021 | 1 | baseline |
| GLD | vol | **hmm** | 1.6165 | 24.021 | 1 | baseline |
| GLD | dollar | carry | 1.6200 | 23.993 | 1 | completed |
| GLD | dollar | kmeans2stage | 1.6200 | 23.993 | 1 | completed |
| GLD | dollar | **always_long** | 1.6200 | 23.993 | 1 | baseline |
| GLD | dollar | **hmm** | 1.6200 | 23.993 | 1 | baseline |
| QQQ | vol | carry | 1.1099 | 22.830 | 1 | completed |
| QQQ | vol | kmeans2stage | 1.1099 | 22.830 | 1 | completed |
| QQQ | vol | **always_long** | 1.1099 | 22.830 | 1 | baseline |
| QQQ | vol | **hmm** | 1.1099 | 22.830 | 1 | baseline |
| QQQ | dollar | carry | 1.1094 | 22.873 | 1 | completed |
| QQQ | dollar | kmeans2stage | 1.1094 | 22.873 | 1 | completed |
| QQQ | dollar | **always_long** | 1.1094 | 22.873 | 1 | baseline |
| QQQ | dollar | **hmm** | 1.1094 | 22.873 | 1 | baseline |

**Baselines established (per asset, both axes ≈ identical):**
- GLD always_long: Calmar **≈ 1.62**, DA **≈ 24.0**, 1 trade (buy-and-hold).
- QQQ always_long: Calmar **≈ 1.11**, DA **≈ 22.8**, 1 trade (buy-and-hold).
- hmm == always_long on both assets (no separation).

Gate checks: G1 Calmar>3.0 FAIL (max 1.62). G2 trades>80 FAIL (all = 1). G3 lookahead 0 (clean). G4 AUCdiv — the *reported* train/val AUC are the always_long-selected cell's 0.5/0.5, not per-featured-cell; not diagnostic this round.

## 5. Verdict & interpretation — **baseline_tie**, with a precise root cause

Every featured labeler AND hmm collapse to **exactly** the always_long result
(same Calmar to 4 dp, same DA, 1 trade). Crucially, `n_cells = 5–9` per train run
proves the featured labelers DID fit and save real cells with non-degenerate
predictions — the collapse is **not** a labeling failure.

**Root cause is ⑧ sizing/threshold, not ① axis or ② labels.** The infer rule
$w_t=\min(1,(p_t-\tau)\cdot200)$ saturates to full long the instant $p_t>\tau+0.005$
(since $0.005\cdot200=1$). Isotonic-calibrated XGBoost probabilities on these
drift-heavy ETFs sit far above $\tau$ ($\tau=0.35$ GLD, $0.15$ QQQ) for essentially
the whole OOS window, so the position pins at 100% long, never crosses back below
$\tau$ → **1 trade = buy-and-hold**, identically for every cell. The ramp slope 200
with a low fixed $\tau$ makes the strategy structurally always-on; the labeler can
never express itself.

This is the most valuable Round-1 finding: **chasing ① axes or ② labelers is futile
until ⑧ stops saturating.** The frontier worth attacking is the sizing/gate.

- Multiplicity: 16 cells; no winner, so no multiplicity concern. Replicated across
  2 tickers × 2 axes — the collapse is robust, not a fluke.
- Negative control note: TLT not run this round (deferred to once active signal exists).

## 6. Next (Round 2 idea)

**Attack ⑧ sizing in `trainer.realistic_cstats` + the infer ramp so the calibrated
probability genuinely modulates exposure.** Two low-DoF, a-priori options:
1. **De-saturate the ramp**: replace $\min(1,(p-\tau)\cdot200)$ with a de Prado-style
   bounded map $w=\,$ sign$\cdot(2\Phi(z)-1)$, $z=(p-0.5)/\sqrt{p(1-p)}$, so $w<1$
   unless conviction is genuinely extreme → forces partial/exit positions → trades>1.
2. **Raise the entry threshold into the probability mass** (a-priori, e.g. the VAL
   median of $p$) so low-conviction bars go flat — a binary trend-vs-stand-aside gate
   with a wide neutral dead-zone (catalog ⓪/⑧). Target: trades >> 1 AND DA below the
   always_long DA, on QQQ first (drift-heavy, more directional signal than GLD).

Best (asset, axis) frontier for rounds 2+: **QQQ on the `vol` axis** (most bars, clear
drift, lowest buy-hold Calmar so most headroom for drawdown control to add Calmar),
with GLD `vol` as the replication check.
