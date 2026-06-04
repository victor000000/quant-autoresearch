# Autoresearch Round {ROUND} — {AXIS} × {LABELER} on {TICKER}

| field | value |
|---|---|
| round | {ROUND} |
| date | {YYYY-MM-DD} |
| git commit | `{SHA}` |
| ticker(s) | {TICKER} |
| module(s) under test | ① Custom Axis · ② Unsupervised Labeling |
| verdict | **{keep \| discard \| leak \| overfit \| timeout \| crash}** |

> One report per round, committed to git (`autoresearch/reports/round_{ROUND}.md`). Math is written
> in `$…$`/`$$…$$` so it renders; every formula is paired with a plain-language description.

---

## 1. Hypothesis
*Plain-language, falsifiable.* e.g. "On {TICKER}, an information-driven **{AXIS}** axis lowers return
kurtosis versus tick bars, and **{LABELER}** regime labels produce a tradable signal, so REAL OOS
Calmar exceeds the current best of {PRIOR_BEST} and the always-long baseline."

Wang module(s) exercised, and whether this is a **featured** method or a **baseline** comparator.

## 2. Method

**① Custom axis — {AXIS}.** Sampling rule (one bar emitted when the accumulator crosses threshold $\theta$, with $\theta$ chosen on TRAIN data to yield $\approx$ {TARGET_BARS} bars):

- dollar: $\sum_i p_i v_i \ge \theta$  (description: sample per fixed traded *dollar* volume)
- vol: $\sum_i (\Delta \log p_i)^2 \sqrt{v_i} \ge \theta$  (sample per fixed *realized-variance* budget — Wang's 3rd axis)
- logdollar: $\sum_i \log(1 + p_i v_i) \ge \theta$  (compresses heavy-tailed dollar volume)
- entropy (information-driven): $\sum_i \big[-\sum_k \hat p_k \log \hat p_k\big] \ge \theta$ over the return-bucket distribution since the last bar (sample per fixed *information/surprise*)
- tick: $\text{count} \ge \theta$ · range: $|p_t - p_{\text{last}}|/p_{\text{last}} \ge \theta\%$

**② Unsupervised labeling — {LABELER}.** Label $y_t \in \{-1\text{ (ignore)}, 0, 1\}$ from forward
metrics $r^{f}_{t}=\log p_{t+h}-\log p_t$ and $\sigma^{f}_{t}=\mathrm{std}(r_{t+1:t+h})$. *All clustering /
thresholds fit on TRAIN only.* e.g.

- carry: $y_t = 1 \iff \sigma^{f}_{t} \le \mathrm{median}_{\text{train}}(\sigma^{f})$
- triple-barrier: barriers $\{+u\sigma_t,\,-l\sigma_t,\,H\text{ bars}\}$; $y_t=1$ if the upper barrier is hit first
- kmeans2stage / bgm / agglomerative: cluster $[\,r^{f}_t,\,|r^{f}_t|\,]$ within the low-$\sigma^f$ regime; up-cluster $\to 1$
- *(baseline)* hmm: 3-state GaussianHMM on $[r_t,|r_t|]$; always-long: $y_t\equiv1$

**Downstream (HELD FIXED — controls for the ①×② effect):** 80 features → correlation filter to 20
($|\rho|>0.90$ drop, top-20 by variance) → XGBoost ($\text{depth}=3,\ \eta=0.03,\ n=200$,
`scale_pos_weight`$=N_-/N_+$) → isotonic calibration → threshold $\tau_{\text{{TICKER}}}$ (same value
used in training selection and live execution).

**Splits.** train $\le$ {TRAIN_END}, val $\le$ {VAL_END}, test $\le$ {TEST_END} (OOS).

## 3. Configuration
```
{cfg_string}   e.g. vol_carry_f100_corr20_iso_ma50  | thresh=0.35 | seeds=1
```

## 4. Results — phase-appropriate metrics

Each backtest is hard-capped at **5 minutes**; if exceeded it is cancelled/deleted via the QC Cloud API
(`delete_backtest`) and the cell is recorded as `timeout`. Definitions:

$$\text{Calmar} = \frac{\text{CAGR}}{\text{MaxDD}},\qquad
d_t = 1 - \frac{E_t}{\max_{s\le t} E_s},\qquad
\text{DA} = \sum_{t} d_t,\qquad
\text{AUCdiv} = \lvert \text{AUC}_{\text{train}} - \text{AUC}_{\text{val}} \rvert$$

where $E_t$ is the equity curve, $d_t\in[0,1]$ the underwater fraction, and **DA (Drawdown Area)** the area
under the underwater curve — *lower is better* (captures depth × duration, not just the worst point).

**TRAIN — fit diagnostics ("middle" metrics, not for selection):**

| metric | value |
|---|---|
| AUC train | {tauc} |
| label balance | {balance} |

**VAL — model/config selection ("middle" metrics; pick min DA s.t. AUCdiv < 0.05 — Wang uses DA):**

| metric | value | role |
|---|---|---|
| **DA (val)** | {da_val} | **primary selector (minimize)** |
| AUC divergence | {aucdiv} | overfit guard (G4 `<0.05`) |
| synthetic Calmar (val) | {synth_cal} | secondary (within-cell only; NOT comparable across axes) |

**TEST / OOS — FINAL reported metrics (REAL QC `SetHoldings` backtest):**

| metric | value | gate |
|---|---|---|
| **REAL Calmar (OOS)** | {real_calmar} | G1 `>3.0` → {pass/fail} |
| **DA (OOS)** | {da_test} | lower = better |
| CAGR / MaxDD | {cagr}% / {mdd}% | — |
| trades (orders) | {trades} | G2 `>80` → {pass/fail} |
| lookahead audit | {violations} | G3 `0` → {pass/fail} |
| AUC divergence | {aucdiv} | G4 `<0.05` → {pass/fail} |

*Note:* synthetic Calmar is used only for within-axis internal checks; **cross-axis comparison and the final
verdict use REAL test-phase Calmar + DA only.**

## 5. Verdict & interpretation
- **{KEEP/DISCARD}** because {reason}. Lift vs always-long baseline: {Δ}. Lift vs HMM baseline: {Δ}.
- Multiplicity: this is cell {k}/{K} of the sweep — treat as *confirmed* only if it replicates across
  ≥2 tickers or ≥2 seeds.

## 6. Next
Follow-up hypothesis / parameter to vary next round, and which `techniques.json` item it advances.
