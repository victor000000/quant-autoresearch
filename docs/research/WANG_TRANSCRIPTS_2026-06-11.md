# Wang Yiming transcripts (2026-06-11) — deep read + performance levers

Two new Wang Yiming (DCP co-founder, ex-Knight Capital) lectures were transcribed
(faster-whisper `small`; proper nouns garbled but technical substance intact) and
filed at `refs/uni_yt/`:

- **BV1VZLi6PE2B** — 《机器学习构建量化模型的课程答疑》 (ML-quant course Q&A, 51m)
- **BV1PQEQ6NETz** — 《金融数据联合概率分布的研究》 (joint probability distribution of
  financial data, 66m)

**Headline:** the transcripts *corroborate* our existing pipeline far more than they
extend it — info-density event-clock, trend-segmentation labels, β200 long-only
routing, HMM weakness, strict OOS, "data processing is the ceiling" all match what
we already do/know. There is **no new mechanism**. But three genuinely new, in-rule,
non-closed levers surfaced; one is already implemented here.

---

## 1. What each lecture actually teaches

### BV1PQEQ6NETz — joint probability distribution
A conceptual lecture (ends in a course pitch) with a real technical spine, demoed on
**a single index (CSI 300) using two of its OWN variables** (daily return × 20-day
realised vol) — i.e. natively single-ticker:

1. Estimate the **joint density** of (return, vol) via **Gaussian KDE**.
2. Collapse to **marginals** (integrate one out) and, the headline, **conditionals**
   P(return | vol-bucket) — the state-dependent return distribution.
3. Discretise the joint plane into **quadrant regime labels** (vol hi/lo × return
   up/down → 4/6/8 states): a non-parametric, unsupervised regime *labeler*.
4. Application = **dynamic risk budgeting**: compute **VaR/ES conditional on the
   current vol regime** rather than full-sample (his red-vs-blue curves).

Strongest claim: **asymmetric/non-linear tail dependence** — "on a crash, vol spikes
far faster than on a rally" — which linear/IC factor tests miss but a joint
distribution captures. He also notes: **rank (Spearman/Kendall) correlation, not
Pearson** ("Pearson assumes normality"); **HMM has a real-time boundary problem**
(smoothing is globally optimal but the *current* state is unreliable — corroborates
our sticky-HMM rejection); and to get a *forward* conditioning variable you need
**implied vol**, not historical vol (historical vol → contemporaneous, no edge).

### BV1VZLi6PE2B — ML course Q&A
~60% course-selling, ~40% technique (some deliberately withheld). The meatiest
exchange is a caller doing *exactly our loop* (custom event-axis + unsupervised
labels on US ETFs). Substance:

- **Axis objective is uniform per-bar INFORMATION DENSITY**; normality is only a
  byproduct ("the money you earn in the tails you give back in the central peak").
  Fix the clock first — factors that flicker on/off fail because their *time-bar
  baseline has no statistical meaning*, "even a non-linear model can't recover
  structure on a bad baseline."
- **Label to match how you trade**: trend-segmentation labels so intra-trend
  pullbacks don't flip Y and holding stays continuous (= our `trend_leg`).
- **Long-only is the fix for label imbalance** on up-drifters (QQQ/TQQQ/gold) — don't
  rebalance to 0.5, don't train long-short (= our β200 routing).
- **Drawdown AREA metric** ("DNA/DA"): the orange area under the drawdown curve = the
  trader's/capital's *psychological-scar area*; **smaller area ⇒ faster recovery**.
  Presented as his proprietary differentiator.
- Features from **OHLCV only** → "thousands" via higher-order math, **topology (TDA)**
  and **fractional differentiation**. **Non-linear dim-reduction has live-trading
  pitfalls** (corroborates our linear-reduce preference). Tick-resampling only helps
  at true HF. Strict OOS on independent data; model training is "half the work" —
  combination + live inference is the hard half.

---

## 2. Levers, classified against our frontier

### NEW + actionable (worth building/racing)
| Lever | Maps to | Status here |
|---|---|---|
| **Drawdown-area / Ulcer / Pain / time-under-water metric** | honesty/eval | **IMPLEMENTED this session** (`lb.metrics`) — see §3 |
| **Conditional-ES (regime-dependent) sizing** — size so per-regime tail risk is constant, vs our raw vol-overlay | sizing | race-ready idea; novel vs current sizers |
| **Joint-KDE quadrant regime as a FEATURE** (discretised cell id + conditional moments), distinct from our parametric `regime_gmm` | features | novel; feed the cell as a *categorical feature*, not a hard gate |
| **Rank (Spearman/Kendall) correlation reduce** ("don't use Pearson") | reduce | cheap; competes with already-winning `infogain` |

### Corroborates what we already do/know (no action)
Info-density event-clock; `trend_leg` trend-segmentation; β200 long-only routing;
HMM real-time-boundary weakness; linear-reduce preference; strict OOS; data-prep as
the ceiling; CDF bet sizing; confidence-filtered execution.

### Closed / out-of-rule (do NOT re-grind — confirmed by the frontier briefing)
- **Fractional-diff features** — closed (correlation-filter crowd-out destroyed TLT).
- **HMM family** — closed (sticky-HMM predictable-not-profitable).
- **Long-short on up-drifters** — closed (median real Calmar ≈ 0).
- **Cross-asset / copula-across-assets, HRP, cross-sectional rotation** — violate the
  single-ticker rule; usable only as exogenous *features*, and cross-asset *price*
  features are already closed in every form.
- **CDF-transform of a single feature before the tree** — inert: XGBoost is invariant
  to monotonic per-feature transforms. Only *constructed joint/copula interaction*
  features (e.g. a tail-co-occurrence indicator of two of the asset's OWN variables)
  could add signal.
- **Implied-vol-conditioned regime** — needs options data (not in QC access); a
  historical-vol regime cell is contemporaneous and has no edge by itself (Wang's own
  caveat). Genuinely new distributional work (copulas / tail-dependence / CDF-ladder
  labeler) remains unexplored but mostly needs the forward (IV) channel or a clever
  past-only construction.

---

## 3. Implemented this session — drawdown-shape metrics (`lb.metrics`)

Wang's most-emphasized proprietary metric is the **drawdown area**. Our honesty stack
had Calmar / MaxDD / Sharpe / DSR but **nothing capturing drawdown duration×depth** —
Calmar sees only the single deepest trough and is blind to a long shallow bleed. Added
`src/lb/metrics.py` (pure stdlib, post-hoc on the realised curve → no leak), re-exported
from `scripts/audit/stats_rigor.py`:

- `ulcer_index` — Martin (1989) RMS drawdown `sqrt(mean d_t²)`
- `pain_index` — mean |drawdown| = **drawdown area / N** (Wang's 回撤面积, Becker Pain)
- `max_dd_duration` — longest time-under-water (points below a prior peak)
- `martin_ratio` — annualised CAGR / Ulcer (the Calmar analogue that rewards fast recovery)
- `max_drawdown`, `underwater`

Tested (`tests/test_drawdown_metrics.py`, 5 cases + 13 self-test checks in
`stats_rigor`). **It changes the book ranking in a meaningful, correct way** — computed
on the real OOS curves in `results/series_cache.json`:

| tk | Calmar | Martin | MaxDD% | Pain% | Ulcer% | UW_dur |
|----|-------:|-------:|-------:|------:|-------:|-------:|
| GLD | 18.13 | 43.80 | 2.80 | 0.89 | 1.16 | 33 |
| USO | 9.16 | **48.79** | 14.70 | 1.33 | 2.76 | 17 |
| UUP | 5.08 | 17.77 | 1.53 | 0.27 | 0.44 | 54 |
| IWM | 2.49 | 9.16 | 4.53 | 0.83 | 1.23 | 46 |
| HYG | 7.08 | 33.51 | 3.79 | 0.43 | 0.80 | 15 |
| TIP | 4.20 | 12.23 | 3.46 | 0.83 | 1.19 | 42 |
| DBC | 3.46 | **7.06** | 11.63 | 4.70 | 5.69 | **138** |

- **DBC**: Calmar mid-pack but **last on Martin** — underwater 138/224 points (62% of
  the time), Pain 4.70%. Calmar hides how punishing it is to *hold*.
- **USO**: **#1 on Martin** (short drawdowns, fast recovery) — matches "USO is now the
  stronger engine."
- **IWM vs DBC**: Calmar ranks IWM worst; Martin/Pain correctly rank DBC worse.

Use Martin/Pain alongside Calmar when weighting/ranking book members; they penalise the
slow-bleed names (DBC, IWM, decayed-UUP) that a single-trough Calmar flatters.

---

## 4. Next race-ready levers (for the loop, in EV order)

1. ~~**Conditional-ES regime sizer** (`cond_es`)~~ — **BUILT + RACED 2026-06-11, REJECTED.**
   Tail analog of `cdf_overlay`: throttle the cdf bet when recent left-tail loss (fast
   ES, worst-20%) spikes above baseline (slow ES), floor 0.4; causal/leak-safe; added
   identically to `sizing_ext._size` and `infer.py.tmpl._size`. Results: **GLD** cond_es
   2.37 ≈ champion dd_overlay 2.40 (redundant — dd_overlay already protects downside,
   cond_es just cuts trades 1157→852); **USO** cond_es 2.46 **< cdf_plain 2.69** —
   tail-throttling *fights* mean-reversion (it cuts the bet into the oversold dips
   `revert` harvests). **Mechanism:** ES-throttling is redundant on trend+dd_overlay and
   anti-synergistic with reversion — a sizer reshapes risk but doesn't add signal. Kept
   in `sizing_ext.py` for any future tail-sensitive momentum name. (Note: this was also
   the first live backtest through the restructured `src/lb` pipeline — end-to-end OK.)
2. **Joint-KDE quadrant cell as a categorical feature** — TRAIN-fit KDE on (return,
   vol), emit the discretised cell id (+ conditional quantile of return given the vol
   bucket) as past-only features; race under `reduce=infogain` on GLD. Distinct from
   the parametric `regime_gmm` gate.
3. **Rank-correlation reduce** (`reduce=spearman`) — cheap; A/B vs `infogain`. Low EV
   (infogain already won the 80-feat panel) but Wang-prescribed and unbuilt.

All three are single-ticker, past-only, and fit the 64k QC budget. None needs new data.
Build candidate modules: sizer → `sizing_ext.py`; KDE feature → `ml_ext.py`/`features.py`;
rank reduce → `ml_ext.py`. Race each A/B vs the champion with the full keep-gate
(trades>80, val_auc>0.52, beats-champion, DSR/Bonferroni, permute-collapse).
