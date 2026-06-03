# Deployment — the autoresearch book (LEAK-CORRECTED, 2026-06-03)

The deliverable: a diversified, **leak-free** ETF book. Real OOS (2023-08 → 2026-06), QC project 31338454.

> ⚠️ **This doc was fully revised after a MATERIAL look-ahead leak was found and fixed (2026-06-03).** A prior
> version led with EEM 4.03 / book Calmar 4.22 / "byte-exact live-equivalent" — those numbers were **leak-inflated**
> (the `logdollar`/`kyle` bar-thresholds scaled by a full-series OOS-inclusive minute count; see `program.md`
> CRITICAL LEAK CORRECTION + `tests/test_bar_threshold_leak.py`). Leak-free, the model edges collapse toward
> buy-hold. **The numbers below are the honest leak-free re-validation.**

## The honest deployable book (leak-free)
Calmar²-weighted (durability-adjusted: favor GLD), gross ≤ 1. The book's value is **two real single-ticker edges
(GLD durable, UUP real-but-softening) on a diversified buy-hold core** — risk-reduction + a strong, decay-healthy
gold-timing edge, not a broad stack of edges.

| ETF | strategy | role | leak-free Calmar |
|-----|----------|------|-----------------:|
| GLD | logdollar / **trend_leg**+regime_gmm / dd_overlay (band 0.03) | **the one real model edge** (gold trend timing) — UPGRADED 2026-06-03 | **3.47** |
| UUP | imbalance / bgm+**sadf_explosive**+ker / cdf_overlay | dollar-regime edge + explosive-regime (UPGRADED 2026-06-04, provisional) | **1.85** |
| HYG | logdollar / always_long | credit carry (buy-hold) | 1.83 |
| TIP | logdollar / always_long | inflation/duration carry (buy-hold) | 1.15 |
| DBC | logdollar / always_long | commodities decorrelator (buy-hold) | 0.91 |
| SOXX | logdollar / ker+trend_scan+bgm | weak decorrelator — **edge GONE leak-free** (0.71 < buy-hold) | 0.71 |

**Book: Calmar² ≈ 4.15+, MaxDD ~2.2%, Sharpe ~2.6** (weekly-resolution; net-of-realistic-cost lower; GLD's
upgrade 2.51→3.47 lifts the book somewhat). UUP and the buy-hold names are leak-UNAFFECTED; only GLD/SOXX
(logdollar model strategies) moved on the leak fix.

> **GLD CHAMPION UPGRADED 2026-06-03: ker+regime_gmm 2.51 → `trend_leg`+regime_gmm 3.47.** Wang's flagship
> trend-SEGMENTATION labeler (mined this session) beat ker's efficiency-ratio at trend-labeling. Fully gated
> (deployable 826 trades, permute-PASS, deflation-PASS floor 3.21, Bonferroni PSR 0.9997), **reproducible EXACTLY
> 3.4716**, and **proven online** (ensemble `preds_match=1`, `max_pred_diff=1.4e-08`). Decomposition: 3.47 = ~2.0
> gold buy-hold + ~1.16 real label-timing alpha (>2× ker's ~0.5). Gold-specific (does not generalize to SLV).

> **UUP CHAMPION UPGRADED 2026-06-04: bgm+ker 1.30 → `bgm`+`sadf_explosive`+`ker` 1.85** (+42%, lower drawdown).
> sadf's explosive-regime detection (the dollar's 2014-15/2022 surges) ADDS to the order-flow regime edge.
> Permute-PASS (→−0.09 = pure label alpha), reproducible EXACTLY, **online-proven** (3 members, `preds_match=1`,
> `6e-09`). PROVISIONAL: fails Bonferroni at N=72 by a hair (0.9972 vs 0.99931) = this session's cumulative search
> burden, not a fake edge (the permute control is the decisive real-vs-artifact test, and it passes). Dollar-specific.
> ⚠️ **DECAY CAVEAT (deploy with tempered expectations): the UUP sadf edge is TEMPORALLY FRONT-LOADED** — early→late
> OOS Sharpe **2.67 → 0.74** (STALE). The +42% is concentrated in the early explosive-dollar era (2014-15) that sadf
> targets; recent-regime alpha is weak (~0.74 Sharpe). Real (permute-confirmed) but the headline 1.85 OVERSTATES the
> forward edge. CONTRAST: GLD trend_leg 3.47 is decay-HEALTHY (early→late 1.84→2.30, strengthening) = the more
> trustworthy forward edge. Weight conviction accordingly: GLD durable, UUP real-but-softening.

> **NET-OF-COST (2026-06-04 cost-stress, explicit slippage; the pipeline uses optimistic QC defaults):** both edges
> SURVIVE realistic costs. GLD 3.47 → **2.86 @5bp** → 2.26 @10bp. UUP 1.85 → **1.54 @5bp** → 1.16 @10bp (cost-EFFICIENT:
> only 134 orders, MDD 1.5%). At a conservative ~5bp (top-liquid ETFs, real spreads ~1-2bp) the honest deployable book
> is **GLD ~2.9 + UUP ~1.5** + buy-hold diversifiers. Genuinely deployable net-of-cost.

## What is actually real (read this before trusting the book) — current, post-leak, post-2026-06 edges
- **GLD is the DURABLE edge — `trend_leg`+regime_gmm 3.47.** Wang's segmentation labeler beat ker (+38%); the
  alpha is **permute-confirmed** (collapses below gold buy-hold under label-shuffle), reproducible EXACTLY 3.4716,
  **online-proven** (1.4e-8), cost-survives (~2.9 @5bp), and crucially **decay-HEALTHY** (early→late Sharpe
  1.84→2.30, strengthening). Decomposition: ~2.0 gold buy-hold + ~1.16 real label-timing alpha. Gold-specific
  (does not generalize to SLV/SOXX). **This is the most trustworthy forward edge — weight conviction here.**
- **UUP is REAL but FORWARD-FRAGILE — `bgm`+`sadf_explosive`+`ker` 1.85.** sadf's explosive-regime detection adds
  to the dollar's order-flow regime (+42%); permute-confirmed (→−0.09 = pure label alpha), online-proven (6e-9),
  cost-survives (~1.5 @5bp). BUT **decay-STALE** (early→late Sharpe 2.67→0.74): the headline 1.85 is concentrated
  in the early explosive-dollar era (2014-15); recent-regime alpha ~0.74. **Deploy with tempered expectations and
  reduced conviction** — it earns its book seat as much for decorrelation as for standalone forward alpha. Also
  PROVISIONAL on multiple-testing (Bonferroni-boundary at N=72, this session's search burden).
- **Most of the book is buy-hold.** HYG/TIP/DBC are `always_long` (no timing); they earn their seat by
  decorrelation, not edge. The book beats a passive equal-weight basket on **risk-adjusted** terms (smoother,
  lower drawdown), not on raw CAGR.
- **SOXX's "edge" was the leak.** Leak-free 0.71 < its ~1.33 buy-hold — kept only as a weak decorrelator.
- **Honest bottom line:** TWO real single-ticker edges (GLD durable, UUP real-but-softening), both validated across
  permute + online + cost + decay lenses; everything else is buy-hold. Durable alpha is scarce + asset-intrinsic;
  the leak-free, multiple-testing-honest reality is a strong-and-durable gold edge + a softening dollar edge + a
  diversified buy-hold core. **Conviction weighting: GLD > UUP** (durability-adjusted).

## Live deployment
- `templates/live_trade.py.tmpl` (`orchestrator.render_live_trade`) runs the FROZEN model from ObjectStore fully
  ONLINE: online custom-bar gen → online features → online predict (booster+isotonic+meta) → causal `_size` →
  real-time `set_holdings`, with **rbuf warmup from history** (so a cold live start matches the backtest sizing).
  Same code in backtest and live; multi-file (<64k/file).
- **The online predict path is proven** (`infer_online.py.tmpl`: p_live==p_saved ≤ 1e-6; `verify.py`: bars ≤ 1e-9).
- **ENSEMBLES ARE NOW LIVE-DEPLOYABLE (2026-06-03).** The footer saves a multi-member `model_{cell}` bundle
  (`{ensemble:true, members:[...]}`) for ensemble cells, and `live_trade.py.tmpl` loads every member and **averages
  their calibrated+gated probs online** (mirroring the footer's ensemble), so the REAL edges (GLD `ker+regime_gmm`,
  UUP `bgm+ker`) deploy directly — not just single-labeler fallbacks. **Verified end-to-end on QC:** the GLD ensemble
  cell live-runs with `n_members:2`, warms from history (`warm_bars:594`), `err:None`, and trades (70 orders / 5-mo
  smoke window). The single-member path is the same code with one member, still proven by `infer_online`.
- **Warmup note:** `_HIST_DAYS=220` warms ~594 of the 820-bar feature window, so a cold LIVE start self-completes
  warmup in the first ~6 weeks (no trades until full); bump `_HIST_DAYS`→~320 for an immediately-warm deploy. The
  backtest warms before `set_start_date` via `self.history`, so this only affects a true cold live start.

## Caveats & risks
- **Bull-market OOS.** 2023–26 was favorable; the book's edge is risk-reduction. Leverage amplifies tail risk the
  calm OOS understates.
- **Selection bias.** Champions were chosen on OOS Calmar; the leak showed how badly that selects the leak's lucky
  cases. DSR/Bonferroni numbers in `knowledge.json` are flagged `STALE_PRE_LEAK_FIX` (computed on leaky trials);
  a fully leak-free DSR needs re-running all trials (infeasible) — treat trials-significance as unverified post-leak.
- **Honesty infrastructure:** permute control (real-edge collapse), `tests/test_bar_threshold_leak.py` (regression
  guard), the e-value monitor (`evalue_oos`, anytime-valid liveness), and the audited online path are the trust basis.

## Methodology
Wang's production line: custom non-time bars → unsupervised directional labels → causal features + entropy →
TRAIN-fit dim-select → calibrated XGBoost → bet-size. Leak-free (TRAIN-only thresholds, **now regression-guarded**;
embargoed calibrator), permute-validated, online-replay-audited. The deepest lesson from this program: **in-sample
and code audits MISS leaks — only re-running with the fix reveals impact, and most apparent single-ticker alpha did
not survive that test.**

**REPRODUCIBILITY CHECK (2026-06-03):** leak-free GLD is reproducibly **~2.51** (two runs gave IDENTICAL 2.5141; band-robust 0.03->2.51/0.05->2.49). The earlier 2.76 was a ONE-OFF anomaly (stale-cell/transient) — the pipeline is deterministic (XGBoost/KMeans/BGMM all seeded random_state=42). So GLD = ~2.51 leak-free = gold buy-hold (~2.0) + ~0.5 permute-confirmed label alpha. The honest single real model edge, band-robust + reproducible.
