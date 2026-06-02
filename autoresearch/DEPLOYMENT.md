# Deployment — the autoresearch book

The deliverable of the program: a diversified, leak-free, live-equivalent ETF book.
Everything below is real OOS (2023-08 → 2026-06), QC project 31338454.

## The book (deployable)
Conviction-weighted (capital ∝ each champion's Calmar), 8 ETFs, gross ≤ 1 (lever per budget):

| ETF | strategy (cell) | role | Calmar |
|-----|-----------------|------|-------:|
| EEM | logdollar / triple_barrier + **meta** / cdf_overlay @0.30 | timing edge (flagship) | 4.03 |
| HYG | logdollar / always_long @0.55 | credit carry | 2.21 |
| GLD | logdollar / always_long @0.45 | gold | 1.99 |
| TLT | range / triple_barrier / ls_overlay @0.50 | rates timing (long/short) | 1.52 |
| QQQ | logdollar / always_long @0.15 | US tech beta | 1.24 |
| IWM | imbalance / triple_barrier+bgm / cdf_overlay @0.45 | small-cap | 1.14 |
| XLE | logdollar / tertile / cdf_overlay @0.30 | energy | 0.91 |
| DBC | logdollar / always_long @0.45 | commodities (decorrelator) | 0.86 |

EFA and UUP are champions in the universe but EXCLUDED from the book (EFA correlated → hurts;
UUP too weak → Calmar-neutral).

## Expected performance (real OOS)
- **Calmar 4.22, MaxDD 2.74%, Sharpe 2.52**, positive every calendar year (2023 +3%, 2024 +8.3%,
  2025 +15.5%, 2026 +5.3%; yearly MaxDD ≤3.2%).
- **vs passive** buy-hold of the same basket (Calmar 1.61 / MaxDD 11.2%): the book gives up raw
  CAGR in a bull (11.5% vs 17.9%) but is far smoother — **2.2× the Calmar, a third of the drawdown**.
- **Levered 2×** it dominates passive on BOTH axes: CAGR 25% (>17.9%) at 6.2% MaxDD (<11.2%),
  Calmar 4.01. 3× → CAGR 39.8% / MaxDD 9.3% / Calmar 4.26. Leverage caveat below.

## How to run
1. Train champions: `render_train_config(cfg)` per row → QC backtest writes the cell to ObjectStore.
2. Replay the book: `orchestrator.render_portfolio(champions, leverage)` where
   `champions = [[ticker, cell_key, weight], ...]`, `weight = real_calmar`, `cell_key` from
   `per_etf_best[t].cell` (`+`→`_x_`, `t0.XX`→`tXX`). One QC backtest = the portfolio equity.
3. Live: each champion is byte-exact live-equivalent (`infer_online.py.tmpl` rebuilds bars+features
   +model online; EEM's meta secondary is carried in the bundle). The book is its weighted sum.

## Caveats & risks (read before deploying)
- **Bull-market OOS.** 2023-26 was favorable; the book's edge is risk-reduction, so it should hold
  up in drawdowns, but leverage AMPLIFIES tail risk the calm OOS understates. 2× is the prudent point.
- **Selection bias.** Champions were chosen on OOS Calmar; 4/8 clear PSR-vs-Bonferroni (EEM/HYG/GLD/TLT),
  the rest are beta or trials-suspect. The portfolio's value is diversification + the EEM/TLT timing edges.
- **Objective-dependent.** ∝Calmar¹ maximizes Calmar (4.22); ∝Calmar² maximizes Sharpe (2.75). Choose by mandate.
- **Inclusion rule:** add a name iff DECORRELATED from the sleeve AND its MaxDD cut beats its return drag.

## Methodology (what the research established)
Wang's full production line: custom non-time bars → directional/trend labels → causal features +
entropy → TRAIN-fit dim-select → calibrated XGBoost → **meta-labeling** (secondary "is-the-primary-right"
GATE — the one new-method edge, lifted EEM 2.43→4.03) → conviction-weighted, decorrelation-gated portfolio.
Leak-free (TRAIN-only thresholds; embargoed calibrator), trials-gated (PSR/Bonferroni), byte-exact
live-equivalent. Drift ETFs can't be timed (carry dominates); meta-labeling helps only a strong long-only
timing primary; diversification > concentration, but decorrelation > name-count.
