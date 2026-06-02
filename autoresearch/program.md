# autoresearch

Real ETF ML on QuantConnect, built as Wang's full production line. Each round: pick the
weakest ETF, think hard along Wang's pipeline, race two hypotheses on the 2 QC nodes, keep
the winner iff it beats that ETF's best on real OOS. Never stop.

## wang's production line (the workflow we follow)
Data → **custom axis** → **labels** → **features** → **dim-reduce** → **train** → **combine**
→ **infer/live**. Each round co-designs the steps together, never one in isolation.
1. **data** — minute bars; analyse the differenced distribution + autocorrelation (time axis hides risk).
2. **axis ①** — non-time bars (dollar/vol/range/imbalance/entropy/fracdiff/dc…); a good axis makes the
   first difference closer to normal / less autocorrelated → cleaner ML.
3. **labels ②** — directional/trend labels (triple_barrier, dc_trend, regimes…); label density follows trend.
4. **features ③** — integer-order + custom-window stats, RSI/Bollinger, **entropy**; past-only.
5. **dim-reduce** — fit on TRAIN; we *select* (correlation-prune→20), not project, so live inference is stable
   (Wang's warning about linear projection breaking live inference doesn't bite a selector).
6. **train** — XGBoost depth3/lr0.03/n200, scale_pos_weight; isotonic calibration on VAL; anti-overfit via AUCdiv.
7. **combine ⑧** — sizing + optional cross-axis ensemble averaging.
8. **infer/live** — frozen-model online replay, proven live-equivalent.

## metric (locked)
Real backtest, OOS 2023-08 → 2026-06. **Calmar** = CAGR/MaxDD (up). **DA** = Σ(1−E_t/peak) (down).
Scorer, splits, execution are LOCKED. Train/val metrics are only knobs to pick within a run.

## loop
1. pick the weakest ETF — lowest real OOS Calmar with trades>80.
2. **think.** read the causal graph + that ETF's findings; design the two best hypotheses by hand along
   Wang's pipeline (co-design axis×labeler×features×sizing). `target_next.py` ranks; Claude decides.
3. race: `run_autoresearch_round.py '<A>' '<B>'` — one ETF, 2 nodes.
4. score real OOS Calmar + DA. deployable = both legs done, trades>80.
5. keep iff deployable AND beats the ETF's best. else discard. log both.
6. record: `render_round.py` → report; update `knowledge.json.causal_graph`; `render_causal_graph.py
   --inject`; `render_index.py`; commit.
7. re-rank → next weakest. never stop.

## rules
- only real OOS Calmar+DA crown a winner. buy-and-hold (1 trade) is a ceiling, not a result.
- no lookahead: features past-only; fit on TRAIN (calibrator on VAL, last ~200 bars embargoed); the future
  label filters only FIT bars, never which OOS bars trade; **bar thresholds fit on TRAIN minutes, not the full
  series.** a too-good KEEP must beat an `always_long` control.
- **trust = trials-adjusted.** a kept edge must clear PSR vs Bonferroni-by-N_trials (`assess_dsr.py`), else
  selection bias. Today 4/7 champions clear it (EEM/GLD/HYG/TLT); QQQ/IWM/XLE don't.
- **A/B every NEW method vs the champion on real OOS; revert if it loses.** simple is best — a "correct"
  technique can still destroy edge (sample-uniqueness weights hurt both timing edges → reverted).
- long/short allowed on every ETF; confirm on ≥2 runs (it's TLT's champion, but drags EEM/IWM).

## state (2026-06)
- Leaderboard: **EEM 4.03** (timing + meta-labeling, Calmar>3, significant), HYG 2.21 / GLD 1.99 / QQQ 1.24
  (drift), TLT 1.52 (timing, significant), IWM 1.14 (axis-bound), XLE 0.91 (ceiling).
- **Meta-labeling WON** (Wang's trading-decision 2nd model, `triple_barrier_meta`): a secondary 'is-the-primary-
  right?' model GATES the signal → EEM 2.43→4.03 (Calmar>3), PROVEN live-equivalent (infer_online carries the
  secondary; byte-exact 0.0). LONG-ONLY-specific (hurt TLT's long/short); needs a STRONG primary (failed IWM's
  weak one). Single-config swaps had converged; new METHODS from Wang's course are the lever. (cross-asset SPY
  feats failed — needs pairs.)
- **PORTFOLIO (Wang endpoint ⑨⑩):** the 7 champions combined, conviction-weighted (∝Calmar), give a deployable
  book at real OOS **Calmar 3.36, MaxDD 3.4%, Sharpe 2.55** — diversification crushes drawdown.
- Verified leak-free (6-agent audit); OOS replay byte-identical to a fully-online rebuild — incl. the EEM
  meta champion (primary+secondary+gate online, max_pred_diff 0.0). Every champion is live-equivalent.

## setup
QC project 31338454, creds `qc/.creds.json`. py3.11 / pandas 2.3.3 / sklearn 1.6 / xgboost 3.
A hypothesis = `{ticker, axis, labeler, thresh, sizing}`. Modules: `autoresearch/modules/`. Findings &
prior work: `knowledge.json` (+ causal graph) and `_archive/`. Wang's course: `pdfs/`.
