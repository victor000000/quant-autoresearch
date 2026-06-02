# autoresearch

Real ETF ML on QuantConnect. Each round: pick the weakest ETF, think hard, race two
hypotheses on the 2 QC nodes, keep the winner if it beats that ETF's best. Never stop.

## metric (the only fixed thing)
Real backtest, OOS 2023-08 → 2026-06. **Calmar** = CAGR/MaxDD (up). **DA** = Σ(1−E_t/peak) (down).
Scorer, splits, and real execution are LOCKED. Train/val metrics are just knobs to pick within a run.

## loop
1. pick the weakest ETF — lowest real OOS Calmar with trades>80.
2. **think.** read the causal graph + that ETF's findings, then design the two best hypotheses by hand.
   `target_next.py` only ranks candidates — Claude decides. Modules connect: co-design
   axis ① × labeler ② × features ③ × sizing ⑧, don't tweak one in isolation.
3. race: `run_autoresearch_round.py '<A>' '<B>'` — one ETF, 2 nodes.
4. score real OOS Calmar + DA. deployable = both legs done, trades>80.
5. keep iff deployable AND beats the ETF's best (after the leak-check). else discard. log both.
6. record: `render_round.py` → report; update `knowledge.json.causal_graph`; `render_causal_graph.py
   --inject`; `render_index.py`; commit.
7. re-rank → next weakest. never stop.

## rules
- only real OOS Calmar+DA crown a winner. buy-and-hold (1 trade) is a ceiling, not a result.
- no lookahead: features past-only; fit on TRAIN (calibrator on VAL, last ~200 bars embargoed).
  the future label filters only FIT bars, never which OOS bars trade. **bar thresholds fit on TRAIN
  minutes, not the full series.** a too-good KEEP must beat an `always_long` control. check this every round.
- long/short is allowed on EVERY ETF — try it even on up-drifters. simpler is better. confirm on ≥2 runs.
- **trust = trials-adjusted.** a kept edge must clear PSR vs Bonferroni-by-N_trials (`assess_dsr.py`),
  else it's selection bias. Today only 4/7 champions clear it (EEM/GLD/HYG/TLT); QQQ/IWM/XLE don't.

## verified (2026-06)
- pipeline is leak-free (6-agent audit). the OOS backtest is a prediction-replay that is PROVEN
  byte-identical (max diff 2.3e-8) to a fully-ONLINE run that rebuilds bars+features+frozen-model live
  (`infer_online.py.tmpl` + the per-cell model bundle) — i.e. live-trading-equivalent.

## setup
QC project 31338454, creds `qc/.creds.json`. py3.11 / pandas 2.3.3 / sklearn 1.6 / xgboost 3.
A hypothesis = `{ticker, axis, labeler, thresh, sizing}`. Live modules, findings, and prior work:
`knowledge.json` and `_archive/`.
