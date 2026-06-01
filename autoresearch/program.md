# autoresearch

Real ETF ML pipeline on QuantConnect. Each round: pick the weakest ETF, race two hypotheses on the
2 QC nodes, keep the winner if it beats that ETF's best. Never stop.

## metric — the only fixed thing
Real backtest, OOS 2023-08 → 2026-06: **Calmar** = CAGR/MaxDD (up), **DA** = Σ(1 − E_t/peak) (down).
The scorer, the splits, and real execution are LOCKED. Everything else — including the train/val
"middle" metrics — is a knob you may tune.

## loop
1. pick the weakest ETF — lowest real OOS Calmar with trades>80.
2. target: `target_next.py` → enumerate interventions from causal-graph FINDING nodes, rank, take the top 2.
3. race: `run_autoresearch_round.py '<A>' '<B>'` — one ETF, 2 nodes.
4. score real OOS Calmar + DA. deployable = both legs done, trades>80.
5. keep iff deployable and beats the ETF's best (after the leak-check). else discard. log both.
6. record: `render_round.py` → report; update `knowledge.json.causal_graph`; `render_causal_graph.py --inject`; commit.
7. re-rank → next weakest. never stop.

## rules
- only real OOS Calmar+DA crown a winner. train/val metrics just select within a run.
- deployable = >80 OOS trades. buy-and-hold (1 trade) is a ceiling, not a result.
- no lookahead: features past-only; fit on TRAIN (calibrator on VAL, last ~200 bars embargoed). the future
  label filters only the FIT bars, never which OOS bars trade. a too-good KEEP must beat an always_long control.
- axis ① and labeling ② matter most. shorting is allowed. simpler is better. confirm on ≥2 runs.

## setup
QC project 31338454, creds `qc/.creds.json`. py3.11 / pandas 2.3.3 (`.ffill()`) / sklearn 1.6 / xgboost 3.
A hypothesis = `{ticker, axis, labeler, thresh, sizing}`. Live axes/labelers/sizings, findings, and prior
work: `knowledge.json` and `_archive/`.
