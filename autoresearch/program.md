# autoresearch

Real ETF ML on QuantConnect, built as Wang's production line. Pick the weakest ETF, think hard,
race two hypotheses on the 2 nodes, keep the winner iff it beats that ETF's best on real OOS.

**Do not stop exploring.** There's always another axis, label, feature, dim-reduce, model,
combination, or scale to try. Convergence means try something *different*, not stop.

## wang's philosophy (the why)
1. **Resample off the clock.** Non-time bars (dollar/vol/…) make returns closer to IID-normal → cleaner ML.
2. **Label unsupervised; the label may look ahead.** Labels are the *training target*, so an unsupervised
   detector (trend-scanning, change-point, clustering — NOT HMM) using the *full* series is fine. Causality
   lives in the downstream SUPERVISED model, which predicts the label from past-only features on OOS.
3. **Rich features, then reduce.** Many features (integer-diff + frac-diff + entropy + freq), then a learned
   (non-linear) dim-reduce — fit on TRAIN.
4. **Combine across scales/models.** Multi-scale stacking + bet-sizing/leverage; diversify to cut drawdown.
5. Aim Calmar > 3, fully reproducible, deployable live.

## metric (locked)
Real OOS Calmar = CAGR/MaxDD. Splits train→2021-08, val→2023-08, test→2026-06. Scorer + execution LOCKED.

## loop
1. pick the weakest ETF (lowest real OOS Calmar, trades>80).
2. think — read the causal graph + findings; co-design axis × label × features × dim-reduce × model × sizing.
3. race `run_autoresearch_round.py '<A>' '<B>'`; keep iff deployable (trades>80) AND beats the ETF's best.
4. record: render_round → knowledge.json.causal_graph → render_causal_graph → render_index → commit. repeat.

## rules
- only real OOS Calmar crowns a winner; a too-good keep must beat an `always_long` control.
- features past-only; the unsupervised label may use the future (it's the target) but must only filter FIT bars,
  never which OOS bars trade. bar thresholds on TRAIN minutes. calibrator on embargoed VAL.
- trust = trials-adjusted (PSR vs Bonferroni-by-N_trials). A/B every new method vs the champion; revert if it loses.

## best so far · keep pushing
EEM 4.03 (meta-labeling, Calmar>3, live-equivalent) · 8-ETF book 4.22 (conviction, 2× dominates passive).
Priorities: **new custom axes** + **new unsupervised labels** (trend-scanning, change-point, clustering — NOT
HMM). Also: richer features, learned dim-reduce, multi-scale stacking, meta-labeling on more primaries,
decorrelated ETFs. Search papers for every module.

## setup
QC project 31338454, creds `qc/.creds.json`. Hypothesis = `{ticker, axis, labeler, thresh, sizing}`.
Modules `autoresearch/modules/`; findings `knowledge.json` (+ causal graph); deploy `DEPLOYMENT.md`;
Wang's course + Q&A `pdfs/`, `docs/legacy/wang_qa_questions.md`, `uni/transcripts/`.
