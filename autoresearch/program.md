# autoresearch

Real ETF ML on QuantConnect, built as Wang's production line. Pick the weakest ETF, think hard,
race two hypotheses on the 2 nodes, keep the winner iff it beats that ETF's best on real OOS.

**Do not stop exploring.** There's always another axis, label, feature, dim-reduce, model, or
scale to try. Convergence means try something *different*, not stop. **Simple is best** — prefer the
small change that hardens honesty over a framework that adds horsepower.

## backtest contract (the invariant — never break it)
The real OOS backtest is **online, leak-free, and uses ONLY the QC-ObjectStore-trained model.** Audited
+ proven (2026-06-02, see `BACKTEST_AUDIT.md`):
- `infer.py` (the headline metric) holds **no model** — it loads saved predictions+thresh+sizing from the
  cell and applies a causal `_size`. `portfolio.py` is the same. Nothing is re-fit on test.
- Every `.fit` lives in `footer.py` (training): model on TRAIN, calibrator/early-stop on embargoed VAL,
  meta on purged-CV within TRAIN. Test enters ONLY as predict_proba/transform — no future-label trade selection.
- Bars/features/predict/set_holdings run causally bar-by-bar. Proven: `verify.py` bars byte-identical
  (max_lc_diff≤1e-9); `infer_online.py` rebuilds everything online and asserts p_live==p_saved
  (max_pred_diff≤1e-6; live: GLD 2.1e-8, UUP 0.0). Bar thresholds = TRAIN-only minutes, fail-loud.
- Keep it this way: features past-only; thresholds/scaler/reduce/calibrator fit on TRAIN(+embargoed VAL) only;
  before trusting a champion's Calmar, its `infer_online` must show preds_match=1.

## wang's philosophy (the backbone)
1. **Resample off the clock** — non-time bars (dollar/vol/range/…) → returns closer to IID → cleaner ML.
2. **Label unsupervised; the label may look ahead** (it's the target). Causality lives in the SUPERVISED
   model predicting that label from past-only features. (Detectors: trend-scan, change-point, clustering — NOT HMM.)
3. **Rich features, then reduce** (fit on TRAIN). 4. **Combine across scales** + bet-size; diversify to cut drawdown.
5. Aim Calmar > 3, reproducible, deployable.

## metric (locked)
Real OOS Calmar = CAGR/MaxDD. Splits train→2021-08, val→2023-08, test→2026-06. Deterministic (fixed seeds):
same code+data ⇒ same result. But TEST_END clamps to available data, so the **OOS window grows and stored
Calmars go STALE** — a weak edge lucky over a short window decays as the window extends. **Re-validate before trusting.**

## loop
1. Pick the weakest ETF (lowest real OOS Calmar, trades>80). **Re-validate its stored best first** over the current window.
2. Think — read the provenance graph + findings; co-design axis × label × features × reduce × model × sizing.
3. Race `run_autoresearch_round.py '<A>' '<B>'`.
4. Record: render_round → knowledge.json (provenance graph) → render_causal_graph → render_index → commit.

## rules (honesty > horsepower)
- A keep must be: **deployable** (trades>80) AND **Calmar > 0** (a negative Calmar loses money — never keep it,
  even if it beats a worse control) AND **> the re-validated prev best** AND **val_auc > 0.52** (val_auc≈0.5 ⇒ the
  Calmar is a window/path artifact, not an edge) AND beat an `always_long` control.
- **Trust = trials-deflated.** Many configs tried ⇒ the max is upward-biased; require the edge to clear the
  best-of-N-trials noise (Deflated Sharpe / `deflated_audit.py`), not a raw threshold. A SEARCHED model edge that
  fails deflation is a selection artifact; an `always_long` baseline carries no selection bias.
- **durable > lucky.** Drift/long-biased cells persist; two-sided timing with val_auc≈0.5 decays. Don't deploy a Calmar you can't reproduce today.
- A/B every new method vs the champion; revert if it loses (sample-uniqueness HURT, meta-labeling collapsed, trend_scan won on GLD).

## best so far · keep pushing
Re-validated + deflation-audited 2026-06-02 (the old "EEM 4.03 / book 4.22" were STALE artifacts). Honest state:
durable single-asset ML alpha is SCARCE — only **GLD trend_scan (2.16)** + **UUP imbalance_bgm (1.08)** survive
both re-validation and the deflated-Calmar selection-bias audit; EEM/TLT/IWM/DBC/XLE timing all collapse to
≤buy-hold or fail deflation. Everything else is best held PASSIVELY. **Deployable book = DECORRELATED CORE
(GLD/UUP/TIP/DBC/HYG), weight ∝ Calmar², Calmar 3.53, MaxDD 2.1%, Sharpe 2.06, positive every year.**
Decorrelation (not member-count) is the lever; dropping correlated high-MDD equities ~doubled Calmar.
Priorities: (1) squeeze GLD/UUP; (2) find a NEW durable, DECORRELATED edge (new axis / unsupervised label) that
JOINS the core; (3) book dials. Don't grind ceilinged buy-hold names.

## setup
QC project 31338454, creds `qc/.creds.json`. Hypothesis = `{ticker, axis, labeler, thresh, sizing}`.
Modules `autoresearch/modules/`; findings `knowledge.json` (provenance graph); deploy `DEPLOYMENT.md`;
reviews `RESEARCH_REVIEW.md` / `RESEARCH_REVIEW_v2.md`; backtest audit `BACKTEST_AUDIT.md`;
Wang's course `pdfs/`, `docs/legacy/wang_qa_questions.md`, `uni/transcripts/`.
