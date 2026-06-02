# autoresearch

Single-ticker ETF ML on QuantConnect, Wang's pipeline. Each round: pick the weakest ticker, race
two hypotheses on the 2 nodes, keep the winner iff it beats that ticker's best on real OOS Calmar.

**Do not stop exploring.** Always another axis, label, feature, reduce, model, sizing to try.
**Simple is best.** **Single-ticker only — no cross-ticker ensembling.**

## the loop
1. Pick the weakest ticker (lowest real OOS Calmar). Re-validate its stored best first — records go stale.
2. Think — read the provenance graph + findings; co-design one ticker's `axis × label × features × reduce × model × sizing`.
3. Race: `run_autoresearch_round.py '<A>' '<B>'` (the driver auto-updates the report).
4. Keep iff **deployable (trades>80) AND Calmar>0 AND > re-validated best AND val_auc>0.52 AND beats `always_long` AND survives deflation**. Else discard. Record → commit.

## never break — backtest contract (audited clean, see BACKTEST_AUDIT.md)
Real OOS backtest is **online, leak-free, model-only-from-QC-ObjectStore.** `infer.py` holds no model (replays
saved predictions + causal `_size`); every `.fit` is in `footer.py` on TRAIN(+embargoed VAL) only; test enters
only via predict. Proven: `verify.py` bars ≤1e-9, `infer_online.py` p_live==p_saved ≤1e-6. Features past-only,
thresholds TRAIN-only. Don't trust a champion until its `infer_online` shows preds_match=1.

## wang's backbone
Resample off the clock → label **unsupervised** (the label may look ahead; causality lives in the supervised
model on past-only features) → **rich features then reduce** (fit on TRAIN) → **bet-size.** Detectors:
trend-scan / change-point / clustering — **NOT HMM.** Aim Calmar > 3, reproducible, deployable.

## why the gates (hard-won)
- **Records go stale.** OOS window grows as data arrives → a short-window-lucky edge decays. Re-validate before trusting.
- **Trust = trials-deflated** (Deflated Sharpe / `deflated_audit.py`): the max of N tries is upward-biased; a searched
  edge must clear the best-of-N noise. `always_long` baselines carry no selection bias.
- **Durable > lucky:** drift/long-biased edges persist; two-sided timing with val_auc≈0.5 decays. A/B every new method vs the champion.

## honest state (2026-06-02)
Durable single-ticker alpha is **scarce** — only **GLD trend_scan (2.16)** and **UUP imbalance_bgm (1.08)** survive
re-validation + deflation; EEM/TLT/IWM/DBC/XLE timing collapse to ≤buy-hold or fail deflation; the rest is best
held passively. (The old "EEM 4.03" was a stale window artifact.) Find **new durable single-ticker edges** — a new
axis or unsupervised label. The deployable book is just a downstream combination of single-ticker champions — not the research target.

## setup
QC project 31338454, creds `qc/.creds.json`. Hypothesis = `{ticker, axis, labeler, thresh, sizing}`.
Code `autoresearch/modules/` · findings `knowledge.json` (provenance graph) · audit `BACKTEST_AUDIT.md` ·
reviews `RESEARCH_REVIEW*.md` · Wang's course `pdfs/`, `docs/legacy/wang_qa_questions.md`.
