# autoresearch

Single-ticker ETF ML on QuantConnect, Wang's pipeline. Each round: pick the **weakest** ticker, race
two hypotheses on the 2 nodes, keep the winner iff it beats that ticker's best on real OOS Calmar.

**Do not stop exploring.** Always another axis, label, feature, reduce, model, sizing to try.
**Simple is best.** **Single-ticker only — no cross-ticker ensembling.**

## the loop
1. **Pick the weakest ticker** (lowest real OOS Calmar — never the strongest). Re-validate its stored best first — records go stale.
2. **Think** — read the provenance graph + findings; co-design one ticker's `axis × label × features × reduce × model × sizing`.
3. **Build a new method.** The edge comes from a *better method in some module*, not another permutation of old ones — invent one (a new label, feature, sizer, …) and A/B it against the champion.
4. **Race:** `run_autoresearch_round.py '<A>' '<B>'` (the driver auto-updates the report).
5. **Keep** iff **deployable (trades>80) AND Calmar>0 AND > re-validated best AND val_auc>0.52 AND beats `always_long` AND survives deflation.** Else discard. Record → commit.

## never break — backtest contract (audited clean, see BACKTEST_AUDIT.md)
Real OOS backtest is **online, leak-free, model-only-from-QC-ObjectStore.** `infer.py` holds no model (replays
saved predictions + causal `_size`); every `.fit` is in `footer.py` on TRAIN(+embargoed VAL) only; test enters
only via predict. Proven: `verify.py` bars ≤1e-9, `infer_online.py` p_live==p_saved ≤1e-6. Features past-only,
thresholds TRAIN-only. Don't trust a champion until its `infer_online` shows preds_match=1.

## wang's backbone
Resample off the clock → label **unsupervised** (the label may look ahead; causality lives in the supervised
model on past-only features) → **rich features then reduce** (fit on TRAIN) → **bet-size.** Fixed downstream:
`StandardScaler → reduce_dims(corr,20) → XGBoost(depth 3, lr .03, n 200) → isotonic calibrate (VAL).`
Detectors: trend-scan / change-point / clustering — **NOT HMM.** Aim Calmar > 3, reproducible, deployable.

## why the gates (hard-won)
- **Records go stale.** OOS window grows as data arrives → a short-window-lucky edge decays. Re-validate before trusting.
- **Trust = trials-deflated** (Deflated Sharpe / `deflated_audit.py`): the max of N tries is upward-biased; a searched
  edge must clear the best-of-N noise. `always_long` baselines carry no selection bias.
- **Durable > lucky:** drift/long-biased edges persist; two-sided timing with val_auc≈0.5 decays. A/B every new method vs the champion.
- **New methods help only where there's structure** (val_auc>0.6). On val_auc≈0.5 drifters no method beats buy-hold — don't grind them.

## honest state (2026-06-03)
Durable single-ticker alpha is **scarce** — two edges survive re-validation + deflation:
**GLD `ker+trend_scan` 3.20** (gold's tradeable trend) and **UUP `bgm+ker` 1.30** (dollar regime). Everything else
is buy-hold drift (QQQ/EEM/EFA/HYG/TIP/DBC/XLE/IWM/SLV) or no-edge (TLT, negative under every method). The KER
labeler (Kaufman efficiency-ratio) was the breakthrough — it lifted *both* edges; the deployable book is just a
downstream combination of these champions, not the research target.

**Closed levers — don't re-grind:** model capacity (depth-5 overfits 3.20→1.71; depth-3 optimal) · model family
(sklearn ExtraTrees is platform-blocked in QC's runtime — XGBoost only) · meta-labeling (weaker than KER on GLD,
over-filters UUP, decayed on EEM) · VR + SPY-cross-asset features (crowd the correlation-select, regress) · universe
siblings (SLV doesn't generalize — GLD's edge is gold-idiosyncratic, not a precious-metals property). A real
cross-asset edge needs a 2-symbol *pairs* strategy, which violates single-ticker.

**Open frontier — new method per module.** The remaining alpha is a genuinely *new method* in some module, tested on
the weakest names: new **labels** (built `ker` ✓ win, `accel`/`sharpe_scan` valid but no win yet), new **bar axes**,
new **features** (non-crowding family), new **reducers** (vs corr-20), new **sizers** (drawdown-aware vs `cdf_overlay`).

## setup
QC project 31338454, creds `qc/.creds.json`. Hypothesis = `{ticker, axis, labeler, thresh, sizing[, max_depth]}`.
Modules `autoresearch/modules/` (bar_builder · features · labeler · trainer) + templates (header/footer/infer/verify) ·
findings `knowledge.json` (provenance graph) · audit `BACKTEST_AUDIT.md` · reviews `RESEARCH_REVIEW*.md` ·
Wang's course `pdfs/`, `docs/legacy/wang_qa_questions.md`.
