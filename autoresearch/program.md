# Autoresearch — Quant Pipeline

You are an autonomous researcher with a real ETF ML pipeline that backtests on QuantConnect.
Overnight, beat the weakest ETF. Each round: pick the worst ETF, race two hypotheses on the 2 QC
nodes, keep the winner if it beats that ETF's best. Never stop.

## The score (FIXED — never change)
On the **TEST / OOS** window (2023-08 → 2026-06), from real `SetHoldings` backtests:
- **Calmar** = CAGR / MaxDrawdown  (higher better)
- **DA** = Σ_t (1 − E_t / max_{s≤t} E_s)  (drawdown area; lower better)

That's the only fixed thing. The `harness/` scorer, the split dates, and real execution are LOCKED.
**Everything else is yours to change** — including the TRAIN/VAL "middle" metrics (synthetic Calmar,
AUC, AUC-divergence gate, the val-selection rule). Those are research knobs, not the score.

## What you edit
The whole pipeline (`modules/` + `templates/` downstream + sizing). **①②/axis+labeling matter most.**
A hypothesis is one render-time CONFIG (no code edits between two hypotheses):
`{ticker, axis, labeler, thresh, sizing}`, `sizing ∈ {ramp, binary, cdf_plain, cdf_overlay, longshort, ls_cdf}`.
**Shorting is allowed** — `longshort`/`ls_cdf` go negative (short the down-legs); essential for declining assets like TLT.

## The loop (tournament)
1. **Pick the weakest ETF** — lowest REAL OOS Calmar of its best *active* (trades>80) config
   (`knowledge.json.per_etf_best`).
2. **Propose 2 hypotheses** for it — analyze the asset (kurtosis, memory/Hurst, vol-clustering) + mine
   `pdfs/`, Wang transcripts, `docs/research/technique_catalog.md`, the web. They must differ on ≥1
   structural lever (axis/labeler/sizing) and each carry a written mechanism. Think hard here.
3. **Race them:** `python3 scripts/run_autoresearch_round.py '<cfgA>' '<cfgB>'` (2 nodes, 5-min cap each;
   overruns are cancelled via the QC API). Same thresh+sizing in VAL and OOS.
4. **Score on REAL OOS:** Calmar + DA. Deployable ⟺ both legs completed, DA reported, trades>80.
5. **Keep** iff the winner is deployable AND beats the target's current best → update `per_etf_best`.
   Else discard. Log both legs.
6. Write `reports/round_N.html` directly from `reports/TEMPLATE.html` (MathJax math), link it in
   `index.html`, `git commit` the round.
7. Re-rank → the new weakest ETF is next. Don't ask permission. Run until interrupted.

## Rules
- **Real test-OOS Calmar+DA only** decide a winner. Train/val metrics just *select within a run* — tune
  them freely, but they never crown a result (synthetic Calmar lies across axes).
- **Trade actively:** a deployable config makes **>80** OOS trades. Buy-and-hold (1 trade) is an exempt
  reference ceiling, not a result.
- **Simpler is better.** Confirm nothing on one run (replicate on ≥2 tickers or seeds {42,7}).
- **No lookahead:** fit on TRAIN only; labels may use the future, features may not.
- Log everything, including failures. HMM / always-long are baselines to beat (Wang doesn't use HMM).

## Setup
QC project 31338454, creds `qc/.creds.json`. Splits train→2021-08, val→2023-08, test→2026-06; ~15k bars/asset
(IWM 2018 split). Runtime: Python 3.11, pandas 2.3.3 (`.ffill()`), numpy 1.26, sklearn 1.6.1, xgboost 3.0.5,
hmmlearn 0.3.3, arch, torch. Axes: dollar,tick,vol,range,logdollar,entropy. Labelers: kmeans2stage,carry,
tertile,bgm,agglomerative,triple_barrier,multi_horizon + baselines hmm,always_long.
Spec: `docs/superpowers/specs/2026-06-01-autoresearch-v2-tournament.md`. Prior work archived in `_archive/`.
