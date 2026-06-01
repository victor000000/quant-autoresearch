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
`{ticker, axis, labeler, thresh, sizing}`, `sizing ∈ {ramp, binary, cdf_plain, cdf_overlay, longshort, ls_cdf, ls_overlay}`.
**Shorting is allowed** — `longshort`/`ls_cdf` go negative (short the down-legs); essential for declining assets like TLT.

## The loop (tournament)
1. **Pick the weakest ETF** — lowest real OOS Calmar with trades>80 (`per_etf_best`).
2. **Target** (think hard) — run `python3 scripts/target_next.py`, then enumerate candidate interventions,
   each from a **FINDING** node of the causal graph with a one-line mechanism; rank by *expected gain ×
   confidence* and *edge-disambiguation*; the 2 hypotheses are the top two (differ on ≥1 lever). Propose
   from the graph, not from scratch.
3. **Race** — `python3 scripts/run_autoresearch_round.py '<A>' '<B>'` (one ETF, 2 nodes).
4. **Score real OOS** — Calmar + DA. Deployable ⟺ both legs done, DA reported, trades>80.
5. **Keep** iff deployable AND beats the ETF's best — but **leak-check first** (see Rules); then update
   `per_etf_best`, else discard. Log both legs.
6. **Record** — `python3 scripts/render_round.py '<spec json>'` (hero + stat cards + table, shared
   `style.css`), append the round's node/edges to `knowledge.json.causal_graph`
   (KEEP→milestone, new mechanism→finding), run
   `python3 scripts/render_causal_graph.py --inject reports/round_N.html --label "round N" --highlight "<ids>" --note "<path>"`,
   `git commit`.
7. **Re-rank** → next weakest. Never stop.

## Rules
- **Real OOS Calmar+DA only** crown a winner. Train/val metrics only *select within a run* — tune freely.
- **Trade actively:** deployable = **>80** OOS trades. Buy-and-hold (1 trade) is a reference ceiling, not a result.
- **No lookahead (every round):** features past-only; model/scaler/dim-reduce/calibrator fit on TRAIN
  (calibrator on labeled VAL). The future label may filter only the bars used to **fit** — never which
  VAL/TEST bars get a position. Check the footer's `n_pred ≈ n_test_bars` sentinel; if a KEEP looks too good
  (Calmar ≫ buy-hold / DA ≪ buy-hold / trades ~80), it must beat an `always_long` control before it counts.
- **Simpler is better.** Confirm nothing on one run (replicate on ≥2 tickers or seeds {42,7}).
- Log everything, including failures. HMM / always-long are baselines to beat (Wang doesn't use HMM).

## Setup
QC project 31338454, creds `qc/.creds.json`. Splits train→2021-08, val→2023-08, test→2026-06; ~15k bars/asset
(IWM 2018 split). Runtime: Python 3.11, pandas 2.3.3 (`.ffill()`), numpy 1.26, sklearn 1.6.1, xgboost 3.0.5,
hmmlearn 0.3.3, arch, torch. Axes: dollar,tick,vol,range,logdollar,entropy. Labelers: kmeans2stage,carry,
tertile,bgm,agglomerative,triple_barrier,multi_horizon + baselines hmm,always_long.
Spec: `docs/superpowers/specs/2026-06-01-autoresearch-v2-tournament.md`. Prior work archived in `_archive/`.
