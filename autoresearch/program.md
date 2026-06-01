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
2. **Propose 2 hypotheses** for it — **reason from the causal graph** (`knowledge.json.causal_graph`,
   rendered at `reports/causal_graph.html`): derive each hypothesis from a **FINDING** node (the mechanism
   hubs), not from scratch. Also analyze the asset (kurtosis, memory/Hurst, vol-clustering) + mine `pdfs/`,
   Wang transcripts, `docs/research/technique_catalog.md`, the web. They must differ on ≥1 structural lever
   (axis/labeler/sizing) and each carry a written mechanism tracing back to a finding. Think hard here.
3. **Race them:** `python3 scripts/run_autoresearch_round.py '<cfgA>' '<cfgB>'` (2 nodes, 5-min cap each;
   overruns are cancelled via the QC API). Same thresh+sizing in VAL and OOS.
4. **Score on REAL OOS:** Calmar + DA. Deployable ⟺ both legs completed, DA reported, trades>80.
5. **Keep** iff the winner is deployable AND beats the target's current best → update `per_etf_best`.
   Else discard. Log both legs.
6. Write `reports/round_N.html` directly from `reports/TEMPLATE.html` (MathJax math), link it in
   `index.html`. **Update the causal graph:** append the round's node(s) + causal edge(s) to
   `knowledge.json.causal_graph` (types: finding/round/milestone/decision; a KEEP becomes a milestone, a
   new mechanism becomes a finding hub), then run
   `python3 scripts/render_causal_graph.py --inject reports/round_N.html --label "round N" --highlight "<new node ids>" --note "<reasoning path>"`
   — this regenerates `reports/causal_graph.html` AND embeds the up-to-date graph in the round report, **red-ringing
   this round's new/changed nodes** and printing the reasoning path so a reader can see what changed. `git commit` the round.
7. Re-rank → the new weakest ETF is next. Don't ask permission. Run until interrupted.

## Rules
- **Real test-OOS Calmar+DA only** decide a winner. Train/val metrics just *select within a run* — tune
  them freely, but they never crown a result (synthetic Calmar lies across axes).
- **Trade actively:** a deployable config makes **>80** OOS trades. Buy-and-hold (1 trade) is an exempt
  reference ceiling, not a result.
- **Simpler is better.** Confirm nothing on one run (replicate on ≥2 tickers or seeds {42,7}).
- **No lookahead (G3 invariant):** features causal (past-only); the model/scaler/dim-reduce/calibrator
  fit on TRAIN (calibrator on labeled VAL). The future-derived **label may filter only the bars used to
  FIT** — it must **never** decide which VAL/TEST bars get a position. TEST predictions + the VAL synth
  metric are emitted for **every causal bar** (`fv & mask`), not just labeled (`y>=0`) bars. *(Audit 2026-06-01:
  `ex = fv & (y>=0) & te_m` violated this — it let forward-derived labels pick OOS trades with hindsight.
  Severe for `multi_horizon` (kept only all-horizon-agreement bars → the bogus XLE +2.50) and `tertile`;
  ~harmless for `triple_barrier`/clustering labelers which label every bar. Fixed in `footer.py.tmpl`.)*
- Log everything, including failures. HMM / always-long are baselines to beat (Wang doesn't use HMM).

## Setup
QC project 31338454, creds `qc/.creds.json`. Splits train→2021-08, val→2023-08, test→2026-06; ~15k bars/asset
(IWM 2018 split). Runtime: Python 3.11, pandas 2.3.3 (`.ffill()`), numpy 1.26, sklearn 1.6.1, xgboost 3.0.5,
hmmlearn 0.3.3, arch, torch. Axes: dollar,tick,vol,range,logdollar,entropy. Labelers: kmeans2stage,carry,
tertile,bgm,agglomerative,triple_barrier,multi_horizon + baselines hmm,always_long.
Spec: `docs/superpowers/specs/2026-06-01-autoresearch-v2-tournament.md`. Prior work archived in `_archive/`.
