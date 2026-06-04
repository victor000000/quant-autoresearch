# Quant Autoresearch Workspace

An autonomous ML quant-research loop (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch))
applied to ETF trading on QuantConnect. An agent edits the pipeline, backtests on QC, scores on **REAL
out-of-sample Calmar + Drawdown Area**, keeps-if-better, and writes a tech report each round.

**Start here:** [`program.md`](program.md) — the one-page contract (goal, what's
editable, the loop, the rules).

## Layout

The repo **root IS the project** (flattened 2026-06-04 — `autoresearch/` promoted to root). Note: the QuantConnect ObjectStore keys are still namespaced `autoresearch/{TICKER}/...` (a QC data path, deliberately unchanged).

| Path | Purpose |
|---|---|
| `program.md` | The contract / instructions (read first). |
| `harness/` | **LOCKED scorer** — `qc_client.py`, `orchestrator.py`, `evaluator.py` (gates G0–G4), `constants.py`. |
| `templates/` | `header/footer/infer` rendered into the QC `main.py` (footer downstream + infer sizing are editable). |
| `modules/` | **Editable pipeline** — `bar_builder.py` ① axis, `labeler.py` ② labels, `features.py` ③, `trainer.py` ④–⑧. |
| `reports/` | **Per-round tech reports as HTML** (`round_N.html`, MathJax math) + `index.html` + `TEMPLATE.html`. |
| `knowledge.json` · `techniques.json` | Shared memory (`per_etf_best`, findings, dead-ends, idea queue). |
| `results.tsv` · `results/*.csv` | Result logs (`results/round_results.csv` + the axis-label CSVs). |
| `scripts/` | QC drivers: `run_autoresearch_round.py` (v2 tournament, 2-node A/B), `run_axis_label_parallel.py` (full sweep, 2-node), `run_axis_label_study.py` (serial), `_minify_check.py`; `diag/` scratch. |
| `qc/` | QC Cloud API client + `.creds.json` (gitignored). |
| `docs/research/` | Mined technique catalog + strategy-type spec. |
| `docs/superpowers/` | Experiment design + specs. `docs/legacy/` | older session summaries. |
| `uni/`, `uni_yt/` | Wang ("uni 的量化日记") course transcripts (idea source). |
| `pdfs/` (gitignored) · `qa_doc/` | AFML/MLAM/causal-investing + Wang course PDF + Q&A. |
| `_archive/` (gitignored) | Prior experiments, predecessor pipelines, corrupt-git backup, reset memory. |

## Running

```bash
cd /home/ubuntu/lb
# v2 tournament round: pick the weakest ETF, race two hypotheses on the 2 QC nodes, keep the winner.
# explicit (two CONFIG JSONs, both targeting the same ETF):
python3 scripts/run_autoresearch_round.py \
  '{"ticker":"TLT","axis":"vol","labeler":"carry","thresh":0.55,"sizing":"binary"}' \
  '{"ticker":"TLT","axis":"dollar","labeler":"kmeans2stage","thresh":0.45,"sizing":"binary"}'
# auto (pick weakest ETF from knowledge.json, read its two configs from hypotheses.json):
python3 scripts/run_autoresearch_round.py
```

A CONFIG is `{ticker, axis, labeler, thresh, sizing}` (`sizing ∈ ramp|binary|cdf_plain|cdf_overlay`); the driver
renders one TRAIN → one INFER per hypothesis with the SAME thresh+sizing on VAL and OOS, then keeps the winner
iff it is deployable (G2 trades>80, DA reported) and beats that ETF's current best. It logs both legs to
`results/round_results.csv` and updates `knowledge.json.per_etf_best`; the HTML report + git commit
are done per round by the human/opus. (The full-sweep driver `run_axis_label_parallel.py` is still available.)

QC project 31338454; each backtest is hard-capped at 5 min (auto-deleted on overrun). Splits: train ≤ 2021-08,
val ≤ 2023-08, test ≤ 2026-06 (OOS). Universe = 7 core ETFs (QQQ IWM EEM XLE HYG TLT GLD).

## Status (rounds so far)

Six rounds done. Per-ETF best **active** (>80-trade) real OOS Calmar (the v2 leaderboard, lowest = next target):
EEM 1.33 · HYG 1.26 · QQQ 1.10 · GLD 0.78 · XLE 0.72 · IWM 0.65 · **TLT −0.15** (weakest link → round-1 v2 target).
Rounds 5–6 showed single-asset long-only ML ties buy-hold because the label is washed out (overlay) or reverts
to always-long (overlay off); v2 attacks one ETF at a time with label-driven `binary` sizing so the model can go
FLAT. No single ETF clears G1 > 3.0 yet. See `reports/index.html` and `knowledge.json`.
