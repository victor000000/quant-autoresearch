# Quant Autoresearch Workspace

An autonomous ML quant-research loop (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch))
applied to ETF trading on QuantConnect. An agent edits the pipeline, backtests on QC, scores on **REAL
out-of-sample Calmar + Drawdown Area**, keeps-if-better, and writes a tech report each round.

**Start here:** [`program.md`](program.md) — the one-page contract (goal, what's
editable, the loop, the rules).

## Layout

The repo uses an editable install (`pip install -e .` via `pyproject.toml`): all importable code lives under `src/lb/` (the `lb` package); mutable state (`knowledge.json`, `results/`, etc.) stays at the repo root, and `lb.paths.ROOT` is derived from the source tree so paths resolve correctly from the checkout. Note: the QuantConnect ObjectStore keys are still namespaced `autoresearch/{TICKER}/...` (a QC data path, deliberately unchanged).

| Path | Purpose |
|---|---|
| `program.md` | The contract / instructions (read first). |
| `src/lb/harness/` | **LOCKED scorer** — `qc_client.py`, `orchestrator.py`, `psuf.py` (cell-key helpers), `constants.py`. |
| `src/lb/templates/` | `header/footer/infer` `*.py.tmpl` rendered into the QC `main.py` (footer downstream + infer sizing are editable). |
| `src/lb/modules/` | **Editable pipeline** — `bar_builder.py` + `bar_ext.py` ① axis, `labeler.py` ② labels, `features.py` + `ml_ext.py` ③, `trainer.py` + `sizing_ext.py` ④–⑧. |
| `src/lb/console/` | Flask report server (entry point `lb-report`; restart after any `console/` change). |
| `src/lb/cli.py` · `src/lb/describe.py` | Entry-point shims (`lb-round` tournament driver; describe utilities). |
| `reports/` | **Per-round tech reports as HTML** (`round_N.html`, MathJax math) + `index.html` + `TEMPLATE.html`. |
| `knowledge.json` · `techniques.json` | Shared memory (`per_etf_best`, findings, dead-ends, idea queue). |
| `results.tsv` · `results/*.csv` | Result logs (`results/round_results.csv` + the axis-label CSVs). |
| `scripts/` | QC drivers at top level: `run_round.py` (v2 tournament, 2-node A/B), `run_axis_label_parallel.py` (full sweep, 2-node), `run_axis_label_study.py` (serial); grouped sub-dirs: `research/` (sweep/screen/decay scripts), `audit/` (honesty/DSR/leak scripts), `diag/` (render + report tools). |
| `hypotheses.json` | Per-ticker queued configs the driver reads each round. |
| `qc/` | QC Cloud API client + `.creds.json` (gitignored). |
| `docs/analysis/` | Analysis writeups — `HONEST_AUDIT.md`, `BACKTEST_AUDIT.md`, `DEPLOYMENT.md`, `CHAMPION_DECAY.md`, `RESEARCH_REVIEW*.md`. |
| `docs/research/` · `docs/superpowers/` · `docs/legacy/` | Mined technique catalog + strategy spec · experiment design + specs · older session summaries. |
| `refs/` | Reference material — `pdfs/` (AFML/MLAM/causal PDFs) and third-party course transcripts (`uni/`, `uni_yt/`, `qa_doc/`) are gitignored / excluded from the public repo. |
| `scratch/` (gitignored) | `_archive/` (prior experiments, predecessor pipelines) + `data_cache/`. |

**Root holds only:** the two entry docs (`README.md`, `program.md`), the operational state (`knowledge.json`, `techniques.json`, `hypotheses.json`, `results.tsv`), and the output dirs (`reports/`, `results/`, `docs/`). All importable code is under `src/lb/`; everything else is grouped under `docs/`, `refs/`, `scratch/`.

## Running

```bash
cd /home/ubuntu/lb
# v2 tournament round: pick the weakest ETF, race two hypotheses on the 2 QC nodes, keep the winner.
# explicit (two CONFIG JSONs, both targeting the same ETF):
python3 scripts/run_round.py \
  '{"ticker":"TLT","axis":"vol","labeler":"carry","thresh":0.55,"sizing":"binary"}' \
  '{"ticker":"TLT","axis":"dollar","labeler":"kmeans2stage","thresh":0.45,"sizing":"binary"}'
# auto (pick weakest ETF from knowledge.json, read its two configs from hypotheses.json):
python3 scripts/run_round.py
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
