# Quant Autoresearch Workspace

An autonomous ML quant-research loop (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch))
applied to ETF trading on QuantConnect. An agent edits the pipeline, backtests on QC, scores on **REAL
out-of-sample Calmar + Drawdown Area**, keeps-if-better, and writes a tech report each round.

**Start here:** [`autoresearch/program.md`](autoresearch/program.md) — the one-page contract (goal, what's
editable, the loop, the rules).

## Layout

| Path | Purpose |
|---|---|
| `autoresearch/` | **The project.** |
| `autoresearch/program.md` | The contract / instructions (read first). |
| `autoresearch/harness/` | **LOCKED scorer** — `qc_client.py`, `orchestrator.py`, `evaluator.py` (gates G0–G4), `constants.py`. |
| `autoresearch/templates/` | `header/footer/infer` rendered into the QC `main.py` (footer downstream + infer sizing are editable). |
| `autoresearch/modules/` | **Editable pipeline** — `bar_builder.py` ① axis, `labeler.py` ② labels, `features.py` ③, `trainer.py` ④–⑧. |
| `autoresearch/reports/` | **Per-round tech reports as HTML** (`round_N.html`, MathJax math) + `index.html` + `TEMPLATE.html`. |
| `autoresearch/knowledge.json` · `techniques.json` | Shared memory (findings, dead-ends, idea queue). |
| `autoresearch/results.tsv` · `*_results.csv` | Result logs. |
| `scripts/` | QC drivers: `run_axis_label_study.py` (serial), `run_axis_label_parallel.py` (2-node), `_minify_check.py`; `diag/` scratch. |
| `qc/` | QC Cloud API client + `.creds.json` (gitignored). |
| `docs/research/` | Mined technique catalog + strategy-type spec. |
| `docs/superpowers/` | Experiment design + specs. `docs/legacy/` | older session summaries. |
| `uni/`, `uni_yt/` | Wang ("uni 的量化日记") course transcripts (idea source). |
| `pdfs/` (gitignored) · `qa_doc/` | AFML/MLAM/causal-investing + Wang course PDF + Q&A. |
| `_archive/` (gitignored) | Prior experiments, predecessor pipelines, corrupt-git backup, reset memory. |

## Running

```bash
cd /home/ubuntu/lb
# one experiment, both QC nodes:
python3 scripts/run_axis_label_parallel.py QQQ,IWM,EEM,XLE,HYG,TLT,GLD <axis> <labelers_csv>
```

QC project 31338454; each backtest is hard-capped at 5 min (auto-deleted on overrun). Splits: train ≤ 2021-08,
val ≤ 2023-08, test ≤ 2026-06 (OOS). Universe = 7 core ETFs (QQQ IWM EEM XLE HYG TLT GLD).

## Status (rounds so far)

Best **active** (>80-trade) config: EEM dollar Calmar 1.33 / GLD-complex on vol ~1.6. No single ETF clears the
G1 > 3.0 gate yet — the frontier is cross-asset pairs (archive: best pair 2.91). See `reports/index.html` and
`autoresearch/knowledge.json`.
