# Quant Workspace — Wang ML Pipeline

Per-folder map. Run scripts from this root directory.

## Top-level

| Path | Purpose |
|---|---|
| `pdfs/` | Reference books (AFML, CFI, MLAM, Wang course slides) |
| `yiweihua_books/` | 姜伟生 Iris series (218 PDF chapters) — preserved |
| `qa_doc/` | `wang_qa.tex/pdf`, V4-V6 final results, AFML excerpts |
| `qc/` | QuantConnect API helpers + active state |
| `qc/api.py` | Raw QC Cloud API (submit/read/poll) |
| `qc/inflight.json` | Active backtests being polled |
| `qc/last_results.json` | Most-recent results cache |
| `qc/results/` | Archived JSON result dumps (15 files) |
| `qc/.creds.json` | QC credentials (mode 0o600, never commit) |
| `scripts/` | `run_*.py` — backtest submission scripts |
| `tools/` | QC reporting + reorg utilities |
| `lean_workspace/` | LEAN/QuantConnect projects |
| `lean_workspace/_pipeline_v*_train\|infer\|ens/` | Active pipelines |
| `lean_workspace/0[1-8]_*/` | Numbered stage scaffolds |
| `lean_workspace/qqq_*/`, `slx_*/`, etc. | Historical experiments |
| `uni_videos/` | Today's Bilibili video (BV1VZLi6PE2B) + transcripts |
| `uni_yt/` | Prior Wang video transcripts (large-v3) |
| `data_cache/` | Downloaded reference data |
| `.venv-bili/` | Python venv for yutto (bilibili dl) |

## Running

Scripts assume cwd = workspace root:

```bash
cd /home/txy/lb
python3 scripts/run_v4_train.py
```

QC API uses `qc/api.py`; inflight tracked in `qc/inflight.json`.

## Pipeline versions tested

- v3: RV/Vol axis + 3-state [r,|r|] HMM
- **v4: Dollar bars + Trend Scanning** — winner for liquid stable ETFs
- v5: dollar + CUSUM
- **v6: Tick-count bars + TS** — winner for volume-drift ETFs
- v7: Renko + TS
- v8: EWMA-adaptive dollar + TS
- v9: dollar + K-bar persistence
- v10: dollar + extreme-move
- v11: dollar + combined (TS ∧ TB)
- v12: dollar + KMeans regime
- v14: range bars + TS
- v15: bull-bear segmentation labels

Per-asset best `Cal` (val-honest):
XLE 2.25 (v4) · GLD 1.31 (v4) · IWM 0.88 (v4) · QQQ 0.86 (v4) · EEM 0.64 (v6) · HYG 0.40 (v4) · TLT 0.25 (v6)

See `qa_doc/V4_V6_FINAL_RESULTS.md` for full report.
