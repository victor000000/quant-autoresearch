# Wang's pipeline — canonical spec from the full corpus (2026-06-11)

Five parallel deep-reads covered the ENTIRE Wang corpus for the first time: 16 uni_yt
lecture transcripts + the course PDF (`wang_course_2026-06.pdf`) + the QA doc
(`wang_qa.tex`). **Provenance finding: only 9 of 16 lectures are actually Wang Yiming**
(the series mixes in vendor talks — FPGA hardware, FOF SaaS, platform demos, options
tools, LLM tooling — all triaged out). The genuine Wang sources: BV1a6enzLEVw,
BV1bpstzBEVm, BV1FnjuzUEJD (the 6-stage teardown), Ca1n2jgKjrs (entropy),
q922tUTWmCY (arbitrage-as-trend), tGzbK8c3R_A (vol forecasting), uVnOeOcoivw (the
pipeline spine), vL8DY2NP96I (autoencoders), WkZ6TUg_gK4 (frequency domain), plus the
2 lectures transcribed 2026-06-11 (ML-QA, joint distributions) and the course PDF + QA doc.

## 1. Wang's canonical pipeline (merged across all sources)

```
① DATA: minute OHLCV; reject the time axis ("congenitally broken — no statistical
   properties; intra-bar info density non-uniform")
② CUSTOM AXIS: resample by cumulative turnover (dollar bars) / volume / price-action;
   threshold rolling-standardized for secular drift; SELECT the axis by pushing the
   1st-difference return distribution toward NORMAL (minimize kurtosis / central peak;
   peak height ∝ chop frequency = where trend P&L bleeds); distribution SHAPE routes
   the mechanism (normal→trend, high-peak/bimodal→event/reversion)
③ LABELS ("重中之重"): discrete trend-leg labels via probability clustering (GMM) or
   change-point; whole leg labeled through pullbacks; trend-degree = the differencing
   ORDER (sweep 5..9) — a ladder of label strengths; label-mean ≈ 0.5 gate; NEVER naive
   next-bar sign; he REJECTS AFML triple-barrier and OLS-t trend-scan
④ FEATURES: fractional-differencing family (d≈0.7–0.8, 5 rolling windows; "production
   uses fracdiff, integer-diff loses history") + entropy (sample/approximate; r via
   range) + frequency-domain complement (VMD+NRBO boundary-safe; avoid FFT) + regime-
   as-FEATURE (HMM state, entropy-of-state) — 300–400 dims "enough"; standardize all
⑤ REDUCE: purpose = cut constraint terms to suppress overfit; linear PCA at variance
   threshold (99.99% → ~35 dims) AND nonlinear VAE (latent-dim by MSE elbow); may
   CONCATENATE PCA⊕VAE; pick the SMALLEST dim within tolerance; "linear reduction has
   a live-inference pitfall that cost us real money"
⑥ MODEL: discriminative tree (XGBoost/LGB/CatBoost/NGBoost), classification > regression,
   never NN-final ("LSTM as final is 非常烂"; LSTM only as encoder); consistency +
   repeatability are hard requirements (no LLMs/generative)
⑦ CALIBRATION/SIZING: the PRIMARY model's confidence "is NOT accuracy — you cannot size
   off it"; size from a SECONDARY (meta) model's probability; manage by Calmar
   (annualized/maxDD; ann.return ≈ Calmar × DD-budget); DA (drawdown area) for recovery
⑧ VALIDATION: "OOS-of-OOS" (the backtest window never touched by train OR inner
   val/test); A/B one stage at a time, never pre-judge; overfit fought at EVERY stage;
   claims event bars ≈ IID license stratified KFold
⑨ COMBINATION (his "no grail" core): ensemble the trend-degree ladder + multiple axes
   on the SAME asset — disagreement = chop = flat, agreement = trend = hold; 2–3 models
   suffice; select lower-dim-among-consistent, never top-Calmar; rank (not Pearson)
   correlation for book decorrelation; multi-asset multi-axis book to scale AUM
⑩ LIVE: train-once-freeze; rebuild bars+features online from the minute stream;
   real-time resample + infer; shadow models; guarded incremental retrain (3–6 mo,
   "retraining can break the balance")
```

His own emphasis ranking: (1) whole-pipeline 全局观/协同 (the part "no one else
teaches"); (2) the custom axis (Calmar 1.39→5.63 on RB from the axis alone);
(3) labels are pivotal; (4) non-linearity is the reason ML wins; (5) no grail — combine;
(6) OOS-of-OOS honesty.

## 2. Module-by-module: BUILT / EXCEEDS / DIVERGES / CLOSED-BY-EXPERIMENT / OPEN

The agents flagged many "MISSING" items that we have in fact ALREADY RACED TO CLOSURE —
this table is the reconciled truth (closure evidence in parentheses):

| Wang instruction | Status here |
|---|---|
| Dollar/turnover event bars, TRAIN-fit threshold | **BUILT** (logdollar = champion; imbalance 2nd) |
| Trend-leg labels through pullbacks, change-point | **BUILT** (`trend_leg` = champion labeler) |
| GMM probability-clustering labels | **BUILT** (`regime_gmm`/`bgm`) |
| Entropy features (sample/dispersion) | **BUILT** (8 in the base-80 panel) |
| log-diff base-e + StandardScaler | **BUILT** |
| PCA reduce, TRAIN-fit frozen | **BUILT — and our PCA-on-base lead (3.84) is directly validated: Wang's own demo reduce champion is linear PCA** |
| Information-gain selection | **BUILT** (`infogain`) |
| Discriminative tree classifier, no NN-final | **BUILT** (XGBoost d3) |
| Calmar headline + drawdown-area metric | **BUILT** (`lb.metrics` Pain×N ≡ his DA; Ulcer/Martin beyond it) |
| OOS-of-OOS + A/B one-variable discipline | **BUILT** (real online replay; A/B driver) |
| Probability output | **BUILT** |
| Calibration (isotonic/Venn-Abers), permute control, DSR/PBO/SPA/MinBTL/LOND/N_eff, leak teeth | **EXCEEDS — Wang specifies nothing comparable; every agent converged on this** |
| Sticky Gaussian-HMM forward-filter labeler (K=3, obs=[r,|r|], sticky .95, τ=.5) | **CLOSED-BY-EXPERIMENT** (built incl. sticky floor; GLD 0.40 @ val_auc 0.88 — predictable-not-profitable; HMM family closed) |
| VAE / autoencoder reduce (incl. 512-dim rich panel) | **CLOSED-BY-EXPERIMENT** (ae_np 2.97 < 4.02; AE 2.44→1.11; wangrich panels dilute under EVERY reducer; QC blocks torch) |
| Fracdiff features (d sweep) | **CLOSED-leaning** (crowd-out under correlation; wangrich×pca 2.60 < 3.84; panel-dilution generalizes — only a small TRUE-FFD-under-infogain probe remains, flagged low-EV) |
| Trend-degree ladder SOFT ensemble | **CLOSED-BY-EXPERIMENT** (tleg fast+mid+slow 3.66 < 4.02); unanimity hard-AND variant unraced (backlog, medium) |
| Multi-axis same-ticker combination | **CLOSED conditionally** (netting 3.57 < 4.02; reopen iff a 2nd same-ticker leg reaches ≥2.5 — peer-strength requirement, Wang's own implicit assumption) |
| Rolling-standardized bar threshold | **PART-RACED** (`logdollar_rc` USO 1.79 < 3.85 — smoothing killed the reversion concentration; unraced on GLD-trend where his drift argument is stronger) |
| Regime-as-FEATURE | **PART-RACED** (`features=regime` GLD 1.58 < 2.40 — diluted; HMM-state/entropy-of-state variant unraced, close cousin) |
| Meta-model probability as SIZING source | **PART-RACED** (meta-labeling raced as a gate and closed; "meta-prob drives size" variant unraced — sizing class carries the "sizers add no signal" prior) |
| Secondary crash/regime switch via vol FORECAST (FI-GARCH rolling threshold) | **OPEN** (unbuilt; sizing-class prior caps EV) |
| Frequency-domain features (VMD+NRBO online-safe; CWT+SWT; avoid FFT) | **OPEN** (unbuilt; boundary-effect handling is the leak-relevant part) |
| **1st-diff kurtosis/normality axis DIAGNOSTIC + shape ROUTER** | **OPEN — genuinely new** (we select axes by Calmar/val_auc only; his bar-quality objective kurt→0, peak_frac→0.383 and normal→trend / bimodal→reversion routing never implemented) |
| **Near-Gaussian LOW-density clocks (RealVar@5k, Vol@5k)** | **OPEN — genuinely new** (QA doc: his combined-recipe winner is RealVar@5k, NOT LogDollar@20k; we never raced deliberately low-N near-IID bars) |
| **Variance-threshold / minimum-K PCA** (99.99% → dynamic dims; smallest-K-within-tolerance) | **OPEN — cheap knob on the live PCA lead** (we fix K=20) |
| SHAP importance (± PCA cascade) | OPEN (medium; infogain is our supervised filter already) |
| Label-balance gate (mean≈0.5) + label-visualization QC | OPEN (cheap loop hygiene) |
| Calmar×DD-budget sizing rule; same-axis signal netting; lower-dim-among-consistent selection; plateau-parameter preference | OPEN (small, cheap) |
| Stratified KFold on event bars ("≈IID") | **REJECTED — keep purged/embargoed CV** (our evidence: overlap IS the signal; uniqueness weights hurt; stratified shuffle would leak serial structure) |
| Triple-barrier + OLS-t trend-scan "don't use" | **NOTED — partially agree by results** (our champions use trend_leg/revert, not triple_barrier; but trend_scan won SOXX historically — we keep both as library options) |
| Cross-ticker: multi-asset book, CS/rank path, spread-as-synthetic-instrument, index-constituent alignment | **OUT-OF-RULE** (single-ticker mandate; spread-as-one-tradeable would be a user policy decision) |
| Options IV / vol-surface / GNN transmission features ("the only forward-looking features") | **OUT-OF-DATA** (needs options modality — user decision; consistent with our closed exogenous-channel work) |

## 3. The honest bottom line

**We have already built — and in most cases raced beyond — Wang's pipeline.** Every
load-bearing stage of his spec exists here, usually with stronger honesty machinery than
he teaches. His three "crown jewels" that we lack are: (a) the **within-ticker ensemble
philosophy** (raced in two forms, both lost to the single-model champion — his demo gains
likely reflect his weaker baseline); (b) **VAE/nonlinear reduction** (raced, lost,
platform-blocked); (c) the **meta-model-sizing** doctrine (gate form raced and closed).
The corpus's genuinely NEW, never-raced, in-rule items are **axis-quality science**:

1. **Bar-quality diagnostic + mechanism router** — score every axis by 1st-diff excess
   kurtosis & central-peak mass (target normal: kurt→0, peak_frac→0.383); route
   normal→trend / bimodal→reversion. Host-side, zero QC cost, complements val_auc.
2. **RealVar@5k / Vol@5k near-Gaussian low-density clocks** — his QA-doc combined-recipe
   winner; a deliberately near-IID bar regime we never tried (our axes run dense).
3. **Variance-threshold + minimum-K PCA** on the live GLD PCA lead (cheap knob).
4. **PCA live-inference stability guard** — his "lost real money" warning; our OOS replay
   is immune (transform runs only in the training algo) but any LIVE deploy of the PCA
   config must freeze loadings — add an explicit check to `live_trade` rendering.
5. Loop hygiene: label-balance gate, label visualization, plateau-parameter preference,
   Calmar×DD-budget brief.

Items 1–3 are the recommended next races (after the 311 screen completes). Everything
else from the corpus is either built, closed by experiment with stronger evidence than
Wang's demos, out-of-rule, or out-of-data. Full per-lecture extraction tables live in the
five agent reports (session transcripts); garble glossaries included there.

## 4. Course-PDF deep crack (read first-hand, all 7 pages)

The syllabus (`wang_course_2026-06.pdf`: Shanghai 2026-06-13/14, ¥24,800) names his 7
"first-ever-public" levers; each decodes via the lecture corpus and reconciles to us:
① Calmar>3 full training = the QA 5-module recipe → we hold an HONEST equivalent (GLD
pca 3.84, permute+IAU-replication+deflation — gates he doesn't teach). ② custom-axis
system → built; the real secret is M3's last bullet "implement OTHER axes whose 1st-diff
approaches normal" = axis DESIGN against a normality objective (our queued race #1).
③ trendiness labels with "density tuned by trend behavior" = the trend-degree ladder
knob (soft ensemble raced/lost; ladder-knob sweep open). ④ nonlinear DR = VAE (raced/
closed here) — but M6 spends 2/4 bullets on "the problem LINEAR DR causes in live
inference and how to fix it" → the PCA-live-deploy FREEZE GUARD is a prerequisite for
crowning the GLD pca config into live trading (backtest immune; live_trade render must
freeze loadings). ⑤ secondary decision model = meta-prob-as-sizing (gate form raced/
closed; sizing-source variant open, low prior). ⑥ capital-scale = smoother equity via
multi-asset/axis/period COMBINATION → scale (in-rule version = our decorrelated book).
⑦ full code = n/a. M8 confirms his cross-asset-features lane ≡ our closed exogenous
channel. Core-value page: the claimed moat is whole-pipeline 统筹/协同 — i.e. the system,
not any module; this repo is that system with a stronger honesty layer.
