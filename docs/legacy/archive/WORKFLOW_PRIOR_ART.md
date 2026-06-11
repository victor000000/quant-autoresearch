# Prior Art: Who Else Runs This Workflow?

Deep-research sweep (2026-06-04): 5 search angles → 22 sources fetched → 82 claims →
25 adversarially verified (3-vote panels, 24 confirmed / 1 killed). Venues searched:
YouTube/bilibili/Zhihu, SSRN, arXiv, Google Scholar, books, quant blogs (Hudson & Thames,
QuantStart, QuantInsti, Quantopian-legacy), GitHub.

## Bottom line

**No public source runs the complete 7-step pipeline + the cross-sectional "trend-on-the-spread"
extension as one named workflow.** That synthesis appears unique. But **every individual component
is off-the-shelf**, almost all from López de Prado's *Advances in Financial Machine Learning* (AFML).
The genuinely novel, no-precedent-found piece is **the spread gated on POSITIVE autocorrelation**.

## It's AFML underneath

Steps 1, 2, 3, 6 are López de Prado's methods, not original:

- **Trend-segmentation labeling (step 2) == AFML "trend scanning"** (Lecture 3/10 + *Machine Learning
  for Asset Managers*): fit look-forward regressions t→t+L, pick max-t-value slope, emit {−1,0,1}.
  *Verified 3-0.* Even the "alternative to triple-barrier" framing is canonical AFML.
- Information-driven bars (1), fractional differentiation (3), meta-labeling (6): AFML chapters.

## (a) Full AFML-pipeline teachers / coders

| Source | Overlaps Wang on | Diverges / missing |
|---|---|---|
| **Hudson & Thames — `mlfinlab` + `ArbitrageLab`** (the canonical public impl) | steps **1, 2, 3, 6** in one library (info-driven bars + trend-scanning + fracdiff + triple-barrier/meta-labeling) | no spread-as-trend, no VMD, no VAE, no GARCH gate. (Paywalled/deprecated post-2021; historical code + docs still corroborate.) |
| **`boyboi86/AFML`** (GitHub) | AFML base (triple-barrier, fracdiff, NCO clustering), Ch. 2–16, sklearn/statsmodels only | explicitly lacks trend-segmentation, VMD, VAE, GARCH, DBSCAN, spread framing |

## (b) Closest single match — custom bars + trend-segmentation *together*

- **Yoo/Lim et al. (2024), "Deep neural network… directional predictability of multi-stock returns"**
  (ScienceDirect S2199853124002324). Best config **"DB-TSC" literally pairs dollar bars with
  trend-scanning labels** = Wang steps 1+2 fused. **Divergence:** DNNs, not Wang's tree-only step 5.
  *Verified 3-0.* The single nearest published instance of the distinctive bars+trend-seg pairing.

## (c) "Trend on the spread" — nobody matches the mechanism (firmest finding)

Three stat-arb works share only the **clustering/pair-formation** step and **all gate the opposite way**:

| Source | Clustering | Signal / gate (≠ Wang) |
|---|---|---|
| Han-He-Toh, EJOR 2023 ("Pairs Trading via Unsupervised Learning") | K-Means / DBSCAN / agglomerative | cross-sectional 1-month **momentum reversal** — no A−βB spread, no autocorr gate |
| Korniejczuk, arXiv 2406.10695 (2024) | graph / Signed-Laplacian / SPONGE | 5-day cluster-mean **reversion** |
| Hudson & Thames **ArbitrageLab** | PCA + OPTICS/DBSCAN | **cointegration + Hurst<0.5 + half-life** = explicit **mean-reversion** |

→ **Every public cross-sectional method gates on mean-reversion/cointegration — the statistical
OPPOSITE of Wang's positive-autocorrelation / trend-persistence gate.** No source gates a constructed
spread on positive autocorrelation and runs it through a trend-segmentation pipeline. *Verified 3-0.*

## Fragments of the exotic sub-stack (each isolated, never combined his way)

- **Springer 2025 crypto paper** (s40854-025-00866-w): info-driven bars — but triple-barrier (rejected by
  Wang) + Transformers (rejected by Wang).
- **VMD-GARCH-LSTM paper** (Wiley ijmm/2710277): only the VMD+GARCH sub-stack, single-asset regression, LSTM.
- **Borsa Istanbul 2021 paper** (s40854-021-00243-3): only the VAE dimensionality-reduction piece (step 4).

## Implications for this project

- The parts we implement (bars, trend labels, fracdiff, meta-labeling, tree models) are **independently
  validated public methods** — real, not folklore.
- The **"trend on the spread, gated on positive autocorrelation"** idea (our internal "wall-breaker")
  has **no public precedent found** — it is the one differentiated, defensible edge: a no-new-data escape
  from the single-ticker no-structure wall, and this prior-art gap is an extra reason to prioritize it.

## Open questions (unresolved by the sweep)

1. Does Wang's own `uni 的量化日记` / DCP or any Chinese educator publish the identical 7-step combo?
   (Chinese-language coverage was thin — Zhihu/bilibili hit anti-scrape walls; absence-of-evidence only.)
2. Any public source that gates a spread A−βB on **positive** autocorrelation (ADF/DW/ACF) then runs a
   trend-segmentation pipeline? **None found.**
3. Anyone combining VMD (over FFT) + FIGARCH/GARCH sizing + tree-based trend-segmentation together?
4. Repos/courses doing trend-scanning on info-driven bars feeding **trees specifically** (vs. the
   DNN/Transformer choices in the matched papers)?

## Key sources

- Hudson & Thames mlfinlab — https://github.com/hudson-and-thames/mlfinlab ; https://hudsonthames.org/mlfinlab/
- Trend-scanning docs — https://random-docs.readthedocs.io/en/latest/implementations/labeling_trend_scanning.html
- H&T meta-labeling + dollar-bar normality — https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/
- DB-TSC paper — https://www.sciencedirect.com/science/article/pii/S2199853124002324
- Han-He-Toh pairs via unsupervised learning — https://www.sciencedirect.com/science/article/abs/pii/S037722172200769X
- Korniejczuk graph-clustering stat-arb — https://arxiv.org/abs/2406.10695
- ArbitrageLab ML pairs selection — https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/ml_approach/ml_based_pairs_selection.html
- boyboi86/AFML — https://github.com/boyboi86/AFML
