# New custom-axis + unsupervised-labeler research (2026-06-09)

# New custom-axis + unsupervised-labeling methods (2026-06-09)

Grounded against the live code: `modules/bar_builder.py` `_AXES_ORDER` = **27 axes built** (22 of them in `BUILDER_CLASSES` = online byte-exact deployable; `entropy, fracdiff, diurnal, kalman, newma` are dormant/batch because they carry a 2nd fitted param). `modules/labeler.py` `LABELERS` = **53 registry entries** (≈45 distinct mechanisms; incl. `hmm`/`sticky_hmm` baselines + the triple_barrier/tleg/ker ladders). The prompt's "21 axes / 45 labelers" ≈ this online/distinct subset. Confirmed `signature_lead_lag` (Lévy area) already lives at `modules/features.py:119-136`.

---

## 1. Bottom line

**Raw NOVEL_BUILDABLE verdicts: 12 axes + 21 labeler verdicts.** After collapsing duplicates (matrix-profile FLUSS proposed 3×, RQA 2×, ClaSP-segmentation folded into CLaP) → **12 genuinely-new axes and ~17 distinct new labeler mechanisms** that are QC-buildable (numpy-only, XGB target, no torch/HMM/sklearn-model) and unsupervised.

Honest prior, though: **novelty ≠ edge.** Component-selection history is unambiguous — only `logdollar`+`imbalance` axes have ever carried edges; "axis choice is asset physics, not tuning," and most microstructure clocks densify in the same stress regions as `vol`/`amihud` (so they coarsen where those coarsen and carry nothing new). So the **12 axes have a structurally LOW prior** regardless of citation quality; the only bet worth making is a *new information channel* orthogonal to vol/CUSUM/imbalance (exchangeability, efficiency-ratio, sign-sequence predictability). **Labelers have the better prior** — `meta-labeling`, `trend_leg`, `revert`, `ker`, `sadf_explosive` all show new labelers CAN carry where `val_auc>0.6` — but the trend-segmentation / change-point / clustering buckets are saturated ("swapping a mechanism's implementation always loses"). The labelers with real EV are the ones **outside** those buckets: a new *target principle* (cost-aware Bellman action path) and a new *structure* (subsequence shape-recurrence).

Net: of ~29 distinct candidates, **2–3 clear a build-worthy threshold** (Section 4); the rest are catalogued so the space is provably mapped.

---

## 2. New AXIS candidates (NOVEL_BUILDABLE, kind=axis) — ranked by EV

Online = single TRAIN scalar threshold, byte-exact append-OOS replay → `BUILDER_CLASSES`. Dormant = needs a 2nd frozen param (like `newma`/`kalman`/`diurnal`) → leak-safe but batch-only, caps deployability.

**A1. conformal test martingale axis** — *online; value 0.70 / novelty 0.85 (top axis)*
- Source: Vovk, "Retrain or not retrain: Conformal test martingales for change-point detection," PMLR v152 2021 (arXiv:2102.10439); Volkhonskiy et al. 2017 inductive-conformal-martingale CPD.
- NEW channel: distribution-free **exchangeability-violation** measured as a gambler's capital process on conformal p-values. Orthogonal to every existing change clock — `newma` is kernel-MMD on embeddings, `zcusum` is CUSUM-of-mean, `vol`/`icss`-family is 2nd-moment.
- Build: rolling past-only window (≤500) of conformity scores `s_t=|r_t−median(past r)|/MAD(past r)`; conformal p-value `p_t=(#past≥s_t [+tie])/(n+1)` (past-only ranks → causal); betting update `M_t=M_{t-1}·f(p_t)` with a Simple-Jumper `f` whose jump rate is a FIXED constant; accumulate `logM`; **emit + reset M=1 when logM ≥ τ** (τ = single TRAIN scalar to target bar count). Fixed jump rate + past-only ranks + lone TRAIN scalar → byte-exact.
- Why not one of our 27: none of `newma/zcusum/vol/entropy` is a betting process on rank-exchangeability; it fires on *any* distributional non-exchangeability, not a moment.

**A2. MOSUM moving-sum mean-shift axis** — *online; 0.55 / 0.55 (cleanest change axis)*
- Source: Eichinger & Kirch, Bernoulli 24(1) 2018; Cho-Kirch-Meier `mosum` R pkg (JSS 2021).
- NEW channel: fixed-bandwidth contrast of two *adjacent trailing* windows — spikes only in a G-neighborhood of a local mean shift, naturally multi-scale via bandwidth G. `zcusum` accumulates an unbounded global path; MOSUM is localized + refractory.
- Build: per minute on `r=Δlog close`, keep trailing ring sums `S_left`[k−2G,k−G], `S_right`[k−G,k], fixed G; `M_k=(S_right−S_left)/(σ̂·√(2/G))`, σ̂ = causal running vol; **emit when |M_k|≥T**, impose G-length refractory as reset. Backward windows only → causal. Optional G-ladder for multi-scale.
- Why not one of our 27: it's a *localized/refractory* mean-shift contrast; `zcusum` is the *cumulative* path. Value rests on G-scale-selectivity decorrelating from CUSUM.

**A3. vratio (variance-ratio / price-efficiency clock)** — *online; 0.50 / 0.72*
- Source: Lo & MacKinlay, Rev. Fin. Studies 1988.
- NEW channel: dimensionless martingale-deviation `VR(q)=Var(q-ret)/(q·Var(1-ret))` — samples fast in trending (VR>1) or mean-reverting (VR<1), slow under clean random walk. A price-*discovery-efficiency* clock; the existing vol family measures the variance LEVEL, not the cross-horizon RATIO.
- Build: ring of last q log-closes; `r1=lc[t]−lc[t−1]`, `rq=lc[t]−lc[t−q]`; EWMA `V1=(1−a)V1+a·r1²`, `Vq=(1−a)Vq+a·rq²` (fixed a, a hyperparam like fracdiff's d); `VR=Vq/(q·V1)`; **accumulate `|VR−1|`, emit at TRAIN threshold**. Recursive EWMA + scalar → byte-exact.
- Why not one of our 27: `vol/volofvol/semivar/signedjumpvar` are variance levels; `fracdiff` is differencing-memory; `hurst_persist` is a labeler. No cross-horizon ratio clock exists.

**A4. flowent_clock (order-flow transition-entropy-rate)** — *online (w/ online trailing mean); 0.45 / 0.78*
- Source: fresh arXiv (2512.x); mechanism stands on Kolmogorov-Sinai entropy-rate of a sign Markov chain.
- NEW channel: nonlinear **sign-sequence predictability** (conditional entropy of the tick-rule sign chain), not net flow magnitude. `entropy` axis is a marginal 5-bucket Shannon surprise; `vpin` is BVC toxicity magnitude; `imbalance/tickimb/volumeimb` accumulate NET sign; `ofsc` is a LINEAR serial-corr *labeler*.
- Build: `s_t=sign(Δlog close)∈{−1,0,+1}`; update 3×3 transition counts `C[prev,cur]`; conditional entropy `H_t=−Σ_i π_i Σ_j p(j|i)log p(j|i)`; maintain **online trailing mean `H̄`**, accumulate `max(0, H̄−H_t)` (low entropy-rate = structured flow); emit at TRAIN threshold. Fixed sign-states + online H̄ + scalar → byte-exact. (A TRAIN-frozen baseline instead of online H̄ would drop it to dormant.)
- Why not one of our 27: no axis measures nonlinear predictability of the *sequence* of signs; it is sign-permutation-invariant so it densifies on informed-flow concentration.

**A5. HFD (Higuchi fractal-dimension roughness clock)** — *online; 0.50 / 0.50*
- Source: Higuchi, Physica D 31 1988; Gómez-Gómez et al., Fractals 24 2016. `NEW_METHODS_BACKLOG` flags it explicitly as "missing online anti-persistence axis."
- NEW channel: curve-length-vs-scale **path roughness**, different small-sample bias than DFA. `range`/`run` are linear displacement; `wavelet` is band energy; `hurst_persist`(DFA) is a labeler.
- Build: past-only rolling log-close window; for k=1..8 build interleaved subsampled curves, sum normalized lengths L(k); `D=slope(log L(k) vs log(1/k))` via one `np.linalg.lstsq`; **accumulate `max(0, D_t−1)`** (excess roughness above smooth=1) to TRAIN threshold. No 2nd fitted param (accumulate excess, not |D−baseline|) → online. A/B vs `hurst_persist`.
- Why not one of our 27: there is no fractal-dimension sampling clock; the roughness geometry is distinct from variance and band-energy.

**A6. surprise_clock (Bayesian KL posterior-shift)** — *DORMANT (2-param); 0.50 / 0.60*
- Source: Itti & Baldi, Vision Research 49(10) 2009.
- NEW channel: `KL(posterior‖prior)` parameter-belief revision (predictive-coding / learning-rate clock). Distinct from `entropy` (static TRAIN-frozen bucket surprise), `newma` (kernel-MMD), deferred `bocpd` (run-length posterior).
- Build: per-minute NIG conjugate update of return mean/precision with a **frozen forgetting factor γ≈0.985** (else posterior variance→0 and the clock DIES over growing OOS — the documented staleness mode); closed-form `KL(NIG_t‖NIG_{t-1})`; accumulate to TRAIN threshold. γ + prior = 2nd param → dormant tier, same as `entropy`.
- Why not one of our 27: belief-revision KL has no analog; but the forgetting factor makes it a 2-param axis (not pure `BUILDER_CLASSES`).

**A7. turbulence_clock (Mahalanobis financial-turbulence)** — *DORMANT (frozen μ,Σ⁻¹); 0.45 / 0.60*
- Source: Kritzman & Li, FAJ 66(5) 2010.
- NEW channel: multivariate distance-from-fitted-joint-distribution (correlation-ellipsoid break). No axis computes a multivariate Mahalanobis distance.
- Build: TRAIN-freeze `μ, Σ⁻¹` over `x_t=[Δlogc,|Δlogc|,k-bar ret,sign·log(1+vol),short-RV]`; per minute `d_t=(x_t−μ)Σ⁻¹(x_t−μ)ᵀ`; accumulate, emit at TRAIN θ. Frozen μ,Σ⁻¹ → dormant/batch like `newma`.
- Why not one of our 27: off-diagonal correlation breakdown is genuinely new — BUT the proposed FV is vol-dominated, so **validate decorrelation of emission times vs the `vol` axis before crowning** (only the off-diagonal is orthogonal).

**A8. exvol (MDH excess-volume / information-arrival)** — *online (ratio form); 0.40 / 0.50*
- Source: Dey-Wang / Gallo, J. Fin. Markets 13(3) 2010.
- NEW channel: order flow arriving WITHOUT a commensurate price move (volume residual conditional on volatility). Build online: `s_t=vol_t/(β·EWMA_rv+ε)` (β a single frozen TRAIN scalar), accumulate `max(s_t−1,0)`, emit at θ. (Exact `E[vol|RV]` regression form = dormant.)
- Why not one of our 27: `dollar/tick/logdollar` are raw volume level; vol-family is squared returns. **Caveat:** high excess-volume ≈ inverse of `amihud` (|ret|/$vol) — require decorrelation check vs `amihud`/`chl`/`vpin`.

**A9. ofpersist (order-flow sign-persistence / long-memory)** — *online; 0.40 / 0.40*
- Lag-1 volume-weighted autocovariance of the tick-rule sign sequence: `A+=w_t·s_t·s_{t-1}` (clamp negatives to 0 so chop stalls), emit at θ. `imbalance/tickimb/volumeimb` accumulate NET flow; `vpin` is the fraction; the sign *serial-correlation* is `ofsc` (a labeler). Discount: overlaps `run` (directional persistence) — A/B vs `run`.

**A10. innov (studentized-innovation forecast-surprise)** — *online; 0.40 / 0.45*
- `r̂` = EWMA forecast, EWMA var `v`; `e=|r−r̂|/√(v+ε)`; accumulate, emit at T. New vs inventory = subtracting an ADAPTIVE forecast before measuring surprise (`zcusum` = const-zero mean). Caveat: studentized increment ~|N(0,1)|≈0.8/step → near-constant rate, degenerates toward a tick clock except in regime shifts when σ̂ lags. Thin, concentrated edge.

**A11. SlopEn (slope-symbol surprise)** — *DORMANT (clean form); 0.38 / 0.43*
- Cuesta-Frau, Entropy 21(12) 2019. 5-symbol slope-direction+amplitude alphabet. True surprise form `−log(p_word)` needs a TRAIN-frozen word histogram → 2nd param → dormant. Third entropy-family construct; closest to existing `entropy` axis → incremental.

**A12. e-detector axis (anytime-valid e-process)** — *online (running-normalizer); 0.30 / 0.40 (lowest)*
- Shin-Ramdas-Rinaldo, NEJSDS 2023 (arXiv:2203.03532). Mixture e-process `e←e·Emix(z)`, emit at log(e)≥T. **Strongest skeptic mark:** the entire selling point is Ville-validity, which is *discarded* once you fit a scalar threshold to target bar count — stripped of validity it's `exp(running LLR)` ≈ the CUSUM/GLR that `zcusum` already embodies. High reparameterization risk.

---

## 3. New LABELER candidates (NOVEL_BUILDABLE, kind=labeler) — ranked by EV

All: unsupervised structure → per-bar 0(short)/1(long)/−1(no-trade) target; features past-only; label may look ahead bounded by `_EMBARGO=max(200,horizons)`; numpy-only, XGB head.

**L1. dp_oracle_label (cost-aware Bellman perfect-foresight oracle)** — *0.80 / 0.80 (top labeler)*
- Source: "Label-Driven Optimization of Trading Models," Mathematics 13(23):3889 2025; "dynamic threshold breakout labeling," 2024.
- NEW structure: a **friction-aware globally-optimal ACTION path** (Bellman optimality with an explicit per-flip cost), not a path statistic. The single genuinely-missing property in the book: **cost / trade-frequency awareness in the target.**
- Build: per anchor t, backward DP over `[t,t+H]`, H≤200, states `s∈{−1,0,+1}`: step reward `s·lr[k+1]`, transition penalty `c·|s_k−s_{k-1}|`; `V[k][s]=reward(s,k)+max_{s'}(V[k+1][s']−c·|s−s'|)`; forward-decode optimal path; **label[t]=optimal action at anchor** (+1→1, −1→0, flat→−1). O(3·H)/anchor numpy (reuse `jump_model`'s Viterbi machinery). Sweep cost c, H; pick TRAIN-balanced.
- Why not one of our 53: verified ≠ `triple_barrier` (first-touch fixed barriers), `trend_leg`/`trend_scan` (slope/zigzag), `revert`/`turn_scan` (reversal timing), `mfe_mae` (excursion ratio), `jump_model` (Viterbi minimizing ‖z−centroid‖²+λ·switches, no PnL/cost). It is NOT in the saturated trend/change-point/cluster buckets.

**L2. matrix-profile FLUSS arc-curve segmentation** *(collapses mp_fluss_seg / mpseg / FLUSS-CAC, proposed 3×)* — *0.70 / 0.80*
- Source: Gharghabi, Ding, Yeh, Keogh, "Matrix Profile VIII / Domain-Agnostic Online Semantic Segmentation," DMKD 33 2019. `NEW_METHODS_BACKLOG` lists it as a recognized-but-unbuilt gap.
- NEW structure: model-free segmentation by **subsequence-recurrence GEOMETRY** (arc-crossing density) — where the local SHAPE VOCABULARY changes. Every existing break detector is a parametric test (`changepoint`=mean, `icss_var`=variance, `bocpd_label`=Gaussian run-length, `sadf_explosive`=unit-root, `bde_cusum`=recursive residual). `visgraph` is amplitude-HVG; `sliced_wasserstein` is OT on sorted windows — neither is self-similarity.
- Build: z-normalize length-m close subsequences; matrix-profile NN index via MASS (`numpy.fft` sliding dot-products) + trivial-match exclusion; FLUSS arc curve `AC[i]=#NN-arcs crossing i` ÷ idealized parabolic count; cut at deepest corrected minima → segments; **label each segment's bars by sign of segment net forward return** (1/0), −1 near boundaries/short segments. Sweep m, #regimes.
- Why not one of our 53: subsequence self-similarity is absent. **Caveat:** O(N²) self-join — cap to the ~few-thousand-bar series (seconds w/ FFT MASS); offline batch is fine for a target.

**L3. dpc_regime (density-peak clustering)** — *0.58 / 0.74*
- Source: Rodríguez & Laio, Science 2014.
- NEW structure: ρ (local density) vs δ (distance-to-nearest-higher-density) decision graph — nonparametric density-mode regimes that **natively isolate low-density transition/outlier bars as a no-trade regime** (no existing clusterer does this). `regime_gmm/bgm` assume Gaussian; `kmeans2stage/agglomerative/tertile/setar` are centroid/threshold.
- Build: reuse `regime_gmm` causal FV; on TRAIN (subsample N≤Nmax) pairwise dist; `d_c`=TRAIN distance quantile; `ρ_i=Σ_j exp(−(d_ij/d_c)²)`; `δ_i`=min over higher-ρ neighbors; `γ=ρδ`; centers=top-k γ (k from γ-gap); assign each bar to nearest higher-density neighbor's regime; flag low-ρ bars no-trade; **regime→direction by sign of TRAIN mean fwd-return**; |mean|<floor or isolated → 0.
- Why not one of our 53: density-mode + native outlier-no-trade is new. **Caveat:** O(N²) (same as in-tree `agglomerative`); must beat `bgm` on decorrelation — its no-trade behavior is the differentiator.

**L4. l1tf_label (L1 / total-variation trend filtering)** — *0.55 / 0.68*
- Source: Kim, Koh, Boyd, Gorinevsky, "ℓ1 Trend Filtering," SIAM Review 51(2) 2009.
- NEW structure: a GLOBAL convex program with TV-of-slope L1 penalty that places knots by sparsity over the whole window simultaneously — continuous piecewise-LINEAR (vs `changepoint`'s piecewise-constant). `trend_leg`=greedy zigzag, `trend_scan`=forward-OLS t-stat, `kllt`=Kalman RTS, `accel`=2nd-diff.
- Build: `x*=argmin ½‖y−x‖²+λ‖D2 x‖₁` via ~50-iter ADMM (pentadiagonal Thomas factorization once; soft-threshold z-update); slope `s_t=x*[t+1]−x*[t]`; **label=sign(s_t)**, 0 where |s_t|<TRAIN tertile. Ladder fast/mid/slow via fixed λ grid.
- Why not one of our 53: distinct global-convex inductive bias. **HONEST DISCOUNT:** trend-segmentation is the most saturated bucket (~12 trend labelers); earns its place only if global-convex decorrelates from greedy on whipsaw assets. A/B vs `trend_leg`.

**L5. ggs (Greedy Gaussian Segmentation)** — *0.50 / 0.50*
- Source: Hallac, Nystrup, Boyd, ADAC 2019.
- NEW structure: contiguous JOINT mean+covariance regularized-MLE segmentation; the novel niche is segmenting on **cross-feature COVARIANCE shifts** (`regime_gmm/bgm` ignore contiguity; `changepoint`=mean; `icss_var`=variance).
- Build: small causal `X=[lr,|lr|,short-vol]`; greedy add breakpoint maximizing `Σ −n_s/2·logdet(Σ_s+λI)`, refine to fixed point; **label by segment joint state** (up/low-vol→1, down/high-vol→0, ambiguous→−1). `np.linalg.slogdet` only.
- Why not one of our 53: covariance-shift segmentation is new. **Discount:** in 1-D it collapses toward `bocpd_label` — value needs multivariate (now feasible since cross-asset features are allowed).

**L6. spectral_regime (graph-Laplacian eigen-clustering)** — *0.45 / 0.50*
- Normalized-Laplacian eigengap clustering (manifold/non-convex geometry) vs centroid/Gaussian-density. Build: kNN-sparse affinity W → `L=I−D^{-1/2}WD^{-1/2}` → `np.linalg.eigh` bottom-k → KMeans the embedding → regime→dir by TRAIN mean fwd-return. **Moderate novelty:** new ALGORITHM inside the saturated cluster scaffold (reuses `regime_gmm`'s map-then-1NN). Cap N for O(N²).

**L7. tda_sublevel_regime (0-D sublevel-set persistence)** — *0.45 / 0.40*
- Source: Gidea & Katz, Physica A 2018 (crash-precursor Lp norms). 0-D sublevel persistence via union-find merge tree (sort heights, union components, record birth/death) → features = total/max persistence + landscape L1/L2 → gate forward-return sign by persistence regime. **Novelty discount:** the identical 0-D mechanism is already spec'd as the deferred `persistence` AXIS — this is its labeler form, not a new discovery. (Full Vietoris-Rips = QC-infeasible, see §5.)

**L8. clap_state (CLaP recurring-state-detection)** *(ClaSP segmentation = its subroutine)* — *0.45 / 0.80 (most novel mechanism)*
- Source: Ermshaus-Schaefer-Leser, PVLDB 2025 (CLaP); Schaefer-Ermshaus-Leser, CIKM 2021 (ClaSP).
- NEW structure: **state RE-OCCURRENCE / recurrence identity** — segment then MERGE into a small alphabet of REVISITED states. Absent: every regime labeler clusters i.i.d. points or labels segments independently; none does contiguous-segment-then-merge into recurring ids.
- Build: (1) segment log-close via ClaSP — kNN self-join → binary classification-score profile → recursive split (pure-numpy kNN, NO ROCKET/sklearn); (2) iteratively merge the pair with highest mutual classifier confusion until gain<thr → state alphabet; (3) **state→direction by sign of TRAIN mean fwd-return**; transition/short states → 0.
- Why not one of our 53: recurrence-identity is genuinely new. **Heaviest build; gate hard + permute-test the recurrence-direction assumption** (a recurring state can flip sign on recurrence — unproven).

**L9. RCMSE (refined-composite multiscale sample-entropy)** — *0.42 / 0.55*
- Costa MSE 2002 / RCMSE 2014. Adds amplitude-tolerance template-matching (SampEn) vs `perment`'s ordinal-only entropy + a multiscale coarse-graining curve. Reuses `perment`'s gate-low-complexity-then-forward-sign template → moderate. **Caveat:** SampEn O(w²)/scale/bar → cap w≤120, S≤5.

**L10. KCPD (kernel multiple-change-point penalized-DP)** — *0.40 / 0.50*
- Arlot-Celisse-Harchaoui, JMLR 2019. Exact-Gram RKHS multi-break DP with model-selection penalty that auto-chooses #breaks. Build: `G_ij=exp(−‖r_i−r_j‖²/2σ²)`, DP minimizing within-segment kernel variance + `pen(D)`. New METHOD in a covered class; O(N²) → cap ~1500/segment.

**L11. OPTN (ordinal-pattern transition-network entropy)** — *0.40 / 0.50*
- Transition entropy of the ordinal-motif Markov chain (irreversibility/dominant flows) that `perment`'s static frequency histogram discards (two series with equal permutation entropy differ in transition entropy). Gate low-transition-entropy → forward sign. Shares `perment`'s gate → novelty is the statistic, not the gating.

**L12. RQA determinism/laminarity** *(collapses RQA + rqa_det_label, proposed 2×)* — *0.40 / 0.70*
- Source: Bastos & Caiado, Physica A 390 2011; Marwan et al., Phys. Rep. 438 2007. DET (diagonal-line fraction = predictability) / LAM (vertical = critical-transition warning) from a phase-space recurrence matrix. Build: Takens-embed returns (m,τ); rolling `R_ij=1[‖x_i−x_j‖<ε]`, ε=causal pct for fixed recurrence rate; DET/LAM via run-length counts; **gate high-DET→forward sign, LAM-spike→−1**. **Caveats:** notoriously sensitive to (m,τ,ε,l_min); on a 1-D embedding correlates with `run`/`ker` autocorrelation → demand orthogonality vs `perment`/`run`. Predictability-class-adjacent to `perment`.

**L13. levy_area_label (path-signature Lévy-area price-vs-flow)** — *0.40 / 0.45 — partly ALREADY_HAVE*
- Source: Chevyrev & Kormilitzin, arXiv:1603.03788. **The math already exists in-tree** at `modules/features.py:119-136` (`signature_lead_lag`, returns `0.5·(s12−s21)/sd`) — currently a cross-asset FEATURE. Newness = labeler-role + price-vs-flow. Build: `X=cumsum(Δlogclose)`, `Y=cumsum(sign(Δclose)·vol)`; `A=0.5·(s12−s21)/(…)`; gate on |A|≥TRAIN-q, direction from forward return (rotation *magnitude* = gate, NOT rotation sign = direction — that mapping is economically unmotivated). **Low marginal EV:** would surface via `features=signature` if predictive; close-only flow proxy shares the sign(Δp) driver → partial degeneracy.

**L14–L17 (tail, compact):**
- **RuLSIF density-ratio change** (0.35/0.35) — Liu-Yamada-Sugiyama, Neural Networks 43 2013. New divergence ESTIMATOR; mechanism (two-window shift→segment) is the most-mined family (`changepoint/cusum_regime/sliced_wasserstein/newma`). Incremental — A/B before adopting.
- **louvain_net_regime** (0.35/0.55) — Blondel 2008. Greedy-modularity community detection on a kNN similarity graph; new PARTITIONING but in the saturated cluster bucket; "regime mechanism is asset-intrinsic" → expect it to lose to `bgm`.
- **ticc_mrf_regime** (0.32/0.58) — Hallac et al. (TICC). Inverse-covariance/partial-correlation-network regimes (genuinely new channel) BUT requires a hand-rolled ADMM graphical-lasso nested in EM×Viterbi — heaviest build, real 64k-render + runtime risk; low value-per-effort.
- **Chen graph-based change-point** (0.30/0.45) — kNN/MST edge-count two-sample scan. Distribution-free multivariate break detector (fires on dependence breaks univariate detectors can't see) — but the 6th change-point labeler; build ONLY paired with cross-asset features + strict uniqueness check vs `changepoint`/`cusum_regime`.

---

## 4. The 2–3 highest-EV to build first

Picked for **least redundancy with the saturated buckets** + buildability, not just citation quality.

**① dp_oracle_label (L1) — build first.** It is the only candidate that adds a *property the book lacks* (cost/trade-frequency awareness) via a *target principle* not in any existing bucket. Cheapest heavy-lifter (O(3H) numpy DP, reuses `jump_model` Viterbi), clean 3-class XGB target.
- **A/B race:** on **GLD** (structured, `val_auc>0.6`, `trend_leg` champion 3.47) and **USO** (oil mean-reversion, `revert` champion 2.18). For each ticker, hold the full pipeline fixed and swap only the labeler: `dp_oracle_label` vs the asset's champion. Sweep cost c (bps) and H; keep TRAIN-balanced.
- **Permute control (the decisive test):** scramble the optimal-action labels (permuted-label harness) and re-run — a real edge must collapse to ≈BH, as proven before (UUP real 1.30 vs permuted −0.08; sadf real 1.85 vs −0.09). Then gate: beats-BH baseline + decay + cost + DSR (`N_eff`-deflated for search burden) before any book consideration.

**② conformal test martingale axis (A1) — build second.** The only *online, top-value, genuinely-new-channel* axis; the exchangeability betting process is orthogonal to every existing clock, and online byte-exact replay keeps it deployable.
- **A/B race:** axes only carry edges on `logdollar`/`imbalance`, so the honest test is **head-to-head against the asset's champion axis** with the champion labeler fixed: on **GLD** run `conformal_martingale + trend_leg` vs `logdollar + trend_leg`; on **UUP** run `conformal_martingale + (bgm+sadf+ker)` vs `imbalance + (bgm+sadf+ker)`.
- **Control:** axes can't take a label-permute; instead require (a) **decorrelation of emission times** vs `logdollar`/`vol`/`zcusum` (if it samples in the same places it carries nothing new), and (b) the standard decay/cost/DSR gate. Prior is LOW — treat as a channel-existence probe.

**③ matrix-profile FLUSS arc-curve labeler (L2) — build third (optional).** Strongest peer-reviewed source, a genuinely new STRUCTURE (shape recurrence) outside the parametric-break and clustering buckets, and explicitly backlogged.
- **A/B race:** on the weakest *structured* name — race `mp_fluss` vs `changepoint` and `sliced_wasserstein` (the nearest existing segmenters) on **GLD**/**SLV**/**XME**-class; cap N + window m for the O(N²) join and the 64k file limit.
- **Control:** permute the *segment boundaries* (random cut positions, same count) — a real shape-recurrence edge must beat random segmentation; plus the uniqueness/decorrelation check vs `changepoint`'s segmentation.

Rationale for ordering: dp_oracle has the highest value AND the cleanest "new property" argument; conformal is the single defensible new *axis channel* (and axes are the bigger blind spot); matrix-profile is high-value but O(N²)-gated so it's the optional third.

---

## 5. Rejected — the space is mapped

**ALREADY_HAVE (mechanism already in the book):**
- **rollbounce** (Roll/EDGE serial-covariance transient-impact, axis) — covered by `chl` (effective-spread) + `amihud`.
- **ghperm** (Glosten-Harris permanent-impact / adverse-selection, axis) — covered by `kyle`/`amihud` price-impact.
- **modwt_trend** (labeler) — `wavelet` axis + the trend-labeler family already encode MODWT-style multi-resolution trend.
- **l1tf** (bare framing, labeler) — overlaps the trend family. *Nuance:* the specific global-convex TV-penalty build (`l1tf_label`, L4) was judged distinct enough to be NOVEL, but it lands in the most-saturated bucket — hence it's a §3 tail entry, not a first build.
- **levy_area_label** (partial) — the Lévy-area math already exists as a cross-asset FEATURE at `modules/features.py:119-136`; novel only in labeler-role (kept in §3 with low EV).

**QC_INFEASIBLE (violates the hard constraints):**
- **Topological persistence — full persistence-landscape / Vietoris-Rips** (labeler) — needs `ripser`/`gudhi` (non-numpy, QC-blocked). Only the **0-D sublevel-set variant** (`tda_sublevel_regime`, L7) is numpy-buildable, and even that duplicates the deferred `persistence` axis mechanism.

**DUBIOUS (buildability or distinctness doesn't hold up — also shows the boundary):**
- **reservoir_state_regime** (Echo-State random reservoir, labeler) — random recurrent reservoir is torch-shaped / overfit-prone, doesn't cleanly satisfy the XGB-only + numpy constraints.
- **bhp** (boosted Hodrick-Prescott trend, labeler) — boosted-HP collapses into the saturated trend family.
- **wpc** (weighted-price-contribution stealth price-discovery, axis) — needs intraday cross-section / OHLC the close-only bar contract forbids.
- **Bubble Entropy** (almost-parameter-free rank-swap complexity, labeler) — overlaps the entropy/`perment` family; marginal.

This catalogue (12 axes + ~17 labelers NOVEL_BUILDABLE; 4 ALREADY_HAVE; 1 QC_INFEASIBLE family; 4 DUBIOUS) demonstrates the single-ticker axis/labeler design space is now substantially enumerated: the remaining novelty is concentrated in **cost-aware targets** and **shape-recurrence / exchangeability channels**, which is exactly where the three first-builds sit.