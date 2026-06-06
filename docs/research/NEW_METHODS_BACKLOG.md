# New axes + labelers backlog (deep-specs)

_Workflow explore-axes-labelers (wksd2pm3d), 2026-06-06. 12 web-search agents → dedupe vs existing → 14 deep-specs, all online-causal + leak-safe + numpy._

## Summary

Synthesized 14 deep-specs (cross-checked against the live pipeline: 21 registered axes in _AXES_ORDER, ~40 labelers, the uniform update(ts,close,vol) builder contract, scalar BUILDER_CLASSES vs the multi-param entropy/fracdiff verify path, and the _safe_thresh/_train_minute_mask/_minute_log_returns helpers). All 14 are online-causal and leak-safe under the audited rate-on-TRAIN-then-scale recipe; none is a rename of an existing method. I recommend 11 new axes and 7 new labelers, dropping only the two near-duplicate labelers the specs themselves flag (LZ-labeler and persistence-entropy-label both collapse to perment) and the ICSS/SETAR axis halves (redundant with volofvol/spectral or a category error). The highest-leverage picks open mechanism families the program has ZERO coverage of: seasonality (diurnal), self-excitation/aftershock memory (Hawkes), online kernel-MMD full-distribution change (NEWMA), state-space trend (Kalman), liquidity-cost effective-spread (chl), probabilistic regime-age (BOCPD), and topological depth (persistence). The cheapest wins are pure scalar accumulators that mirror vol/ddonset (semivar, chl, signedjumpvar) and the entropy-style frozen-profile axis (diurnal). Bias the queue toward tail/crash and seasonality mechanisms, which align with the only-ever SPY crash-veto life and attack the documented open/close-churn confound in every minute clock.

## Top 3 to build first

- semivar (downside RS- clock): HIGH priority at the LOWEST build cost — a ~15-line scalar accumulator that mirrors vol/ddonset, zero deps, BUILDER_CLASSES byte-exact, and it directly extends the only-ever SPY crash-veto life from the sampling side.
- chl (Abdi-Ranaldo effective-spread clock): HIGH priority, scalar/O(1), opens a liquidity-COST dimension orthogonal to all 3 existing liquidity axes and maps cleanly to the illiquid commodity/energy/FX deep-sweep fits (USO/UCO/GSG/DJP/GDX/XME/AGQ/UUP/HYG).
- diurnal clock + diurnal_anomaly label: HIGH priority, near-free build (entropy-style frozen profile), and it is the program's FIRST seasonality method — attacks the open/close U-shape confound in every minute clock and yields both a new axis and a new labeler from one fit.

## Recommended AXES (modules/bar_builder.py)

### semivar (downside realised-semivariance / bad-vol clock)  [high]
**Why:** First downside-asymmetry sampling axis. vol() sums RS+ and RS- together (and sqrt-vol-weights), averaging the documented good/bad-vol asymmetry away; semivar isolates the bad-vol half (Patton-Sheppard) and concentrates bars on down-minute crash build-ups. Directly extends the only-ever SPY crash-veto life from the sampling side. Lowest build cost in the whole backlog: a scalar accumulator that mirrors vol/ddonset, zero deps, BUILDER_CLASSES-compatible (byte-exact replay).

**Impl:** class SemivarBarBuilder: thresh,cum=0,last_lc=None. update: lc=log(close); if last_lc is not None: r=lc-last_lc; if r<0: cum+=r*r; last_lc=lc; if cum>=thresh: cum=0; emit{ts_close,log_close}. _make_builder 'semivar': ret=_minute_log_returns(c); tr=_train_minute_mask; terms=np.where(np.isfinite(ret)&(ret<0),ret*ret,np.nan); keep=tr&np.isfinite(terms); total=np.mean(terms[keep])*len(c); return SemivarBarBuilder(_safe_thresh(total,target_bars)). Register in _AXES_ORDER/AXES/BUILDER_CLASSES. Add tests/test_bar_threshold_leak teeth check.

**Source:** Patton & Sheppard 2015 REStat 97(3):683 (RS-=Σr²·1{r<0}); Barndorff-Nielsen, Kinnebrock & Shephard 2010 SSRN 1262194

### chl (Abdi-Ranaldo effective-spread liquidity clock)  [high]
**Why:** New liquidity-COST dimension orthogonal to all 3 existing liquidity axes (amihud=impact/dollar, kyle=√-share lambda, vpin=BVC toxicity); none estimates the effective spread and chl uses no volume — pure close-vs-range-center geometry. A cost clock densifies bars exactly in illiquidity/stress windows where adverse selection and reversals cluster, targeting the less-liquid commodity/energy/FX deep-sweep fits (USO/UCO/GSG/DJP/GDX/XME/AGQ/UUP/HYG). Scalar threshold, O(1)/min, zero deps.

**Impl:** Non-overlapping W=30-min pseudo-day blocks: track running max/min/last of log-close; on block completion eta_cur=(max+min)/2,c_cur=last; if prev block exists: term=(c_prev-eta_prev)*(c_prev-eta_cur); cum+=sqrt(4*term) if term>0 else 0; shift prev<-cur, reset block; emit at cum>=thresh. _make_builder 'chl' replays the IDENTICAL block loop over minutes writing per-block terms[i] at block-end, keep=tr&isfinite(terms), total=mean(terms[keep])*(len(c)/W), _safe_thresh. Scalar .thresh -> BUILDER_CLASSES. Caveat (not a leak): high/low proxied by max/min of minute closes.

**Source:** Abdi & Ranaldo 2017 RFS 30(12):4437 (eta=(H+L)/2; S²=4(C-eta_t)(C-eta_{t+1})); SSRN 2725981

### diurnal (deseasonalised time-of-day RV clock)  [high]
**Why:** Opens an ENTIRELY new mechanism family — the program has ZERO seasonality/calendar methods (spectral is FFT power, not calendar-aligned; zcusum standardises by a TRAILING window, not minute-of-day). Removes the deterministic open/close U-shape that dominates every dollar/vol clock, so bars sample on diurnally-anomalous activity (information shocks) instead of predictable churn. Near-free build: mirrors the shipped entropy axis (TRAIN-frozen profile array passed to the constructor). Pairs with the diurnal-anomaly labeler below.

**Impl:** class DiurnalVolBarBuilder(threshold,prof,gmean): prof=len-1440 TRAIN mean|r| per minute-of-day. update: r=lc-last_lc; m=ts.hour*60+ts.minute; rt=r/(prof[m] or gmean); cum+=rt*rt; emit at cum>=thresh. _fit_diurnal_axis: tr=_train_minute_mask; mod=(ts vectorised astype('datetime64[m]') -> minute-of-day); prof[m]=mean(|r|) over tr&(mod==m) (>=20 obs else gmean); total=mean((r[tr]/prof[mod])**2)*len(c). Multi-param (frozen array) -> entropy-style verify path, store T in .thresh. Optional FFF smoother = one np.linalg.lstsq of log(prof) on [1,n/N1,n²/N2,cos/sin(2πpn/N)].

**Source:** Andersen & Bollerslev 1997 J.Emp.Finance (Flexible Fourier Form intraday deseasonalisation)

### kalman (local-linear-trend filtered-slope axis)  [high]
**Why:** No state-space estimator exists (trend_leg=greedy zigzag, trend_scan=OLS t-stat, accel=finite-diff). The Kalman LLT gives an MLE-bandwidth, noise-filtered causal slope WITH an uncertainty band; at steady state it is a frozen-gain Holt double-EMA so the online builder is two scalar IIR updates. Emits on slope-strength rising-edge (|z| hysteresis) to stay distinct from spectral's zero-crossing. trend_leg already wins GLD 3.47 — a cleaner adaptive slope should lift the trend-momentum cohort (GLD/IAU/XME/GSG/SPXL/QLD/UUP).

**Impl:** class KalmanTrendBarBuilder(threshold,k0,k1,sig_slope): update: pred=level+slope; v=lc-pred; level=pred+k0*v; slope+=k1*v; z=slope/sig_slope; if armed and |z|>=thresh: armed=False; emit; if |z|<0.5*thresh: armed=True. _fit_kalman_axis: R=var(minute returns on tr); vectorised 4x4 grid-MLE over (q_level,q_slope) log-spaced (one python loop, length-G numpy state); Riccati-iterate to steady-state gain (k0,k1) and sig_slope=sqrt(P_pred_ss[1,1]); binary-search delta on a TRAIN builder sim to target_bars*train_frac. Multi-param -> store (k0,k1,sig_slope) like fracdiff; thresh=delta. GUARD: sig_slope from Riccati fixed point, NOT a full-series slope std.

**Source:** Harvey 1989 (LLT); Rauch-Tung-Striebel 1965; statsmodels statespace_local_linear_trend

### hawkes (self-exciting intensity / compensator clock)  [high]
**Why:** New mechanism: aftershock MEMORY. Every existing axis hard-resets (imbalance/run/vpin) or is memoryless (jump); none carries a decaying self-excitation kernel. lambda_t=mu+Σα·exp(-β(t-t_i)) over jump events stays elevated for an aftershock window, so the clock over-samples the post-jump cascade window the jump axis is silent through (markets ~80% endogenous; equity branching ratio n~0.7-0.9). Strongest on event-driven/leveraged/energy names (SPY/QQQ macro, SPXL/QLD/UCO/AGQ, USO/XLE/DJP); pairs with tail labelers (crash_ahead, sadf_explosive, rskew).

**Impl:** Markov state S (O(1)): on each minute decay cum+=mu*dt+(S/β)(1-exp(-β·dt)); S*=exp(-β·dt); event if |L|/sigma_bipower>=ev_k (trailing K=60 bipower sigma, same as jump) -> S+=alpha; emit when cum>=Q, carry cum-=Q. dt in minutes (overnight gaps decay S~0). _fit_hawkes_axis (TRAIN-only, pure numpy, no scipy): ev_k=quantile(|L_train|,0.98); 6x6 grid over (n,tau): beta=1/tau, mu=lambda_bar*(1-n), alpha=n*beta; argmax exact Ozaki-Ogata 1979 O(N_ev) loglik (recursion R_i=exp(-β·dt)(1+R_{i-1})); Q=mean(TRAIN lambda)*len(c)/target_bars. 5-param -> entropy-style verify path; store Q in .thresh. Fit-time guard: reject n pinned at 0 or ~1.

**Source:** Laub-Taimre-Pollett 2015 (arXiv:1507.02822); Bacry-Mastromatteo-Muzy 2015 (arXiv:1502.04592); Ozaki-Ogata 1979

### newma (dual-EWMA online kernel-MMD distributional-change clock)  [high]
**Why:** All existing break clocks (zcusum/logdollar-cusum/cusum_regime/changepoint/bde_cusum) are FIRST-MOMENT (mean) detectors — blind to a pure variance/skew/kurtosis/correlation regime change. NEWMA embeds returns via random Fourier features and compares two time-localized kernel mean embeddings (an O(1) MMD, no Gram matrix), so it ticks on full-distribution dislocations the mean detectors miss. New mechanism family; samples densely at higher-moment regime flips on UUP/TLT/SPY/QQQ/USO/GLD. Medium compute (per-minute 48-dim recursion) — vectorize the embedding.

**Impl:** FROZEN seeded W=randn(m=24,p=4)/sqrt(sigma²_TRAIN) (median heuristic on a TRAIN subsample). update: x_t=last p returns; feat=concat(cos(W@x),sin(W@x))/sqrt(m); ewma+=l*(feat-ewma); ewma2+=L*(feat-ewma2); stat=||ewma-ewma2||; rising-edge stat>=thresh & prev<thresh -> emit. _select_optimal_parameters(B): bisection _convert_parameters (x(1-x)^B=ff(1-ff)^B) + grid-min the paper error bound -> (L,l); B~len(c)/target_bars. Threshold binary-searched on the TRAIN stat series to target_bars (NOT a full-series quantile). Multi-param (store W,l,L) -> entropy verify path. scipy/sklearn replaced by manual pairwise dists + bisection.

**Source:** Keriven-Garreau-Poli 2020 IEEE TSP (arXiv:1805.08061); ref impl lightonai/newma

### signedjumpvar (RS+ minus RS- signed-jump-variation clock)  [medium]
**Why:** Genuinely new: a sign(r)·r² accumulator with |theta| crossing — signs by VARIANCE (the diffusive IV cancels in RS+ - RS-, isolating directional jump runs), orthogonal to the flow-signed imbalance/tickimb/volumeimb family AND to the unsigned vol axis. Fires on persistent net up/down variance imbalance. Sibling of semivar (same Patton-Sheppard family) so lower marginal novelty — build AFTER semivar; together with jump and volofvol it forms a tail trio. Scalar, O(1)/min, BUILDER_CLASSES-compatible.

**Impl:** class SignedJumpVarBarBuilder: update: r=lc-last_lc; theta+=r*abs(r) (=r²·1{r>0}-r²·1{r<0}); if abs(theta)>=thresh: theta=0; emit. _make_builder 'signedjumpvar' (mirror imbalance branch): term=ret*np.abs(ret); sigma=std(term[tr&finite]); thresh=sigma*sqrt(len(c)/target_bars). PREFER TRAIN-bisection variant (run-axis style) because term has nonzero mean (skew drift) -> tighter bar-count control. Single scalar threshold.

**Source:** Patton & Sheppard 2015 REStat 97(3):683 (signed jump variation)

### bocpd (Bayesian online changepoint run-length clock)  [medium]
**Why:** New mechanism: probabilistic regime-AGE. Unlike the heuristic mean-only detectors, BOCPD maintains a calibrated posterior over run-length via the Adams-MacKay recursion with a Normal-Gamma predictive, detecting MEAN AND VARIANCE shifts jointly, and uniquely yields an online run-length axis plus a regime-age feature for the XGBoost gate. Best on variance/regime-switching names (TLT/UUP, SPY/QQQ crash precursors, USO/UCO/AGQ). Cost center: runs on the raw minute feed at O(N·Rmax) so it is the slowest axis — mitigate with Rmax=300 truncation + refractory debounce.

**Impl:** Log-space _BOCPD.step(x): lp=logStudentT(x; mu,kap,al,be) (df=2al, scale²=be(kap+1)/(al·kap)); growth=logR+lp+log(1-H); cp=logsumexp(logR+lp+logH); R=concat([cp],growth)-logsumexp; NG update mu'=(kap·mu+x)/(kap+1),kap'+1,al'+0.5,be'+=kap(x-mu)²/(2(kap+1)), prepend prior; truncate to Rmax. Axis: x=(lc-prev)/sd (sd frozen TRAIN), emit when P_reset>=thresh AND since_emit>=refractory (state NEVER reset on emit). Calibrate thresh from TRAIN P_reset rate * full_len/train_len (NOT full-series quantile). gammaln=np.vectorize(math.lgamma); numpy only.

**Source:** Adams & MacKay 2007 (arXiv:0710.3742); financial app arXiv:2307.02375

### persistence_clock (0-D sublevel-set persistence depth clock)  [medium]
**Why:** Brings the topological-depth mechanism tractably (linear-time 0-D PL persistence via a monotone extrema stack — NOT ripser/gudhi O(N^3)). Accumulates resolved valley prominences (saddle minus trough depth, finalized causally when price first rises above the lower flanking saddle), distinct from ddonset (single running-peak drawdown threshold) and crash_ahead (forward-return tail). Targets deep-drawdown / nested-oscillation names (SPY/QQQ/XLE/XME/SOXX, SSO/SPXL/QLD/AGQ/UCO, GLD/USO). Scalar, O(N·alpha(N)), zero deps.

**Impl:** class PersistClockBuilder: monotone stack st of confirmed extrema; on a turning point push prev extremum and _resolve: while len(st)>=3 and current saddle c>=low=min(saddleL,saddleR): cum+=(low-trough); pop the higher saddle+trough, keep lower saddle. emit at cum>=thresh. _make_builder 'persistence' replays _resolve offline writing terms[i]=resolved depth at minute i; total=mean(terms[tr&finite])*len(c); _safe_thresh. Reset stack on invalid print (fit==replay). Unit-test stack bookkeeping vs brute-force prominence on random walks.

**Source:** Glisse 2023 (arXiv:2301.04745, linear-time 0-D PL persistence); Banana Trees arXiv:2405.17920; Edelsbrunner-Harer

### lz (Lempel-Ziv LZ76 compressibility / entropy-rate clock)  [medium]
**Why:** Distinct from entropy (zeroth-order marginal Shannon surprise — a repeating up/down cycle scores HIGH every minute) and perment (ordinal): LZ measures redundancy of the SEQUENCE (that same cycle is highly compressible -> LOW LZ), i.e. the entropy RATE / weak-form predictability. Ticks densely in locally-compressible windows, over-sampling exploitable regimes to clean up val_auc~0.5 drifters (UUP/TLT). Compute caution (not leak): O(W²) sequential parse, run only on a STRIDE grid; numpy only.

**Impl:** lz76_norm(b): Kaspar-Schuster k_max counting loop over sentinel-prefixed binary b -> c*log2(n)/n. class LZComplexityBarBuilder(threshold,theta): ring buffer W=120 of (r>theta) symbols; every STRIDE=20 minutes (ring full) seq=oldest->newest; cum+=max(0,1-lz76_norm(seq)); emit at cum>=thresh. _make_builder 'lz': theta=TRAIN median of minute returns; MIRROR the builder loop over TRAIN summing inc=max(0,1-lz_norm); total=TRAIN_sum*(len(c)/TRAIN_minutes); _safe_thresh (fit==replay). Store theta (frozen) like entropy; thresh scalar. Keep W<=128, STRIDE>=20; subsample symbols if QC build times out.

**Source:** Lempel & Ziv 1976 IT-22:75; Kaspar-Schuster 1987 PRA 36:842; Fiedor 2014 (arXiv:1310.5540); NeuroKit2 reference loop

### tlb (Three-Line-Break self-adapting reversal clock)  [medium]
**Why:** New emission geometry: the reversal distance ADAPTS to recent leg lengths (break beyond the extreme of the last 3 variable-size lines), so it demands a large counter-move in a strong trend and flips easily in chop — automatic volatility/trend-strength normalization no fixed-threshold clock (dc percentage, range span) has. Emits on continuations too (denser in trends). Best on regime-oscillating turning-point names (TLT/UUP/EEM/HYG) with a trend-persistence component for GLD/SSO/SPXL/AGQ. Scalar g floor, O(1)/min.

**Impl:** class ThreeLineBreakBarBuilder(threshold=g): mode in {+1,-1}, deque of last <=4 line-closes. up-mode: continuation if lc>=last+g (new line, emit); reverse if lc<=min(lines) (break low of last 3, flip, emit); symmetric down-mode. NOTE pure native TLB on 1-min data emits hundreds of thousands of lines -> a TRAIN-fit min-line-size g is REQUIRED for bar-count control (the spec's 'only constant is 3' is overstated at minute freq, but g is leak-safe). _fit_tlb_axis: seed g=std(TRAIN returns)*sqrt(len/target_bars); bracket+bisect g on a TRAIN close-only sim to target_bars*train_frac (count monotone-decreasing in g). Scalar .thresh=g -> BUILDER_CLASSES byte-exact.

**Source:** Nison 1994 'Beyond Candlesticks'; StockCharts ChartSchool 'Three Line Break'

## Recommended LABELERS (modules/labeler.py)

### kllt (Kalman RTS-smoothed-slope sign label)  [high] — trend-momentum (linear-Gaussian state-space; matched look-ahead target for the kalman filtered-slope axis)
**Why:** The genuinely novel piece is the matched pair: the causal filtered slope (axis feature) is the natural predictor of the look-ahead RTS-smoothed slope sign (target) — same model and params, only the information set differs. The MMSE-optimal smoothed slope under an explicit noise model beats trend_scan's windowed-OLS t-stat and trend_leg's greedy zigzag on the trend-momentum cohort where trend_leg already wins (GLD). Pure numpy O(N), trivial.

**Impl:** R=var(lr[tr_m&fv]); 2D q-grid MLE on lc[tr_m] (forward Kalman, max LL); refit forward filter + RTS smoother (backward: C_t=P_t T'(P_pred_{t+1})^{-1}; x_s_t=x_t+C_t(x_s_{t+1}-x_pred_{t+1})) over FULL lc with best q -> smoothed slope ss. thr sweep TRAIN quantiles {0.4,0.5,0.6} of ss[tr_m&fv]; y=(ss>=thr).astype(int) where finite else -1; require TRAIN balance in (0.1,0.9). Smoother look-ahead is the legitimate target. Return (y, f'kllt_q{ql}_{qs}_thr', None).

**Source:** Harvey 1989 (LLT); Rauch-Tung-Striebel 1965

### diurnal_anomaly (seasonal information-shock label)  [high] — seasonality / intraday periodicity — trade only bars whose activity is anomalous FOR ITS TIME-OF-DAY
**Why:** No existing labeler is calendar/seasonality-aware. Isolates true information shocks (off the open/close U-shape) from predictable churn, then labels them directionally — near-balanced by construction. On macro/event names (TLT/GLD/UUP) shocks arrive off the U-shape (FOMC ~14:00, CPI/NFP 08:30), plausibly reviving the weak EEM/TLT timing edges. Pairs with the diurnal axis. Needs a trivial bar_ts plumbing thread-through (footer calls labelers without bar_ts today; ~2-line change).

**Impl:** mod=minute-of-day(bar_ts); s2[m]=per-minute-of-day TRAIN mean of lr² over (tr_m&fv) with gmean fallback; z=lr²/s2[mod]; for H in [50,100,200], q in (0.6,0.7,0.8): cut=quantile(z[tr_m&fv&finite],q); abnormal=fv&finite(fwd_ret[H])&(z>=cut); y=-1; y[abnormal&(fwd_ret>0)]=1; y[abnormal&(fwd_ret<=0)]=0; pick (H,q) with TRAIN balance in (0.2,0.8). Vectorize minute-of-day via astype('datetime64[m]'). Thread bar_ts into the canonical signature or stash a module global.

**Source:** Andersen & Bollerslev 1997 (FFF intraday periodicity)

### rskew (forward realized-skewness sign label)  [medium] — tail / crash-asymmetry via the 3rd standardized realized moment
**Why:** No existing labeler targets the 3rd realized moment. Scale-free and drift-free, so it isolates down-tail asymmetry independent of net move or magnitude — a window can be strongly left-skewed yet have positive net return (crash_ahead labels that 0; rskew labels it 1). Order-invariant unlike mfe_mae's path extrema. Targets the crash-asymmetry cohort already showing tail life (SPY/QQQ/IWM, SPXL/SSO/QLD/UCO/AGQ); deploy with a crashveto sizer. O(N) prefix sums, trivial.

**Impl:** S2=cumsum(lr²),S3=cumsum(lr³). For H in [40,80,160]: rv=S2[t+H+1]-S2[t+1]; sr3=S3[...]; sk=sqrt(H)*sr3/rv**1.5 where rv>1e-12. med=median(sk[tr_m&fv]); sc=sk-med; for q in (0.3,0.4,0.5): cut=quantile(|sc|[tr_m&fv],q); y=-1; y[sc<=-cut]=1 (left-skewed); y[sc>=cut]=0; pick most TRAIN-balanced in (0.2,0.8). Register LABELERS['rskew'].

**Source:** Amaya, Christoffersen, Jacobs & Vasquez 2015 JFE 118(1):135

### icss_var (CUSUM-of-squares variance-break onset-direction label)  [medium] — volatility-regime structural break in unconditional variance (Inclan-Tiao Dk sup-test, signed)
**Why:** No dedicated variance-break labeler exists (bde_cusum/cusum_regime/changepoint target the MEAN; regime_gmm/hmm fit iid mixtures but never DATE variance breaks). The NOVEL head is onset-direction (variance falling=calm/risk-on vs rising=turbulence), not the calm/turbulent LEVEL split (that overlaps the existing carry labeler). Deploy as a META/sizing gate (de-risk before turbulence, Moreira-Muir) on a directional champion — the meta pattern that won EEM 2.43->4.03. O(H·N), same cost as changepoint.

**Impl:** csq=cumsum(lr²) prefix; for H in [60,120,200], forward window: Cn=csq[t+1+H]-csq[t+1]; maxD,argm via inner loop over split m of D=Ck/Cn-m/H; IT=sqrt(H/2)*maxD; dir=v2-v1 (post-break var minus pre). for q in (0.5,0.6,0.7): cut=quantile(IT[tr_m&fv],q) (or fixed 1.358); strong=fv&(IT>=cut); y[strong&(dir<0)]=1; y[strong&(dir>=0)]=0; -1 else; keep TRAIN balance in (0.2,0.8). A/B vs carry mandatory. (Skip the axis half — redundant with volofvol + O(W)/min compute risk.)

**Source:** Inclan & Tiao 1994 JASA 89:913; deploy rationale Moreira & Muir 2017

### bocpd_label (Bayesian break-ahead / vol-regime label)  [medium] — probabilistic regime change — variance-aware break precursor / new-segment vol regime
**Why:** The directional head overlaps cusum_regime, but the break-ahead head (will a structural break occur within H bars) and the vol-regime head (forward realized vol of the new segment in its upper regime) are distinct — they exploit BOCPD's unique joint mean+variance detection as a crash precursor, complementary to crash_ahead. Cheap as a labeler (O(N_bars·Rmax) over ~20k bars). Best on TLT/UUP (where cusum_regime/bgm found life) and SPY/QQQ/USO/UCO/AGQ vol regimes. Also expose run-length as a causal regime-age feature.

**Impl:** Fit prior(mu0,sd from lr[tr_m]) + hazard (TRAIN segment count in {10,20,40}) on TRAIN; replay _BOCPD forward over FULL lr (causal forward pass) -> cp[t]=P_reset, rl[t]=MAP run-length. Break-ahead head: y=1 if any cp[t+1..t+H]>=cut else 0; cut=TRAIN quantile sweep {.5,.6,.7}; keep TRAIN balance in (0.2,0.8). Vol-regime head: y=1 if forward-segment realized vol in upper TRAIN regime. Return standard triple; add rl[t] to features.py as regime-age.

**Source:** Adams & MacKay 2007 (arXiv:0710.3742)

### setar (observable-threshold AR regime label)  [medium] — observable-driven (non-latent) nonlinear regime: AR dynamics switch on a lagged-return threshold
**Why:** Distinct from the latent-Markov stack (hmm/sticky_hmm/regime_gmm/bgm/jump_model need EM/Viterbi) and from dc_trend/cusum_regime: regime is a deterministic threshold on z=y_{t-d}, and it carries regime-SPECIFIC AR coefficients (a fitted momentum-vs-reversion sign per regime) NO existing labeler has — the momentum-vs-reversion switch EEM/TLT's two opposing edges need. Cheapest regime method (no EM, ~30 lines, sub-second). EFFICACY risk is real (tiny AR(1) coeffs on minute returns) so A/B + permuted-label control are mandatory. Build only the labeler (the axis is a category error / spectral-redundant).

**Impl:** y=lr; for d in (1,2,3): z=lag(y,d) (zero-pad, no wraparound); for p in (1,2): X=[1,y_{t-1..t-p}] on tr; cand c=quantile(z[tr],linspace(.15,.85,17)); per c split z<=c, OLS lstsq each regime, pick min pooled SSR -> (c,bl,bh). Causal label: Xf full-series lag matrix (zero-pad prefixes), yhat=where(z<=c, Xf@bl, Xf@bh); lab=(yhat>0).astype(int) where finite else -1. for H in [50,100,200]: select on VAL directional acc - 0.02*p, TRAIN balance (0.2,0.8). Bounded reach H<=200<=embargo.

**Source:** Tong & Lim 1980 JRSS-B 42(3):245; Tsay 1989 JASA 84:231; Hansen 2000 Econometrica 68(3):575

### tlb_reversal (leg-exhaustion reversal-vs-continuation label)  [medium] — turning-point / leg-exhaustion — will the next Three-Line-Break event be a reversal (1) or continuation (0)
**Why:** A different target column from dc_reversal (direction of the next turn) and dc_trend (current trend state): it predicts reversal-VS-continuation = leg exhaustion, a tail/turning-point target on the adaptive-leg geometry. Secondary to the TLB axis (the genuinely new piece), but cheap to add once the builder exists. Best on names that whipsaw a fixed-delta dc (the dc_reversal TLT motivation): TLT/UUP/EEM/HYG/choppy mean-reverters.

**Impl:** Reuse ThreeLineBreakBarBuilder. For k in (0.5,1,1.5,2,3): g=k*std(lr[tr_m]); replay builder over bars recording line_dir/line_bar; is_rev[j]=1 if dir flips; for each bar t set y[t]=is_rev[next_line] if (next_line_bar - t)<=200 (embargo floor, exactly the dc_reversal 993edcf fix) else -1; keep TRAIN balance (0.25,0.75) maximizing closeness to 0.5. Return (y, f'tlb_reversal_k{k}', 200).

**Source:** Nison 1994; StockCharts ChartSchool 'Three Line Break'

## Coverage gaps (future pass)

Method families in the shortlist but NOT yet deep-spec'd (next pass should spec these): (1) Markov-Switching Multifractal (Calvet-Fisher) — multi-timescale long-memory volatility cascade with a closed-form online filter, distinct from our single-frequency iid vol regimes; the only proposed long-memory-vol mechanism. (2) Higuchi fractal dimension — fast path-roughness persistence axis that complements DFA-based hurst_persist with different small-sample bias and a missing online ANTI-persistence axis. (3) Path-signature (truncated iterated integrals) — causal price-volume LEAD-LAG / order-sensitive geometry no existing axis captures (sliced_wasserstein/fracdiff/spectral are order- or lag-blind). (4) Matrix-Profile discord + FLOSS arc-curve — subsequence self-similarity novelty clock + shape-based segmentation (watch the O(N^2) join; cap history/stride). (5) Dispersion entropy — amplitude-aware symbolic complexity, sharper and cheaper than perment/SampEn. Broader mechanism gaps untouched by the entire shortlist: (a) cross-sectional / multi-asset SPREAD methods — the Wang cross-sectional escape from the single-ticker wall (needs user OK + new data); (b) options-implied / VIX external-data crash signals flagged in the SPY crash-veto memory (needs new data feed); (c) JOINT price-volume distributional clocks (NEWMA with a volume channel appended to the delay vector); (d) smooth-transition LSTAR/ESTAR continuous regime WEIGHT as a soft bet-sizer input (vs SETAR's hard threshold); (e) Renko fixed-brick as a simpler distinct sibling of TLB. Meta/infrastructure gap (from the efficiency-review memory, orthogonal to any single method): formal multiple-testing control — DSR/PBO + decay/survival — is the principled fix for the recurring leaderboard-staleness problem and should gate every new axis/labeler before it enters the book.