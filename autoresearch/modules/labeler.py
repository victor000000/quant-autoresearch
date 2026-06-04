"""Module ② — Unsupervised Labeling.

Featured methods (Wang's): kmeans2stage, carry, tertile, bgm, agglomerative,
triple_barrier, multi_horizon.
Baseline comparator (NOT featured — Wang does NOT use HMM): hmm.

CANONICAL SIGNATURE (every NEW labeler, and the uniform-wrapped registry entries):
    generate_labels_<name>(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                           horizons=[50,100,200]) -> (labels|None, cfg:str, horizon|None)
    labels: np.array int, -1=ignore, 0=short/flat, 1=long.
    Best horizon chosen by TRAIN/VAL label balance in (0.2, 0.8).

LEGACY EXCEPTION: generate_labels_carry keeps its historic
    (fwd_vol, tr_m, va_m, fv, horizons) signature, but is also exposed through a
    uniform wrapper in the LABELERS registry.

CAUSALITY: every fitted parameter (cluster centers, HMM Baum-Welch params,
thresholds, scalers, medians, percentiles, rolling vol) is fit on tr_m ONLY,
then applied to val/test. Forward returns/vol ARE the prediction target and may
be used freely for LABELS. No .shift(-N), no [::-1] on time series, no bfill,
no OR across split masks.

CONCATENATION CONVENTION: this file is concatenated with the other modules into
ONE QC script sharing ONE global namespace. Do NOT import sibling modules.
Top-level imports here are ONLY numpy/pandas/math; sklearn/hmmlearn are imported
INSIDE functions (or guarded) so `python3 -m py_compile` always succeeds and a
missing library degrades gracefully at runtime.
"""
import numpy as np
import pandas as pd
import math


# --------------------------------------------------------------------------- #
# Forward metrics                                                             #
# --------------------------------------------------------------------------- #
def compute_forward_metrics(lc, lr, horizons=[50, 100, 200]):
    """Compute forward returns and volatility at multiple horizons.

    Returns:
        fwd_ret: dict {horizon: np.array of forward log-returns (lc[t+h]-lc[t])}
        fwd_vol: dict {horizon: np.array of forward realized vol (std of lr window)}
    NaN where the forward window does not fit (last `horizon` bars).
    """
    N = len(lc)
    fwd_ret = {}
    fwd_vol = {}
    for fk in horizons:
        fr = np.full(N, np.nan)
        fv = np.full(N, np.nan)
        for t in range(N - fk):
            wr = lr[t + 1:t + fk + 1]
            if len(wr) < 2:
                continue
            fr[t] = lc[t + fk] - lc[t]
            fv[t] = float(np.std(wr))
        fwd_ret[fk] = fr
        fwd_vol[fk] = fv
    return fwd_ret, fwd_vol


# --------------------------------------------------------------------------- #
# Featured: KMeans two-stage (vol regime -> direction)                        #
# --------------------------------------------------------------------------- #
def generate_labels_kmeans_two_stage(lc, lr, tr_m, va_m, te_m, fv,
                                      fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """KMeans two-stage labeling: vol cluster -> direction cluster.

    Stage 1: KMeans(K=2) on forward volatility (fit on TRAIN) -> low-vol regime.
    Stage 2: KMeans(K in {2,3}) on standardized [fwd_ret, |fwd_ret|] within the
             low-vol regime (scaler + clusters fit on TRAIN) -> up-cluster.

    Returns (best_labels, best_cfg, best_horizon). best_horizon chosen by VAL
    label balance in (0.2, 0.8).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as FS

    N = len(lc)
    best_score = -999
    best_labels = None
    best_cfg = ""
    best_horizon = None

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk])
        fwd_abs_k = np.abs(fwd_ret[fk])

        # Stage 1: vol regime, fit on TRAIN only.
        fv_clean = fwd_vol[fk][tr_m & fvd_k & fv]
        if len(fv_clean) < 30:
            continue
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(
            fv_clean.reshape(-1, 1))
        cv_vol = km_vol.predict(fwd_vol[fk][fvd_k & fv].reshape(-1, 1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low = (cv_vol == lo_vol)
        s2_mask_tr = tr_m[fvd_k & fv] & is_low

        fwd_cl = fwd_ret[fk][fvd_k & fv]
        fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        vf_tr = np.column_stack([fwd_cl[s2_mask_tr], fwd_abs_cl[s2_mask_tr]])
        if len(vf_tr) < 60:
            continue

        # Stage 2 scaler fit on TRAIN low-vol only.
        fs_dir = FS().fit(vf_tr)
        vf_all = np.column_stack([fwd_cl[is_low], fwd_abs_cl[is_low]])
        vf_all_z = fs_dir.transform(vf_all)

        for nc in [2, 3]:
            km_dir = KMeans(n_clusters=nc, random_state=42, n_init=5).fit(
                fs_dir.transform(vf_tr))
            cv_dir = km_dir.predict(vf_all_z)
            up_c = int(np.argmax(km_dir.cluster_centers_[:, 0]))
            dir_labels = np.zeros(cv_dir.shape[0], dtype=int)
            dir_labels[cv_dir == up_c] = 1

            full_labels = np.full(N, -1, dtype=int)
            full_labels[np.where(fvd_k & fv)[0][is_low]] = dir_labels
            y = full_labels
            ly = y >= 0

            vx = fv & ly & va_m
            if vx.sum() < 30:
                continue
            val_balance = y[vx].mean()
            if 0.2 < val_balance < 0.8:
                cfg = f"km2_f{fk}_c{nc}"
                score = min(val_balance, 1 - val_balance)
                if score > best_score:
                    best_score = score
                    best_labels = y
                    best_cfg = cfg
                    best_horizon = fk

    return best_labels, best_cfg, best_horizon


# --------------------------------------------------------------------------- #
# Featured: Carry (legacy signature preserved)                                #
# --------------------------------------------------------------------------- #
def generate_labels_carry(fwd_vol, tr_m, va_m, fv, horizons=[50, 100, 200]):
    """Carry-inspired labels: long when forward vol is below the TRAIN median.

    LEGACY SIGNATURE (kept as-is). Median threshold fit on TRAIN only.
    Returns (best_labels, best_cfg, None).
    """
    N = len(fwd_vol[horizons[0]])
    best_labels = None
    best_cfg = ""
    best_score = -999

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_vol[fk])
        sel_tr = tr_m & fvd_k & fv
        med_v = float(np.median(fwd_vol[fk][sel_tr])) if sel_tr.sum() > 10 else 0
        if med_v <= 0:
            continue

        y_carry = np.full(N, -1, dtype=int)
        y_carry[fvd_k & fv & (fwd_vol[fk] <= med_v)] = 1
        y_carry[fvd_k & fv & (fwd_vol[fk] > med_v)] = 0

        ly = y_carry >= 0
        vx = fv & ly & va_m
        if vx.sum() < 30:
            continue

        balance = y_carry[vx].mean()
        if 0.2 < balance < 0.8:
            score = min(balance, 1 - balance)
            if score > best_score:
                best_score = score
                best_labels = y_carry
                best_cfg = f"carry_f{fk}"

    # Fallback: if no horizon passed the strict 0.2-0.8 balance gate (e.g. a very
    # noisy asset like XLE), use the longest usable horizon's TRAIN-median split
    # unconditionally. Returning None here would crash the run downstream.
    if best_labels is None:
        for fk in reversed(horizons):
            fvd_k = ~np.isnan(fwd_vol[fk])
            sel_tr = tr_m & fvd_k & fv
            if sel_tr.sum() > 10:
                med_v = float(np.median(fwd_vol[fk][sel_tr]))
                if med_v > 0:
                    y_carry = np.full(N, -1, dtype=int)
                    y_carry[fvd_k & fv & (fwd_vol[fk] <= med_v)] = 1
                    y_carry[fvd_k & fv & (fwd_vol[fk] > med_v)] = 0
                    best_labels, best_cfg = y_carry, f"carry_f{fk}_fallback"
                    break

    return best_labels, best_cfg, None


def generate_labels_carry_uniform(lc, lr, tr_m, va_m, te_m, fv,
                                  fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """Uniform-signature wrapper around the legacy carry labeler (for registry)."""
    return generate_labels_carry(fwd_vol, tr_m, va_m, fv, horizons=horizons)


# --------------------------------------------------------------------------- #
# Featured: Tertile (extreme-move quantile labels)                            #
# --------------------------------------------------------------------------- #
def generate_labels_tertile(lc, lr, tr_m, va_m, te_m, fv,
                            fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """Quantile tertile labels: top tertile=long(1), bottom tertile=short(0),
    middle tertile skipped (-1). Tertile boundaries fit on TRAIN only.

    Avoids the noisy middle 33% where direction is ambiguous, raising purity.
    Returns (best_labels, best_cfg, best_horizon).
    """
    N = len(lc)
    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk])

        fwd_train = fwd_ret[fk][tr_m & fvd_k & fv]
        if len(fwd_train) < 200:
            continue
        top_t = float(np.percentile(fwd_train, 67))
        bot_t = float(np.percentile(fwd_train, 33))

        valid = fv & fvd_k
        fr_k = fwd_ret[fk]
        y_tertile = np.full(N, -1, dtype=int)
        y_tertile[valid & (fr_k >= top_t)] = 1
        y_tertile[valid & (fr_k <= bot_t)] = 0
        # Middle tertile remains -1 (skip).

        ly = y_tertile >= 0
        tx = fv & ly & tr_m
        vx = fv & ly & va_m
        if tx.sum() < 100 or vx.sum() < 20:
            continue

        balance = y_tertile[tx].mean()
        if 0.3 < balance < 0.7:
            cfg = f"tertile_f{fk}"
            score = min(balance, 1 - balance)
            if score > best_score:
                best_score = score
                best_labels = y_tertile
                best_cfg = cfg
                best_horizon = fk

    return best_labels, best_cfg, best_horizon


# --------------------------------------------------------------------------- #
# Featured: Bayesian Gaussian Mixture with posterior threshold                #
# --------------------------------------------------------------------------- #
def generate_labels_bgm(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                        horizons=[50, 100, 200], post_thresh=0.40):
    """Bayesian Gaussian Mixture labels with POST_THRESH (Wang technique).

    Vol-regime gate (KMeans K=2 on fwd vol, TRAIN-fit) -> BGMM on
    [fwd_ret, |fwd_ret|] over the low-vol regime (TRAIN-fit, sparse Dirichlet
    prior). Label 1 only where the assigned cluster is the up-cluster AND its
    posterior > post_thresh; else 0. Returns (best_labels, best_cfg, best_horizon).
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import BayesianGaussianMixture

    N = len(lc)
    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk])
        fwd_abs_k = np.abs(fwd_ret[fk])

        # Vol-regime gate, fit on TRAIN.
        fv_clean = fwd_vol[fk][tr_m & fvd_k & fv]
        if len(fv_clean) < 30:
            continue
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(
            fv_clean.reshape(-1, 1))
        cv_vol = km_vol.predict(fwd_vol[fk][fvd_k & fv].reshape(-1, 1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low = (cv_vol == lo_vol)
        s2_mask_tr = tr_m[fvd_k & fv] & is_low

        fwd_cl = fwd_ret[fk][fvd_k & fv]
        fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        bgm_tr = np.column_stack([fwd_cl[s2_mask_tr], fwd_abs_cl[s2_mask_tr]])
        bgm_all = np.column_stack([fwd_cl[is_low], fwd_abs_cl[is_low]])
        if len(bgm_tr) < 100:
            continue

        for K in [3, 4, 5]:
            try:
                bgm = BayesianGaussianMixture(
                    n_components=K, covariance_type='full',
                    weight_concentration_prior=0.1,
                    random_state=42, max_iter=300, n_init=3)
                bgm.fit(bgm_tr)  # TRAIN-fit only.

                up_c = int(np.argmax(bgm.means_[:, 0]))
                assign = bgm.predict(bgm_all)
                posteriors = bgm.predict_proba(bgm_all)
                up_post = posteriors[:, up_c]

                labels = np.zeros(len(bgm_all), dtype=int)
                labels[(assign == up_c) & (up_post > post_thresh)] = 1

                full_labels = np.full(N, -1, dtype=int)
                full_labels[np.where(fvd_k & fv)[0][is_low]] = labels

                y = full_labels
                ly = y >= 0
                tx = fv & ly & tr_m
                vx = fv & ly & va_m
                if tx.sum() < 100 or vx.sum() < 20:
                    continue

                balance = y[tx].mean()
                if 0.2 < balance < 0.8:
                    cfg = f"bgm_f{fk}_K{K}_pt{post_thresh}"
                    quality = min(balance, 1 - balance)
                    n_labeled = int(ly.sum())
                    score = quality + 0.01 * min(n_labeled / 1000, 1.0)
                    if score > best_score:
                        best_score = score
                        best_labels = y
                        best_cfg = cfg
                        best_horizon = fk
            except Exception:
                continue

    return best_labels, best_cfg, best_horizon


def _jump_viterbi(z, centroids, lam):
    """Viterbi DP: assign each row of z to a state minimizing sum ||z-centroid||^2 + lam*(state changes).
    The lam term is the JUMP PENALTY -> persistent regimes. O(Nn*K^2), K small."""
    Nn = z.shape[0]
    K = centroids.shape[0]
    cost = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)   # Nn x K
    off = lam * (1.0 - np.eye(K))                                       # off[k,j] = lam if j!=k else 0
    dp = cost[0].copy()
    back = np.zeros((Nn, K), dtype=np.int64)
    for i in range(1, Nn):
        m = dp[None, :] + off                                          # m[k,j] = dp_prev[j] + jump
        j = m.argmin(axis=1)
        dp = cost[i] + m[np.arange(K), j]
        back[i] = j
    states = np.zeros(Nn, dtype=np.int64)
    states[-1] = int(dp.argmin())
    for i in range(Nn - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def _jump_fit(z, tr_mask, K, lam, n_iter=6):
    """Coordinate descent for the Statistical Jump Model: alternate Viterbi assignment with a
    centroid M-step fit on TRAIN rows only (leak-safe). Returns the full state sequence or None."""
    tr_idx = np.where(tr_mask)[0]
    if len(tr_idx) < K:
        return None
    step = max(1, len(tr_idx) // K)
    pick = [min(i * step, len(tr_idx) - 1) for i in range(K)]
    cents = z[tr_idx[pick]].astype(float).copy()
    for _ in range(n_iter):
        st = _jump_viterbi(z, cents, lam)
        newc = cents.copy()
        for k in range(K):
            mk = tr_mask & (st == k)
            if mk.any():
                newc[k] = z[mk].mean(axis=0)
        if np.allclose(newc, cents):
            cents = newc
            break
        cents = newc
    return _jump_viterbi(z, cents, lam)


def generate_labels_jump_model(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[50, 100]):
    """Statistical Jump Model (Shu & Mulvey 2024, arXiv:2402.05272) — PERSISTENT-regime label, mined
    2026-06-03. Penalized k-means: cluster a per-bar regime feature [fwd_ret, fwd_vol] into K states with
    an explicit JUMP PENALTY (lambda) on state TRANSITIONS, via coordinate descent (Viterbi assignment +
    TRAIN-fit centroid M-step). The jump penalty yields PERSISTENT regimes (no per-bar flip-flop) — the
    key difference from `bgm`, which clusters the forward-return distribution memorylessly. The up-state
    (TRAIN highest mean fwd_ret) -> label 1, else 0. Centroids + state-direction fit on TRAIN; the forward
    feature defines only the TARGET (G3-ok). Tests whether persistence beats bgm on regime assets (UUP).
    Returns (labels, cfg, horizon)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for fk in horizons:
        if fk not in fwd_ret:
            continue
        sel = (~np.isnan(fwd_ret[fk])) & fv
        idx = np.where(sel)[0]
        if len(idx) < 200:
            continue
        fr = fwd_ret[fk][sel]
        fvol = fwd_vol[fk][sel]
        feat = np.column_stack([fr, fvol]).astype(float)
        tr_sub = tr_m[sel]
        if int(tr_sub.sum()) < 100:
            continue
        mu = feat[tr_sub].mean(axis=0)
        sd = feat[tr_sub].std(axis=0) + 1e-9
        z = (feat - mu) / sd                                # standardized on TRAIN
        base = float(np.mean(np.sum(z[tr_sub] ** 2, axis=1))) + 1e-9   # per-bar cost scale
        for K in (2, 3):
            for lam_mult in (0.0, 3.0):                      # 0 = plain k-means, 3*base = persistent
                st = _jump_fit(z, tr_sub, K, lam_mult * base)
                if st is None:
                    continue
                tr_st = st[tr_sub]
                tr_fr = fr[tr_sub]
                means = [float(tr_fr[tr_st == k].mean()) if (tr_st == k).any() else -1e18 for k in range(K)]
                up = int(np.argmax(means))
                y = np.full(N, -1, dtype=int)
                y[idx] = (st == up).astype(int)
                ly = y >= 0
                tx = fv & ly & tr_m
                vx = fv & ly & va_m
                if tx.sum() < 100 or vx.sum() < 20:
                    continue
                bal = float(y[tx].mean())
                if 0.2 < bal < 0.8:
                    score = min(bal, 1 - bal)
                    if score > best_score:
                        best_score, best, best_cfg, best_h = score, y, f"jump_f{fk}_K{K}_lam{lam_mult}", fk
    if best is None:
        return None, "jump_model_no_balanced", None
    return best, best_cfg, best_h


# --------------------------------------------------------------------------- #
# Featured: Agglomerative (Ward)                                              #
# --------------------------------------------------------------------------- #
def generate_labels_agglomerative(lc, lr, tr_m, va_m, te_m, fv,
                                  fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """Agglomerative (Ward) regime labels.

    AgglomerativeClustering has no native predict; to keep this causal we fit
    Ward on TRAIN low-vol points, compute TRAIN cluster centroids, then assign
    ALL low-vol points by nearest TRAIN centroid (1-NN forward assignment). The
    up-cluster is the TRAIN centroid with the highest mean forward return.
    Returns (best_labels, best_cfg, best_horizon).
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering

    N = len(lc)
    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk])
        fwd_abs_k = np.abs(fwd_ret[fk])

        fv_clean = fwd_vol[fk][tr_m & fvd_k & fv]
        if len(fv_clean) < 30:
            continue
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(
            fv_clean.reshape(-1, 1))
        cv_vol = km_vol.predict(fwd_vol[fk][fvd_k & fv].reshape(-1, 1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low = (cv_vol == lo_vol)
        tr_low = tr_m[fvd_k & fv] & is_low

        fwd_cl = fwd_ret[fk][fvd_k & fv]
        fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        agg_all = np.column_stack([fwd_cl[is_low], fwd_abs_cl[is_low]])
        agg_tr = np.column_stack([fwd_cl[tr_low], fwd_abs_cl[tr_low]])
        if len(agg_tr) < 100:
            continue

        for K in [2, 3, 4]:
            try:
                agg = AgglomerativeClustering(n_clusters=K, linkage='ward')
                tr_lab = agg.fit_predict(agg_tr)  # TRAIN-fit only.

                # TRAIN cluster centroids + up-cluster by TRAIN mean fwd return.
                centroids = []
                cl_means = []
                for c in range(K):
                    sel = (tr_lab == c)
                    if sel.sum() == 0:
                        centroids.append(np.array([np.inf, np.inf]))
                        cl_means.append(-np.inf)
                    else:
                        centroids.append(agg_tr[sel].mean(axis=0))
                        cl_means.append(float(fwd_cl[tr_low][sel].mean()))
                centroids = np.array(centroids)
                up_c = int(np.argmax(cl_means))

                # 1-NN forward assignment of ALL low-vol points to TRAIN centroids.
                d = np.linalg.norm(agg_all[:, None, :] - centroids[None, :, :], axis=2)
                assign = np.argmin(d, axis=1)
                dir_labels = np.zeros(len(agg_all), dtype=int)
                dir_labels[assign == up_c] = 1

                full_labels = np.full(N, -1, dtype=int)
                full_labels[np.where(fvd_k & fv)[0][is_low]] = dir_labels

                y = full_labels
                ly = y >= 0
                tx = fv & ly & tr_m
                vx = fv & ly & va_m
                if tx.sum() < 100 or vx.sum() < 20:
                    continue

                balance = y[tx].mean()
                if 0.2 < balance < 0.8:
                    cfg = f"agg_f{fk}_K{K}_ward"
                    score = min(balance, 1 - balance)
                    if score > best_score:
                        best_score = score
                        best_labels = y
                        best_cfg = cfg
                        best_horizon = fk
            except Exception:
                continue

    return best_labels, best_cfg, best_horizon


# --------------------------------------------------------------------------- #
# Featured: Triple-barrier (AFML)                                             #
# --------------------------------------------------------------------------- #
def generate_labels_triple_barrier(lc, lr, tr_m, va_m, te_m, fv,
                                   fwd_ret, fwd_vol, horizons=[50, 100, 200],
                                   U=2.0, L=2.0):
    """Triple-barrier labeling (Lopez de Prado, AFML).

    For each bar t: upper barrier = +U*sigma_t, lower barrier = -L*sigma_t,
    vertical barrier = H bars ahead (H taken from `horizons`). Walking forward
    over the cumulative log-return path:
        label 1 if the upper barrier is hit first within H bars,
        label 0 if the lower barrier is hit first OR the vertical barrier is
                hit with a non-positive cumulative return (timeout-down).
    A timeout with positive cumulative return is also labeled 1 (path closed up).

    sigma is a causal rolling vol of `lr`; its window is calibrated so that the
    rolling-vol distribution is anchored to the TRAIN segment. Using FORWARD path
    information for the LABEL itself is allowed (labels are the target). The only
    fitted quantity (vol scale) is taken from TRAIN.
    Returns (best_labels, best_cfg, best_horizon).
    """
    N = len(lc)
    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    # Causal rolling vol of returns (only past/current bars).
    vol_window = 50
    sigma_series = pd.Series(lr).rolling(vol_window, min_periods=vol_window).std()
    sigma = sigma_series.to_numpy()
    # Fall back / floor the rolling vol using a TRAIN-fit median scale so that
    # early bars and NaNs get a sensible, TRAIN-derived sigma rather than 0.
    tr_idx = np.where(tr_m)[0]
    if len(tr_idx) > 0:
        tr_sigma = sigma[tr_idx]
        tr_sigma = tr_sigma[~np.isnan(tr_sigma)]
        sigma_floor = float(np.median(tr_sigma)) if len(tr_sigma) > 0 else 0.0
    else:
        sigma_floor = 0.0
    if not np.isfinite(sigma_floor) or sigma_floor <= 0:
        # Degenerate TRAIN vol (cannot occur under footer's tr_m.sum()>=500 guard):
        # use a fixed epsilon, NEVER a full-series median (which would include OOS).
        sigma_floor = 1e-6
    sigma = np.where(np.isnan(sigma) | (sigma <= 0), sigma_floor, sigma)

    for H in horizons:
        y_tb = np.full(N, -1, dtype=int)
        for t in range(N - H):
            if not fv[t]:
                continue
            s = sigma[t]
            up_b = U * s
            lo_b = -L * s
            cum = 0.0
            hit = None
            for j in range(t + 1, t + H + 1):
                cum += lr[j]
                if cum >= up_b:
                    hit = 1
                    break
                if cum <= lo_b:
                    hit = 0
                    break
            if hit is None:  # vertical barrier (timeout)
                hit = 1 if cum > 0 else 0
            y_tb[t] = hit

        ly = y_tb >= 0
        tx = fv & ly & tr_m
        vx = fv & ly & va_m
        if tx.sum() < 100 or vx.sum() < 20:
            continue

        balance = y_tb[tx].mean()
        if 0.2 < balance < 0.8:
            cfg = f"tb_H{H}_U{U}_L{L}"
            score = min(balance, 1 - balance)
            if score > best_score:
                best_score = score
                best_labels = y_tb
                best_cfg = cfg
                best_horizon = H

    return best_labels, best_cfg, best_horizon


# --------------------------------------------------------------------------- #
# Featured: Multi-horizon consensus                                           #
# --------------------------------------------------------------------------- #
def generate_labels_multi_horizon(lc, lr, tr_m, va_m, te_m, fv,
                                  fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """Multi-horizon agreement labels.

    For each horizon h, the per-horizon sign label is 1 if fwd_ret[h] >= 0 else 0
    (only where the forward window is valid). The consensus label is 1 only if
    ALL horizons agree on up, 0 only if ALL agree on down, and -1 (ignore) on
    any disagreement or where any horizon's forward window is missing. This is a
    pure label-target construction (forward returns allowed); nothing is fit, so
    there is no leakage. Returns (consensus_labels, cfg, None).
    """
    N = len(lc)

    up_stack = []
    down_stack = []
    valid_stack = []
    for h in horizons:
        fr = fwd_ret[h]
        valid = fv & ~np.isnan(fr)
        up_stack.append(valid & (fr >= 0))
        down_stack.append(valid & (fr < 0))
        valid_stack.append(valid)

    all_valid = np.logical_and.reduce(valid_stack)
    all_up = np.logical_and.reduce(up_stack)
    all_down = np.logical_and.reduce(down_stack)

    y = np.full(N, -1, dtype=int)
    y[all_valid & all_up] = 1
    y[all_valid & all_down] = 0

    ly = y >= 0
    tx = fv & ly & tr_m
    vx = fv & ly & va_m
    if tx.sum() < 100 or vx.sum() < 20:
        return None, "", None

    balance = y[tx].mean()
    if not (0.2 < balance < 0.8):
        return None, "", None

    hs = "+".join(str(h) for h in horizons)
    return y, f"mh_consensus_{hs}", None


# --------------------------------------------------------------------------- #
# BASELINE comparator: HMM (Wang does NOT use HMM)                            #
# --------------------------------------------------------------------------- #
def generate_labels_hmm(lc, lr, tr_m, va_m, te_m, fv,
                        fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """HMM regime labels via hmmlearn.GaussianHMM (3-state) on [r, |r|].

    BASELINE comparator ONLY — Wang does NOT use HMM; treat as a yardstick, not
    a featured method.

    - Baum-Welch (fit) on the TRAIN segment ONLY, params='stmc' (start, trans,
      means, covars).
    - Per Wang's "HMM can't do online computation" warning, decoding uses a
      CAUSAL forward filter (the hmmlearn forward-pass posterior alpha,
      normalized), NOT Viterbi and NOT a full-sequence smoothed posterior on
      TEST. Each bar's state is argmax of its forward-filtered posterior.
    - State -> direction mapping uses TRAIN forward-return statistics only: the
      state with the highest TRAIN mean fwd_ret is "up" (label 1), the rest 0.
    - Missing library: return (None, 'hmm_unavailable', None).
    Returns (labels|None, cfg, horizon|None).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None, "hmm_unavailable", None

    N = len(lc)

    # Observation matrix: [r, |r|]. Strictly past/current — causal features.
    obs = np.column_stack([lr, np.abs(lr)]).astype(float)
    bad = ~np.isfinite(obs).any(axis=1)
    if bad.any():
        obs[bad] = 0.0

    tr_idx = np.where(tr_m)[0]
    if len(tr_idx) < 200:
        return None, "hmm_insufficient_train", None
    obs_tr = obs[tr_idx]

    try:
        model = GaussianHMM(
            n_components=3, covariance_type='full',
            params='stmc', init_params='stmc',
            n_iter=100, random_state=42)
        model.fit(obs_tr)  # Baum-Welch on TRAIN ONLY.
    except Exception:
        return None, "hmm_fit_failed", None

    # Causal forward-filter posterior over the FULL sequence (online, no Viterbi,
    # no backward smoothing). hmmlearn exposes the scaled forward lattice.
    try:
        framelogprob = model._compute_log_likelihood(obs)
        log_startprob = np.log(model.startprob_ + 1e-300)
        log_transmat = np.log(model.transmat_ + 1e-300)
        n_states = model.n_components
        log_alpha = np.full((N, n_states), -np.inf)
        log_alpha[0] = log_startprob + framelogprob[0]
        for t in range(1, N):
            for j in range(n_states):
                log_alpha[t, j] = (np.logaddexp.reduce(
                    log_alpha[t - 1] + log_transmat[:, j]) + framelogprob[t, j])
        # Per-bar forward-filtered posterior = normalize alpha across states.
        log_norm = np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        post = np.exp(log_alpha - log_norm)
        states = np.argmax(post, axis=1)
    except Exception:
        return None, "hmm_decode_failed", None

    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    for fk in horizons:
        fr = fwd_ret[fk]
        fvd_k = ~np.isnan(fr)

        # State -> direction using TRAIN forward-return stats ONLY.
        state_dir = {}
        for s in range(model.n_components):
            sel = tr_m & fvd_k & fv & (states == s)
            state_dir[s] = float(np.mean(fr[sel])) if sel.sum() > 0 else -np.inf
        up_state = int(max(state_dir, key=state_dir.get))

        y = np.full(N, -1, dtype=int)
        valid = fv & fvd_k
        y[valid] = 0
        y[valid & (states == up_state)] = 1

        ly = y >= 0
        tx = fv & ly & tr_m
        vx = fv & ly & va_m
        if tx.sum() < 100 or vx.sum() < 20:
            continue

        balance = y[tx].mean()
        if 0.2 < balance < 0.8:
            cfg = f"hmm3_f{fk}_up{up_state}"
            score = min(balance, 1 - balance)
            if score > best_score:
                best_score = score
                best_labels = y
                best_cfg = cfg
                best_horizon = fk

    if best_labels is None:
        return None, "hmm_no_balanced_horizon", None
    return best_labels, best_cfg, best_horizon


def generate_labels_always_long(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                horizons=[50, 100, 200]):
    """BASELINE: always long. Labels every bar 1 (single class) so the footer
    bypasses the model and emits a constant long signal == buy-and-hold. This is
    the floor the featured labelers must beat (alongside the hmm baseline)."""
    n = len(lc)
    return np.ones(n, dtype=int), "always_long", None


# --------------------------------------------------------------------------- #
# Featured: Causal-feature GMM regime labeler (regime_gmm)                     #
# --------------------------------------------------------------------------- #
def _causal_regime_features(lc, lr):
    """Cheap CAUSAL feature panel for regime detection (past/current bars only).

    Six columns, each strictly causal (rolling/lagged, no .shift(-N), no reversal):
        0  mom20      : 20-bar log-return momentum  lc[t]-lc[t-20]
        1  vol50      : 50-bar rolling std of lr (realized vol)
        2  volratio   : (10-bar std)/(100-bar std) — vol expansion/contraction
        3  zmom20     : 20-bar momentum z-scored over a 100-bar window (trend strength)
        4  madist50   : lc[t] - 50-bar moving average of lc (distance from trend)
        5  absmom5    : |lc[t]-lc[t-5]| — short-horizon move magnitude

    A regime in this space is e.g. "calm uptrend", "high-vol selloff",
    "low-vol drift" — exactly the states that map to a directional stance. These
    are a deliberate subset of features.py's panel, recomputed here because the
    labeler receives only (lc, lr), not the full feature matrix (same pattern as
    the hmm baseline which builds [r,|r|] internally). NaN where a window does
    not fit; the caller restricts to fully-formed rows. Nothing is fit here.
    """
    N = len(lc)
    slr = pd.Series(lr)
    slc = pd.Series(lc)

    mom20 = np.full(N, np.nan)
    mom20[20:] = lc[20:] - lc[:-20]

    absmom5 = np.full(N, np.nan)
    absmom5[5:] = np.abs(lc[5:] - lc[:-5])

    vol50 = slr.rolling(50, min_periods=50).std().to_numpy()
    std10 = slr.rolling(10, min_periods=10).std().to_numpy()
    std100 = slr.rolling(100, min_periods=100).std().to_numpy()
    volratio = std10 / (std100 + 1e-12)

    m_mom = pd.Series(mom20).rolling(100, min_periods=100).mean().to_numpy()
    s_mom = pd.Series(mom20).rolling(100, min_periods=100).std().to_numpy()
    zmom20 = (mom20 - m_mom) / (s_mom + 1e-12)

    ma50 = slc.rolling(50, min_periods=50).mean().to_numpy()
    madist50 = lc - ma50

    X = np.column_stack([mom20, vol50, volratio, zmom20, madist50, absmom5])
    return X


def generate_labels_regime_gmm(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                               horizons=[50, 100, 200]):
    """Causal-feature Gaussian-Mixture REGIME labeler (directional, TRAIN-fit).

    Contrast with `bgm`, which clusters on FORWARD returns [fwd_ret,|fwd_ret|];
    that makes the per-bar regime a function of the future. Here the per-bar
    regime is a CAUSAL function of features only:

      1. Build a causal feature panel (_causal_regime_features) from (lc, lr).
      2. Fit a StandardScaler + GaussianMixture (full covariance) on the TRAIN
         causal rows ONLY. This is the only fitted object; it never sees the
         future and never sees val/test.
      3. Assign EVERY causal bar (train/val/test) its regime by GMM.predict on
         the causally-scaled features — a pure forward application of the frozen
         model. Each bar's regime is therefore decided by its own past/current
         features, exactly like an online detector would decide it live.
      4. Map regime -> direction using TRAIN forward-return means ONLY: a regime
         whose TRAIN mean fwd_ret > 0 is "long" (label 1), else "short/flat" (0).
         Forward returns are used solely to define the TARGET (allowed by G3);
         they do NOT pick which OOS bars trade — every causal bar gets a label,
         and the footer emits a prediction for every causal bar.

    Because direction is a CAUSAL regime -> a TRAIN-fixed sign map, the XGBoost
    downstream learns "which causal feature regime tends to go up next", which is
    a genuinely directional, two-sided target (a high-vol-selloff regime maps to
    0/short, a calm-uptrend regime to 1/long). This is built to give TLT a real
    directional, short-capable signal and to pair with the directional imbalance/
    tickimb/volumeimb axes (regimes are cleaner on a directional clock).

    Sweeps K in {2,3,4} and the horizon used for the TRAIN sign map; selects the
    most TRAIN-balanced configuration in (0.2, 0.8). Returns
    (best_labels, best_cfg, best_horizon). Missing sklearn -> (None, ..., None).
    """
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler as FS
    except ImportError:
        return None, "regime_gmm_unavailable", None

    N = len(lc)
    X = _causal_regime_features(lc, lr)
    # Rows with a fully-formed causal feature vector (windows have warmed up).
    row_ok = np.isfinite(X).all(axis=1)

    tr_fit = tr_m & row_ok          # TRAIN rows used to fit scaler + GMM
    if int(tr_fit.sum()) < 200:
        return None, "regime_gmm_insufficient_train", None

    fs = FS().fit(X[tr_fit])        # scaler fit on TRAIN only
    Xz_tr = fs.transform(X[tr_fit])

    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    for K in [2, 3, 4]:
        try:
            gmm = GaussianMixture(
                n_components=K, covariance_type='full',
                reg_covar=1e-5, max_iter=300, n_init=3, random_state=42)
            gmm.fit(Xz_tr)          # TRAIN-fit ONLY
        except Exception:
            continue

        # Forward application: regime for EVERY causal row (frozen model).
        regime = np.full(N, -1, dtype=int)
        try:
            regime[row_ok] = gmm.predict(fs.transform(X[row_ok]))
        except Exception:
            continue

        for fk in horizons:
            fr = fwd_ret[fk]
            fvd_k = np.isfinite(fr)

            # Regime -> direction using TRAIN forward-return means ONLY.
            up_regimes = set()
            for c in range(K):
                sel = tr_m & row_ok & fvd_k & fv & (regime == c)
                if sel.sum() < 20:
                    # Too few TRAIN points to trust this regime's sign -> short/flat.
                    continue
                if float(np.mean(fr[sel])) > 0.0:
                    up_regimes.add(c)

            # Per-bar label is the CAUSAL regime mapped through the TRAIN sign map.
            valid = fv & row_ok & fvd_k
            y = np.full(N, -1, dtype=int)
            y[valid] = 0
            for c in up_regimes:
                y[valid & (regime == c)] = 1

            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 200 or vx.sum() < 20:
                continue

            balance = float(y[tx].mean())
            if 0.2 < balance < 0.8:
                cfg = f"regimegmm_f{fk}_K{K}_up{len(up_regimes)}"
                score = min(balance, 1 - balance)
                if score > best_score:
                    best_score = score
                    best_labels = y
                    best_cfg = cfg
                    best_horizon = fk

    if best_labels is None:
        return None, "regime_gmm_no_balanced_cfg", None
    return best_labels, best_cfg, best_horizon


# --------------------------------------------------------------------------- #
# Featured: CUSUM change-point regime labeler (cusum_regime)                  #
# --------------------------------------------------------------------------- #
def _cusum_change_points(z, h, k):
    """Online two-sided (symmetric) CUSUM change-point detector (de Prado's
    "CUSUM filter", AFML Ch.2/Ch.17 change-point tests).

    z : standardized increments (z-scored log-returns). k is a small reference /
    slack value (the per-step drift we tolerate as noise); only the part of an
    increment beyond +/-k accumulates, so the filter ignores pure noise and fires
    only on a PERSISTENT directional run (a regime shift). h is the decision
    threshold: a change point is declared the bar the running sum crosses +/-h,
    after which both accumulators reset (a fresh regime begins).

        S+_t = max(0, S+_{t-1} + z_t - k)      ->  fires when S+_t >= h
        S-_t = min(0, S-_{t-1} + z_t + k)      ->  fires when S-_t <= -h

    Strictly online/causal: each step uses only z_t and the running sums, never a
    future value. Returns the sorted list of change-point bar indices.
    """
    cps = []
    s_pos = 0.0
    s_neg = 0.0
    n = len(z)
    for t in range(n):
        v = z[t]
        if not np.isfinite(v):
            v = 0.0
        s_pos = max(0.0, s_pos + v - k)
        s_neg = min(0.0, s_neg + v + k)
        if s_pos >= h or s_neg <= -h:
            cps.append(t)
            s_pos = 0.0
            s_neg = 0.0
    return cps


def _calibrate_cusum_h(z_tr, k, target_seg):
    """Bisect the CUSUM threshold h on the TRAIN increments z_tr so the detector
    yields ~target_seg segments (= target_seg-1 change points) over TRAIN.

    Monotone: larger h -> fewer change points. Fit on TRAIN ONLY; the resulting
    h is then applied forward unchanged. Returns a positive float h.
    """
    lo, hi = 0.5, 5000.0

    def n_cp(h):
        return len(_cusum_change_points(z_tr, h, k))

    # Make sure the bracket straddles the target (defensive widening).
    for _ in range(8):
        if n_cp(hi) + 1 > target_seg:
            hi *= 4.0
        else:
            break
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n_cp(mid) + 1 > target_seg:   # too many segments -> raise h
            lo = mid
        else:                            # too few segments -> lower h
            hi = mid
    h = 0.5 * (lo + hi)
    return h if (np.isfinite(h) and h > 0) else 1.0


def generate_labels_cusum_regime(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                 horizons=[50, 100, 200]):
    """CUSUM change-point REGIME labeler (directional, causal, TRAIN-fit).

    Built for two-sided assets (TLT): an online CUSUM filter segments the return
    path into regimes, and each bar's label is the SIGN of that regime's drift,
    estimated CAUSALLY from the bars already seen inside the regime. Because the
    sign is a running (online) drift estimate that resets at each detected change
    point, the per-bar label is a strict function of PAST/CURRENT bars only -- it
    is exactly what a live detector would output, and it is defined for every
    causal bar (including a regime that lives entirely in the OOS segment, which a
    per-segment TRAIN map could never sign).

    Pipeline:
      1. Standardize log-returns with the TRAIN mean/std ONLY:
             z_t = (lr_t - mu_train) / sd_train.
         (mu_train, sd_train are the only globally fitted scalars; both TRAIN-only.)
      2. Slack k = K_FRAC * (TRAIN mean |z|) -- a TRAIN-fit noise floor so the
         filter accumulates only persistent drift, not single-bar noise.
      3. Threshold h is bisected on the TRAIN z-series to yield ~target_seg
         regimes (_calibrate_cusum_h). h is frozen and applied forward.
      4. Run the frozen (h, k) CUSUM forward over the FULL z-series to get change
         points and contiguous regime ids (a pure forward application).
      5. PER-BAR LABEL = sign of a CAUSAL EWMA drift of lr within the regime,
         re-initialised at the regime's start:  label 1 if EWMA-drift > 0 else 0.
         This uses NO forward information at all -> unambiguously G3-clean. The
         EWMA (span ~ EWMA_ALPHA) adapts within a regime and FORGETS the stale
         start, so it stays accurate even when the frozen TRAIN threshold sizes an
         OOS regime imperfectly (a simple cumulative mean degrades badly when one
         regime accidentally spans a true flip; the EWMA does not). The estimate
         resets at each detected change point, so it is exactly what a live online
         detector would output bar-by-bar.

    Forward returns are used ONLY to PICK the configuration (which TRAIN segment
    count / which `fk` produces the most TRAIN-balanced, validating labels) -- i.e.
    purely for model selection on TRAIN/VAL balance, never to decide which OOS bar
    trades. Every causal bar receives a label and the footer emits a prediction
    for every causal bar; the forward target only tunes (target_seg, fk).

    Co-design note: this pairs naturally with the DIRECTIONAL axes (imbalance /
    tickimb / volumeimb / fracdiff). Those clocks already concentrate bars on
    directional runs, so the CUSUM regimes are cleaner and the running-drift sign
    is sharper -- the same axis-labeler synergy that boosted bgm on TLT, here made
    explicitly directional and short-capable.

    Sweeps target_seg in {10, 20, 40} (TRAIN regime granularity) and the horizon
    fk. Each config must first be TRAIN-balanced in (0.2, 0.8); among those, the
    winner maximises VAL directional agreement (causal sign vs realised forward
    sign) minus a parsimony penalty on regime count, so coarse/persistent regimes
    that generalise OOS beat over-segmented ones that memorise TRAIN. The forward
    return is used ONLY for this holdout config selection, never to gate OOS bars.
    Returns (best_labels, best_cfg, best_horizon).
    """
    K_FRAC = 0.10      # slack as a fraction of TRAIN mean |z| (noise floor)
    EWMA_ALPHA = 0.05  # within-regime drift forgetting factor (causal)

    N = len(lc)
    tr_idx = np.where(tr_m)[0]
    if len(tr_idx) < 200:
        return None, "cusum_insufficient_train", None

    lr_tr = lr[tr_idx]
    lr_tr = lr_tr[np.isfinite(lr_tr)]
    if len(lr_tr) < 100:
        return None, "cusum_insufficient_train", None

    mu_tr = float(np.mean(lr_tr))
    sd_tr = float(np.std(lr_tr))
    if not np.isfinite(sd_tr) or sd_tr <= 0:
        return None, "cusum_degenerate_train", None

    # TRAIN-fit standardization, applied forward to the full series.
    z = (lr - mu_tr) / sd_tr
    z = np.where(np.isfinite(z), z, 0.0)

    # Slack k from TRAIN-only |z| (noise floor); guard tiny/degenerate scales.
    z_tr = z[tr_idx]
    mean_abs_z_tr = float(np.mean(np.abs(z_tr))) if len(z_tr) else 1.0
    k = K_FRAC * mean_abs_z_tr
    if not np.isfinite(k) or k <= 0:
        k = 0.05

    best_labels = None
    best_cfg = ""
    best_score = -999
    best_horizon = None

    for target_seg in [10, 20, 40]:
        # Threshold fit on TRAIN z only, then frozen.
        h = _calibrate_cusum_h(z_tr, k, target_seg)

        # Forward application of the frozen (h, k) detector over the FULL series.
        cps = _cusum_change_points(z, h, k)

        # Contiguous regime ids + CAUSAL EWMA-drift sign within each regime.
        run_sign = np.zeros(N, dtype=int)
        cur_seg = 0
        ewma = 0.0
        started = False
        cp_set = set(cps)
        for t in range(N):
            r = lr[t] if np.isfinite(lr[t]) else 0.0
            if not started:                       # re-init EWMA at regime start
                ewma = r
                started = True
            else:
                ewma = EWMA_ALPHA * r + (1.0 - EWMA_ALPHA) * ewma
            run_sign[t] = 1 if ewma > 0.0 else 0  # causal within-regime drift sign
            if t in cp_set:                       # close regime AFTER labeling bar t
                cur_seg += 1
                started = False
        n_seg = cur_seg + 1

        for fk in horizons:
            fr = fwd_ret[fk]
            fvd_k = np.isfinite(fr)

            # Per-bar label = CAUSAL running-drift sign on every causal bar.
            valid = fv & fvd_k
            y = np.full(N, -1, dtype=int)
            y[valid] = run_sign[valid]

            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 200 or vx.sum() < 20:
                continue

            # Gate on TRAIN label balance (avoid a degenerate one-sided target).
            balance = float(y[tx].mean())
            if not (0.2 < balance < 0.8):
                continue

            # CONFIG SELECTION (allowed: forward returns may TUNE the target on a
            # holdout, just not pick which OOS bars trade). Score = how well the
            # CAUSAL regime sign agrees with the realised forward direction on VAL,
            # minus a parsimony penalty on regime count: coarse, persistent regimes
            # generalise OOS; over-segmentation memorises TRAIN and decays on TEST.
            agree_va = (y[vx] == (fr[vx] > 0).astype(int))
            va_dir_acc = float(np.mean(agree_va)) if vx.sum() > 0 else 0.0
            score = va_dir_acc - 0.0008 * n_seg
            if score > best_score:
                best_score = score
                best_labels = y
                best_cfg = f"cusum_f{fk}_seg{target_seg}_n{n_seg}"
                best_horizon = fk

    if best_labels is None:
        return None, "cusum_no_balanced_cfg", None
    return best_labels, best_cfg, best_horizon


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
# Every value is callable with the CANONICAL uniform signature:
#   fn(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons)
#       -> (labels|None, cfg, horizon|None)
# Names exactly as specified. carry is exposed through its uniform wrapper.
# hmm is a BASELINE comparator (Wang does NOT use HMM); the other 7 are featured.
def generate_labels_triple_barrier_tight(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol):
    """Tighter triple-barrier (U=L=1.5 sigma vs the default 2.0): labels trigger on
    smaller moves -> a sharper, more frequent directional signal. Same forward-path
    labeling (allowed for a target); same TRAIN-fit vol scale. Module-①② lever."""
    return generate_labels_triple_barrier(
        lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, U=1.5, L=1.5)


def generate_labels_triple_barrier_meta(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                        horizons=[50, 100, 200]):
    """Meta-labeling PRIMARY: identical triple-barrier labels — the footer adds the
    SECONDARY 'is-the-primary-right' model on top (Wang's trading-decision 2nd model).
    Distinct name so the renderer's labeler-pruner keeps it (and, via its call below,
    generate_labels_triple_barrier) and the cell gets its own key (champion untouched)."""
    return generate_labels_triple_barrier(
        lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=horizons)


def generate_labels_triple_barrier_tight_meta(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                              horizons=[50, 100, 200]):
    """Meta-labeling primary with TIGHTER barriers (U=L=1.5σ vs 2.0): sharper, denser
    directional labels feed the secondary 'is-the-primary-right' model (footer adds it).
    Distinct name so the pruner keeps it (+ generate_labels_triple_barrier) and the cell
    gets its own key. Tests Wang ④ label-density × meta on the EEM edge."""
    return generate_labels_triple_barrier(
        lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=horizons, U=1.5, L=1.5)


def generate_labels_triple_barrier_ae(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                      horizons=[50, 100, 200]):
    """Same triple-barrier labels; the footer routes this cell through a NONLINEAR
    AUTOENCODER dim-reduction (Wang ⑥) instead of the linear correlation-select. Distinct
    name → own cell key + the pruner keeps it (+ generate_labels_triple_barrier)."""
    return generate_labels_triple_barrier(
        lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=horizons)


def generate_labels_dc_reversal(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol):
    """Directional-Change REVERSAL label — MEAN-REVERSION target. Run the delta-reversal
    directional-change process; label each bar with the direction of the NEXT confirmed
    directional change (1 = next reversal is UP / near a bottom, 0 = next reversal is DOWN
    / near a top). Uses the forward DC as the TARGET (allowed — labels may use the future);
    the model predicts the coming turning point from CAUSAL features. Motivated by the
    finding that TLT is mean-reverting (trend-following dc_trend whipsawed). delta = TRAIN
    bar-return std * k. Returns (labels, cfg, None)."""
    N = len(lc)
    tr_idx = np.where(tr_m & fv)[0]
    if len(tr_idx) < 50:
        return None, "", None
    sigma = float(np.nanstd(lr[tr_idx]))
    if not np.isfinite(sigma) or sigma <= 0:
        return None, "", None
    for k in (3.0, 5.0, 8.0):
        delta = k * sigma
        mode, ext = 1, float(lc[0])
        ev_idx, ev_dir = [], []
        for t in range(N):
            p = float(lc[t])
            if mode == 1:
                if p > ext:
                    ext = p
                elif p <= ext - delta:
                    mode, ext = -1, p
                    ev_idx.append(t); ev_dir.append(-1)   # downward reversal
            else:
                if p < ext:
                    ext = p
                elif p >= ext + delta:
                    mode, ext = 1, p
                    ev_idx.append(t); ev_dir.append(1)    # upward reversal
        if len(ev_idx) < 20:
            continue
        y = np.full(N, -1, dtype=int)
        j = 0
        for t in range(N):
            while j < len(ev_idx) and ev_idx[j] <= t:
                j += 1
            if j < len(ev_idx):
                y[t] = 1 if ev_dir[j] == 1 else 0          # direction of the NEXT reversal
        sel = y[tr_idx][y[tr_idx] >= 0]
        bal = float(sel.mean()) if len(sel) else 0.5
        if 0.25 < bal < 0.75:
            return y, f"dc_reversal_k{k}", None
    return None, "dc_reversal_unbalanced", None


def generate_labels_dc_trend(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol):
    """Directional-Change TREND-STATE label — UNSUPERVISED and fully CAUSAL (no forward
    returns). Run a delta-reversal directional-change process over the BAR log-prices and
    label each bar with its current trend mode (1 = up-trend, 0 = down-trend). The mode at
    bar t depends only on prices up to t. delta = TRAIN bar-return std * k, k chosen for a
    balanced up/down split on TRAIN. Aimed at two-sided assets: long up-trends, short downs.
    Returns (labels, cfg, None)."""
    N = len(lc)
    tr_idx = np.where(tr_m & fv)[0]
    if len(tr_idx) < 50:
        return None, "", None
    sigma = float(np.nanstd(lr[tr_idx]))
    if not np.isfinite(sigma) or sigma <= 0:
        return None, "", None
    for k in (3.0, 5.0, 8.0, 12.0):
        delta = k * sigma
        mode, ext = 1, float(lc[0])
        y = np.full(N, -1, dtype=int)
        for t in range(N):
            p = float(lc[t])
            if mode == 1:
                if p > ext:
                    ext = p
                elif p <= ext - delta:
                    mode, ext = -1, p
            else:
                if p < ext:
                    ext = p
                elif p >= ext + delta:
                    mode, ext = 1, p
            y[t] = 1 if mode == 1 else 0
        bal = float(y[tr_idx].mean())
        if 0.25 < bal < 0.75:
            return y, f"dc_trend_k{k}", None
    return None, "dc_trend_unbalanced", None


def generate_labels_trend_scan(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                               horizons=[50, 100, 200]):
    """TREND-SCANNING label (Lopez de Prado, AFML Ch.5) — UNSUPERVISED, NOT HMM. For each bar t,
    fit OLS of forward log-price over several horizons L; take each fit's slope t-statistic; pick
    the horizon with the MAX |t-stat|; label by the SIGN of that most-significant trend (1=up,
    0=down). Discovers the dominant forward trend per bar with NO predefined barrier. Uses the
    future (allowed — the label is the target; the supervised model predicts it causally on OOS).
    Fully vectorised per horizon (convolution + rolling sums). Returns (labels, cfg, None)."""
    N = len(lc)
    Ls = [20, 40, 80]                       # trend horizons in BARS (~4/8/16 trading days on dollar bars)
    csum = np.concatenate([[0.0], np.cumsum(lc)])
    csum2 = np.concatenate([[0.0], np.cumsum(lc * lc)])
    best_abs_t = np.zeros(N)
    best_sign = np.full(N, -1, dtype=int)
    for L in Ls:
        n = L + 1
        if n >= N:
            continue
        x = np.arange(n, dtype=float); w = x - x.mean(); sxx = float((w * w).sum())
        if sxx <= 0:
            continue
        T = N - n + 1                       # windows [t, t+L] for t = 0..T-1
        num = np.convolve(lc, w[::-1], mode="valid")        # Σ_k lc[t+k]·w[k]
        beta = num / sxx
        wsum = csum[n:n + T] - csum[0:T]
        wsum2 = csum2[n:n + T] - csum2[0:T]
        ym = wsum / n
        Syy = wsum2 - n * ym * ym
        SSE = np.maximum(Syy - beta * beta * sxx, 0.0)
        se = np.sqrt(SSE / max(1, n - 2) / sxx)
        tval = np.where(se > 1e-12, beta / se, 0.0)
        at = np.abs(tval)
        upd = at > best_abs_t[:T]
        best_abs_t[:T] = np.where(upd, at, best_abs_t[:T])
        best_sign[:T] = np.where(upd, (beta > 0).astype(int), best_sign[:T])
    y = np.full(N, -1, dtype=int)
    lab = (best_abs_t > 0) & fv
    y[lab] = best_sign[lab]
    tx = fv & (y >= 0) & tr_m
    vx = fv & (y >= 0) & va_m
    if tx.sum() < 200 or vx.sum() < 30:
        return None, "trendscan_insufficient", None
    bal = float(y[tx].mean())
    if not (0.1 < bal < 0.9):               # natural balance ok (don't force 0.5); reject only degenerate
        return None, "trendscan_degenerate", None
    return y, "trendscan_L20_40_80", None


def generate_labels_crash_ahead(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                horizons=[50, 100, 200]):
    """TAIL-RISK label: y=1 if the forward return over horizon H is a CRASH (< -k*sigma_H),
    else 0. A model predicts imminent drawdowns from CAUSAL features; paired with the
    'crashveto' sizing the strategy stays FULLY LONG (capturing carry) and flattens ONLY the
    predicted-crash bars -> aims to beat buy-and-hold on a DRIFT asset by cutting MaxDD while
    keeping CAGR (directional timing instead forfeits the carry, R90/R98). sigma of forward
    returns is fit on TRAIN only; k is swept for a learnable crash base-rate (~5-35%, prefer
    ~15%). Forward returns define the TARGET (allowed). Returns (labels, cfg, horizon)."""
    N = len(lc)
    best = None
    for H in horizons:
        fr = fwd_ret[H]
        fvd = np.isfinite(fr)
        tr_fr = fr[tr_m & fvd & fv]
        if len(tr_fr) < 200:
            continue
        sd = float(np.std(tr_fr))
        if not np.isfinite(sd) or sd <= 0:
            continue
        for k in (1.0, 1.25, 1.5):
            thr = -k * sd
            y = np.full(N, -1, dtype=int)
            valid = fv & fvd
            y[valid] = 0
            y[valid & (fr < thr)] = 1                  # crash ahead within H bars
            tx = fv & (y >= 0) & tr_m
            vx = fv & (y >= 0) & va_m
            if tx.sum() < 200 or vx.sum() < 30:
                continue
            base = float(y[tx].mean())
            if 0.05 < base < 0.35:                     # learnable: not too rare, not too common
                score = -abs(base - 0.15)              # prefer ~15% crash rate
                if best is None or score > best[0]:
                    best = (score, y, f"crash_H{H}_k{k}", H)
    if best is None:
        return None, "crash_unbalanced", None
    return best[1], best[2], best[3]


def generate_labels_ker(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                        horizons=[20, 40, 80]):
    """Kaufman Efficiency-Ratio 'clean-trend' label (Kaufman, 'Smarter Trading' 1995) — UNSUPERVISED,
    non-HMM, orthogonal to trend_scan (which uses an OLS t-stat). For horizon H the forward
    Efficiency Ratio is KER = |lc[t+H]-lc[t]| / sum_{i in (t,t+H]} |lc[i]-lc[i-1]|  in [0,1]:
    high KER = an EFFICIENT (low-noise) directional move, low KER = chop. Label 1 if the forward
    move is an efficient UPtrend (KER>=cut and lc[t+H]>lc[t]), 0 if efficient DOWNtrend, -1 (ignore)
    if choppy. The cutoff is a TRAIN quantile chosen to balance the label set. Forward info defines
    only the TARGET (G3-ok); the SUPERVISED model predicts it from past-only features. Sweeps H and
    the quantile; picks the most TRAIN-balanced config. Returns (labels, cfg, horizon)."""
    N = len(lc)
    abs_dl = np.abs(np.diff(lc, prepend=lc[0]))     # |bar move|; abs_dl[i] = |lc[i]-lc[i-1]|
    cum = np.cumsum(abs_dl)                          # for O(1) gross path
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        ker = np.full(N, np.nan)
        net = np.full(N, np.nan)
        idx = np.arange(N - H)
        net_v = lc[idx + H] - lc[idx]
        gross_v = cum[idx + H] - cum[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ker[idx] = np.abs(net_v) / np.where(gross_v > 1e-12, gross_v, np.nan)
        net[idx] = net_v
        trsel = tr_m & fv & np.isfinite(ker)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.4, 0.5, 0.6, 0.7):
            cut = float(np.quantile(ker[trsel], q))
            y = np.full(N, -1, dtype=int)
            clean = fv & np.isfinite(ker) & (ker >= cut)
            y[clean & (net > 0)] = 1
            y[clean & (net <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"ker_H{H}_q{q}_cut{round(cut,3)}", H
    if best is None:
        return None, "ker_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_trend_leg(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                              horizons=[40, 80]):
    """Wang's FLAGSHIP connected-leg trend-SEGMENTATION label (course Module 4,
    '根据趋势行为自定义标签的密度' / customize label density by trend behaviour) — mined 2026-06-03 as
    the primary Wang labeler we had never built. UNSUPERVISED, non-HMM.

    Segment the forward window into directional LEGS with a CUSUM/zig-zag reversal rule and label by
    the FIRST leg's direction. Walk forward from t tracking the running extreme; the leg is confirmed
    REVERSED when price retraces from that extreme by more than rev = krev*sigma*sqrt(elapsed_bars)
    (random-walk-scaled). The first leg's signed magnitude (extreme - start) is the readout: label 1
    if the first clean leg is UP, 0 if DOWN, -1 (ignore) if its magnitude is below a TRAIN quantile
    (the 'label density' knob — only label where a trend leg is clean).

    Distinct from ker (endpoint efficiency ratio over a FIXED horizon) and trend_scan (OLS t-stat over
    a fixed horizon): trend_leg ADAPTS the effective horizon to the actual trend structure (how far the
    first move runs before a confirmed reversal), and from turn_scan (which reads reversal TIMING, not
    leg direction). sigma (reversal scale) and the magnitude cutoff are TRAIN-fit; the forward window
    defines only the TARGET (G3-ok) — the supervised model predicts it from past-only features.
    Returns (labels, cfg, horizon)."""
    N = len(lc)
    trr = lr[tr_m & np.isfinite(lr)]
    # fallback is dead code under the footer's >=500-TRAIN-bar guard, but keep it TRAIN-masked
    # too (defensive: a full-series std here would be an OOS-inclusive leak if ever reached).
    sigma = float(np.std(trr)) if trr.size > 50 else (float(np.std(trr)) if trr.size > 1 else 1e-4)
    if not np.isfinite(sigma) or sigma <= 0:
        return None, "trend_leg_degenerate", None
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        for krev in (2.0, 3.5):
            legnet = np.full(N, np.nan)        # signed magnitude of the first forward leg
            for t in range(N - H):
                start = lc[t]
                peak = start
                trough = start
                peak_i = t
                trough_i = t
                done = False
                for j in range(t + 1, t + H + 1):
                    p = lc[j]
                    if p > peak:
                        peak = p
                        peak_i = j
                    if p < trough:
                        trough = p
                        trough_i = j
                    thr = krev * sigma * ((j - t) ** 0.5)
                    if (peak - p) >= thr and peak_i > t:       # up-leg confirmed-reversed at its peak
                        legnet[t] = peak - start
                        done = True
                        break
                    if (p - trough) >= thr and trough_i > t:   # down-leg confirmed-reversed at its trough
                        legnet[t] = trough - start
                        done = True
                        break
                if not done:
                    legnet[t] = lc[t + H] - start              # no reversal within H -> net over H
            mag = np.abs(legnet)
            trsel = tr_m & fv & np.isfinite(legnet)
            if int(trsel.sum()) < 100:
                continue
            for q in (0.4, 0.5, 0.6):
                cut = float(np.quantile(mag[trsel], q))
                y = np.full(N, -1, dtype=int)
                clean = fv & np.isfinite(legnet) & (mag >= cut)
                y[clean & (legnet > 0)] = 1
                y[clean & (legnet <= 0)] = 0
                ly = y >= 0
                tx = fv & ly & tr_m
                vx = fv & ly & va_m
                if tx.sum() < 100 or vx.sum() < 20:
                    continue
                bal = float(y[tx].mean())
                if 0.2 < bal < 0.8:
                    score = min(bal, 1 - bal)
                    if score > best_score:
                        best_score, best, best_cfg, best_h = score, y, f"trendleg_H{H}_k{krev}_q{q}", H
    if best is None:
        return None, "trend_leg_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_accel(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                          horizons=[20, 40, 80]):
    """Trend-ACCELERATION label — UNSUPERVISED, non-HMM. Targets ACCELERATING moves, orthogonal to
    trend LEVEL (trend_scan), QUALITY (KER), and REGIME (bgm). For horizon H, split the forward
    window in half: m1 = lc[t+H/2]-lc[t], m2 = lc[t+H]-lc[t+H/2]. Label 1 if accelerating UP
    (m2>0 and m2>m1), 0 if accelerating DOWN (m2<0 and m2<m1), -1 (ignore) if decelerating/mixed.
    Forward info defines only the TARGET (G3-ok); the supervised model predicts it from past
    features. Sweeps H; picks most TRAIN-balanced. Returns (labels, cfg, horizon)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        h2 = H // 2
        if N <= H or h2 < 1:
            continue
        idx = np.arange(N - H)
        m1 = lc[idx + h2] - lc[idx]
        m2 = lc[idx + H] - lc[idx + h2]
        yy = np.full(N - H, -1, dtype=int)
        yy[(m2 > 0) & (m2 > m1)] = 1
        yy[(m2 < 0) & (m2 < m1)] = 0
        y = np.full(N, -1, dtype=int)
        y[idx] = yy
        ly = y >= 0
        tx = fv & ly & tr_m
        vx = fv & ly & va_m
        if tx.sum() < 100 or vx.sum() < 20:
            continue
        bal = float(y[tx].mean())
        if 0.2 < bal < 0.8:
            score = min(bal, 1 - bal)
            if score > best_score:
                best_score, best, best_cfg, best_h = score, y, f"accel_H{H}", H
    if best is None:
        return None, "accel_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_revert(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                           horizons=[20, 40, 80]):
    """Mean-REVERSION / contrarian label — UNSUPERVISED, non-HMM. Every other label targets trend
    CONTINUATION; this targets REVERSALS. For horizon H, trail = lc[t]-lc[t-H] (past move), fwd =
    lc[t+H]-lc[t] (forward move). Label 1 = BOUNCE (trail DOWN, fwd UP), 0 = FADE (trail UP, fwd DOWN);
    -1 (ignore) for CONTINUATIONS (trail & fwd same sign) or weak moves. The supervised model then
    predicts — from PAST-ONLY features (which include the trailing move / oversold state) — whether an
    extended move REVERSES. This is the natural target for MEAN-REVERTERS (TLT rates, IWM) where every
    trend-shape label (ker/trend_scan/accel) fails. cut = TRAIN quantile of |fwd| (keep strong reversals,
    balanced). Forward info defines only the TARGET (G3-ok); features stay past-only. Sweeps H and q.
    Orthogonal to ALL existing labels (they label continuation; this labels the turn)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= 2 * H:
            continue
        idx = np.arange(H, N - H)
        fwd = lc[idx + H] - lc[idx]                       # forward net move
        trail = lc[idx] - lc[idx - H]                     # trailing net move (past; selection only)
        fwd_full = np.full(N, np.nan)
        tr_sign = np.zeros(N)
        fwd_full[idx] = fwd
        tr_sign[idx] = np.sign(trail)
        trsel = tr_m & fv & np.isfinite(fwd_full)
        if int(trsel.sum()) < 100:
            continue
        afwd = np.abs(fwd_full)
        for q in (0.4, 0.5, 0.6, 0.7):
            cut = float(np.quantile(afwd[trsel], q))
            y = np.full(N, -1, dtype=int)
            strong = fv & np.isfinite(fwd_full) & (afwd >= cut)
            y[strong & (tr_sign < 0) & (fwd_full > 0)] = 1     # bounce: was falling, now rising
            y[strong & (tr_sign > 0) & (fwd_full < 0)] = 0     # fade: was rising, now falling
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"revert_H{H}_q{q}_cut{round(cut, 4)}", H
    if best is None:
        return None, "revert_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_turn_scan(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                              horizons=[20, 40, 80]):
    """Forward TURNING-POINT / extremum-TIMING label — UNSUPERVISED, non-HMM. Orthogonal to every
    existing label by the QUANTITY it reads off the forward window: trend_scan=slope significance,
    ker=path efficiency, accel=curvature, mfe_mae=excursion MAGNITUDE, revert=trail-vs-fwd sign.
    This one reads the POSITION (timing) of the forward EXTREMUM to detect a local reversal we are
    sitting ON: a V (forward MIN occurs EARLY, then price ends higher) = a TROUGH -> label UP (1);
    a Λ (forward MAX early, then ends lower) = a PEAK -> label DOWN (0); else -1 (still trending /
    extremum at the far end). Built on this session's finding that edges RESOLVE at turning points,
    and matched to regime-OSCILLATING assets (UUP/dollar) whose tradeable signal IS the regime turn.
    Distinct from `revert` (which keys off the PAST trailing move); turn_scan uses ONLY the forward
    shape. Sweeps H and the 'early' fraction; picks the most TRAIN-balanced. Forward info is the
    TARGET only (G3-ok); the supervised model predicts it from past-only features. Returns (y,cfg,H)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H + 1:
            continue
        M = N - H
        seg = np.empty((M, H), dtype=float)             # seg[t, k-1] = lc[t+k], k=1..H
        for k in range(1, H + 1):
            seg[:, k - 1] = lc[k:k + M]
        terminal = seg[:, -1] - lc[:M]                  # forward net move
        amin = np.argmin(seg, axis=1)                   # position (0-based) of forward MIN
        amax = np.argmax(seg, axis=1)                   # position of forward MAX
        for frac in (0.25, 0.4, 0.5):
            early = max(1, int(H * frac))
            yy = np.full(M, -1, dtype=int)
            yy[(amin < early) & (terminal > 0)] = 1     # V: bottomed early then rose -> trough -> UP
            yy[(amax < early) & (terminal < 0)] = 0     # Λ: peaked early then fell  -> peak  -> DOWN
            y = np.full(N, -1, dtype=int)
            y[:M] = yy
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"turnscan_H{H}_f{frac}", H
    if best is None:
        return None, "turnscan_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_perment(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                            horizons=[20, 40, 80]):
    """INFO-THEORETIC predictability label — permutation entropy (Bandt-Pompe 2002), UNSUPERVISED,
    non-HMM. For horizon H, take the forward log-price window, embed it into consecutive ORDINAL
    triples (d=3), histogram the 6 ordinal patterns, and compute normalized Shannon (permutation)
    entropy PE in [0,1]: LOW PE = an ordinally STRUCTURED / predictable forward path, HIGH PE = random
    chop. Trade only the structured windows (PE <= TRAIN-quantile cut), labelled by the SIGN of the
    terminal forward move (1=up / 0=down); -1 (ignore) when PE is high. Distinct from `ker` (which
    reads MAGNITUDE path-efficiency): PE reads ORDINAL pattern only, so it can KEEP a volatile-but-
    monotone climb that ker's net/gross ratio rejects (low ker, low PE). Forward info is the TARGET
    only (G3-ok); the supervised model predicts it from past-only features. Sweeps H and the quantile;
    picks the most TRAIN-balanced. Returns (labels, cfg, horizon)."""
    N = len(lc)
    d = 3
    logd = math.log(6.0)                                # log(d!) for d=3, the PE normalizer
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H + 1 or H < d + 1:
            continue
        M = N - H
        seg = np.empty((M, H), dtype=float)             # seg[t,k] = lc[t+1+k], k=0..H-1
        for k in range(H):
            seg[:, k] = lc[k + 1:k + 1 + M]
        terminal = seg[:, -1] - lc[:M]                  # forward net move
        counts = np.zeros((M, 8), dtype=np.float64)     # 8 comparison-codes (6 are valid orderings)
        for j in range(H - d + 1):
            a = seg[:, j]
            b = seg[:, j + 1]
            c = seg[:, j + 2]
            code = ((a < b).astype(int) << 2) | ((a < c).astype(int) << 1) | (b < c).astype(int)
            for cc in range(8):
                counts[:, cc] += (code == cc)
        tot = counts.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = counts / np.where(tot[:, None] > 0, tot[:, None], np.nan)
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
        pe = np.full(N, np.nan)
        pe[:M] = ent / logd                             # normalized permutation entropy ~[0,1]
        term_full = np.full(N, np.nan)
        term_full[:M] = terminal
        trsel = tr_m & fv & np.isfinite(pe)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.3, 0.4, 0.5):
            cut = float(np.quantile(pe[trsel], q))
            y = np.full(N, -1, dtype=int)
            structured = fv & np.isfinite(pe) & (pe <= cut) & np.isfinite(term_full)
            y[structured & (term_full > 0)] = 1
            y[structured & (term_full <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"perment_H{H}_q{q}_cut{round(cut, 3)}", H
    if best is None:
        return None, "perment_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_sharpe_scan(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                horizons=[20, 40, 80]):
    """Risk-adjusted forward-trend label — UNSUPERVISED, non-HMM. For horizon H the forward
    Sharpe = (lc[t+H]-lc[t]) / (per-bar vol * sqrt(H)): a move measured RELATIVE TO its own
    volatility. Orthogonal to KER (path-efficiency |net|/gross), trend_scan (OLS t-stat) and
    accel (curvature) — a move can be efficient yet low-Sharpe (large but choppy-vol) or vice
    versa. Label 1 if |Sharpe|>=cut and up, 0 if down, -1 (ignore) if weak. Cut = TRAIN quantile
    of |Sharpe| (balances the set). Forward info defines only the TARGET (G3-ok); the supervised
    model predicts it from past-only features. Sweeps H and q. Returns (labels, cfg, horizon)."""
    N = len(lc)
    dl = np.diff(lc, prepend=lc[0])                  # per-bar move; dl[i]=lc[i]-lc[i-1]
    cdl = np.cumsum(dl)                              # O(1) window sum (== net move)
    cdl2 = np.cumsum(dl * dl)                        # O(1) window sum-of-squares (for vol)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        idx = np.arange(N - H)
        net_v = lc[idx + H] - lc[idx]
        s1 = cdl[idx + H] - cdl[idx]
        s2 = cdl2[idx + H] - cdl2[idx]
        mean = s1 / H
        var = np.maximum(s2 / H - mean * mean, 0.0)
        denom = np.sqrt(var) * float(np.sqrt(H))    # forward-window return vol
        shp = np.full(N, np.nan)
        net = np.full(N, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            shp[idx] = np.where(denom > 1e-12, net_v / denom, np.nan)
        net[idx] = net_v
        trsel = tr_m & fv & np.isfinite(shp)
        if int(trsel.sum()) < 100:
            continue
        ashp = np.abs(shp)
        for q in (0.4, 0.5, 0.6, 0.7):
            cut = float(np.quantile(ashp[trsel], q))
            y = np.full(N, -1, dtype=int)
            clean = fv & np.isfinite(shp) & (ashp >= cut)
            y[clean & (net > 0)] = 1
            y[clean & (net <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"sharpe_scan_H{H}_q{q}_cut{round(cut,3)}", H
    if best is None:
        return None, "sharpe_scan_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_ofsc(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                         horizons=[50, 100, 200]):
    """Order-flow SERIAL-CORRELATION label — UNSUPERVISED, non-HMM, info-flow class (AFML Ch.19.6.5;
    Toth, Eisler, Lillo, Kockelkoren & Bouchaud 2011). Targets the one signal class none of the built
    flow methods touch: every built order-flow method (imbalance/tickimb/volumeimb/vpin) is a bar AXIS
    that NETS signed flow and discards its serial dependence. Here the forward tick-rule sign sequence
    f_j = sign(lr_j) over the window [t+1, t+H] is read for its PERSISTENCE P = mean of the lag-1..5
    CENTERED autocorrelations: P>0 = persistent directional flow (informed / order-splitting), P<0 =
    mean-reverting chop, P~0 = unstructured noise. Orthogonal to the trend/regime labelers, which read
    price-PATH geometry (efficiency / OLS slope / segmentation) and ignore the sign sequence's own serial
    structure. Label 1 if P>=cut and net>0 (persistent up-flow), 0 if P>=cut and net<=0, -1 (ignore) if
    P<cut (no flow persistence = unlearnable). cut = TRAIN quantile of P (balances the set). Forward info
    defines only the TARGET (G3-ok); the supervised model predicts P-regime+direction from past-only
    features. O(1)-per-window via per-lag cumsum (no recompute). Sweeps H and q. Returns (labels, cfg, H)."""
    N = len(lc)
    s = np.sign(lr).astype(float)                       # tick-rule sign of bar returns (0 stays 0)
    cs = np.concatenate([[0.0], np.cumsum(s)])          # O(1) window sum of signs
    cs2 = np.concatenate([[0.0], np.cumsum(s * s)])     # O(1) window sum of sign^2
    KMAX = 5
    prodcs = {}                                         # per-lag cumsum of s_j * s_{j+k}
    for k in range(1, KMAX + 1):
        pk = s[:-k] * s[k:]
        prodcs[k] = np.concatenate([[0.0], np.cumsum(pk)])
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        lo = np.arange(N - H)
        hi = lo + H
        acc = np.zeros(N - H)                            # sum of finite lag-k autocorrs
        cnt = np.zeros(N - H)                            # count of finite lags
        for k in range(1, KMAX + 1):
            n = H - k
            if n <= 5:
                continue
            sumA = cs[hi - k] - cs[lo]                   # A = s[lo:hi-k]
            sumB = cs[hi] - cs[lo + k]                   # B = s[lo+k:hi]
            sumA2 = cs2[hi - k] - cs2[lo]
            sumB2 = cs2[hi] - cs2[lo + k]
            sumAB = prodcs[k][hi - k] - prodcs[k][lo]    # sum_j s_j s_{j+k} over the window
            mA = sumA / n
            mB = sumB / n
            cov = sumAB / n - mA * mB
            vA = np.maximum(sumA2 / n - mA * mA, 0.0)
            vB = np.maximum(sumB2 / n - mB * mB, 0.0)
            den = np.sqrt(vA * vB)
            with np.errstate(divide="ignore", invalid="ignore"):
                ak = np.where(den > 1e-12, cov / den, np.nan)
            good = np.isfinite(ak)
            acc[good] += ak[good]
            cnt[good] += 1.0
        P = np.full(N, np.nan)
        net = np.full(N, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            P[lo] = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        net[lo] = lc[lo + H] - lc[lo]
        trsel = tr_m & fv & np.isfinite(P)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.4, 0.5, 0.6):
            cut = float(np.quantile(P[trsel], q))
            y = np.full(N, -1, dtype=int)
            pers = fv & np.isfinite(P) & (P >= cut)
            y[pers & (net > 0)] = 1
            y[pers & (net <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"ofsc_H{H}_q{q}_cut{round(cut, 3)}", H
    if best is None:
        return None, "ofsc_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_bde_cusum(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                              horizons=[100, 200]):
    """Brown-Durbin-Evans recursive-CUSUM STRUCTURAL-BREAK label — UNSUPERVISED, non-HMM,
    structural-break class (Brown, Durbin & Evans 1975 JRSS-B; AFML Ch.17.3.1). Fits an expanding
    simple-linear OLS (y~a+b*j) to the forward log-price window and accumulates STANDARDIZED RECURSIVE
    RESIDUALS into a CUSUM; a stable trend produces tiny residuals (no break), a trend ONSET / regime
    change produces a large CUSUM excursion. break_stat = max|CUSUM|/sqrt(Nw) measures structural
    instability. Distinct from the trend labelers (ker/trend_leg/sadf): those reward a SUSTAINED clean
    move; BDE fires at the forecast-error instability of trend ONSET — orthogonal (a clean linear trend
    scores LOW here, a flat->ramp scores HIGH). Label 1 if break_stat>=cut and net>0, 0 if break_stat>=cut
    and net<=0, -1 (ignore) if break_stat<cut (stable, no break). cut = TRAIN quantile of break_stat
    (balances). Recursive residuals + net are read off the forward window = the TARGET (G3-ok). Vectorized
    O(Nw)-per-step across windows via per-step running sums. Sweeps H and q. Returns (labels, cfg, H)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H + 10:
            continue
        k0 = max(4, H // 10)
        Nw = H - k0
        nwin = N - H
        x = np.arange(H, dtype=float)
        rows = np.arange(nwin)
        Y = lc[1 + rows[:, None] + np.arange(H)[None, :]]    # (nwin, H): Y[t,j]=lc[t+1+j]
        cumY = np.cumsum(Y, axis=1)                          # cumY[:,s-1] = sum_{j<s} Y
        cumXY = np.cumsum(x[None, :] * Y, axis=1)
        csx = np.cumsum(x)
        csx2 = np.cumsum(x * x)
        Wmat = np.zeros((nwin, Nw))
        for si in range(Nw):
            s = k0 + si
            n = float(s)
            Sx = csx[s - 1]
            Sxx = csx2[s - 1]
            Sy = cumY[:, s - 1]
            Sxy = cumXY[:, s - 1]
            denom = n * Sxx - Sx * Sx
            if abs(denom) < 1e-12:
                continue
            b = (n * Sxy - Sx * Sy) / denom
            a = (Sy - b * Sx) / n
            pred = a + b * x[s]
            xbar = Sx / n
            Sxx_c = Sxx - Sx * Sx / n
            d = 1.0 / n + ((x[s] - xbar) ** 2) / Sxx_c if Sxx_c > 1e-12 else 1.0 / n
            Wmat[:, si] = (Y[:, s] - pred) / float(np.sqrt(1.0 + d))
        sig = np.sqrt(np.maximum(np.mean(Wmat * Wmat, axis=1), 1e-24))     # (nwin,) recursive-resid std
        cusum = np.cumsum(Wmat, axis=1) / sig[:, None]
        stat = np.max(np.abs(cusum), axis=1) / float(np.sqrt(Nw))         # break_stat per window
        bstat = np.full(N, np.nan)
        net = np.full(N, np.nan)
        bstat[rows] = stat
        net[rows] = lc[rows + H] - lc[rows]
        trsel = tr_m & fv & np.isfinite(bstat)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.5, 0.6, 0.7):
            cut = float(np.quantile(bstat[trsel], q))
            y = np.full(N, -1, dtype=int)
            brk = fv & np.isfinite(bstat) & (bstat >= cut)
            y[brk & (net > 0)] = 1
            y[brk & (net <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"bde_cusum_H{H}_q{q}_cut{round(cut, 3)}", H
    if best is None:
        return None, "bde_cusum_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_changepoint(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                horizons=[40, 80, 150]):
    """Mean-shift CHANGE-POINT label — UNSUPERVISED, non-HMM. Wang's EXPLICITLY PREFERRED labeler
    ("还不如直接用change point" — he rejects triple-barrier and dislikes the OLS-t trend_scan). For the
    forward window finds the single split m* that maximizes the standardized mean-shift of per-bar moves
    (CUSUM-of-mean: |mean(seg2)-mean(seg1)| * sqrt(m(H-m)/H) / pooled_std), then labels by the POST-CHANGE
    regime direction (sign of seg2 mean) — distinct from trend_leg (reversal zig-zag), bde_cusum (recursive-
    residual break) and trend_scan (OLS t-stat): it pins WHERE the drift regime shifts and reads the NEW
    regime. Label 1 if cp_stat>=cut and post-change up, 0 if down, -1 (ignore) if cp_stat<cut (no clear
    shift). cut = TRAIN quantile of cp_stat (balances). Forward window = TARGET (G3-ok). O(H)-per-window via
    cumsum running means. Sweeps H and q. Returns (labels, cfg, horizon)."""
    N = len(lc)
    dl = np.diff(lc, prepend=lc[0])                       # per-bar move; dl[i]=lc[i]-lc[i-1]
    cdl = np.concatenate([[0.0], np.cumsum(dl)])          # cdl[k]=sum dl[0..k-1]
    cdl2 = np.concatenate([[0.0], np.cumsum(dl * dl)])
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H + 2:
            continue
        nwin = N - H
        t = np.arange(nwin)
        w_lo = t + 1                                      # forward window dl[t+1 .. t+H]
        w_hi = t + 1 + H
        tot = cdl[w_hi] - cdl[w_lo]
        gmean = tot / H
        sse = (cdl2[w_hi] - cdl2[w_lo]) - 2.0 * gmean * tot + H * gmean * gmean   # sum (dl-gmean)^2
        psd = np.sqrt(np.maximum(sse / H, 1e-18))
        stat = np.zeros(nwin)
        argm = np.full(nwin, H // 2)
        for m in range(3, H - 2):
            c_m = cdl[w_lo + m] - cdl[w_lo]               # sum of first m moves
            m1 = c_m / m
            m2 = (tot - c_m) / (H - m)
            d = np.abs(m2 - m1) * float(np.sqrt(m * (H - m) / H)) / psd
            better = d > stat
            stat = np.where(better, d, stat)
            argm = np.where(better, m, argm)
        c_am = cdl[w_lo + argm] - cdl[w_lo]
        seg2 = (tot - c_am) / np.maximum(H - argm, 1)     # post-change segment mean (the NEW regime)
        cp = np.full(N, np.nan)
        s2 = np.full(N, np.nan)
        cp[t] = stat
        s2[t] = seg2
        trsel = tr_m & fv & np.isfinite(cp)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.5, 0.6, 0.7):
            cut = float(np.quantile(cp[trsel], q))
            y = np.full(N, -1, dtype=int)
            strong = fv & np.isfinite(cp) & (cp >= cut)
            y[strong & (s2 > 0)] = 1
            y[strong & (s2 <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"changepoint_H{H}_q{q}_cut{round(cut, 3)}", H
    if best is None:
        return None, "changepoint_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_calmar_scan(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                                horizons=[40, 80]):
    """Drawdown-adjusted forward-trend label — UNSUPERVISED, non-HMM. Mined 2026-06-03; targets the
    DEPLOYED objective (Calmar) directly. For horizon H: net = lc[t+H]-lc[t]; walk the forward path and
    track the DIRECTION-AWARE adverse excursion — for an up move the worst peak-to-trough DRAWDOWN, for a
    down move the worst trough-to-peak RUN-UP. CMR = |net| / (adverse + eps): a clean trend (large net,
    small counter-move) scores high in EITHER direction. Orthogonal to sharpe_scan (symmetric vol denom —
    penalizes ALL variance) and ker (path-efficiency |net|/gross): Calmar penalizes only the DOWNSIDE of
    the realized direction, which is exactly what dd_overlay sizing + the Calmar objective reward. Label 1
    if net>0 and CMR>=cut, 0 if net<=0 and CMR>=cut, -1 (ignore) if choppy. Cut = TRAIN quantile of CMR.
    Forward info defines only the TARGET (G3-ok). Returns (labels, cfg, horizon)."""
    N = len(lc)
    eps = 1e-6
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        cmr = np.full(N, np.nan)
        net = np.full(N, np.nan)
        for t in range(N - H):
            start = lc[t]
            peak = start
            trough = start
            maxdd = 0.0     # worst peak-to-trough (long-side adverse)
            maxru = 0.0     # worst trough-to-peak (short-side adverse)
            for j in range(t + 1, t + H + 1):
                p = lc[j]
                if p > peak:
                    peak = p
                if p < trough:
                    trough = p
                dd = peak - p
                ru = p - trough
                if dd > maxdd:
                    maxdd = dd
                if ru > maxru:
                    maxru = ru
            nt = lc[t + H] - start
            adverse = maxdd if nt > 0 else maxru
            net[t] = nt
            cmr[t] = abs(nt) / (adverse + eps)
        trsel = tr_m & fv & np.isfinite(cmr)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.4, 0.5, 0.6, 0.7):
            cut = float(np.quantile(cmr[trsel], q))
            y = np.full(N, -1, dtype=int)
            clean = fv & np.isfinite(cmr) & (cmr >= cut)
            y[clean & (net > 0)] = 1
            y[clean & (net <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"calmar_scan_H{H}_q{q}_cut{round(cut,3)}", H
    if best is None:
        return None, "calmar_scan_no_balanced", None
    return best, best_cfg, best_h


def _adf_tstat(y):
    """ADF t-stat for the explosiveness regression dy = a + beta*y_lag (p=0, no lagged diffs for speed).
    Large POSITIVE t = explosive / super-martingale (bubble/crash); ~0 = random walk; negative = mean-revert."""
    nobs = len(y) - 1
    if nobs < 6:
        return -1e9
    x = y[:-1]
    dy = y[1:] - y[:-1]
    xbar = x.mean()
    dybar = dy.mean()
    xd = x - xbar
    sxx = float((xd * xd).sum())
    if sxx < 1e-12:
        return -1e9
    beta = float((xd * (dy - dybar)).sum()) / sxx
    a = dybar - beta * xbar
    resid = dy - (a + beta * x)
    dof = nobs - 2
    if dof < 1:
        return -1e9
    sig2 = float((resid * resid).sum()) / dof
    if sig2 <= 0.0:
        return -1e9
    se = math.sqrt(sig2 / sxx)
    return beta / se if se > 1e-12 else -1e9


def generate_labels_sadf_explosive(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[100, 200]):
    """Supremum-ADF EXPLOSIVE-regime label (AFML Ch.17.4.2, Phillips-Wu-Yu 2011) — mined 2026-06-03.
    A structurally NOVEL signal: neither trend (trend_leg/ker) nor distributional regime (bgm) — it flags
    whether the forward window is in a price BUBBLE/CRASH (explosive / super-martingale, faster-than-
    random-walk growth or collapse). SADF_t = sup over backward-EXPANDING start points of the ADF t-stat
    (beta/se of dy=a+beta*y_lag) with right end fixed at t+H (coarse 5-point grid for speed). Label 1 if
    SADF>=cut and the forward move is UP, 0 if explosive-DOWN, -1 (ignore) if NOT explosive. cut = TRAIN
    quantile of SADF. de Prado's motivation: the explosive regime is where most participants are caught off
    guard — orthogonal to the trend/regime edges. Cut TRAIN-only; forward window = TARGET only (G3-ok).
    Returns (labels, cfg, horizon)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        tau = max(10, H // 4)
        step = max(1, (H - tau) // 4)
        sadf = np.full(N, np.nan)
        net = np.full(N, np.nan)
        for t in range(N - H):
            y = lc[t:t + H + 1]
            bt = -1e9
            for t0 in range(0, H - tau + 1, step):
                ts = _adf_tstat(y[t0:])
                if ts > bt:
                    bt = ts
            sadf[t] = bt
            net[t] = lc[t + H] - lc[t]
        trsel = tr_m & fv & np.isfinite(sadf)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.5, 0.7, 0.85):
            cut = float(np.quantile(sadf[trsel], q))
            y_lab = np.full(N, -1, dtype=int)
            expl = fv & np.isfinite(sadf) & (sadf >= cut)
            y_lab[expl & (net > 0)] = 1
            y_lab[expl & (net <= 0)] = 0
            ly = y_lab >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y_lab[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y_lab, f"sadf_H{H}_q{q}", H
    if best is None:
        return None, "sadf_no_balanced", None
    return best, best_cfg, best_h


def _dfa_hurst(w):
    """Detrended-Fluctuation-Analysis Hurst exponent of a window (Peng 1994). Higher = more PERSISTENT
    (trending / long-memory); lower = anti-persistent (mean-reverting). The absolute scale is irrelevant
    here (the labeler gates on a TRAIN quantile of H), only the relative ordering matters. Manual linear
    detrend (no polyfit) for speed."""
    n = len(w)
    if n < 8:
        return 0.5
    z = np.cumsum(w - w.mean())
    scales = [max(4, n // 8), max(5, n // 4), max(6, n // 2)]
    logs, logf = [], []
    for s in scales:
        if s < 4 or s > n:
            continue
        nseg = n // s
        if nseg < 1:
            continue
        x = np.arange(s, dtype=float)
        xm = x.mean()
        sxx = float(((x - xm) ** 2).sum())
        if sxx < 1e-12:
            continue
        rms = []
        for i in range(nseg):
            seg = z[i * s:(i + 1) * s]
            sm = seg.mean()
            slope = float(((x - xm) * (seg - sm)).sum()) / sxx
            resid = seg - (slope * (x - xm) + sm)
            rms.append(math.sqrt(float((resid * resid).mean())))
        if rms:
            logs.append(math.log(s))
            logf.append(math.log(sum(rms) / len(rms) + 1e-12))
    if len(logs) < 2:
        return 0.5
    la = np.asarray(logs)
    fa = np.asarray(logf)
    lm = la.mean()
    sxx2 = float(((la - lm) ** 2).sum())
    if sxx2 < 1e-12:
        return 0.5
    return float(((la - lm) * (fa - fa.mean())).sum()) / sxx2


def generate_labels_hurst_persist(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[80, 160]):
    """Forward FRACTAL-PERSISTENCE label (DFA Hurst; Hurst 1951, Peng 1994) — mined 2026-06-04. Directly
    targets PERSISTENCE (the durability property the session's durable edge, trend_leg→gold, exhibits):
    label the forward windows that are most PERSISTENT (high DFA-Hurst = clean long-memory directional move,
    multi-scale) by their net direction. Distinct from ker (endpoint efficiency), trend_leg (segmentation),
    sharpe_scan (vol-ratio): Hurst is a MULTI-SCALE fluctuation exponent. Label 1 if H_exp>=cut and net>0,
    0 if persistent-down, -1 (ignore) if non-persistent/choppy. cut = TRAIN quantile of H_exp (so the absolute
    DFA scale is irrelevant). Forward window = TARGET only (G3-ok). Returns (labels, cfg, horizon)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        hexp = np.full(N, np.nan)
        net = np.full(N, np.nan)
        for t in range(N - H):
            hexp[t] = _dfa_hurst(lc[t + 1:t + H + 1])
            net[t] = lc[t + H] - lc[t]
        trsel = tr_m & fv & np.isfinite(hexp)
        if int(trsel.sum()) < 100:
            continue
        for q in (0.4, 0.5, 0.6):
            cut = float(np.quantile(hexp[trsel], q))
            y = np.full(N, -1, dtype=int)
            persist = fv & np.isfinite(hexp) & (hexp >= cut)
            y[persist & (net > 0)] = 1
            y[persist & (net <= 0)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"hurst_H{H}_q{q}", H
    if best is None:
        return None, "hurst_no_balanced", None
    return best, best_cfg, best_h


def generate_labels_mfe_mae(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                            horizons=[20, 40, 80]):
    """Forward EXCURSION-ASYMMETRY label — UNSUPERVISED, non-HMM. Over horizon H the path
    from t rewards a long by MFE = max(lc[t..t+H]) - lc[t] and punishes it by
    MAE = lc[t] - min(lc[t..t+H]). asym = (MFE - MAE)/(MFE + MAE) in [-1,1] measures how much
    the PATH pays a long before it hurts — orthogonal to net-move (ker/sharpe_scan), OLS slope
    (trend_scan) and curvature (accel): two bars with the SAME net move differ in whether price
    ran for you first or drew down first (path quality a holder actually feels). Label 1 if
    asym>=cut (path favoured up), 0 if asym<=-cut (path favoured down), -1 (ignore) if symmetric.
    cut = TRAIN quantile of |asym| (balances the set). Forward info defines only the TARGET
    (G3-ok); the supervised model predicts it from past-only features. Sweeps H and q."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    for H in horizons:
        if N <= H:
            continue
        base = lc[:N - H]
        fmax = base.copy()                              # forward rolling max/min over [t, t+H]
        fmin = base.copy()                              # basic-numpy (no stride_tricks): H+1 passes
        for k in range(1, H + 1):
            seg = lc[k:k + (N - H)]
            fmax = np.maximum(fmax, seg)
            fmin = np.minimum(fmin, seg)
        mfe = fmax - base                               # >= 0 max favorable excursion
        mae = base - fmin                               # >= 0 max adverse excursion
        asym = np.full(N, np.nan)
        denom = mfe + mae
        with np.errstate(divide="ignore", invalid="ignore"):
            asym[:N - H] = np.where(denom > 1e-12, (mfe - mae) / denom, np.nan)
        trsel = tr_m & fv & np.isfinite(asym)
        if int(trsel.sum()) < 100:
            continue
        aa = np.abs(asym)
        for q in (0.3, 0.4, 0.5, 0.6):
            cut = float(np.quantile(aa[trsel], q))
            y = np.full(N, -1, dtype=int)
            fin = fv & np.isfinite(asym)
            y[fin & (asym >= cut)] = 1
            y[fin & (asym <= -cut)] = 0
            ly = y >= 0
            tx = fv & ly & tr_m
            vx = fv & ly & va_m
            if tx.sum() < 100 or vx.sum() < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"mfe_mae_H{H}_q{q}_cut{round(cut, 3)}", H
    if best is None:
        return None, "mfe_mae_no_balanced", None
    return best, best_cfg, best_h


# ===========================================================================
# Wang's TREND-STRENGTH ENSEMBLE (in-rule lever, 2026-06-04). Wang's core
# regime-adaptation trick: instead of crowning ONE labeler, ensemble the SAME
# trend labeler swept across TREND-STRENGTH (his diff-order 5->9 = coarseness;
# here the horizon H is the coarseness knob). Register fixed-strength variants
# so "tleg_fast+tleg_mid+tleg_slow" trains a model per strength and AVERAGES
# them via the existing "+"-ensemble path = implicit regime adaptation.
# ===========================================================================
# Explicit defs (NOT a closure factory) so the orchestrator's _prune_labelers AST pass
# sees each as a FunctionDef and keeps generate_labels_trend_leg / _ker in the pruned render.
def generate_labels_tleg_fast(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=None):
    return generate_labels_trend_leg(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[20])


def generate_labels_tleg_mid(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=None):
    return generate_labels_trend_leg(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[60])


def generate_labels_tleg_slow(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=None):
    return generate_labels_trend_leg(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[150])


def generate_labels_ker_fast(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=None):
    return generate_labels_ker(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[20])


def generate_labels_ker_mid(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=None):
    return generate_labels_ker(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[60])


def generate_labels_ker_slow(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=None):
    return generate_labels_ker(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[150])


def generate_labels_sliced_wasserstein(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons=[60, 120]):
    """Tail-aware OPTIMAL-TRANSPORT regime label (1-D sliced-Wasserstein k-medians) — mined 2026-06-04.
    The non-HMM, non-Gaussian regime labeler the catalog flags as missing: clusters whole FORWARD return-window
    DISTRIBUTIONS (tails/skew), not summary stats. For scalar returns the sliced-Wasserstein distance collapses
    to the closed form W1(a,b)=mean|sort(a)-sort(b)| (sort-difference), and the W1 barycenter of a cluster is the
    element-wise MEDIAN of its members' sorted windows -> k-medians on sorted windows = exact 1-D OT clustering.
    Distinct from bgm/regime_gmm (Gaussian on [fwd_ret,|fwd_ret|]), agglomerative (Euclidean linkage),
    cusum_regime (change-point): NONE uses a transport/distributional distance, which is what makes it tail-aware
    (separates calm vs crash regimes Gaussian clustering blurs). Centroids + the regime->direction sign-map are
    fit on TRAIN ONLY; each bar is assigned by its own forward window (TARGET only, G3-ok), exactly like bgm.
    Sweeps H (window) and K in {2,3}; keeps the most TRAIN-balanced config. Returns (labels, cfg, horizon)."""
    N = len(lc)
    best, best_cfg, best_score, best_h = None, "", -1.0, None
    rng = np.random.RandomState(42)
    for H in horizons:
        if N <= H + 2:
            continue
        # sorted forward return windows W[t] = sort(lr[t+1 : t+1+H]); net = cumulative forward move.
        W = np.full((N, H), np.nan)
        net = np.full(N, np.nan)
        for t in range(N - H - 1):
            w = lr[t + 1:t + 1 + H]
            if w.shape[0] == H and np.isfinite(w).all():
                W[t] = np.sort(w)
                net[t] = lc[t + H] - lc[t]
        row_ok = np.isfinite(W).all(axis=1) & np.isfinite(net)
        tr_fit = tr_m & fv & row_ok
        if int(tr_fit.sum()) < 200:
            continue
        Wtr = W[tr_fit]
        if Wtr.shape[0] > 2500:                          # bound the k-medians fitting compute
            Wfit = Wtr[rng.choice(Wtr.shape[0], 2500, replace=False)]
        else:
            Wfit = Wtr
        Wok = W[row_ok]
        for K in (2, 3):
            order = np.argsort(Wfit.mean(axis=1))        # deterministic init: K windows spread by mean
            C = np.asarray([Wfit[order[int((j + 0.5) / K * (len(order) - 1))]] for j in range(K)], dtype=float)
            for _ in range(8):                           # Lloyd iterations (W1 assign, median update)
                d = np.empty((Wfit.shape[0], K))
                for c in range(K):
                    d[:, c] = np.abs(Wfit - C[c]).mean(axis=1)
                asg = d.argmin(axis=1)
                newC = C.copy()
                for c in range(K):
                    sel = asg == c
                    if int(sel.sum()) >= 1:
                        newC[c] = np.median(Wfit[sel], axis=0)   # W1 barycenter = elementwise median
                if np.allclose(newC, C, atol=1e-12):
                    break
                C = newC
            # assign EVERY row-ok bar to the frozen TRAIN centroids (per-centroid loop bounds memory)
            regime = np.full(N, -1, dtype=int)
            dok = np.empty((Wok.shape[0], K))
            for c in range(K):
                dok[:, c] = np.abs(Wok - C[c]).mean(axis=1)
            regime[row_ok] = dok.argmin(axis=1)
            # regime -> direction via TRAIN net mean ONLY (leak-safe sign map)
            up = set()
            for c in range(K):
                sel = tr_m & fv & row_ok & (regime == c)
                if int(sel.sum()) < 20:
                    continue
                if float(np.mean(net[sel])) > 0.0:
                    up.add(c)
            valid = fv & row_ok
            y = np.full(N, -1, dtype=int)
            y[valid] = 0
            for c in up:
                y[valid & (regime == c)] = 1
            tx = fv & (y >= 0) & tr_m
            vx = fv & (y >= 0) & va_m
            if int(tx.sum()) < 100 or int(vx.sum()) < 20:
                continue
            bal = float(y[tx].mean())
            if 0.2 < bal < 0.8:
                score = min(bal, 1 - bal)
                if score > best_score:
                    best_score, best, best_cfg, best_h = score, y, f"slicedW_H{H}_K{K}", H
    if best is None:
        return None, "slicedW_no_balanced", None
    return best, best_cfg, best_h


LABELERS = {
    "accel": generate_labels_accel,                   # trend-acceleration (new, non-HMM, orthogonal)
    "ker": generate_labels_ker,                       # Kaufman efficiency-ratio clean-trend (new, non-HMM)
    "trend_leg": generate_labels_trend_leg,           # Wang's flagship connected-leg trend SEGMENTATION (new, non-HMM)
    "sharpe_scan": generate_labels_sharpe_scan,       # risk-adjusted forward-trend (new, non-HMM, vol-normalized)
    "ofsc": generate_labels_ofsc,                     # order-flow SERIAL-CORRELATION / flow-persistence (new, non-HMM, info-flow class)
    "bde_cusum": generate_labels_bde_cusum,           # Brown-Durbin-Evans recursive-CUSUM structural-BREAK / trend-onset (new, non-HMM)
    "changepoint": generate_labels_changepoint,       # Wang-PREFERRED mean-shift CHANGE-POINT (post-change regime direction; new, non-HMM)
    "tleg_fast": generate_labels_tleg_fast,           # Wang trend-strength ladder: trend_leg @ H=20 (fast/fine trend)
    "tleg_mid": generate_labels_tleg_mid,             # Wang trend-strength ladder: trend_leg @ H=60 (mid trend)
    "tleg_slow": generate_labels_tleg_slow,           # Wang trend-strength ladder: trend_leg @ H=150 (slow/coarse trend)
    "ker_fast": generate_labels_ker_fast,             # Wang trend-strength ladder: ker @ H=20
    "ker_mid": generate_labels_ker_mid,               # Wang trend-strength ladder: ker @ H=60
    "ker_slow": generate_labels_ker_slow,             # Wang trend-strength ladder: ker @ H=150
    "calmar_scan": generate_labels_calmar_scan,       # drawdown-adjusted forward-trend (new, non-HMM; targets Calmar/downside)
    "sadf_explosive": generate_labels_sadf_explosive, # Supremum-ADF EXPLOSIVE/bubble regime (new, non-HMM; novel signal)
    "hurst_persist": generate_labels_hurst_persist,   # DFA forward fractal-PERSISTENCE (new, non-HMM; multi-scale)
    "sliced_wasserstein": generate_labels_sliced_wasserstein,  # tail-aware OPTIMAL-TRANSPORT regime (W1 k-medians on sorted forward windows; new, non-HMM)
    "mfe_mae": generate_labels_mfe_mae,               # forward excursion-asymmetry / path-quality (new, non-HMM, orthogonal)
    "revert": generate_labels_revert,                 # mean-reversion / contrarian (new, non-HMM; labels the TURN, not continuation)
    "turn_scan": generate_labels_turn_scan,           # forward extremum-TIMING / V-Λ reversal (new, non-HMM; reads turning-point timing)
    "perment": generate_labels_perment,               # permutation-entropy predictability (new, non-HMM, info-theoretic; ordinal structure)
    "kmeans2stage": generate_labels_kmeans_two_stage,
    "dc_trend": generate_labels_dc_trend,
    "dc_reversal": generate_labels_dc_reversal,
    "carry": generate_labels_carry_uniform,
    "tertile": generate_labels_tertile,
    "bgm": generate_labels_bgm,
    "jump_model": generate_labels_jump_model,         # Statistical Jump Model — PERSISTENT regimes (new, non-HMM)
    "agglomerative": generate_labels_agglomerative,
    "triple_barrier": generate_labels_triple_barrier,
    "triple_barrier_tight": generate_labels_triple_barrier_tight,
    "triple_barrier_meta": generate_labels_triple_barrier_meta,   # same labels; footer adds the meta secondary model
    "triple_barrier_tight_meta": generate_labels_triple_barrier_tight_meta,  # 1.5σ labels + meta secondary
    "triple_barrier_ae": generate_labels_triple_barrier_ae,   # autoencoder dim-reduce (footer routes it)
    "multi_horizon": generate_labels_multi_horizon,
    "crash_ahead": generate_labels_crash_ahead,       # tail-risk target; pair with 'crashveto' sizing.
    "trend_scan": generate_labels_trend_scan,         # AFML trend-scanning (unsupervised, non-HMM).
    "regime_gmm": generate_labels_regime_gmm,         # causal-feature GMM regimes.
    "cusum_regime": generate_labels_cusum_regime,     # CUSUM change-point regimes.
    "hmm": generate_labels_hmm,                       # BASELINE comparator only.
    "always_long": generate_labels_always_long,       # BASELINE: buy-and-hold floor.
}

# Which registry entries are Wang's FEATURED methods vs. BASELINE comparators.
FEATURED_LABELERS = [
    "accel", "ker", "trend_leg", "sharpe_scan", "ofsc", "bde_cusum", "changepoint", "tleg_fast", "tleg_mid", "tleg_slow", "ker_fast", "ker_mid", "ker_slow", "calmar_scan", "sadf_explosive", "hurst_persist", "sliced_wasserstein", "mfe_mae", "revert", "turn_scan", "perment", "kmeans2stage", "carry", "tertile", "bgm",
    "agglomerative", "jump_model", "triple_barrier", "triple_barrier_tight", "triple_barrier_meta",
    "triple_barrier_tight_meta", "triple_barrier_ae", "trend_scan", "multi_horizon",
    "regime_gmm", "cusum_regime",
]
BASELINE_LABELERS = ["hmm", "always_long"]
