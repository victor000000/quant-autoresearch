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
        s_all = sigma[~np.isnan(sigma)]
        sigma_floor = float(np.median(s_all)) if len(s_all) > 0 else 1e-6
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


LABELERS = {
    "kmeans2stage": generate_labels_kmeans_two_stage,
    "carry": generate_labels_carry_uniform,
    "tertile": generate_labels_tertile,
    "bgm": generate_labels_bgm,
    "agglomerative": generate_labels_agglomerative,
    "triple_barrier": generate_labels_triple_barrier,
    "triple_barrier_tight": generate_labels_triple_barrier_tight,
    "multi_horizon": generate_labels_multi_horizon,
    "regime_gmm": generate_labels_regime_gmm,         # causal-feature GMM regimes.
    "cusum_regime": generate_labels_cusum_regime,     # CUSUM change-point regimes.
    "hmm": generate_labels_hmm,                       # BASELINE comparator only.
    "always_long": generate_labels_always_long,       # BASELINE: buy-and-hold floor.
}

# Which registry entries are Wang's FEATURED methods vs. BASELINE comparators.
FEATURED_LABELERS = [
    "kmeans2stage", "carry", "tertile", "bgm",
    "agglomerative", "triple_barrier", "triple_barrier_tight", "multi_horizon",
    "regime_gmm", "cusum_regime",
]
BASELINE_LABELERS = ["hmm", "always_long"]
