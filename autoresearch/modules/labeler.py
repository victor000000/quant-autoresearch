"""Unsupervised labeling methods. Edit freely — any labeling scheme.

Currently: KMeans two-stage, carry, tertile, BGMM with POST_THRESH,
Agglomerative Ward, and Forest of Opinions (label ensemble).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler as FS


def compute_forward_metrics(lc, lr, horizons=[50, 100, 200]):  # Stable default — QQQ uses tick for speed
    """Compute forward returns and volatility at multiple horizons.
    Returns:
        fwd_ret: dict {horizon: np.array of forward returns}
        fwd_vol: dict {horizon: np.array of forward volatility}
    """
    N = len(lc)
    fwd_ret = {}
    fwd_vol = {}
    for fk in horizons:
        fr = np.full(N, np.nan)
        fv = np.full(N, np.nan)
        for t in range(N - fk):
            wr = lr[t+1:t+fk+1]
            if len(wr) < 2:
                continue
            fr[t] = lc[t+fk] - lc[t]
            fv[t] = float(np.std(wr))
        fwd_ret[fk] = fr
        fwd_vol[fk] = fv
    return fwd_ret, fwd_vol


def generate_labels_kmeans_two_stage(lc, lr, tr_m, va_m, te_m, fv,
                                      fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """KMeans two-stage labeling: vol cluster → direction cluster.

    Stage 1: KMeans(K=2) on forward volatility → find low-vol regime
    Stage 2: KMeans(K∈{2,3}) on [fwd_ret, |fwd_ret|] within low-vol → find up-cluster

    Returns:
        best_labels: np.array of -1 (ignore), 0 (short/neutral), 1 (long)
        best_cfg: string describing the winning configuration
        best_horizon: the chosen forward horizon
    """
    N = len(lc)
    best_val_cal = -999
    best_labels = None
    best_cfg = ""
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
        s2_mask_tr = tr_m[fvd_k & fv] & is_low

        fwd_cl = fwd_ret[fk][fvd_k & fv]
        fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        vf_tr = np.column_stack([fwd_cl[s2_mask_tr], fwd_abs_cl[s2_mask_tr]])
        if len(vf_tr) < 60:
            continue

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
            val_balance = y[vx].mean() if vx.sum() > 0 else 0
            if 0.2 < val_balance < 0.8:
                cfg = f"km2_f{fk}_c{nc}"
                quality = min(val_balance, 1 - val_balance)
                score = quality
                if best_val_cal < -99 or score > best_val_cal:
                    best_val_cal = score
                    best_labels = y
                    best_cfg = cfg
                    best_horizon = fk

    return best_labels, best_cfg, best_horizon


def generate_labels_carry(fwd_vol, tr_m, va_m, fv, horizons=[50, 100, 200]):
    """Carry-inspired labels: always long when forward vol is below median."""
    N = len(fwd_vol[horizons[0]])
    best_labels = None
    best_cfg = ""
    best_score = -999

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_vol[fk])
        med_v = float(np.median(fwd_vol[fk][tr_m & fvd_k & fv])) if (tr_m & fvd_k & fv).sum() > 10 else 0
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


def generate_labels_tertile(lc, lr, tr_m, va_m, te_m, fv,
                             fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """Quantile tertile labels: top tertile=long, middle=skip, bottom=no-trade.

    Avoids labeling the noisy middle 33% of returns where direction is ambiguous.
    This should increase signal purity by only trading on extreme moves.
    """
    N = len(lc)
    best_labels = None
    best_cfg = ""
    best_score = -999

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk])

        # Compute tertile boundaries on training data
        fwd_train = fwd_ret[fk][tr_m & fvd_k & fv]
        if len(fwd_train) < 200:
            continue
        top_t = np.percentile(fwd_train, 67)   # top tertile
        bot_t = np.percentile(fwd_train, 33)   # bottom tertile

        y_tertile = np.full(N, -1, dtype=int)
        for i in range(N):
            if not fv[i] or not fvd_k[i]:
                continue
            fr = fwd_ret[fk][i]
            if fr >= top_t:
                y_tertile[i] = 1   # Strong up → long
            elif fr <= bot_t:
                y_tertile[i] = 0   # Strong down → avoid
            # Middle tertile: skip (-1, no label)

        ly = y_tertile >= 0
        tx = fv & ly & tr_m
        vx = fv & ly & va_m
        if tx.sum() < 100 or vx.sum() < 20:
            continue

        balance = y_tertile[tx].mean()
        if 0.3 < balance < 0.7:
            cfg = f"tertile_f{fk}"
            quality = min(balance, 1 - balance)
            if quality > best_score:
                best_score = quality
                best_labels = y_tertile
                best_cfg = cfg

    return best_labels, best_cfg, None


def generate_labels_bgm(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                         k_values=[3, 4, 5], post_thresh=0.40,  # More aggressive: more trades
                         horizons=[50, 100, 200]):
    """Bayesian Gaussian Mixture labels with POST_THRESH (Wang v362 technique).

    BGMM uses a Dirichlet prior (weight_concentration_prior=0.1) to prevent
    cluster over-fragmenting — proven better than KMeans for OOS robustness.

    POST_THRESH: only label as 1 if posterior probability > post_thresh.
    This filters uncertain regime assignments, reducing noise.

    Returns best_labels, best_cfg, best_horizon.
    """
    N = len(lc)
    best_labels = None; best_cfg = ""; best_score = -999

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk]); fwd_abs_k = np.abs(fwd_ret[fk])

        # Vol filter stage (same as KMeans two-stage)
        fv_clean = fwd_vol[fk][tr_m & fvd_k & fv]
        if len(fv_clean) < 30: continue
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(fv_clean.reshape(-1,1))
        cv_vol = km_vol.predict(fwd_vol[fk][fvd_k & fv].reshape(-1,1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low = (cv_vol == lo_vol)

        fwd_cl = fwd_ret[fk][fvd_k & fv]; fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        bgm_input = np.column_stack([fwd_cl[is_low], fwd_abs_cl[is_low]])
        if len(bgm_input) < 100: continue

        for K in [3, 4, 5]:  # Sweep K values for best cluster count
            try:
                bgm = BayesianGaussianMixture(
                    n_components=K, covariance_type='full',
                    weight_concentration_prior=0.1,  # Sparse Dirichlet prior
                    random_state=42, max_iter=300, n_init=3)
                bgm.fit(bgm_input)

                # Find up-cluster: highest mean forward return
                component_means = bgm.means_[:, 0]
                up_c = int(np.argmax(component_means))

                # Get posterior probabilities
                posteriors = bgm.predict_proba(bgm_input)
                up_posterior = posteriors[:, up_c]

                # POST_THRESH: only label if posterior > threshold AND cluster == up
                labels = np.zeros(len(bgm_input), dtype=int)
                mask = (bgm.predict(bgm_input) == up_c) & (up_posterior > post_thresh)
                labels[mask] = 1

                full_labels = np.full(N, -1, dtype=int)
                full_labels[np.where(fvd_k & fv)[0][is_low]] = labels

                y = full_labels; ly = y >= 0
                tx = fv & ly & tr_m; vx = fv & ly & va_m
                if tx.sum() < 100 or vx.sum() < 20: continue

                balance = y[tx].mean()
                if 0.2 < balance < 0.8:
                    cfg = f"bgm_f{fk}_K{K}_pt{post_thresh}"
                    quality = min(balance, 1 - balance)
                    n_labeled = int(ly.sum())
                    # Bonus for more labeled bars (better coverage)
                    score = quality + 0.01 * min(n_labeled / 1000, 1.0)
                    if score > best_score:
                        best_score = score; best_labels = y; best_cfg = cfg
            except Exception:
                continue

    return best_labels, best_cfg, None


def generate_labels_agglomerative(lc, lr, tr_m, va_m, te_m, fv,
                                    fwd_ret, fwd_vol, horizons=[50, 100, 200]):
    """Agglomerative clustering labels (Ward linkage — proven GLD Cal 4.39).

    Unlike KMeans/BGMM which need pre-specified K, agglomerative builds a
    hierarchy of clusters. Ward's method minimizes within-cluster variance,
    producing compact, well-separated regimes.
    """
    N = len(lc)
    best_labels = None; best_cfg = ""; best_score = -999

    for fk in horizons:
        fvd_k = ~np.isnan(fwd_ret[fk]); fwd_abs_k = np.abs(fwd_ret[fk])

        fv_clean = fwd_vol[fk][tr_m & fvd_k & fv]
        if len(fv_clean) < 30: continue
        km_vol = KMeans(n_clusters=2, random_state=42, n_init=5).fit(fv_clean.reshape(-1,1))
        cv_vol = km_vol.predict(fwd_vol[fk][fvd_k & fv].reshape(-1,1))
        lo_vol = int(np.argmin(km_vol.cluster_centers_.flatten()))
        is_low = (cv_vol == lo_vol)

        fwd_cl = fwd_ret[fk][fvd_k & fv]; fwd_abs_cl = fwd_abs_k[fvd_k & fv]
        agg_input = np.column_stack([fwd_cl[is_low], fwd_abs_cl[is_low]])
        if len(agg_input) < 100: continue

        for K in [2, 3, 4]:
            try:
                agg = AgglomerativeClustering(n_clusters=K, linkage='ward')
                labels = agg.fit_predict(agg_input)
                # Find up-cluster: highest mean forward return
                up_c = int(np.argmax([np.mean(fwd_cl[is_low][labels==c]) if np.sum(labels==c)>0 else -999 for c in range(K)]))

                dir_labels = np.zeros(len(agg_input), dtype=int)
                dir_labels[labels == up_c] = 1

                full_labels = np.full(N, -1, dtype=int)
                full_labels[np.where(fvd_k & fv)[0][is_low]] = dir_labels

                y = full_labels; ly = y >= 0
                tx = fv & ly & tr_m; vx = fv & ly & va_m
                if tx.sum() < 100 or vx.sum() < 20: continue

                balance = y[tx].mean()
                if 0.2 < balance < 0.8:
                    cfg = f"agg_f{fk}_K{K}_ward"
                    quality = min(balance, 1 - balance)
                    if quality > best_score:
                        best_score = quality; best_labels = y; best_cfg = cfg
            except Exception:
                continue

    return best_labels, best_cfg, None


def generate_labels_forest(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol,
                            horizons=[50, 100, 200]):
    """Forest of Opinions (Wang v372): ensemble multiple labeling methods.

    Generate labels from Carry + KMeans + Tertile independently, then
    form consensus: label=1 where ≥2 methods agree, label=0 otherwise.
    This filters noise by requiring cross-method agreement.

    Proven: 6 tickers > Calmar 1.0 in v372 (SLV, USO, DBC, GDXJ, XME, EMB).
    """
    N = len(lc)
    all_labels = []

    # Collect labels from each method
    y_carry, _, _ = generate_labels_carry(fwd_vol, tr_m, va_m, fv, horizons)
    if y_carry is not None: all_labels.append(("carry", y_carry))

    y_tertile, _, _ = generate_labels_tertile(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons)
    if y_tertile is not None: all_labels.append(("tertile", y_tertile))

    y_km, _, _ = generate_labels_kmeans_two_stage(lc, lr, tr_m, va_m, te_m, fv, fwd_ret, fwd_vol, horizons)
    if y_km is not None: all_labels.append(("km", y_km))

    if len(all_labels) < 2:
        # Need at least 2 methods for forest
        if all_labels: return all_labels[0][1], f"forest_{all_labels[0][0]}", None
        return None, "", None

    # Ensemble: majority vote (label=1 if ≥ ceil(n/2) methods agree)
    label_stack = np.array([y for _, y in all_labels])  # (M, N)
    # -1 means "no label" for that method; treat as abstain
    n_methods = label_stack.shape[0]
    threshold = 2  # Any 2 methods agree (less conservative, more trades)

    # Count positive votes (1 = long)
    pos_votes = np.sum(label_stack == 1, axis=0)
    # Count total votes (not abstaining)
    total_votes = np.sum(label_stack >= 0, axis=0)

    forest_labels = np.full(N, -1, dtype=int)
    # Label=1 if majority of voting methods say long
    mask_long = (pos_votes >= threshold) & (total_votes >= threshold)
    forest_labels[mask_long] = 1
    # Label=0 if majority say not-long (but some voted)
    mask_short = (total_votes >= threshold) & ~mask_long
    forest_labels[mask_short] = 0

    # Validate
    ly = forest_labels >= 0
    tx = fv & ly & tr_m; vx = fv & ly & va_m
    if tx.sum() < 100 or vx.sum() < 20:
        return None, "", None

    balance = forest_labels[tx].mean()
    if not (0.2 < balance < 0.8):
        return None, "", None

    methods_str = "+".join(m for m, _ in all_labels)
    return forest_labels, f"forest_{methods_str}", None
