"""ml_ext — feature/reduce EXTENSION module (separate QC project file, 2026-06-10).

Wang frontier #5: his production unit is a RICH integer-differencing feature panel
compressed by a NONLINEAR autoencoder (VAE->16). Our prior "reduce closed" verdict
tested only the 80-feature base panel (and a fallback-prone sklearn-MLP AE) on a
reversion name. This module supplies the missing unit, OUT of the concatenated
main.py (64k budget): a ~160-dim integer-diff block + a numpy AE (manual Adam —
torch does not exist on QC) + a PCA control. Same rules as bar_ext.py: top-level
imports numpy/math only, everything else passed in, lint-clean, TRAIN-only fits.
"""
import math

import numpy as np


def _shift_sub(x, w):
    """One stride-w difference: f[i] = x[i] - x[i-w]; NaN head (causal warmup)."""
    out = np.full(len(x), np.nan)
    out[w:] = x[w:] - x[:-w]
    return out


def rich_block(lc, lr):
    """Wang integer-order differencing grid: orders 1..5 x 16 strides x 2 series
    (log-close; cumulative |log-return| = activity clock). 160 features, strictly
    causal (trailing shift-subtract only). Downstream StandardScaler (TRAIN-fit)
    handles scale; NaN warmup rows are dropped by the feature-validity mask."""
    lc = np.asarray(lc, dtype=float)
    act = np.cumsum(np.abs(np.asarray(lr, dtype=float)))
    strides = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
    cols = []
    for series in (lc, act):
        for d in range(1, 6):
            base = series
            for _ in range(d - 1):
                base = _shift_sub(base, 1)         # raise the order at stride 1
            for w in strides:
                cols.append(_shift_sub(base, w))   # stride-w difference of order d
    return np.column_stack(cols).astype(np.float32)


def _adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    p -= lr * mh / (np.sqrt(vh) + eps)


def vix_feats(vix, lr, N=None):
    """VXX (VIX short-term futures ETN) exogenous features — the equity-index reopening
    channel (user: use the tradeable ETN, not the CBOE index). vix = log(VXX close)
    co-indexed with bars (causal market data, carrier from the footer); lr = the
    asset's own bar log-returns. All causal trailing windows; NaN until warm.
    8 features: level z (60/252), change (1/5/20 bars), 252-bar percentile rank,
    variance-risk-premium proxy (VIX/100/sqrt(252) vs trailing realized bar vol), VRP z."""
    import numpy as _np
    v = _np.asarray(vix, dtype=float)
    n = len(v) if N is None else N
    out = []
    def _z(x, w):
        r = _np.full(n, _np.nan)
        for i in range(w, n):
            seg = x[i - w:i + 1]
            m = _np.nanmean(seg); sd = _np.nanstd(seg)
            r[i] = (x[i] - m) / (sd + 1e-9)
        return r
    out.append(_z(v, 60)); out.append(_z(v, 252))
    for k in (1, 5, 20):
        d = _np.full(n, _np.nan); d[k:] = v[k:] - v[:-k]; out.append(d)
    pr = _np.full(n, _np.nan)
    for i in range(252, n):
        seg = v[i - 252:i + 1]
        pr[i] = float(_np.nanmean(seg <= v[i]))
    out.append(pr - 0.5)
    rl = _np.asarray(lr, dtype=float)
    rv = _np.full(n, _np.nan)
    for i in range(20, n):
        rv[i] = float(_np.nanstd(rl[i - 20:i + 1]))
    # VRP proxy on the VXX carrier (log-level, not VIX points): z-difference between the
    # implied-vol carrier and the asset's realized vol — scale-free, split/decay-robust.
    vrp = _z(v, 60) - _z(rv, 60)
    out.append(vrp)
    out.append(_z(_np.where(_np.isfinite(vrp), vrp, _np.nan), 60))
    return _np.column_stack(out).astype(_np.float32)


def cal_feats(ts_np):
    """Calendar/seasonality block (backlog #8): 8 causal columns from bar timestamps
    ONLY — zero price content (immune to panel dilution). sin/cos day-of-week,
    sin/cos month-phase, turn-of-month flag (last 2 + first 3 trading-ish days),
    days-to-month-end (scaled), sin/cos month-of-year. Row-wise in own timestamp =>
    append-invariant by construction."""
    days = ts_np.astype("datetime64[D]")
    d = days.astype("int64")
    dow = (d + 3) % 7                                    # 0 = Monday
    M = days.astype("datetime64[M]")
    dom = (days - M.astype("datetime64[D]")).astype("int64")          # 0-based
    eom = ((M + 1).astype("datetime64[D]") - days).astype("int64")    # days to next month
    tom = ((dom < 3) | (eom <= 2)).astype(np.float32)
    moy = M.astype("int64") % 12
    tp = 2.0 * math.pi
    cols = [np.sin(tp * dow / 7.0), np.cos(tp * dow / 7.0),
            np.sin(tp * dom / 31.0), np.cos(tp * dom / 31.0),
            tom, (eom / 31.0).astype(np.float32),
            np.sin(tp * moy / 12.0), np.cos(tp * moy / 12.0)]
    return np.column_stack(cols).astype(np.float32)


def dd_feats(lc, lr):
    """Drawdown-STATE + trend-age block (backlog #9): 8 causal trailing columns —
    underwater depth (vs running max), underwater duration, spell-recovery slope,
    vol-normalized depth, signed return run-length, trend age (bars since 50-bar
    momentum sign flip), distance from rolling 100-bar max/min. Path-DEPENDENT
    state (not monotone-of-momentum): reversion timing + trend exhaustion inputs."""
    N = len(lc)
    runmax = np.maximum.accumulate(lc)
    depth = lc - runmax                                   # <= 0
    dur = np.zeros(N, np.float32)
    rec = np.zeros(N, np.float32)
    trough = lc.copy()
    for t in range(1, N):
        if depth[t] < -1e-12:
            dur[t] = dur[t - 1] + 1
            trough[t] = min(trough[t - 1], lc[t])
            rec[t] = (lc[t] - trough[t]) / dur[t]
        else:
            trough[t] = lc[t]
    W = 100
    vol = np.full(N, np.nan, np.float32)
    hi = np.full(N, np.nan, np.float32)
    lo = np.full(N, np.nan, np.float32)
    for t in range(W, N):
        w = lr[t - W:t]
        w = w[np.isfinite(w)]
        vol[t] = np.std(w) if len(w) >= 20 else np.nan
        hi[t] = lc[t] - np.max(lc[t - W:t])
        lo[t] = lc[t] - np.min(lc[t - W:t])
    zdep = depth / (vol * np.sqrt(W) + 1e-9)
    run = np.zeros(N, np.float32)
    for t in range(1, N):
        s = np.sign(lr[t]) if np.isfinite(lr[t]) else 0.0
        run[t] = run[t - 1] + s if s != 0 and np.sign(run[t - 1]) in (0.0, s) else s
    mom = np.full(N, np.nan)
    mom[50:] = lc[50:] - lc[:-50]
    age = np.zeros(N, np.float32)
    for t in range(51, N):
        same = np.isfinite(mom[t]) and np.isfinite(mom[t - 1]) and np.sign(mom[t]) == np.sign(mom[t - 1])
        age[t] = age[t - 1] + 1 if same else 0
    cols = [depth, np.log1p(dur), rec, zdep, np.clip(run, -30, 30) / 30.0,
            np.log1p(age), hi, lo]
    return np.column_stack(cols).astype(np.float32)


def calibrate(kind, cs, cy, pv_raw, pe_raw, venn_abers=None):
    """Calibration switch (footer delegates here for 64k budget). Returns
    (pv_cal, pe_cal, cal_or_None) — cal is the isotonic object (the hot bundle
    serializes its thresholds; beta/venn bundles carry no calibrator = A/B-grade).
    'beta' = Kull-Filho-Flach beta calibration: logistic fit on [ln p, -ln(1-p)] —
    matches isotonic accuracy at ~1/20th the variance below ~1000 cal samples
    (backlog #7). Default/unknown kind = isotonic, BIT-EXACT legacy path."""
    pv = np.clip(pv_raw, 1e-6, 1 - 1e-6)
    pe = np.clip(pe_raw, 1e-6, 1 - 1e-6)
    if kind == "venn_abers" and venn_abers is not None:
        return venn_abers(cs, cy, pv), venn_abers(cs, cy, pe), None
    if kind == "beta":
        from sklearn.linear_model import LogisticRegression
        X = np.column_stack([np.log(cs), -np.log(1.0 - cs)])
        lrm = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lrm.fit(X, np.asarray(cy).astype(int))
        def _f(p):
            return lrm.predict_proba(np.column_stack([np.log(p), -np.log(1.0 - p)]))[:, 1]
        return _f(pv), _f(pe), None
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(cs, cy)
    return cal.transform(pv), cal.transform(pe), cal


def xgb_plain(scale_w, md=3):
    """The footer's fixed depth-3 xgb spec WITHOUT early stopping (purged-CV folds +
    meta secondary use it). Centralized here for the 64k budget."""
    import xgboost as _xgb
    return _xgb.XGBClassifier(
        n_estimators=200, max_depth=md, learning_rate=0.03,
        reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_w, objective="binary:logistic",
        eval_metric="auc", tree_method="hist", random_state=42, n_jobs=1,
        base_score=0.5)


class _SeedBag:
    """K-seed probability-averaging wrapper (duck-types predict_proba only)."""

    def __init__(self, ms):
        self.ms = ms

    def predict_proba(self, X):
        out = self.ms[0].predict_proba(X)
        for m in self.ms[1:]:
            out = out + m.predict_proba(X)
        return out / len(self.ms)


def fit_model(model, Xt, yt, Xv, yv, scale_w, md):
    """Capacity-matched supervised model swap (CONFIG['model']). Lives here, NOT in
    footer, for 64k budget reasons. Same depth/lr/n/reg/subsample across families —
    a model-FAMILY A/B at fixed capacity (depth>3 stays closed). Returns (m, errs):
    errs carries any lgbm primary-fit exception (surfaced as a runtime stat by the
    caller) before the plain-fit fallback (fixed 200 trees, no early stop)."""
    errs = []
    if model == "lgbm":
        import lightgbm as _lgb
        m = _lgb.LGBMClassifier(
            n_estimators=200, max_depth=md, learning_rate=0.03,
            reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=scale_w, objective="binary",
            random_state=42, n_jobs=1, verbose=-1)
        try:
            m.fit(Xt, yt.astype(int), eval_set=[(Xv, yv.astype(int))],
                  eval_metric="auc", callbacks=[_lgb.early_stopping(30, verbose=False)])
        except Exception as e:
            errs.append(type(e).__name__ + ":" + str(e)[:70])
            m.fit(Xt, yt.astype(int))
    elif model == "lgbm_bag":
        # backlog #11 seed-bagging: K=5 capacity-IDENTICAL lgbm fits (seeds 42..46),
        # probability-averaged. Adds no capacity and no signal — pure variance/decay
        # hardening of the lgbm crown. Bundle family is non-deployable (A/B grade).
        import lightgbm as _lgb
        ms = []
        for k in range(5):
            mk = _lgb.LGBMClassifier(
                n_estimators=200, max_depth=md, learning_rate=0.03,
                reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=scale_w, objective="binary",
                random_state=42 + k, n_jobs=1, verbose=-1)
            try:
                mk.fit(Xt, yt.astype(int), eval_set=[(Xv, yv.astype(int))],
                       eval_metric="auc", callbacks=[_lgb.early_stopping(30, verbose=False)])
            except Exception as e:
                errs.append(type(e).__name__ + ":" + str(e)[:70])
                mk.fit(Xt, yt.astype(int))
            ms.append(mk)
        m = _SeedBag(ms)
    elif model == "xgb_bag":
        # seed-bagging for the xgb family (USO-side test of the lgbm_bag win):
        # K=5 capacity-identical fits, seeds 42..46, probability-averaged.
        import xgboost as _xgb2
        ms = []
        for k in range(5):
            mk = _xgb2.XGBClassifier(
                n_estimators=200, max_depth=md, learning_rate=0.03,
                reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=scale_w, objective="binary:logistic",
                eval_metric="auc", tree_method="hist", random_state=42 + k, n_jobs=1,
                early_stopping_rounds=30, base_score=0.5)
            mk.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            ms.append(mk)
        m = _SeedBag(ms)
    elif model == "catboost":
        from catboost import CatBoostClassifier as _Cat
        m = _Cat(iterations=200, depth=md, learning_rate=0.03,
                 l2_leaf_reg=2.0, rsm=0.85, bootstrap_type="Bernoulli", subsample=0.85,
                 scale_pos_weight=scale_w, loss_function="Logloss", eval_metric="AUC",
                 random_seed=42, thread_count=1, verbose=False, allow_writing_files=False)
        try:
            m.fit(Xt, yt.astype(int), eval_set=(Xv, yv.astype(int)),
                  early_stopping_rounds=30, verbose=False)
        except Exception as e:
            errs.append(type(e).__name__ + ":" + str(e)[:70])
            m.fit(Xt, yt.astype(int), verbose=False)
    else:
        import xgboost as _xgb
        m = _xgb.XGBClassifier(
            n_estimators=200, max_depth=md, learning_rate=0.03,
            reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=scale_w, objective="binary:logistic",
            eval_metric="auc", tree_method="hist", random_state=42, n_jobs=1,
            early_stopping_rounds=30, base_score=0.5)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    return m, errs


def serialize_model(m):
    """(family, payload) for the hot bundle. hasattr-guarded, NEVER raises across
    model APIs — Lean cannot catch cross-API AttributeErrors inside try/except
    (run d61ec92). Only family='xgb' payloads are live-deployable (infer_online/
    live_trade reconstruct xgb-JSON); others are A/B-replay-grade."""
    if hasattr(m, "get_booster"):
        return "xgb", m.get_booster().save_raw("json").decode("utf-8")
    if hasattr(m, "booster_"):
        return "lgbm", m.booster_.model_to_string()
    return type(m).__name__, ""


def reduce_ml(method, X_train, X_val, X_test, n_components, y_train=None):
    """'pca' (sklearn, TRAIN-fit linear control) or 'ae_np' (numpy nonlinear AE,
    manual Adam backprop, TRAIN-fit). Returns the reduce_dims contract tuple:
    (Xtr, Xv, Xte, nk, tag, kept_idx_placeholder). Raises on failure — the caller
    degrades gracefully."""
    K = int(max(2, min(n_components, X_train.shape[1] - 1)))
    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=K, random_state=42)
        p.fit(X_train)
        return (p.transform(X_train).astype(np.float32), p.transform(X_val).astype(np.float32),
                p.transform(X_test).astype(np.float32), K, f"pca{K}", list(range(K)))
    if method == "minor_pca":
        # The COMPLEMENT of pca: project onto the BOTTOM-K eigenvectors (smallest TRAIN-variance
        # directions). Hypothesis (2026-06-11 variance-structure insight): trend signal lives in
        # HIGH-variance directions (pca harvests it on GLD), mean-reversion signal in LOW-variance
        # directions -> minor_pca should help reversion names (USO) and crater trend (GLD), the
        # mirror of pca. Leak-safe: components fit on TRAIN only, frozen loadings applied to val/test.
        from sklearn.decomposition import PCA
        p = PCA(random_state=42)
        p.fit(X_train)
        comp = p.components_[-K:]                       # (K, F) smallest-eigenvalue eigenvectors
        mean = p.mean_
        def _pr(X):
            return ((np.asarray(X, dtype=np.float64) - mean) @ comp.T).astype(np.float32)
        return (_pr(X_train), _pr(X_val), _pr(X_test), K, f"minorpca{K}", list(range(K)))
    if method == "pls":
        # Backlog #1: SUPERVISED twin of the winning pca mechanism — project onto
        # max label-COVARIANCE directions (NIPALS via sklearn). TRAIN-fit on (X, y),
        # frozen loadings applied to val/test. Same leak contract as pca.
        from sklearn.cross_decomposition import PLSRegression
        if y_train is None:
            raise ValueError("pls reduce requires y_train")
        yc = np.asarray(y_train, dtype=np.float64) - float(np.mean(y_train))
        p = PLSRegression(n_components=K, scale=False)
        p.fit(np.asarray(X_train, dtype=np.float64), yc)
        return (p.transform(X_train).astype(np.float32), p.transform(X_val).astype(np.float32),
                p.transform(X_test).astype(np.float32), K, f"pls{K}", list(range(K)))
    if method == "spca":
        # Backlog #2 (Bair supervised PCA): SCREEN features by TRAIN |corr with y|,
        # keep the top 2K, THEN project with PCA(K) — selection-then-projection, the
        # literal bridge of the mechanism-paired insight. TRAIN-only screen + fit.
        from sklearn.decomposition import PCA
        if y_train is None:
            raise ValueError("spca reduce requires y_train")
        Xt64 = np.asarray(X_train, dtype=np.float64)
        yc = np.asarray(y_train, dtype=np.float64) - float(np.mean(y_train))
        sd = Xt64.std(axis=0)
        sd[sd < 1e-12] = 1e-12
        cors = np.abs((Xt64 - Xt64.mean(axis=0)).T @ yc) / (sd * (np.std(yc) + 1e-12) * len(yc))
        keep = np.argsort(cors)[::-1][:max(2 * K, 10)]
        p = PCA(n_components=K, random_state=42)
        p.fit(Xt64[:, keep])
        def _sp(X):
            return p.transform(np.asarray(X, dtype=np.float64)[:, keep]).astype(np.float32)
        return (_sp(X_train), _sp(X_val), _sp(X_test), K, f"spca{K}", list(range(K)))
    if method == "whiten":
        # PCA-WHITEN: top-K directions but each rescaled to UNIT variance — removes the
        # variance-weighting so low-variance (reversion) directions stand on equal footing with
        # high-variance (trend) ones. Leak-safe: fit TRAIN, frozen transform applied to val/test.
        from sklearn.decomposition import PCA
        p = PCA(n_components=K, whiten=True, random_state=42)
        p.fit(X_train)
        return (p.transform(X_train).astype(np.float32), p.transform(X_val).astype(np.float32),
                p.transform(X_test).astype(np.float32), K, f"whiten{K}", list(range(K)))
    if method in ("vae", "vae_rl"):
        # REAL torch VAE (user 2026-06-11: torch IS available on QC — the old "no-torch"
        # record was false; the numpy-AE closure was a stand-in, not Wang's spec).
        # Wang spec: nonlinear reduce, K latents, KL-weighted, mu-encoding at inference.
        # TRAIN-fit, frozen encode for val/test (pca-identical leak contract). Deterministic.
        import torch
        torch.manual_seed(42)
        Xt = torch.tensor(np.asarray(X_train, dtype=np.float32))
        F = Xt.shape[1]; H = 64; beta = 1e-3
        enc = torch.nn.Sequential(torch.nn.Linear(F, H), torch.nn.Tanh())
        mu_l = torch.nn.Linear(H, K); lv_l = torch.nn.Linear(H, K)
        dec = torch.nn.Sequential(torch.nn.Linear(K, H), torch.nn.Tanh(), torch.nn.Linear(H, F))
        params = list(enc.parameters()) + list(mu_l.parameters()) + list(lv_l.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        n = Xt.shape[0]; bs = min(512, n)
        for _ep in range(60):
            perm = torch.randperm(n)
            for s0 in range(0, n, bs):
                xb = Xt[perm[s0:s0 + bs]]
                h = enc(xb); mu = mu_l(h); lv = lv_l(h)
                z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
                xr = dec(z)
                rec = ((xr - xb) ** 2).mean()
                kl = (-0.5 * (1.0 + lv - mu ** 2 - torch.exp(lv)).sum(dim=1)).mean()
                loss = rec + beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
        def _enc(X):
            with torch.no_grad():
                return mu_l(enc(torch.tensor(np.asarray(X, dtype=np.float32)))).numpy().astype(np.float32)
        Ztr, Zv, Zte = _enc(X_train), _enc(X_val), _enc(X_test)
        if method == "vae":
            return (Ztr, Zv, Zte, K, "vae" + str(K), list(range(K)))
        # vae_rl: append an RL POLICY feature (user: "rl for features"). Tiny policy net
        # p=tanh(w·x) trained on TRAIN to maximize mean(p * reward) - lam*mean(|dp|)
        # (reward = +1/-1 from the TRAIN label; turnover penalty -> smoothed/hysteretic
        # positioning, not plain regression). TRAIN-fit, frozen -> leak-safe.
        torch.manual_seed(43)
        yt = torch.tensor((2.0 * np.asarray(y_train, dtype=np.float32) - 1.0))
        pol = torch.nn.Sequential(torch.nn.Linear(F, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1), torch.nn.Tanh())
        opt2 = torch.optim.Adam(pol.parameters(), lr=1e-3)
        lam = 0.1
        for _ep in range(80):
            p = pol(Xt).squeeze(-1)
            j = (p * yt).mean() - lam * (p[1:] - p[:-1]).abs().mean()
            loss2 = -j
            opt2.zero_grad(); loss2.backward(); opt2.step()
        def _pol(X):
            with torch.no_grad():
                return pol(torch.tensor(np.asarray(X, dtype=np.float32))).numpy().astype(np.float32)
        Ztr = np.hstack([Ztr, _pol(X_train)]); Zv = np.hstack([Zv, _pol(X_val)]); Zte = np.hstack([Zte, _pol(X_test)])
        return (Ztr, Zv, Zte, K + 1, "vaerl" + str(K), list(range(K + 1)))
    if method != "ae_np":
        raise ValueError(method)
    # numpy autoencoder F -> H(tanh) -> K(linear) -> H(tanh) -> F, MSE, Adam, seed 42.
    rng = np.random.default_rng(42)
    Xt = np.asarray(X_train, dtype=np.float64)
    F = Xt.shape[1]
    H = int(max(K * 2, min(64, F // 2)))
    def _init(a, b):
        return rng.normal(0, math.sqrt(2.0 / (a + b)), (a, b))
    W = [_init(F, H), _init(H, K), _init(K, H), _init(H, F)]
    B = [np.zeros(H), np.zeros(K), np.zeros(H), np.zeros(F)]
    M = [np.zeros_like(w) for w in W] + [np.zeros_like(b) for b in B]
    V = [np.zeros_like(w) for w in W] + [np.zeros_like(b) for b in B]
    n = len(Xt)
    bs = min(512, n)
    t = 0
    for epoch in range(60):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            xb = Xt[order[s:s + bs]]
            t += 1
            # forward
            h1 = np.tanh(xb @ W[0] + B[0])
            z = h1 @ W[1] + B[1]
            h2 = np.tanh(z @ W[2] + B[2])
            xr = h2 @ W[3] + B[3]
            # backward (MSE)
            d = 2.0 * (xr - xb) / xb.size
            gW3 = h2.T @ d; gB3 = d.sum(0)
            dh2 = (d @ W[3].T) * (1 - h2 * h2)
            gW2 = z.T @ dh2; gB2 = dh2.sum(0)
            dz = dh2 @ W[2].T
            gW1 = h1.T @ dz; gB1 = dz.sum(0)
            dh1 = (dz @ W[1].T) * (1 - h1 * h1)
            gW0 = xb.T @ dh1; gB0 = dh1.sum(0)
            for p, g, m, v in ((W[0], gW0, M[0], V[0]), (W[1], gW1, M[1], V[1]),
                               (W[2], gW2, M[2], V[2]), (W[3], gW3, M[3], V[3]),
                               (B[0], gB0, M[4], V[4]), (B[1], gB1, M[5], V[5]),
                               (B[2], gB2, M[6], V[6]), (B[3], gB3, M[7], V[7])):
                _adam_step(p, g, m, v, t)
    def enc(Z):
        return (np.tanh(np.asarray(Z, dtype=np.float64) @ W[0] + B[0]) @ W[1] + B[1]).astype(np.float32)
    return (enc(X_train), enc(X_val), enc(X_test), K, f"aenp{K}", list(range(K)))


def _moe_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def moe_law_fit(X, y, K, n_em=12, seed=42):
    """Mixture-of-linear-experts with a learned softmax gate (moe_law, 2026-06-10).
    Regimes are defined by WHICH x->y LAW holds (mixture of regressions), not by
    x-density (regime_gmm) or y-clusters (bgm). Deterministic momentum-tertile init,
    seed only breaks ties. All fits on the TRAIN slice passed in. Returns
    (A, b, s2, W, c, train_resp): experts y~X@A[k]+b[k], noise s2[k]; gate
    softmax(X@W+c); train_resp = final TRAIN responsibilities (n,K)."""
    n, dd = X.shape
    order = np.argsort(X[:, 0], kind="stable")
    r = np.zeros((n, K))
    for k in range(K):
        r[order[(n * k) // K:(n * (k + 1)) // K], k] = 1.0   # tertile init on momentum
    A = np.zeros((K, dd)); b = np.zeros(K); s2 = np.ones(K)
    W = np.zeros((dd, K)); c = np.zeros(K)
    for _ in range(n_em):
        # M-step: weighted ridge OLS per expert
        for k in range(K):
            w = r[:, k] + 1e-9
            Xw = X * w[:, None]
            G = X.T @ Xw + 1e-3 * np.eye(dd)
            A[k] = np.linalg.solve(G, (Xw * y[:, None]).sum(axis=0))
            b[k] = float((w * (y - X @ A[k])).sum() / w.sum())
            res = y - X @ A[k] - b[k]
            s2[k] = max(1e-10, float((w * res * res).sum() / w.sum()))
        # gate: 5 gradient steps toward responsibilities (multinomial CE)
        for _g in range(5):
            P = _moe_softmax(X @ W + c)
            Gr = X.T @ (P - r) / n
            W -= 1.0 * Gr
            c -= 1.0 * (P - r).mean(axis=0)
        # E-step: responsibilities from gate prior x expert likelihood
        P = _moe_softmax(X @ W + c)
        L = np.zeros((n, K))
        for k in range(K):
            res = y - X @ A[k] - b[k]
            L[:, k] = np.log(P[:, k] + 1e-12) - 0.5 * (res * res / s2[k] + math.log(s2[k]))
        L = L - L.max(axis=1, keepdims=True)
        r = np.exp(L)
        r = r / r.sum(axis=1, keepdims=True)
    return A, b, s2, W, c, r


def moe_law_assign(X, y, A, b, s2, W, c):
    """Responsibilities for arbitrary rows (uses the row's own forward y — G3-legal
    target use) + the gate's CAUSAL x-only argmax for the agreement filter."""
    n = len(y)
    K = len(b)
    P = _moe_softmax(X @ W + c)
    L = np.zeros((n, K))
    for k in range(K):
        res = y - X @ A[k] - b[k]
        L[:, k] = np.log(P[:, k] + 1e-12) - 0.5 * (res * res / s2[k] + math.log(s2[k]))
    resp = np.argmax(L, axis=1)
    gate = np.argmax(P, axis=1)
    return resp, gate
