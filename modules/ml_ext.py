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


def reduce_ml(method, X_train, X_val, X_test, n_components):
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
