"""Module ⑤: Model Training + Full Pipeline Integration (Wang workflow).

Orchestrates: dim reduction → model → calibration → ensemble → consensus.
Imports from sibling modules: calibrator, ensembler, consensus.
"""
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# These are imported by the template — NOT by this file directly
# from calibrator import calibrate
# from ensembler import train_ensemble
# from consensus import apply_consensus, consensus_stats


# ------------------------------------------------------------------------- #
# Module ⑧ sizing — de Prado CDF bet-size × causal inverse-vol overlay.       #
# ROUND 4 de-tune of R2 sizing (low-DoF, a-priori, NOT swept): raise floor and    #
# slow the trigger so we stay near-full-long on drift (recover CAGR) and de-lever  #
# only in sharp turbulence; CDF recentered on the per-ETF threshold (not 0.5) so   #
# low-tau drift assets stay long. Used IDENTICALLY here and in the infer template. #
# ------------------------------------------------------------------------- #
VOL_FAST = 10
VOL_SLOW = 60
VOL_FLOOR = 1.0  # ROUND 6: overlay OFF (g==1) -> sizing is LABEL-DRIVEN _cdf_bet(p,tau)


def _cdf_bet(p, thresh):
    """de Prado prob->size: gate at thresh, then 2*Phi(z)-1, z=(p-thresh)/sqrt(p(1-p)).
    CDF recentered on thresh (R4): b=0 at p=thresh, rising to 1 as p->1, so a low
    per-ETF thresh keeps drift assets long. Long-only, clipped to [0,1]."""
    if p <= thresh:
        return 0.0
    pp = min(max(p, 1e-6), 1 - 1e-6)
    z = (pp - thresh) / np.sqrt(pp * (1.0 - pp))
    b = 2.0 * 0.5 * (1.0 + math.erf(z / np.sqrt(2.0))) - 1.0  # 2*Phi(z)-1
    return float(min(1.0, max(0.0, b)))


def _invvol_mult(rbuf):
    """Causal inverse-vol overlay from a trailing buffer of bar log-returns.
    g = clip(slow_vol / fast_vol, VOL_FLOOR, 1.0): de-lever when short-term vol
    spikes above its slower baseline (where drawdowns cluster)."""
    m = len(rbuf)
    if m < VOL_FAST + 2:
        return 1.0
    fast = float(np.std(rbuf[-VOL_FAST:]))
    slow = float(np.std(rbuf[-min(m, VOL_SLOW):]))
    if fast <= 1e-9:
        return 1.0
    return float(min(1.0, max(VOL_FLOOR, slow / fast)))


def realistic_cstats(probs, lc_arr, ma_arr, log_rets, tc=0.0005, thresh=0.45):
    """Realistic backtest statistics with transaction costs.
    `thresh` gates entries; sizing = de Prado CDF bet * causal inverse-vol overlay
    (see _cdf_bet/_invvol_mult). It MUST match the infer template's rule.
    Returns: (calmar, trades, total_return, mdd, annual_return, drawdown_area)
    where drawdown_area DA = Σ_t (1 - E_t/peak_t) on the equity curve (lower=better).
    """
    n = min(len(probs) - 1, len(log_rets) - 1, len(lc_arr) - 1, len(ma_arr) - 1)
    if n < 2:
        return 0, 0, 0, 0, 0, 0.0
    positions = np.zeros(n + 1); last_pos = 0.0; trades = 0
    rbuf = []  # trailing bar log-returns (causal: only past returns)
    for i in range(n):
        g = _invvol_mult(rbuf)
        target = _cdf_bet(probs[i], thresh) * g
        if (last_pos == 0 and target > 0) or (last_pos > 0 and target == 0) or abs(target - last_pos) > 0.01:
            trades += 1
        positions[i] = target; last_pos = target
        # Append the return INTO the current bar AFTER deciding (parity with infer's
        # decide-then-append ordering; fixes the VAL inverse-vol off-by-one lookahead).
        if i - 1 >= 0:
            rbuf.append(float(log_rets[i - 1]))
    if trades < 2: return 0, trades, 0, 0, 0, 0.0
    strat_rets = positions[:-1] * log_rets[1:n + 1]
    for i in range(1, n):
        if abs(positions[i] - positions[i - 1]) > 0.01:  # match trade-count + infer rebalance band
            strat_rets[i] -= tc * abs(positions[i] - positions[i - 1])
    cum = np.cumsum(strat_rets); peak = np.maximum.accumulate(cum); dd = cum - peak
    mdd = abs(float(np.min(dd))) + 1e-9; ann = float(np.mean(strat_rets)) * 880
    da = float(np.sum(1.0 - np.exp(dd)))  # underwater area in fractional-equity terms
    cal = ann / mdd if mdd > 0.001 else 0
    return cal, trades, float(np.sum(strat_rets)), mdd, ann, da


def reduce_dims(X_train, X_val, X_test, method="correlation", n_components=20, y_train=None):
    """Module ④: Dimensionality reduction — variance + correlation + top-K cap, OR a
    NONLINEAR AUTOENCODER (Wang ⑥ first-public: linear vs non-linear dim-reduce), OR
    Wang's INFORMATION-GAIN selection (top-K by mutual-info with the TRAIN label, not
    variance — label-RELEVANT, fixes the corr-filter's label-agnostic crowding that
    made added features hurt). y_train (TRAIN labels) is required for infogain; the
    selection is computed on TRAIN only and the kept_idx is frozen + applied to val/test
    (same leak-safe contract as the correlation path)."""
    F = X_train.shape[1]; variances = np.var(X_train, axis=0)
    if method == "autoencoder":
        # Bottleneck autoencoder (sklearn MLP, no torch): fit X->X on TRAIN, then use the
        # bottleneck (first hidden layer, tanh) activations as the reduced features. A
        # NON-LINEAR embedding vs the linear correlation-SELECT. kept_idx is a placeholder
        # (no column subset); the bundle is skipped for AE cells (A/B uses prediction replay).
        try:
            from sklearn.neural_network import MLPRegressor
            L = int(max(4, min(n_components, F // 3)))
            ae = MLPRegressor(hidden_layer_sizes=(L,), activation="tanh", solver="adam",
                              alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                              random_state=42, early_stopping=False)
            ae.fit(X_train, X_train)                 # reconstruct — TRAIN only
            W0 = ae.coefs_[0]; b0 = ae.intercepts_[0]
            enc = lambda Z: np.tanh(Z @ W0 + b0)     # bottleneck encoder
            return (enc(X_train).astype(np.float32), enc(X_val).astype(np.float32),
                    enc(X_test).astype(np.float32), L, f"ae{L}", list(range(L)))
        except Exception:
            method = "correlation"                   # degrade gracefully to the linear path
    kept = np.ones(F, dtype=bool)
    if method == "variance": kept &= variances > 0.01
    elif method in ("correlation", "infogain", "mrmr"):
        kept &= variances > 1e-6; kept_idx = np.where(kept)[0]
        if len(kept_idx) > 1:
            corr = np.abs(np.corrcoef(X_train[:, kept_idx].T))
            upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
            to_remove = set()
            for i, j in zip(*np.where(upper & (corr > 0.90))):
                orig_i, orig_j = kept_idx[i], kept_idx[j]
                to_remove.add(orig_i if variances[orig_i] < variances[orig_j] else orig_j)
            for idx in to_remove: kept[idx] = False
    kept_idx = np.where(kept)[0]
    # Wang INFORMATION-GAIN top-K: select the n_components MOST label-relevant survivors
    # by mutual information with the TRAIN label (vs the variance top-K below). TRAIN-only
    # (X_train, y_train) -> kept_idx frozen for val/test = leak-safe. Falls through to the
    # variance cap if MI is unavailable or y_train is missing.
    if method == "infogain" and y_train is not None and len(kept_idx) > n_components:
        try:
            from sklearn.feature_selection import mutual_info_classif
            yt = np.asarray(y_train).astype(int)
            mi = mutual_info_classif(X_train[:, kept_idx], yt, random_state=42)
            top_k = kept_idx[np.argsort(mi)[-n_components:]]
            kept = np.zeros(F, dtype=bool); kept[top_k] = True; kept_idx = np.where(kept)[0]
        except Exception:
            pass
    # mRMR (min-Redundancy-Max-Relevance, Peng 2005): IG keeps the n_components most
    # label-relevant features but ignores that they may be redundant with each other; mRMR
    # greedily picks each next feature by relevance(MI w/ TRAIN label) MINUS redundancy
    # (mean |corr| with already-picked). TRAIN-only -> kept_idx frozen for val/test (same
    # leak-safe contract). A principled upgrade over infogain on structured names.
    if method == "mrmr" and y_train is not None and len(kept_idx) > n_components:
        try:
            from sklearn.feature_selection import mutual_info_classif
            yt = np.asarray(y_train).astype(int)
            rel = mutual_info_classif(X_train[:, kept_idx], yt, random_state=42)  # relevance (TRAIN)
            C = np.abs(np.corrcoef(X_train[:, kept_idx].T))                        # redundancy (TRAIN)
            C = np.atleast_2d(C)
            sel, cand = [], list(range(len(kept_idx)))                            # positions within kept_idx
            while len(sel) < n_components and cand:
                if not sel:
                    p = int(cand[int(np.argmax(rel[cand]))])
                else:
                    best_sc, p = -1e18, cand[0]
                    for f in cand:
                        sc = float(rel[f]) - float(np.mean([C[f, s] for s in sel]))
                        if sc > best_sc:
                            best_sc, p = sc, f
                sel.append(p); cand.remove(p)
            top_k = kept_idx[np.array(sel, dtype=int)]
            kept = np.zeros(F, dtype=bool); kept[top_k] = True; kept_idx = np.where(kept)[0]
        except Exception:
            pass
    if len(kept_idx) > n_components:
        top_k = kept_idx[np.argsort(variances[kept_idx])[-n_components:]]
        kept = np.zeros(F, dtype=bool); kept[top_k] = True; kept_idx = np.where(kept)[0]
    nk = len(kept_idx)
    # kept_idx (FINAL, after corr-prune AND top-K cap) is returned so the online
    # infer can reproduce the projection: ((feat_row - mean)/scale)[kept_idx].
    return (X_train[:, kept_idx].astype(np.float32), X_val[:, kept_idx].astype(np.float32),
            X_test[:, kept_idx].astype(np.float32), nk, f"{method}{nk}", kept_idx.tolist())


def _pred_list(probs, lc_arr, ma_arr, times):
    out = []
    for i in range(min(len(probs), len(times))):
        try: t = times[i]; ts_str = t.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(t, 'strftime') else str(t)[:19].replace(' ', 'T')
        except: ts_str = str(times[i])[:19].replace(' ', 'T')
        above = bool(lc_arr[i] > ma_arr[i]) if i < len(ma_arr) and i < len(lc_arr) else True
        out.append({"time": ts_str, "pred": float(probs[i]), "above_ma": above})
    return out


def train_and_evaluate(feats, y, tr_m, va_m, te_m, lc, lr, bar_ts, fv,
                        bar_type="vol", label_cfg=""):
    """Full Wang pipeline ④⑤⑥⑦⑧: dim reduce → model → calibrate → ensemble → consensus.

    Sweeps: dim methods, ensemble sizes, calibration methods, MA periods, inversion.
    Returns best config by realistic_cstats Calmar.

    DEAD CODE — DO NOT USE. This legacy path is LEAKY: it selects test bars by a future
    label (ex = fv & ly & te_m) and calibrates on non-embargoed VAL. The live footer
    (_run_cell) uses the embargoed, no-label-filter path instead. Guarded fail-loud so a
    future caller can never silently reintroduce the leak.
    """
    raise RuntimeError("trainer.train_and_evaluate is dead, LEAKY legacy code (selects OOS "
                       "bars by future label + non-embargoed calibrator). Use the footer "
                       "_run_cell path. Refusing to run.")
