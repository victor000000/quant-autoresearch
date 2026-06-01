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
# A-PRIORI params (NOT swept): VOL_FAST=20, VOL_SLOW=100, VOL_FLOOR=0.25,     #
# leverage cap 1.0. Used IDENTICALLY by realistic_cstats (synth/val) and the  #
# standalone infer template (real OOS) so train/infer stay consistent.        #
# ------------------------------------------------------------------------- #
VOL_FAST = 20
VOL_SLOW = 100
VOL_FLOOR = 0.25


def _cdf_bet(p, thresh):
    """de Prado prob->size: gate at thresh, then 2*Phi(z)-1, z=(p-.5)/sqrt(p(1-p)).
    Long-only, clipped to [0,1]. Gentle slope => partial sizes, not saturation."""
    if p <= thresh:
        return 0.0
    pp = min(max(p, 1e-6), 1 - 1e-6)
    z = (pp - 0.5) / np.sqrt(pp * (1.0 - pp))
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
        rbuf.append(float(log_rets[i + 1]) if i + 1 < len(log_rets) else 0.0)
    if trades < 2: return 0, trades, 0, 0, 0, 0.0
    strat_rets = positions[:-1] * log_rets[1:n + 1]
    for i in range(1, n):
        if abs(positions[i] - positions[i - 1]) > 0.001:
            strat_rets[i] -= tc * abs(positions[i] - positions[i - 1])
    cum = np.cumsum(strat_rets); peak = np.maximum.accumulate(cum); dd = cum - peak
    mdd = abs(float(np.min(dd))) + 1e-9; ann = float(np.mean(strat_rets)) * 880
    da = float(np.sum(1.0 - np.exp(dd)))  # underwater area in fractional-equity terms
    cal = ann / mdd if mdd > 0.001 else 0
    return cal, trades, float(np.sum(strat_rets)), mdd, ann, da


def reduce_dims(X_train, X_val, X_test, method="correlation", n_components=20):
    """Module ④: Dimensionality reduction — variance + correlation + top-K cap."""
    F = X_train.shape[1]; variances = np.var(X_train, axis=0)
    kept = np.ones(F, dtype=bool)
    if method == "variance": kept &= variances > 0.01
    elif method == "correlation":
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
    if len(kept_idx) > n_components:
        top_k = kept_idx[np.argsort(variances[kept_idx])[-n_components:]]
        kept = np.zeros(F, dtype=bool); kept[top_k] = True; kept_idx = np.where(kept)[0]
    nk = len(kept_idx)
    return (X_train[:, kept_idx].astype(np.float32), X_val[:, kept_idx].astype(np.float32),
            X_test[:, kept_idx].astype(np.float32), nk, f"{method}{nk}")


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
    """
    N = len(lc); ly = y >= 0
    tx = fv & ly & tr_m; vx = fv & ly & va_m; ex = fv & ly & te_m
    if tx.sum() < 200 or vx.sum() < 30 or ex.sum() < 30:
        return 0, "insufficient_data", None, None, 0, 0, 0

    best_real_cal = -999; best_cfg = ""; best_val_preds = None; best_test_preds = None
    best_train_auc = 0; best_val_auc = 0; best_trades = 0

    # Phase 1 sweep: dim reduction + single model + calibration
    # (Ensemble and consensus added later once base signal is strong)
    for dim_method in ["correlation"]:  # correlation filter is most effective
        for n_comp in [20]:
            sc = StandardScaler()
            Xt_raw = sc.fit_transform(feats[tx]); Xv_raw = sc.transform(feats[vx]); Xe_raw = sc.transform(feats[ex])
            Xt, Xv, Xe, nk, dim_name = reduce_dims(Xt_raw, Xv_raw, Xe_raw, method=dim_method, n_components=n_comp)

            # Ensemble sweep: single (n=1) vs multi-seed (n=5)
            for n_seeds in [1, 5]:  # Both options — sweep picks best per ETF
                seeds = [42, 43, 45, 46, 47][:n_seeds]
                pv_all, pe_all = [], []
                # Compute class weight: balance minority (label=1) vs majority (label=0)
                n_pos = max(1, int(y[tx].sum())); n_neg = max(1, int((1-y[tx]).sum()))
                scale_w = n_neg / n_pos  # >1 means minority class gets more weight
                for seed in seeds:
                    m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                        reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
                        scale_pos_weight=scale_w,
                        objective="binary:logistic", eval_metric="auc", tree_method="hist",
                        random_state=seed, n_jobs=1, early_stopping_rounds=30, base_score=0.5)
                    m.fit(Xt, y[tx], eval_set=[(Xv, y[vx])], verbose=False)
                    pv_all.append(m.predict_proba(Xv)[:, 1])
                    pe_all.append(m.predict_proba(Xe)[:, 1])
                pv_raw = np.mean(pv_all, axis=0); pe_raw = np.mean(pe_all, axis=0)

                try:
                    train_auc = roc_auc_score(y[tx], m.predict_proba(Xt)[:, 1])
                    val_auc = roc_auc_score(y[vx], pv_raw)
                except ValueError: train_auc = 0.5; val_auc = 0.5

                # Isotonic calibration
                try:
                    from sklearn.isotonic import IsotonicRegression
                    cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                    cal.fit(np.clip(pv_raw, 1e-6, 1-1e-6), y[vx])
                    pv_cal = cal.transform(np.clip(pv_raw, 1e-6, 1-1e-6))
                    pe_cal = cal.transform(np.clip(pe_raw, 1e-6, 1-1e-6))
                except Exception: pv_cal, pe_cal = pv_raw, pe_raw

                # Consensus filter (Module ⑧): only keep high-agreement predictions
                cons_suf = ""
                if n_seeds >= 3:
                    min_pv = np.min(np.array(pv_all), axis=0)
                    avg_pv = np.mean(np.array(pv_all), axis=0)
                    cons_mask = (min_pv > 0.5) & (avg_pv > 0.55)
                    pv_cal = np.where(cons_mask, pv_cal, 0.0)
                    # Same for test
                    min_pe = np.min(np.array(pe_all), axis=0)
                    avg_pe = np.mean(np.array(pe_all), axis=0)
                    cons_mask_e = (min_pe > 0.5) & (avg_pe > 0.55)
                    pe_cal = np.where(cons_mask_e, pe_cal, 0.0)
                    cons_suf = "_cons"

                ens_suf = f"_ens{n_seeds}" if n_seeds > 1 else ""

                for ma_period in [50, 100, 0]:
                    if ma_period == 0: ma = np.zeros_like(lc); suf = "_noma"
                    else: ma = pd.Series(lc).rolling(ma_period, min_periods=ma_period).mean().to_numpy(); suf = f"_ma{ma_period}"
                    vi = np.where(vx)[0]; ei = np.where(ex)[0]

                    for inv, vp, ep in [(False, pv_cal, pe_cal), (True, 1-pv_cal, 1-pe_cal)]:
                        inv_suf = "_inv" if inv else ""
                        rc, nt, _, _, _, _ = realistic_cstats(vp[:-1], lc[vi][:-1], ma[vi][:-1], lr[vi][1:])
                        if nt >= 2 and rc > best_real_cal:
                            best_real_cal = rc; best_trades = nt
                            best_cfg = f"{bar_type}_{label_cfg}_{dim_name}_iso{ens_suf}{suf}{inv_suf}"
                        best_train_auc = train_auc; best_val_auc = val_auc
                        best_val_preds = _pred_list(vp[:-1], lc[vi][:-1], ma[vi][:-1], bar_ts[vi][:-1])
                        best_test_preds = _pred_list(ep[:-1], lc[ei][:-1], ma[ei][:-1], bar_ts[ei][:-1])

    return best_real_cal, best_cfg, best_val_preds, best_test_preds, best_train_auc, best_val_auc, best_trades
