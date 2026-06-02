"""Module ⑦: Multi-Seed Ensemble (Wang workflow).
Train XGBoost with multiple seeds, average probabilities.
Separates signal from noise — proven: GLD single-seed 3.18 → ENS5 3.54.

Method: N seeds (default 5), average or median probabilities.
"""
import numpy as np
import xgboost as xgb


SEEDS = [42, 43, 45, 46, 47, 48]  # Proven seed set from Wang experiments


def train_ensemble(X_train, y_train, X_val, y_val, X_test,
                   n_seeds=5, xgb_params=None):
    """Train N XGBoost models with different seeds, return ensemble predictions.

    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test: test data (no labels needed)
        n_seeds: number of ensemble members (max 6)
        xgb_params: dict of XGBoost parameters (uses defaults if None)

    Returns:
        pv_ens, pe_ens: ensemble-averaged validation and test probabilities
        pv_all, pe_all: list of per-seed probabilities
        seeds_used: list of seeds actually used
    """
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            reg_alpha=1.0, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_jobs=1, early_stopping_rounds=30, base_score=0.5
        )

    seeds_used = SEEDS[:n_seeds]
    pv_all = []
    pe_all = []

    for seed in seeds_used:
        params = dict(xgb_params)
        params["random_state"] = seed
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pv_all.append(m.predict_proba(X_val)[:, 1])
        pe_all.append(m.predict_proba(X_test)[:, 1])

    # Average ensemble
    pv_ens = np.mean(pv_all, axis=0)
    pe_ens = np.mean(pe_all, axis=0)

    return pv_ens, pe_ens, pv_all, pe_all, seeds_used


def ensemble_aggregate(probs_list, method="mean"):
    """Aggregate multiple probability vectors into one.

    Args:
        probs_list: list of (N,) probability arrays
        method: "mean", "median", "min", "max"

    Returns:
        aggregated: (N,) array
    """
    probs = np.array(probs_list)
    if method == "mean":
        return np.mean(probs, axis=0)
    elif method == "median":
        return np.median(probs, axis=0)
    elif method == "min":
        return np.min(probs, axis=0)
    elif method == "max":
        return np.max(probs, axis=0)
    return np.mean(probs, axis=0)
