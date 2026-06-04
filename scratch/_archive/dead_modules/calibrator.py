"""Module ⑥: Probability Calibration (Wang workflow).
Calibrate XGBoost predicted probabilities to reduce overconfidence.
CRITICAL for ensemble quality — uncalibrated probs cluster at extremes.

Methods: IsotonicRegression (non-parametric, proven v362),
         Platt scaling (sigmoid), and no calibration (baseline).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calibrate(probs_train, y_train, probs_val, probs_test, method="isotonic"):
    """Calibrate predicted probabilities.

    Args:
        probs_train: (N,) model predictions on training data
        y_train: (N,) true labels on training data
        probs_val: (M,) predictions on validation data
        probs_test: (K,) predictions on test data
        method: "isotonic", "platt", or "none"

    Returns:
        cal_val, cal_test: calibrated probabilities
        method_name: string describing calibration used
    """
    if method == "none":
        return probs_val, probs_test, "nocal"

    if method == "isotonic":
        try:
            cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            cal.fit(probs_train, y_train)
            return cal.transform(probs_val), cal.transform(probs_test), "iso"
        except Exception:
            return probs_val, probs_test, "iso_failed"

    if method == "platt":
        try:
            # Platt scaling: fit logistic regression on raw scores
            lr = LogisticRegression(C=1.0, fit_intercept=True)
            lr.fit(probs_train.reshape(-1, 1), y_train)
            pv = lr.predict_proba(probs_val.reshape(-1, 1))[:, 1]
            pe = lr.predict_proba(probs_test.reshape(-1, 1))[:, 1]
            return pv, pe, "platt"
        except Exception:
            return probs_val, probs_test, "platt_failed"

    return probs_val, probs_test, f"unknown_{method}"
