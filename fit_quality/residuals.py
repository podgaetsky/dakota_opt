"""Ordered-curve residual diagnostics: evidence of structure, never proof of iid noise."""

import numpy as np
from scipy import stats


def _association(x, y):
    if len(x) < 4 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return {"spearman_rho": None, "p_value": None}
    test = stats.spearmanr(x, y)
    return {"spearman_rho": float(test.statistic), "p_value": float(test.pvalue)}


def diagnose(data):
    x, residual = data["x"], data["residual"]
    pred, obs = data["predicted"], data["observed"]
    n = len(residual)
    # Whiten first when full covariance is provided; raw residual trends still
    # matter for physical discrepancy and are reported separately.
    z = data["standardized"] if data["standardized"] is not None else residual
    centered = z - np.mean(z)
    denom = float(centered @ centered)
    lags = min(20, n // 4)
    acf = [float(centered[:-i] @ centered[i:] / denom) if denom > 0 else 0.0
           for i in range(1, lags + 1)]
    lag1 = acf[0] if acf else None
    lb = float(n * (n + 2) * sum(v * v / (n - j) for j, v in enumerate(acf, start=1))) if acf else None
    nz = centered[centered != 0]
    signs = nz > 0
    npos, nneg = int(signs.sum()), int((~signs).sum())
    if npos > 0 and nneg > 0 and len(signs) > 2:
        runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
        expected = 1 + 2 * npos * nneg / (npos + nneg)
        variance = 2 * npos * nneg * (2 * npos * nneg - npos - nneg) / ((npos + nneg) ** 2 * (npos + nneg - 1))
        runs_p = float(2 * stats.norm.sf(abs((runs - expected) / np.sqrt(variance)))) if variance > 0 else None
    else:
        runs, runs_p = None, None
    cal = None
    if n >= 4 and np.ptp(pred) > 0:
        slope, intercept, correlation, p_value, se = stats.linregress(pred, obs)
        cal = {"intercept": float(intercept), "slope": float(slope), "pearson_r": float(correlation),
               "r_squared": float(correlation ** 2), "slope_standard_error": float(se),
             "slope_95_interval": None,
             "caveat": "Secondary diagnostic only. Regression interval withheld: independence, model form and homoscedasticity are unverified; R² is not physical agreement."}
    structure = ((lag1 is not None and abs(lag1) > 1.96 / np.sqrt(n)) or
                 (lb is not None and stats.chi2.sf(lb, len(acf)) < .05) or
                 (runs_p is not None and runs_p < .05))
    corr_length = next((float(x[j] - x[0]) for j, v in enumerate(acf, 1) if abs(v) < np.exp(-1)), None)
    standard = data["standardized"]
    return {"raw_residual_mean": float(np.mean(residual)), "raw_residual_variance": float(np.var(residual, ddof=1)) if n > 1 else None,
            "skewness": float(stats.skew(z)) if n >= 3 and np.std(z) > 0 else None,
            "excess_kurtosis": float(stats.kurtosis(z)) if n >= 4 and np.std(z) > 0 else None,
            "lag1_autocorrelation": lag1, "acf_lags_1_to_n": acf, "ljung_box_q": lb,
            "ljung_box_p_value_exploratory": float(stats.chi2.sf(lb, len(acf))) if lb is not None else None,
            "runs_count": runs, "runs_p_value_exploratory": runs_p,
            "correlation_length_x_approx": corr_length, "independence_status": "WARN" if structure else "UNKNOWN",
            "independence_note": "Independent-noise assumption is questionable. Use a covariance-aware likelihood."
            if structure else "No strong structure detected; independence is not established by these tests.",
            "diagnosed_series": "covariance-whitened" if standard is not None else "raw residual",
            "standardized_outliers_abs_gt_3": int(np.sum(np.abs(standard) > 3)) if standard is not None else None,
            "variance_trends": {key: _association(values, np.abs(residual)) for key, values in
                                (("x", x), ("fitted", pred), ("observed", obs))},
            "calibration": cal,
            "caveat": "Exploratory ordered-residual tests share data and assumptions; no automatic outlier removal or model acceptance."}
