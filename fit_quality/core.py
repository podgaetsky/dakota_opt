"""Pure data-level metrics and Gaussian likelihoods; no Dakota dependencies."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution


def read_curve(path):
    """Accept strictly increasing finite x,y CSV; do not interpolate silently."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not {"x", "y"}.issubset(row) for row in rows):
        raise ValueError(f"Expected nonempty x,y CSV: {path}")
    data = np.array([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Curve x must be strictly increasing and x,y finite")
    return data


def read_noise(path):
    """Noise JSON: {sigma}, {sigma_i}, or {covariance}; no implicit noise."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Noise model must be a JSON object")
    return data


def aligned(reference, prediction):
    ref, pred = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if ref.ndim != 2 or pred.shape != ref.shape or ref.shape[1] != 2 or not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError("Reference and prediction must contain aligned finite x,y pairs")
    if np.any(np.diff(ref[:, 0]) <= 0) or not np.allclose(ref[:, 0], pred[:, 0], atol=1e-10, rtol=1e-8):
        raise ValueError("Curve grids differ: resampling requires a documented, explicit choice")
    return ref, pred


def gaussian_noise(noise, n):
    """Return (name, Cholesky factor, pointwise SD). Reject ambiguous models."""
    if noise is None:
        return None, None, None
    choices = [key for key in ("sigma", "sigma_i", "covariance") if key in noise]
    if len(choices) != 1 or set(noise) != set(choices):
        raise ValueError("Choose exactly one of sigma, sigma_i, covariance")
    if choices[0] == "sigma":
        sd = np.full(n, float(noise["sigma"]))
        name = "iid Gaussian"
    elif choices[0] == "sigma_i":
        sd = np.asarray(noise["sigma_i"], dtype=float)
        name = "heteroscedastic Gaussian"
    else:
        cov = np.asarray(noise["covariance"], dtype=float)
        if cov.shape != (n, n) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Covariance must be finite, symmetric and N by N")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as error:
            raise ValueError("Covariance must be positive definite") from error
        return "correlated Gaussian", chol, np.sqrt(np.diag(cov))
    if sd.shape != (n,) or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("Noise SD must be finite, positive and have one entry per ordinate")
    return name, np.diag(sd), sd


def analyze_curve(reference, prediction, *, noise=None, parameter_count=None, weights=None,
                  chi2_test_justified=False, nrmse_scale=None):
    """Return (metrics, diagnostics_input). No formal p-value without explicit eligibility.

    parameter_count is the number of parameters *fitted on these data*, not
    merely the number listed in a model. Use None when unknown.
    """
    ref, pred = aligned(reference, prediction)
    x, y, fitted = ref[:, 0], ref[:, 1], pred[:, 1]
    residual = y - fitted
    abs_error = np.abs(residual)
    n = len(x)
    if parameter_count is not None and (type(parameter_count) is not int or parameter_count < 0):
        raise ValueError("parameter_count must be a nonnegative integer or omitted")
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(sse / n))
    scale = float(np.ptp(y) if nrmse_scale is None else nrmse_scale)
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weights must be a nonnegative finite N-vector with positive mass")
    integrated = (float(np.sqrt(np.trapezoid(w * residual ** 2, x) / np.trapezoid(w, x)))
                  if n > 1 and np.trapezoid(w, x) > 0 else None)
    metrics = {"n": n, "sse": sse, "rms": rmse, "rmse": rmse,
               "mae": float(np.mean(abs_error)), "median_absolute_error": float(np.median(abs_error)),
               "max_absolute_error": float(np.max(abs_error)), "absolute_error_quantiles":
               {str(q): float(np.quantile(abs_error, q / 100)) for q in (90, 95, 99)},
               "mean_signed_error": float(np.mean(residual)),
               "normalized_rmse": rmse / scale if np.isfinite(scale) and scale > 0 else None,
               "normalized_rmse_scale": "target range" if nrmse_scale is None else "user supplied",
               "relative_l2_error": float(np.linalg.norm(residual) / np.linalg.norm(y)) if np.linalg.norm(y) > 0 else None,
               "x_integrated_rmse": integrated, "x_integral_weights": "user supplied" if weights is not None else "uniform in x",
               "pointwise_weighted_rmse": float(np.sqrt(np.dot(w, residual ** 2) / np.sum(w))),
               "parameter_count": parameter_count}
    # Explicit polynomial SD(x), SD(y_pred) or SD(y_obs) is evaluated on the
    # current curve; no automatic noise fitting from these residuals.
    if noise is not None and "sigma_function" in noise:
        if set(noise) != {"sigma_function"}:
            raise ValueError("sigma_function cannot be combined with another noise specification")
        function = noise["sigma_function"]
        if function.get("variable") not in ("x", "predicted", "observed"):
            raise ValueError("sigma_function.variable must be x, predicted, or observed")
        coordinate = {"x": x, "predicted": fitted, "observed": y}[function["variable"]]
        coefficients = np.asarray(function["coefficients"], dtype=float)
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError("sigma_function.coefficients must be finite polynomial coefficients")
        noise = {"sigma_i": np.polyval(coefficients, coordinate).tolist()}
        metrics["sigma_function_applied"] = function
    model, chol, sd = gaussian_noise(noise, n)
    metrics["likelihood"] = model or "not supplied"
    metrics["chi2_test_eligibility"] = bool(chi2_test_justified)
    whitened = None
    if model:
        whitened = solve_triangular(chol, residual, lower=True, check_finite=True)
        statistic = float(whitened @ whitened)
        logdet = 2 * float(np.log(np.diag(chol)).sum())
        loglike = -.5 * (statistic + n * np.log(2 * np.pi) + logdet)
        dof = n - parameter_count if parameter_count is not None else None
        metrics.update({"chi2": statistic, "degrees_of_freedom": dof if dof is not None and dof > 0 else None,
                        "reduced_chi2": statistic / dof if dof is not None and dof > 0 else None,
                        "chi2_p_value": float(chi2_distribution.sf(statistic, dof))
                        if chi2_test_justified and dof is not None and dof > 0 else None,
                        "log_likelihood": float(loglike),
                        "noise_weighted_chi2_per_ordinate": statistic / n,
                        "aic": 2 * parameter_count - 2 * loglike if parameter_count is not None else None,
                        "aicc": (2 * parameter_count - 2 * loglike + 2 * parameter_count * (parameter_count + 1) /
                                 (n - parameter_count - 1)) if parameter_count is not None and n > parameter_count + 1 else None,
                        "bic": (parameter_count * np.log(n) - 2 * loglike) if parameter_count is not None else None})
    else:
        metrics.update({key: None for key in ("chi2", "degrees_of_freedom", "reduced_chi2", "chi2_p_value",
                                              "log_likelihood", "noise_weighted_chi2_per_ordinate", "aic", "aicc", "bic")})
    return metrics, {"x": x, "observed": y, "predicted": fitted, "residual": residual,
                     "standardized": whitened, "sigma": sd}
