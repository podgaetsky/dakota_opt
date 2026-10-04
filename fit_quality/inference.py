"""Optional local identifiability and posterior prediction from generic arrays."""

import numpy as np
from scipy.linalg import solve_triangular

from .core import aligned, gaussian_noise


def finite_difference_jacobian(model, theta, lower, upper, relative_step=1e-4):
    """Central differences in normalized coordinates, one-sided at bounds.

    model accepts a physical-parameter vector and returns ordinates. Calls to
    an expensive simulator are the caller's responsibility; this is opt-in.
    """
    theta, low, high = (np.asarray(v, dtype=float) for v in (theta, lower, upper))
    if theta.ndim != 1 or low.shape != theta.shape or high.shape != theta.shape or np.any(high <= low):
        raise ValueError("Expected aligned physical parameters and strict bounds")
    if np.any(theta < low) or np.any(theta > high) or not 0 < relative_step < .1:
        raise ValueError("Parameters must be inside bounds and step must be small positive")
    base = np.asarray(model(theta), dtype=float)
    columns = []
    for j in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[j] = min(high[j], theta[j] + relative_step * (high[j] - low[j]))
        down[j] = max(low[j], theta[j] - relative_step * (high[j] - low[j]))
        column = (np.asarray(model(up)) - np.asarray(model(down))) / ((up[j] - down[j]) / (high[j] - low[j]))
        if column.shape != base.shape or not np.isfinite(column).all():
            raise ValueError("Model returned inconsistent/nonfinite ordinates")
        columns.append(column)
    return np.column_stack(columns)


def identifiability(jacobian, noise=None, bounds=None):
    """Jacobian columns must be derivatives w.r.t. normalized parameters."""
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or not np.isfinite(jac).all() or not jac.size:
        raise ValueError("Jacobian must be a finite N by K matrix")
    n, k = jac.shape
    if bounds is not None:
        b = np.asarray(bounds, dtype=float)
        if b.shape != (k, 2) or np.any(b[:, 1] <= b[:, 0]):
            raise ValueError("Bounds must be K by 2 with positive ranges")
        jac = jac * (b[:, 1] - b[:, 0])[None, :]
    kind, chol, _ = gaussian_noise(noise, n)
    whitened = solve_triangular(chol, jac, lower=True) if chol is not None else jac
    u, singular, vh = np.linalg.svd(whitened, full_matrices=True)
    threshold = max(whitened.shape) * np.finfo(float).eps * singular[0] if singular.size else 0
    rank = int(np.sum(singular > threshold))
    condition = float(singular[0] / singular[-1]) if rank == k else None
    result = {"parameter_count": k, "numerical_rank": rank, "condition_number": condition,
              "singular_values": singular.tolist(), "right_singular_vectors": vh.tolist(),
              "weak_combination_normalized": vh[-1].tolist(),
              "sensitivities_column_norms": np.linalg.norm(whitened, axis=0).tolist(),
              "noise_whitening": kind or "none; geometry only", "status": "WARN" if rank < k or (condition and condition > 1e6) else "CONDITIONAL",
              "caveat": "Local linearization at provided parameters; optimizer variability alone cannot establish intrinsic non-identifiability."}
    if kind:
        fisher = whitened.T @ whitened
        result["fisher_information"] = fisher.tolist()
        if rank == k and n >= k:
            covariance = np.linalg.inv(fisher)
            sd = np.sqrt(np.diag(covariance))
            result["local_covariance_normalized"] = covariance.tolist()
            result["local_correlation"] = (covariance / np.outer(sd, sd)).tolist()
            result["caveat"] += " Fisher covariance assumes valid Gaussian noise, adequate local linearity and no active bounds."
    return result


def posterior_predictive(reference, curves, noise=None, seed=217):
    """Ndraw by N array of *exact model curves*, not GP variances or chain samples."""
    ref = np.asarray(reference, dtype=float)
    curves = np.asarray(curves, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 2 or curves.ndim != 2 or curves.shape[1] != len(ref) or len(curves) < 2 or not np.isfinite(curves).all():
        raise ValueError("Supply >=2 finite posterior model curves with columns matching reference x")
    kind, chol, _ = gaussian_noise(noise, len(ref))
    rng = np.random.default_rng(seed)
    simulated = curves + rng.standard_normal(curves.shape) @ chol.T if chol is not None else None
    residual = ref[:, 1][None, :] - curves
    chi2 = np.sum((solve_triangular(chol, residual.T, lower=True).T) ** 2, axis=1) if chol is not None else None
    latent = np.quantile(curves, [.025, .5, .975], axis=0)
    obs = np.quantile(simulated, [.025, .5, .975], axis=0) if simulated is not None else None
    result = {"latent_interval_95": latent.tolist(), "observation_predictive_interval_95": obs.tolist() if obs is not None else None,
              "observed_fraction_in_predictive_interval": float(np.mean((ref[:, 1] >= obs[0]) & (ref[:, 1] <= obs[2]))) if obs is not None else None,
              "rms_distribution": np.sqrt(np.mean(residual ** 2, axis=1)).tolist(),
              "mae_distribution": np.mean(np.abs(residual), axis=1).tolist(),
              "max_error_distribution": np.max(np.abs(residual), axis=1).tolist(),
              "chi2_distribution": chi2.tolist() if chi2 is not None else None,
              "noise_model": kind,
              "caveat": "Conditional on supplied posterior model curves and measurement noise; model discrepancy and GP uncertainty are not included. Posterior sample quality must be checked separately."}
    return result, latent, obs
