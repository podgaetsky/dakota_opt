"""Opt-in physical-parameter profile by reoptimizing other parameters.

Supply a callable objective of a parameter vector; this is NOT a profile of
fixed-other-parameter slices. The caller controls simulator resource budget.
"""

import numpy as np
from scipy.optimize import minimize


def profile_objective(objective, optimum, bounds, grid_size=15, starts=3, seed=123):
    theta = np.asarray(optimum, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(theta), 2) or np.any(bounds[:, 1] <= bounds[:, 0]) or grid_size < 3 or starts < 1:
        raise ValueError("Provide strict bounds, >=3 grid points and >=1 start")
    rng = np.random.default_rng(seed)
    profiles = []
    for j in range(len(theta)):
        other = [i for i in range(len(theta)) if i != j]
        trace = []
        for fixed in np.linspace(*bounds[j], grid_size):
            def constrained(z):
                candidate = theta.copy()
                candidate[j] = fixed
                candidate[other] = z
                return float(objective(candidate))
            trials = [constrained([])] if not other else [minimize(constrained, start, bounds=bounds[other], method="Powell").fun
                for start in [theta[other], *(rng.uniform(bounds[other, 0], bounds[other, 1]) for _ in range(starts - 1))]]
            trace.append({"fixed_parameter": float(fixed), "best_profile_objective": float(min(trials))})
        profiles.append(trace)
    return {"profiles": profiles, "objective": "supplied by caller (RMS or -2 log likelihood; label before interpreting)",
            "caveat": "Numerical reoptimization may miss other modes; use multiple starts, compare budget and inspect bounds."}


def save_profile_figure(result, output, parameter_names, objective_label):
    """Plot reoptimized profiles; label whether objective is RMS or -2 log L."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if objective_label not in ("RMS", "-2 log likelihood"):
        raise ValueError("Label the actual profiled objective as RMS or -2 log likelihood")
    profiles = result["profiles"]
    if len(parameter_names) != len(profiles):
        raise ValueError("Need one name per profiled parameter")
    figure, axes = plt.subplots(len(profiles), 1, figsize=(7, max(3, 2.7 * len(profiles))),
                                squeeze=False, layout="constrained")
    for axis, name, trace in zip(axes[:, 0], parameter_names, profiles):
        axis.plot([row["fixed_parameter"] for row in trace],
                  [row["best_profile_objective"] for row in trace], "o-")
        axis.set(xlabel=name, ylabel=f"Profile {objective_label}")
    figure.savefig(output, dpi=180)
    plt.close(figure)
