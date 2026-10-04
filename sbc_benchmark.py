"""Independent simulation-based calibration (SBC) for exact linear benchmark.

Each replicate draws a NEW theta from the same bounded uniform prior, observes
fresh Gaussian noise, then evaluates the exact truncated-normal posterior CDF at
the generating theta. Under a calibrated likelihood these PIT values are iid
Uniform(0,1). This tests the mathematical benchmark/likelihood, NOT the GP or
DREAM sampler (those are checked separately against the analytic posterior).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, truncnorm

from mcmc_analyze import analytic_linear_posterior


def run_sbc(repetitions=500, sigma=.05, seed=2748):
    if repetitions < 20 or sigma <= 0:
        raise ValueError("Need at least 20 independent datasets and positive sigma")
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1, 1, 9)  # orthogonal columns => independent posterior marginals
    low, high = np.array([0., -1.]), np.array([2., 1.])
    values = np.empty((repetitions, 2))
    for i in range(repetitions):
        actual = rng.uniform(low, high)
        observed = actual[0] + actual[1] * xs + rng.normal(0, sigma, len(xs))
        mean, covariance = analytic_linear_posterior(xs, observed, sigma)
        sd = np.sqrt(np.diag(covariance))
        values[i] = truncnorm.cdf(actual, (low - mean) / sd, (high - mean) / sd,
                                  loc=mean, scale=sd)
    result = {name: {"D": float(test.statistic), "p_value": float(test.pvalue)}
              for j, name in enumerate(("a", "b"))
              for test in [kstest(values[:, j], "uniform")]}
    return values, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/sbc_benchmark"))
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=.05)
    args = parser.parse_args()
    values, tests = run_sbc(args.repetitions, args.sigma)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "pit.csv", values, delimiter=",", header="a,b", comments="")
    (args.output / "ks.json").write_text(json.dumps(tests, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for j, name in enumerate(("a", "b")):
        axes[j].hist(values[:, j], bins=np.linspace(0, 1, 11), density=True,
                     edgecolor="white", color="steelblue")
        axes[j].axhline(1, c="crimson", ls="--", label="Uniform density")
        axes[j].set(xlabel=f"P({name} < true {name} | independent observations)",
                    ylabel="Density", title=f"SBC PIT: KS D={tests[name]['D']:.3f}, p={tests[name]['p_value']:.3f}")
        axes[j].legend(fontsize=7)
    fig.savefig(args.output / "sbc_pit.png", dpi=180)
    plt.close(fig)
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()