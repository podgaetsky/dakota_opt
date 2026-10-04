"""Diagnostics, posterior predictive and corner plots for Dakota DREAM chains.

Only benchmark mode has an analytic reference; no credible interval claims if
mixing, likelihood assumptions or surrogate accuracy checks are inadequate.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mcmc_simulate import benchmark_model


def read_chains(run, names):
    files = sorted(run.glob("dakota_dream_chain*.txt"))
    if not files:
        raise ValueError("No DREAM chains; use --backend dream for multichain diagnostics")
    # DREAM writes: generation, log posterior, normalized variables (one file/chain).
    raw = [np.loadtxt(path, skiprows=1, ndmin=2) for path in files]
    if len(raw) < 3 or any(chain.shape[1] != len(names) + 2 for chain in raw):
        raise ValueError("DREAM chains missing or unexpected number of columns")
    size = min(chain.shape[0] for chain in raw)
    return np.stack([chain[:size, 2:] for chain in raw]), np.stack([chain[:size, 1] for chain in raw])


def analytic_linear_posterior(xs, ys, sigma):
    """Unbounded flat-prior Gaussian regression; check bounds before comparing."""
    design = np.column_stack((np.ones(len(xs)), xs))
    covariance = sigma ** 2 * np.linalg.inv(design.T @ design)
    mean = np.linalg.solve(design.T @ design, design.T @ ys)
    return mean, covariance


def analyze(run):
    import arviz as az
    import corner

    run = Path(run).resolve()
    settings = json.loads((run / "config.json").read_text())
    names = list(settings["parameters"])
    lower = np.array([settings["parameters"][name]["lower"] for name in names])
    span = np.array([settings["parameters"][name]["upper"] - lower[i]
                     for i, name in enumerate(names)])
    normalized, logp = read_chains(run, names)
    chains = lower + normalized * span
    burn = chains.shape[1] // 2  # conservative fixed 50% warm-up; not automatically valid
    retained = chains[:, burn:, :]
    flat = retained.reshape(-1, len(names))
    inference = az.from_dict(posterior={name: retained[:, :, i] for i, name in enumerate(names)})
    rhat = az.rhat(inference, var_names=names)
    ess = az.ess(inference, var_names=names, method="bulk")
    ess_tail = az.ess(inference, var_names=names, method="tail")
    mcse = az.mcse(inference, var_names=names, method="mean")
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    truth = [1.15, .4] if settings["mode"] == "benchmark" else None
    figure = corner.corner(flat, labels=names, truths=truth, show_titles=True,
                           title_fmt=".3g", quantiles=[.025, .5, .975],
                           levels=[.393, .865],  # Gaussian 1σ/2σ 2D enclosed mass
                           hist_kwargs={"density": True})
    figure.savefig(plots / "corner.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(len(names), 1, figsize=(10, max(3, 2.4 * len(names))),
                                sharex=True, layout="constrained", squeeze=False)
    for j, name in enumerate(names):
        for i, series in enumerate(chains[:, :, j]):
            axes[j, 0].plot(series, lw=.6, alpha=.7, label=f"chain {i + 1}")
        axes[j, 0].axvline(burn, color="black", ls="--", lw=1)
        axes[j, 0].set_ylabel(name)
    axes[0, 0].legend(ncol=4, fontsize=7)
    axes[-1, 0].set_xlabel("DREAM generation (dashed: discard boundary)")
    figure.savefig(plots / "trace.png", dpi=180)
    plt.close(figure)
    az.plot_rank(inference, var_names=names, kind="vlines", figsize=(10, max(3, 2.5 * len(names))))
    figure = plt.gcf()
    figure.savefig(plots / "rank.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    az.plot_autocorr(inference, var_names=names, max_lag=min(80, retained.shape[1] // 3),
                     figsize=(10, max(3, 2.6 * len(names))))
    figure = plt.gcf()
    figure.savefig(plots / "autocorrelation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    with (run / "reference.csv").open(newline="") as stream:
        observed = np.array([[float(r["x"]), float(r["y"])] for r in csv.DictReader(stream)])
    summary = {"method": "Dakota DREAM, native GP emulator of curve ordinates",
               "assumption": "independent Gaussian errors with known sigma and bounded uniform priors",
               "posterior_interpretation": "Provisional until validation.json checks surrogate and chain diagnostics",
               "warmup_discarded_per_chain": burn, "retained_draws": len(flat),
               "rhat": {n: float(rhat[n].values) for n in names},
               "ess_bulk": {n: float(ess[n].values) for n in names},
               "ess_tail": {n: float(ess_tail[n].values) for n in names},
               "mcse_mean": {n: float(mcse[n].values) for n in names},
               "mcse_mean_fraction_of_posterior_sd": {
                   n: float(mcse[n].values / np.std(flat[:, i], ddof=1)) for i, n in enumerate(names)},
               "chain_move_fraction": {n: [float(np.mean(np.diff(retained[c, :, i]) != 0))
                                            for c in range(retained.shape[0])]
                                       for i, n in enumerate(names)},
               "posterior": {n: {"median": float(np.median(flat[:, i])),
                                 "equal_tailed_95": list(map(float, np.quantile(flat[:, i], [.025, .975])))}
                             for i, n in enumerate(names)}}
    if settings["mode"] == "benchmark":
        expected, covariance = analytic_linear_posterior(observed[:, 0], observed[:, 1], settings["sigma"])
        distances = np.minimum((expected - lower) / np.sqrt(np.diag(covariance)),
                               (lower + span - expected) / np.sqrt(np.diag(covariance)))
        if np.min(distances) < 5:
            raise ValueError("Unbounded Gaussian reference invalid close to uniform-prior bounds")
        summary["analytic_gaussian_reference"] = {"mean": expected.tolist(),
                                                 "covariance": covariance.tolist(),
                                                 "boundary_distance_in_sd": distances.tolist()}
        summary["descriptive_marginal_ks_D"] = {
            name: float(stats.kstest(flat[:, i], stats.norm(loc=expected[i],
                          scale=np.sqrt(covariance[i, i])).cdf).statistic)
            for i, name in enumerate(names)}
        # The exact Gaussian benchmark is compared against a separate approximate
        # GP posterior. Correlated MCMC draws invalidate a naive KS p-value.
        predicted = np.array([benchmark_model(dict(zip(names, row)), observed[:, 0])
                              for row in flat[::max(1, len(flat) // 2000)]])
        figure, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.fill_between(observed[:, 0], *np.quantile(predicted, [.025, .975], axis=0),
                        color="steelblue", alpha=.3, label="95% latent curve interval")
        ax.plot(observed[:, 0], np.median(predicted, axis=0), label="Posterior median")
        ax.errorbar(observed[:, 0], observed[:, 1], yerr=settings["sigma"], fmt="ko",
                    ms=3, capsize=2, label="Observations ± 1σ")
        ax.set(xlabel="x", ylabel="curve value", title="Posterior latent-curve check")
        ax.legend()
        figure.savefig(plots / "posterior_predictive.png", dpi=180)
        plt.close(figure)
        figure, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4),
                                    layout="constrained", squeeze=False)
        for i, name in enumerate(names):
            ax = axes[0, i]
            ax.hist(flat[:, i], bins=35, density=True, color="steelblue", alpha=.5, label="DREAM")
            grid = np.linspace(expected[i] - 4 * np.sqrt(covariance[i, i]),
                               expected[i] + 4 * np.sqrt(covariance[i, i]), 150)
            ax.plot(grid, stats.norm.pdf(grid, expected[i], np.sqrt(covariance[i, i])),
                    c="crimson", lw=2, label="Analytic reference")
            ax.set(xlabel=name, ylabel="Density", title=f"Marginal KS D={summary['descriptive_marginal_ks_D'][name]:.3f}")
            ax.legend(fontsize=8)
        figure.savefig(plots / "benchmark_marginals.png", dpi=180)
        plt.close(figure)
    (run / "posterior_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    analyze(args.run)


if __name__ == "__main__":
    main()