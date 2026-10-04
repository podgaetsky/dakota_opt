"""Scientific diagnostic figures for saved generic arrays."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_residuals(data, output):
    x, r, pred = data["x"], data["residual"], data["predicted"]
    z = data["standardized"] if data["standardized"] is not None else r
    fig, axes = plt.subplots(5, 2, figsize=(11, 17), layout="constrained")
    ax = axes.ravel()
    ax[0].plot(x, r, "o-"); ax[0].set(title="Residual vs x", xlabel="x", ylabel="data − model")
    ax[1].plot(x, z, "o-"); ax[1].set(title="Whitened/standardized residual vs x (raw if no noise)", xlabel="x")
    ax[2].scatter(pred, r); ax[2].set(title="Residual vs fitted", xlabel="model")
    ax[3].scatter(pred, np.abs(r)); ax[3].set(title="Absolute residual vs fitted", xlabel="model")
    ax[4].hist(z, bins="auto"); ax[4].set(title="Residual histogram (whitened if noise)")
    if len(z) >= 3 and np.std(z) > 0:
        stats.probplot(z, dist="norm", plot=ax[5])
    ax[5].set_title("Gaussian Q–Q (conditional diagnostic)")
    centered = z - np.mean(z)
    d = centered @ centered
    lags = range(1, min(20, len(z) // 4) + 1)
    acf = [centered[:-lag] @ centered[lag:] / d if d > 0 else 0 for lag in lags]
    ax[6].stem(list(lags), acf); ax[6].set(title="Residual ACF", xlabel="lag")
    ax[7].plot(x, np.cumsum(r)); ax[7].set(title="Cumulative raw residual", xlabel="x")
    ax[8].plot(x, z, "o"); ax[8].axhline(3, ls="--", color="crimson"); ax[8].axhline(-3, ls="--", color="crimson")
    ax[8].set(title="Standardized outliers (±3; if noise supplied)", xlabel="x")
    ax[9].plot(pred, data["observed"], "o")
    lo, hi = min(pred.min(), data["observed"].min()), max(pred.max(), data["observed"].max())
    ax[9].plot([lo, hi], [lo, hi], "--", color="grey"); ax[9].set(title="Observed vs predicted (identity line)", xlabel="predicted", ylabel="observed")
    fig.savefig(output / "residual_diagnostics.png", dpi=155)
    plt.close(fig)


def plot_identifiability(result, output):
    singular = result["singular_values"]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    ax.semilogy(np.arange(1, len(singular) + 1), singular, "o-"); ax.set(title="Whitened Jacobian singular values", xlabel="index")
    bx.bar(range(result["parameter_count"]), result["sensitivities_column_norms"]); bx.set(title="Parameter sensitivities (normalized)")
    correlation = result.get("local_correlation")
    if correlation is not None:
        image = cx.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm"); fig.colorbar(image, ax=cx)
        cx.set_title("Conditional local parameter correlation")
    else:
        cx.text(.05, .5, "Correlation unavailable without full-rank Gaussian Fisher matrix", wrap=True)
        cx.axis("off")
    fig.savefig(output / "identifiability.png", dpi=170)
    plt.close(fig)


def plot_predictive(reference, curves, result, output):
    x, y = reference[:, 0], reference[:, 1]
    latent = np.asarray(result["latent_interval_95"])
    observed = result["observation_predictive_interval_95"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes[0, 0].fill_between(x, latent[0], latent[2], alpha=.3, label="latent 95%")
    axes[0, 0].plot(x, latent[1], label="latent median"); axes[0, 0].scatter(x, y, c="k", s=14, label="observed")
    axes[0, 0].set_title("Posterior latent curves (no measurement noise)"); axes[0, 0].legend()
    if observed is not None:
        pred = np.asarray(observed)
        axes[0, 1].fill_between(x, pred[0], pred[2], alpha=.3, label="observation predictive 95%")
        axes[0, 1].plot(x, pred[1]); axes[0, 1].scatter(x, y, c="k", s=14)
        axes[0, 1].legend()
    axes[0, 1].set_title("Observation predictive (noise added; no discrepancy)")
    differences = y[None, :] - curves
    axes[1, 0].fill_between(x, *np.quantile(differences, [.025, .975], axis=0), alpha=.3)
    axes[1, 0].axhline(0, color="k", ls="--"); axes[1, 0].set_title("Posterior raw residual band")
    axes[1, 1].hist(result["rms_distribution"], bins=20)
    axes[1, 1].set_title("Posterior model-data RMS distribution")
    fig.savefig(output / "posterior_predictive_diagnostics.png", dpi=170)
    plt.close(fig)
