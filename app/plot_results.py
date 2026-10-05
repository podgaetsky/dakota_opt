"""Plot empirical convergence and fitted curves; optional independently refit GP diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from driver import curve, rms


def read_curve(path):
    return np.asarray(curve(path), dtype=float)


def fit_diagnostic_gp(xn, y):
    """Refit an independent deterministic GP for visualization, not Dakota's GP."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    # A free WhiteKernel can misclassify deterministic variation as noise and
    # render flat, uninformative slices. alpha is a numerical nugget only.
    kernel = ConstantKernel(1., (1e-2, 1e2)) * Matern(
        length_scale=np.full(xn.shape[1], .4), length_scale_bounds=(.03, 5.), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                  normalize_y=True, n_restarts_optimizer=2, random_state=42)
    return gp.fit(xn, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    meta = json.loads((run / "config.json").read_text())
    with (run / "evaluations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = list(meta["parameters"])
    good = [r for r in rows if r["status"] == "ok"]
    if not good:
        raise SystemExit("No successful evaluations")
    # CSV is append-locked at completion; include failures on the x-axis so
    # "function evaluation" does not silently mean "successful evaluation".
    X = np.array([[float(r[name]) for name in names] for r in good])
    y = np.array([float(r["objective"]) for r in good])
    ids = np.array([i for i, row in enumerate(rows, 1) if row["status"] == "ok"])
    best_idx = int(np.argmin(y))
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.scatter(ids, y, alpha=.48, s=20, label="Evaluations")
    if len(good) < len(rows):
        failures = [i for i, row in enumerate(rows, 1) if row["status"] != "ok"]
        ax.scatter(failures, np.full(len(failures), np.max(y)), marker="x", color="crimson",
                   label="Failed (penalized; shown at top)")
    # Start at first successful evaluation; failures never become best.
    ax.step(ids, np.minimum.accumulate(y), where="post", color="#b3212d", lw=2, label="Best so far")
    if meta["kind"] == "bo":
        initial = [ids[i] for i, row in enumerate(good) if row.get("phase") == "initial"]
        if initial:
            initial_y = [float(rows[i - 1]["objective"]) for i in initial]
            ax.scatter(initial, initial_y, color="#e39927", s=28,
                       label="Initial design")
    ax.set(xlabel="Function evaluation (completion order)", ylabel="RMS", title="Optimization convergence")
    ax.legend()
    fig.savefig(plots / "convergence.png")
    plt.close(fig)

    fig, axs = plt.subplots(len(names), 1, figsize=(8, max(3, 1.9 * len(names))),
                            sharex=True, layout="constrained", squeeze=False)
    for i, name in enumerate(names):
        axs[i, 0].scatter(ids, X[:, i], c=y, cmap="viridis_r", s=18)
        axs[i, 0].axhline(X[best_idx, i], c="#b3212d", ls="--", lw=1)
        axs[i, 0].set_ylabel(name)
    axs[-1, 0].set_xlabel("Function evaluation (completion order; successful points)")
    fig.savefig(plots / "parameters.png")
    plt.close(fig)

    columns = min(len(names), 4)
    fig, axs = plt.subplots((len(names) + columns - 1) // columns, columns,
                            figsize=(4 * columns, 3.4 * ((len(names) + columns - 1) // columns)),
                            layout="constrained", squeeze=False)
    for i, name in enumerate(names):
        ax = axs.flat[i]
        ax.scatter(X[:, i], y, c=y, cmap="viridis_r", s=23)
        ax.scatter([X[best_idx, i]], [y[best_idx]], marker="*", s=175, c="crimson", edgecolors="k")
        ax.set(xlabel=name, ylabel="RMS" if i % columns == 0 else "", title=f"Projection: {name}")
    for ax in list(axs.flat)[len(names):]:
        ax.set_visible(False)
    fig.savefig(plots / "landscape.png")
    plt.close(fig)

    best = good[best_idx]
    reference = read_curve(run / "reference.csv")
    fitted = read_curve(run / best["curve_file"])
    actual_rms = rms(reference, fitted)
    if not np.isclose(actual_rms, y[best_idx], rtol=1e-6, atol=1e-9):
        raise ValueError(f"Recorded best RMS {y[best_idx]} disagrees with curves ({actual_rms})")
    fig, (ax, residual) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                       height_ratios=[2, 1], layout="constrained")
    ax.plot(reference[:, 0], reference[:, 1], c="black", label="Target", lw=2)
    ax.plot(fitted[:, 0], fitted[:, 1], c="#137c8b", ls="--", label="Best simulation", lw=2)
    ax.legend()
    ax.set(ylabel="Curve value", title=f"Final fit | RMS = {y[best_idx]:.5g}")
    residual.plot(reference[:, 0], fitted[:, 1] - reference[:, 1], c="#b3212d")
    residual.axhline(0, color="gray", lw=.8)
    residual.set(xlabel="x", ylabel="Fit − target")
    fig.savefig(plots / "curve_fit.png")
    plt.close(fig)

    # This reconstruction is a *separate diagnostic GP*, not Dakota's internal GP.
    # Do not treat conditional GP std as parameter credible intervals.
    if meta["kind"] == "bo" and len(good) >= max(6, len(names) + 2):
        lower = np.array([meta["parameters"][name]["lower"] for name in names])
        upper = np.array([meta["parameters"][name]["upper"] for name in names])
        xn = (X - lower) / (upper - lower)
        gp = fit_diagnostic_gp(xn, y)
        if len(names) == 2:
            axis = np.linspace(0, 1, 75)
            xx, yy = np.meshgrid(axis, axis)
            pred, spread = gp.predict(np.column_stack((xx.ravel(), yy.ravel())), return_std=True)
            fig, axs2 = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
            for ax2, field, title in zip(axs2, (pred, spread), ("GP predicted RMS", "GP predictive std")):
                image = ax2.contourf(lower[0] + xx * (upper[0] - lower[0]),
                                     lower[1] + yy * (upper[1] - lower[1]),
                                     field.reshape(xx.shape), levels=18, cmap="viridis")
                ax2.scatter(X[:, 0], X[:, 1], c="white", edgecolors="black", s=15)
                ax2.scatter(*X[best_idx], marker="*", c="crimson", s=155, edgecolors="k")
                ax2.set(xlabel=names[0], ylabel=names[1], title=title)
                fig.colorbar(image, ax=ax2)
            fig.savefig(plots / "landscape_2d_gp.png")
            plt.close(fig)
        fig, axs = plt.subplots(min(len(names), 4), 1, figsize=(8, 2.8 * min(len(names), 4)),
                                layout="constrained", squeeze=False)
        center = xn[best_idx].copy()
        for i, name in enumerate(names[:4]):
            grid = np.linspace(0, 1, 120)
            points = np.tile(center, (len(grid), 1))
            points[:, i] = grid
            mean, std = gp.predict(points, return_std=True)
            xaxis = lower[i] + grid * (upper[i] - lower[i])
            ax = axs[i, 0]
            ax.plot(xaxis, mean, c="#137c8b", label="Diagnostic GP prediction")
            ax.fill_between(xaxis, mean - 1.96 * std, mean + 1.96 * std,
                            color="#137c8b", alpha=.2, label="±1.96 GP std")
            ax.scatter(X[:, i], y, c="black", s=9, alpha=.25, label="Evaluations (projected)")
            ax.axvline(X[best_idx, i], color="crimson", ls="--")
            ax.set(xlabel=name, ylabel="RMS", title=f"Slice through empirical best: {name}")
        axs[0, 0].legend(fontsize=7)
        fig.savefig(plots / "surrogate_diagnostics.png")
        plt.close(fig)
    print(f"Saved figures to {plots}")


if __name__ == "__main__":
    main()