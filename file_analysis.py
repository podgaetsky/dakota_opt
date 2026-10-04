"""File-based Gaussian residual analysis; χ² is NOT the Dakota RMS objective.

The formula assumes known, independent per-ordinate observation SD sigma:
    chi2 = sum(((observed_y - predicted_y) / sigma) ** 2).
A fitted-parameter reduced χ² is descriptive unless the fitted model/noise
assumptions (and the relevant effective degrees of freedom) are justified.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from driver import curve


def analyze_files(reference, prediction, sigma, fitted_parameters=0):
    """Read x,y CSVs; return residual rows and transparent goodness-of-fit metrics."""
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite observation standard deviation")
    if type(fitted_parameters) is not int or fitted_parameters < 0:
        raise ValueError("fitted_parameters must be a nonnegative integer")
    observed = curve(Path(reference))
    predicted = curve(Path(prediction))
    if len(observed) != len(predicted) or any(
        not math.isclose(x, xp, rel_tol=1e-8, abs_tol=1e-10)
        for (x, _), (xp, _) in zip(observed, predicted)
    ):
        raise ValueError("Observation/prediction x grids differ; align before calculating chi-squared")
    residuals = [{"x": x, "observed": y, "predicted": yp, "residual": y - yp,
                  "standardized_residual": (y - yp) / sigma}
                 for (x, y), (_, yp) in zip(observed, predicted)]
    n = len(residuals)
    chi2 = math.fsum(row["standardized_residual"] ** 2 for row in residuals)
    rms = math.sqrt(math.fsum(row["residual"] ** 2 for row in residuals) / n)
    dof = n - fitted_parameters
    result = {"reference": str(Path(reference).resolve()), "prediction": str(Path(prediction).resolve()),
              "sigma": sigma, "n_ordinates": n, "fitted_parameters": fitted_parameters,
              "degrees_of_freedom": dof if dof > 0 else None,
              "chi_squared": chi2, "reduced_chi_squared": chi2 / dof if dof > 0 else None,
              "rms": rms,
              "assumption": "Known identical independent Gaussian measurement SD per ordinate; no covariance or model discrepancy",
              "interpretation": "Reduced chi-squared is descriptive unless noise, model, fitted parameter count and degrees of freedom are justified; no goodness-of-fit p-value is claimed."}
    return result, residuals


def write_analysis(reference, prediction, sigma, output, fitted_parameters=0):
    """Save JSON metrics, residual CSV and a portable two-panel curve/residual PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result, residuals = analyze_files(reference, prediction, sigma, fitted_parameters)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "chi_squared.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "residuals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(residuals[0]))
        writer.writeheader()
        writer.writerows(residuals)
    xs = [row["x"] for row in residuals]
    fig, (ax, rx) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                  height_ratios=(2, 1), layout="constrained")
    ax.errorbar(xs, [row["observed"] for row in residuals], yerr=sigma,
                fmt="ko", capsize=2, label="Observations ± assumed 1σ")
    ax.plot(xs, [row["predicted"] for row in residuals], "C0-", label="Prediction")
    ax.legend()
    ax.set(ylabel="y", title=f"File comparison · χ²={result['chi_squared']:.3g}, RMS={result['rms']:.3g}")
    rx.axhline(0, color="gray", linewidth=1)
    rx.plot(xs, [row["standardized_residual"] for row in residuals], "C1o-")
    rx.set(xlabel="x", ylabel="(data − model) / σ")
    fig.savefig(output / "chi_squared.png", dpi=180)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description="Read curve files and save χ², residual table and plot")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="completed optimization run; analyzes its saved best curve")
    source.add_argument("--reference", type=Path, help="observed x,y CSV; requires --prediction")
    parser.add_argument("--prediction", type=Path, help="predicted x,y CSV")
    parser.add_argument("--sigma", type=float, required=True, help="known observation SD, not variance")
    parser.add_argument("--fitted-parameters", type=int, default=0,
                        help="number of fitted parameters (standalone file mode only)")
    parser.add_argument("--output", type=Path, help="output directory; default RUN/plots or current directory")
    args = parser.parse_args()
    if args.run:
        if args.prediction or args.fitted_parameters:
            parser.error("--run selects the saved best curve and fitted-parameter count automatically")
        run = args.run.resolve()
        best = json.loads((run / "best.json").read_text())
        reference = run / "reference.csv"
        prediction = (run / best["curve_file"]).resolve()
        if not prediction.is_relative_to(run):
            parser.error("best curve must reside inside the run")
        count = len(json.loads((run / "config.json").read_text())["parameters"])
        output = args.output or run / "plots"
    else:
        if not args.prediction:
            parser.error("--reference requires --prediction")
        reference, prediction = args.reference, args.prediction
        count, output = args.fitted_parameters, args.output or Path(".")
    result = write_analysis(reference, prediction, args.sigma, output, count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
