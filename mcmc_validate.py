"""Compare Dakota GP-based DREAM posterior log-density differences with exact runs.

Validates the forward-model emulator at retained posterior draws, using the
same Slurm/local evaluator. A corner plot does not establish this agreement.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, spearmanr

from driver import curve, run_job
from mcmc_analyze import read_chains


def validate(run, samples=16):
    run = Path(run).resolve()
    settings = json.loads((run / "config.json").read_text())
    if settings.get("mcmc_backend", "dream") != "dream":
        raise ValueError("Exact-vs-GP log-density validation requires DREAM chain log densities; "
                         "QUESO chain export is not compatible. Do not use QUESO intervals for inference.")
    names = list(settings["parameters"])
    normalized, surrogate_logp = read_chains(run, names)
    burn = normalized.shape[1] // 2
    xn = normalized[:, burn:].reshape(-1, len(names))
    logs = surrogate_logp[:, burn:].reshape(-1)
    indices = np.linspace(0, len(xn) - 1, min(samples, len(xn)), dtype=int)
    reference = curve(run / "reference.csv")
    xs = np.array([x for x, _ in reference])
    expected = np.array([y for _, y in reference])
    exact = []
    predictions = []
    batch = 1
    while (run / "params" / f"validation_{batch:03d}").exists():
        batch += 1
    validation_dir = run / "params" / f"validation_{batch:03d}"
    validation_dir.mkdir()
    for j, i in enumerate(indices):
        work = validation_dir / f"draw.{j:03d}"
        work.mkdir()
        params = {name: spec["lower"] + xn[i, k] * (spec["upper"] - spec["lower"])
                  for k, (name, spec) in enumerate(settings["parameters"].items())}
        (work / "physical_params.json").write_text(json.dumps(params))
        (work / "reference.csv").write_bytes((run / "reference.csv").read_bytes())
        run_job(work, settings, settings["simulation_script"])
        simulated = curve(work / "curve.csv")
        if len(simulated) != len(expected) or any(not np.isclose(x, target[0], atol=1e-10)
                                                   for (x, _), target in zip(simulated, reference)):
            raise ValueError("Validation curve does not match reference x values")
        model_y = np.array([y for _, y in simulated])
        predictions.append(model_y)
        residual = model_y - expected
        exact.append(float(-.5 * np.sum((residual / settings["sigma"]) ** 2)))
    exact = np.asarray(exact)
    approx = logs[indices]
    # Log likelihoods differ by an additive constant; compare centered values.
    delta = (exact - approx) - np.median(exact - approx)
    discrepancy = float(np.quantile(np.abs(delta), .9))
    raw_delta = exact - approx
    density_metrics = {"log_density_error_rmse_centered": float(np.sqrt(np.mean(delta ** 2))),
                       "log_density_error_mae_centered": float(np.mean(np.abs(delta))),
                       "log_density_error_max_centered": float(np.max(np.abs(delta))),
                       "log_density_spearman": float(spearmanr(exact, approx).statistic)
                       if len(exact) >= 3 and np.std(exact) > 0 and np.std(approx) > 0 else None,
                       "log_density_offset_median": float(np.median(raw_delta)),
                       "predictive_interval_coverage": None,
                       "negative_log_predictive_density": None,
                       "note": "Dakota does not export GP predictive variances here; interval coverage and NLPD unavailable. Metrics are local to these exact posterior-region draws."}
    predictions = np.array(predictions)
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    with (validation_dir / "exact_validation_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("posterior_draw", "x", "predicted_y"))
        for draw, model_y in enumerate(predictions):
            writer.writerows((draw, x, y) for x, y in zip(xs, model_y))
    figure, (ax, residual_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                              height_ratios=(2, 1), layout="constrained")
    for model_y in predictions:
        ax.plot(xs, model_y, c="steelblue", alpha=.16, lw=.9)
    ax.plot(xs, np.median(predictions, axis=0), c="navy", lw=2,
            label="Median of exact posterior-draw simulations")
    ax.errorbar(xs, expected, yerr=settings["sigma"], fmt="ko", ms=3, capsize=2,
                label="Reference ± assumed 1σ")
    ax.legend(fontsize=8)
    ax.set(ylabel="Curve value", title="Exact forward-model checks at posterior draws (not a calibrated band)")
    residual_ax.axhline(0, c="gray", lw=.8)
    for model_y in predictions:
        residual_ax.plot(xs, model_y - expected, c="steelblue", alpha=.16, lw=.9)
    residual_ax.plot(xs, np.median(predictions, axis=0) - expected, c="navy", lw=1.8)
    residual_ax.set(xlabel="x", ylabel="Model − reference")
    figure.savefig(plots / "exact_posterior_curves.png", dpi=180)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")
    ax.scatter(approx - np.median(approx), exact - np.median(exact), c="navy", s=25)
    limits = [min(np.min(approx - np.median(approx)), np.min(exact - np.median(exact))),
              max(np.max(approx - np.median(approx)), np.max(exact - np.median(exact)))]
    ax.plot(limits, limits, ls="--", c="crimson")
    ax.set(xlabel="GP log density (centered)", ylabel="Exact log density (centered)",
           title="Posterior emulator validation")
    figure.savefig(plots / "surrogate_validation.png", dpi=180)
    plt.close(figure)
    summary = json.loads((run / "posterior_summary.json").read_text())
    rhat = summary.get("rhat", {})
    ess = summary.get("ess_bulk", {})
    tail = summary.get("ess_tail", {})
    status = {"validated_samples": len(indices),
              "exact_validation_curves": str((validation_dir / "exact_validation_curves.csv").relative_to(run)),
              "log_density_error_p90": discrepancy,
              **density_metrics,
              "exact_chi2_min": float(-2 * np.max(exact)),
              "chi2_999_threshold": float(chi2.ppf(.999, len(expected))),
              "max_rhat": max(rhat.values()), "min_bulk_ess": min(ess.values()),
              "min_tail_ess": min(tail.values()) if tail else None,
              "surrogate_pass": len(indices) >= 16 and discrepancy < 1.0,
              "chain_pass": (max(rhat.values()) < 1.01 and min(ess.values()) >= 400
                             and bool(tail) and min(tail.values()) >= 400)}
    status["inference_ready"] = status["surrogate_pass"] and status["chain_pass"]
    status["exact_draws_caveat"] = "Require at least 16 exact posterior checks; fewer checks cannot establish local GP agreement."
    status["caveat"] = ("inference_ready only checks numerical mixing and GP agreement at a few points; "
                        "noise covariance, model discrepancy, prior, and observational validity require domain review")
    (run / "validation.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples < 4:
        parser.error("Need >=4 posterior validation samples")
    validate(args.run, args.samples)


if __name__ == "__main__":
    main()