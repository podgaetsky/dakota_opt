"""Independent exact hold-out evaluations of a surrogate (not a training-set plot)."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .core import analyze_curve, read_curve


def validate(exact, predicted, predictive_sd=None):
    """Exact and GP-mean x,y CSVs on the same held-out x grid."""
    metrics, data = analyze_curve(exact, predicted)
    residual = data["residual"]
    result = {"rmse": metrics["rmse"], "mae": metrics["mae"],
              "max_absolute_error": metrics["max_absolute_error"],
              "rank_correlation": float(spearmanr(data["observed"], data["predicted"]).statistic)
              if len(residual) >= 3 and np.std(data["observed"]) > 0 and np.std(data["predicted"]) > 0 else None,
              "standardized_prediction_error_rmse": None, "predictive_95_coverage": None,
              "mean_negative_log_predictive_density": None,
              "caveat": "Use independent exact hold-out simulator points, not GP training data. Coverage and NLPD require GP predictive SD, not observation noise SD."}
    if predictive_sd is not None:
        sd = np.asarray(predictive_sd, dtype=float)
        if sd.shape != residual.shape or not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError("Positive finite GP predictive SD required at every hold-out point")
        standardized = residual / sd
        result.update({"standardized_prediction_error_rmse": float(np.sqrt(np.mean(standardized ** 2))),
                       "predictive_95_coverage": float(np.mean(np.abs(standardized) <= 1.96)),
                       "mean_negative_log_predictive_density": float(np.mean(.5 * (standardized ** 2 + np.log(2 * np.pi * sd ** 2))))})
    return result


def main():
    parser = argparse.ArgumentParser(description="Score GP hold-out predictions against exact simulator curves")
    parser.add_argument("--exact", type=Path, required=True, help="held-out exact x,y CSV")
    parser.add_argument("--predicted", type=Path, required=True, help="surrogate mean x,y CSV")
    parser.add_argument("--gp-sd", type=Path, help="optional comma-separated GP predictive SD vector")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sd = np.loadtxt(args.gp_sd, delimiter=",", ndmin=1) if args.gp_sd else None
    result = validate(read_curve(args.exact), read_curve(args.predicted), sd)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
