"""Discrete likelihood-ratio screening of sampled points, conditional on known noise.

Not calibrated parameter intervals: design is adaptive, unsampled locations are absent.
For defensible intervals/posteriors sample a defined likelihood/prior (e.g. Dakota
Bayesian calibration on individual curve residuals, or replicate-data bootstrap).
"""

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import chi2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--sigma", type=float, required=True, help="known per-point Gaussian observation noise SD")
    parser.add_argument("--confidence", type=float, default=.95)
    args = parser.parse_args()
    if args.sigma <= 0 or not 0 < args.confidence < 1:
        parser.error("sigma must be positive; confidence in (0,1)")
    run = args.run.resolve()
    meta = json.loads((run / "config.json").read_text())
    with (run / "evaluations.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["status"] == "ok"]
    with (run / "reference.csv").open(newline="") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    dof = len(meta["parameters"])
    minimum = min(float(r["objective"]) ** 2 for r in rows)
    cutoff = chi2.ppf(args.confidence, dof)
    nearby = [r for r in rows if count * (float(r["objective"]) ** 2 - minimum) / args.sigma ** 2 <= cutoff]
    output = {
        "interpretation": "Exploratory LR screening only over evaluated candidates; NOT a confidence/credible region",
        "assumptions": "Independent Gaussian residuals with known common sigma; chi-square cutoff is asymptotic",
        "sigma": args.sigma, "confidence": args.confidence, "chi2_threshold": cutoff,
        "candidate_count": len(nearby), "evaluated_count": len(rows),
        "observed_candidate_ranges": {
            name: [min(float(r[name]) for r in nearby), max(float(r[name]) for r in nearby)]
            for name in meta["parameters"]},
    }
    (run / "uncertainty.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()