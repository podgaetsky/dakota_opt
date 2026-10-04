"""Describe repeated independent optimizer runs without conflating reproducibility and identifiability."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(runs):
    if len(runs) < 2:
        raise ValueError("Provide at least two independently initialized runs")
    configs = [json.loads((run / "config.json").read_text()) for run in runs]
    names = list(configs[0]["parameters"])
    if any(list(config["parameters"]) != names or any(
        any(config["parameters"][name][edge] != configs[0]["parameters"][name][edge]
            for edge in ("lower", "upper")) for name in names) for config in configs):
        raise ValueError("Parameter names and bounds must agree across runs (initial points may differ)")
    results = [json.loads((run / "best.json").read_text()) for run in runs]
    params = np.array([[best["parameters"][name] for name in names] for best in results])
    bounds = np.array([[configs[0]["parameters"][name][key] for key in ("lower", "upper")] for name in names])
    normalized = (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    with np.errstate(invalid="ignore"):
        distances = [float(np.linalg.norm(a - b)) for i, a in enumerate(normalized) for b in normalized[i + 1:]]
    trajectories = {}
    for run in runs:
        with (run / "evaluations.csv").open(newline="") as stream:
            trajectories[str(run)] = [float(row["objective"]) for row in csv.DictReader(stream)
                                      if row["status"] == "ok"]
    return {"run_count": len(runs), "runs": [str(run) for run in runs], "best_objectives": [r["objective"] for r in results],
            "parameter_values": {name: params[:, i].tolist() for i, name in enumerate(names)},
            "normalized_pairwise_optima_distances": distances,
            "bound_frequency_within_2_percent": {name: float(np.mean((normalized[:, i] < .02) | (normalized[:, i] > .98))) for i, name in enumerate(names)},
            "convergence_series": trajectories,
            "caveat": "Runs must have independent starts/seeds on identical data/model/noise. Stability is not intrinsic identifiability or proof of global optimum."}


def main():
    parser = argparse.ArgumentParser(description="Compare optimization reproducibility across run folders")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
