"""Replace simulate_curve() with your expensive simulation, leaving CSV columns x,y."""

import csv
import json
import math
import sys
from pathlib import Path


def simulate_curve(params, xs):
    # Demonstration only: deterministic and mildly nonlinear.
    return [params["p1"] * math.exp(-x / params["p3"])
            + params["p2"] * math.sin(params["p4"] * x / 5) for x in xs]


def main():
    work = Path(sys.argv[1])
    params = json.loads((work / "physical_params.json").read_text())
    with (work / "reference.csv").open(newline="") as stream:
        xs = [float(row["x"]) for row in csv.DictReader(stream)]
    with (work / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y"))
        writer.writerows(zip(xs, simulate_curve(params, xs)))


if __name__ == "__main__":
    main()