"""Built-in demo model; new models implement the template's simulation command."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("work", nargs="?", type=Path, help="legacy evaluation directory")
    parser.add_argument("--params", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.work is None and not all((args.params, args.out, args.reference)):
        parser.error("Pass an evaluation directory or --params, --out and --reference")
    params_path = args.params or args.work / "physical_params.json"
    reference_path = args.reference or args.work / "reference.csv"
    output_path = args.out or args.work / "curve.csv"
    params = json.loads(params_path.read_text())
    with reference_path.open(newline="") as stream:
        xs = [float(row["x"]) for row in csv.DictReader(stream)]
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y"))
        writer.writerows(zip(xs, simulate_curve(params, xs)))


if __name__ == "__main__":
    main()