"""Analytically tractable two-coefficient toy response; NOT a hydrodynamic EOS."""

import csv
import json
import sys
from pathlib import Path


def benchmark_model(params, xs):
    return [params["a"] + params["b"] * x for x in xs]


def main():
    work = Path(sys.argv[1])
    params = json.loads((work / "physical_params.json").read_text())
    with (work / "reference.csv").open(newline="") as stream:
        xs = [float(row["x"]) for row in csv.DictReader(stream)]
    with (work / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y"))
        writer.writerows(zip(xs, benchmark_model(params, xs)))


if __name__ == "__main__":
    main()