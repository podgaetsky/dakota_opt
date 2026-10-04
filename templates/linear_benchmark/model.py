"""Editable file-based forward-model example; executed once per Dakota evaluation.

The only required contract is: read WORK/physical_params.json and WORK/reference.csv;
write WORK/curve.csv containing x,y on exactly the same x grid. Replace the
formula below with your own physical simulator and curve-file parser.
"""

import csv
import json
import sys
from pathlib import Path


def simulate_from_files(work):
    work = Path(work)
    params = json.loads((work / "physical_params.json").read_text())
    with (work / "reference.csv").open(newline="") as stream:
        xs = [float(row["x"]) for row in csv.DictReader(stream)]
    with (work / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y"))
        writer.writerows((x, params["a"] + params["b"] * x) for x in xs)
    return work / "curve.csv"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: model.py WORK_DIR")
    print(simulate_from_files(sys.argv[1]))
