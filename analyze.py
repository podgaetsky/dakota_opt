"""Summarize complete run; do not interpret GP variance as parameter confidence."""

import csv
import json
import sys
from pathlib import Path


def main():
    run = Path(sys.argv[1]).resolve()
    with (run / "evaluations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    good = [row for row in rows if row["status"] == "ok"]
    if not good:
        raise SystemExit("No successful objective evaluations")
    best = min(good, key=lambda row: float(row["objective"]))
    names = list(json.loads((run / "config.json").read_text())["parameters"])
    result = {"evaluation_id": best["evaluation_id"], "objective": float(best["objective"]),
              "parameters": {name: float(best[name]) for name in names},
              "curve_file": best["curve_file"], "successful": len(good),
              "failed": len(rows) - len(good)}
    (run / "best.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()