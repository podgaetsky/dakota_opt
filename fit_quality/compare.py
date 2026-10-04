"""Compare independently audited models on identical observations and noise."""

import argparse
import csv
import json
from pathlib import Path

from .core import read_curve, read_noise, analyze_curve


def compare(specs):
    """Specs: list of {model, reference, prediction, noise_model, parameter_count, optional validation_reference/prediction}."""
    rows = []
    signature = None
    for spec in specs:
        ref = read_curve(spec["reference"])
        noise = read_noise(spec["noise_model"]) if spec.get("noise_model") else None
        key = (ref.tobytes(), json.dumps(noise, sort_keys=True))
        if signature is not None and key != signature:
            raise ValueError("Comparative AIC/BIC require identical observations, ordering and declared noise model")
        signature = key
        m, _ = analyze_curve(ref, read_curve(spec["prediction"]), noise=noise,
                             parameter_count=spec.get("parameter_count"))
        row = {"model": spec["model"], "parameter_count": spec.get("parameter_count"),
               **{key: m[key] for key in ("rms", "rmse", "mae", "chi2", "reduced_chi2", "log_likelihood", "aic", "aicc", "bic")}}
        row.update({"validation_rmse": None, "validation_mae": None})
        if spec.get("validation_reference") and spec.get("validation_prediction"):
            val, _ = analyze_curve(read_curve(spec["validation_reference"]), read_curve(spec["validation_prediction"]))
            row.update({"validation_rmse": val["rmse"], "validation_mae": val["mae"]})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare compatible candidate curve models (no Dakota)")
    parser.add_argument("--models", type=Path, required=True, help="JSON list of model specifications")
    parser.add_argument("--output", type=Path, required=True, help="output CSV")
    args = parser.parse_args()
    rows = compare(json.loads(args.models.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    print("Lower AIC/AICc/BIC is conditional on a compatible likelihood; held-out accuracy is complementary; lower RMS alone does not justify extra complexity.")


if __name__ == "__main__":
    main()
