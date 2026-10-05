"""Compare Dakota 6.23 DREAM and QUESO on identical curve-calibration inputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dakota_checks import check_dakota

APP = Path(__file__).resolve().parent
ROOT = APP.parent


def compare(template, dakota, workdir, samples=4000, build_samples=48, seed=491,
            mode="benchmark"):
    """Run both samplers with the same prior, reference, noise and GP budget."""
    dakota = str(Path(dakota).resolve())
    result = check_dakota(dakota, os.environ.copy(), sys.executable)
    if result["queso"] != "available":
        raise RuntimeError(f"QUESO-enabled Dakota required: {result['queso_detail']}")
    template = Path(template).resolve()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    summary = {"dakota_version": result["version"], "reference_template": str(template), "mode": mode,
               "samples": samples, "build_samples": build_samples, "seed": seed,
               "warning": "DREAM and QUESO use different sampling methods; QUESO has one chain and cannot establish multi-chain convergence.",
               "samplers": {}}
    for sampler in ("dream", "queso"):
        command = [sys.executable, str(APP / "mcmc_run.py"), "--mode", mode,
                   "--settings", str(template), "--backend", sampler,
                   "--samples", str(samples), "--build-samples", str(build_samples),
                   "--seed", str(seed), "--validate-samples", "0", "--dakota", dakota,
                   "--workdir", str(workdir)]
        started = time.monotonic()
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if run.returncode:
            raise RuntimeError(f"{sampler} failed (exit {run.returncode}):\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
        runs = [Path(line) for line in run.stdout.splitlines() if line.startswith(str(workdir / "runs"))]
        if not runs or not (runs[0] / "posterior_summary.json").is_file():
            raise RuntimeError(f"{sampler} did not produce a posterior_summary.json")
        path = runs[0]
        posterior = json.loads((path / "posterior_summary.json").read_text())
        settings = json.loads((path / "config.json").read_text())
        inputs = {"reference_sha256": hashlib.sha256((path / "reference.csv").read_bytes()).hexdigest(),
                  "parameters": settings["parameters"], "sigma": settings["sigma"],
                  "seed": settings["seed"], "build_samples": settings["build_samples"]}
        if sampler == "dream":
            common_inputs = inputs
        elif inputs != common_inputs:
            raise RuntimeError(f"Sampler inputs differ: DREAM {common_inputs}, QUESO {inputs}")
        entry = {"run": str(path), "wall_seconds": round(time.monotonic() - started, 3),
                 "posterior": posterior.get("posterior"), "rhat": posterior.get("rhat"),
                 "ess_bulk": posterior.get("ess_bulk"), "retained_draws": posterior.get("retained_draws"),
                 "selected_ordinates": settings["likelihood_data_selection"]["selected_ordinates"],
                 "sigma": settings["sigma"]}
        if "analytic_gaussian_reference" in posterior:
            entry["analytic_gaussian_reference"] = posterior["analytic_gaussian_reference"]
        summary["samplers"][sampler] = entry
    summary["common_inputs"] = common_inputs
    directory = workdir / "runs" / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_sampler_comparison")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, nargs="?", default=ROOT / "templates/demo.json")
    parser.add_argument("--mode", choices=("benchmark", "curve"), default="benchmark")
    parser.add_argument("--dakota", required=True)
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--build-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=491)
    args = parser.parse_args()
    if args.samples < 100 or args.samples % 4 or args.build_samples < 8:
        parser.error("samples must be >=100 and divisible by 4; build-samples must be >=8")
    directory, summary = compare(args.template, args.dakota, args.workdir,
                                 args.samples, args.build_samples, args.seed, args.mode)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {directory / 'comparison.json'}")


if __name__ == "__main__":
    main()
