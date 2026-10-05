"""Dakota DREAM or optional QUESO calibration of curves with native GP emulation."""

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config
from driver import curve
from mcmc_simulate import benchmark_model
from provenance import write_provenance
from resources import cluster_settings, write_scripts
from settings import load_settings

ROOT = Path(__file__).resolve().parent
BENCHMARK = {"a": {"initial": 1.0, "lower": 0.0, "upper": 2.0},
             "b": {"initial": 0.3, "lower": -1.0, "upper": 1.0}}


def calibration_record(observations, sigma):
    """Dakota freeform row: scalar response values then their *variances*."""
    return " ".join([*(row["y"] for row in observations),
                     *(f"{sigma ** 2:.17g}" for _ in observations)]) + "\n"


def input_text(run, parameters, n_obs, samples, build_samples, seed, backend="dream", python_executable=None):
    if backend not in ("dream", "queso"):
        raise ValueError("MCMC backend must be dream or queso")
    names = list(parameters)
    initial = [(p["initial"] - p["lower"]) / (p["upper"] - p["lower"])
               for p in parameters.values()]
    driver = " ".join(map(shlex.quote, (python_executable or sys.executable, str(ROOT / "mcmc_driver.py"))))
    sampler = (f"bayes_calibration dream\n        chain_samples = {samples}\n        chains = 4" if backend == "dream"
               else f"bayes_calibration queso\n        chain_samples = {samples}\n        dram")
    return f"""environment
  tabular_data
    tabular_data_file 'dakota_tabular.dat'

method
    {sampler}
        seed = {seed}
        emulator gaussian_process dakota
            build_samples = {build_samples}
        export_chain_points_file 'chain.dat'

variables
  uniform_uncertain = {len(names)}
    initial_point {' '.join(f'{v:.17g}' for v in initial)}
    lower_bounds {' '.join('0' for _ in names)}
    upper_bounds {' '.join('1' for _ in names)}
    descriptors {' '.join(repr('x_' + n) for n in names)}

interface
  analysis_drivers = '{driver}'
  fork
    asynchronous evaluation_concurrency = {config.CONCURRENCY}
    parameters_file = 'params.in'
    results_file = 'results.out'
    work_directory named '{run / 'params' / 'eval'}'
      directory_tag directory_save

responses
  calibration_terms = {n_obs}
  calibration_data_file = 'calibration.dat'
    freeform
    num_experiments = 1
    variance_type = 'scalar'
  no_gradients
  no_hessians
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "curve"), default="benchmark")
    parser.add_argument("--backend", choices=("dream", "queso"), default=None,
                        help="Dakota Bayesian calibration backend (default dream); QUESO requires a QUESO-enabled Dakota build")
    parser.add_argument("--settings", type=Path, help="editable JSON settings template")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--build-samples", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None, help="independently justified per-ordinate observation SD")
    parser.add_argument("--seed", type=int, default=491)
    parser.add_argument("--reference", type=Path, help="required measured x,y CSV for --mode curve")
    parser.add_argument("--validate-samples", type=int, default=None,
                        help="exact simulation evaluations at posterior draws; 0 skips (not recommended)")
    parser.add_argument("--max-ordinates", type=int, default=None,
                        help="EXPLICIT computational thinning; default full curve unless configured in settings")
    parser.add_argument("--dakota", default=os.environ.get("DAKOTA", "dakota"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cluster", help="named clusters/<name>.json profile")
    parser.add_argument("--workdir", type=Path, default=ROOT, help="project directory for runs/")
    args = parser.parse_args()
    user = load_settings(args.settings, args.reference) if args.settings else {"mcmc": {}, "reference": None, "simulation_script": None, "simulation": None}
    if args.backend is None:
        args.backend = user["mcmc"].get("backend", "dream")
    if args.mode == "curve" and args.reference is None and user.get("reference"):
        args.reference = Path(user["reference"])
    for key, default in (("samples", 24000), ("build_samples", 48),
                         ("validate_samples", 16)):
        if getattr(args, key) is None:
            setattr(args, key, user["mcmc"].get(key, default))
    if args.max_ordinates is None:
        args.max_ordinates = user["mcmc"].get("max_ordinates")
    if args.sigma is None:
        args.sigma = user["mcmc"].get("sigma", .05 if args.mode == "benchmark" else None)
    if args.mode == "curve" and args.sigma is None:
        parser.error("Curve MCMC needs a justified --sigma or mcmc.sigma in settings")
    if args.samples < 100 or args.build_samples < 8 or not math.isfinite(args.sigma) or args.sigma <= 0 or args.validate_samples < 0:
        parser.error("Need samples >= 100, build-samples >= 8 and positive sigma")
    if (args.backend == "dream" and args.samples % 4) or (args.max_ordinates is not None and args.max_ordinates < 2):
        parser.error("DREAM samples must be divisible by four; max-ordinates must be >= 2")
    if args.mode == "curve" and (args.reference is None or not args.reference.is_file()):
        parser.error("--mode curve requires an existing --reference x,y CSV")
    if args.mode == "curve":
        curve(args.reference)  # reject malformed or non-increasing x before Dakota
    if config.BACKEND not in ("local", "slurm", "allocation"):
        parser.error("Invalid config.BACKEND")
    profile = cluster_settings(args.cluster) if args.cluster else {}
    if profile.get("partition"):
        config.PARTITION = profile["partition"]
    parameters = BENCHMARK if args.mode == "benchmark" else config.PARAMETERS
    run_root = args.workdir.resolve() / "runs"
    run = run_root / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_mcmc_" + args.mode)
    suffix = 1
    while run.exists():
        run = Path(f"{run_root / (datetime.now().strftime('%Y-%m-%d_%H%M%S') + '_mcmc_' + args.mode)}_{suffix}")
        suffix += 1
    for folder in ("params", "logs", "results", "plots"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        xs = [-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.]
        true = benchmark_model({"a": 1.15, "b": .4}, xs)
        # Fixed, documented noise realization for a reproducible posterior.
        noise = [.3, -.7, .1, 1.1, -.4, .2, -.8, .6, -.3]
        with (run / "reference.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y"))
            writer.writerows((x, y + args.sigma * e) for x, y, e in zip(xs, true, noise))
    else:
        with args.reference.open(newline="") as stream:
            original = list(csv.DictReader(stream))
        if not original or any("x" not in row or "y" not in row for row in original):
            parser.error("Reference must be nonempty x,y CSV")
        # Explicit, deterministic thinning limits the number of emulated outputs.
        # Independent-noise assumption still needs scientific validation.
        import numpy as np
        selected = (np.arange(len(original)) if args.max_ordinates is None else
                np.unique(np.linspace(0, len(original) - 1,
                          min(len(original), args.max_ordinates), dtype=int)))
        with (run / "reference.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y"))
            writer.writerows((original[i]["x"], original[i]["y"]) for i in selected)
    with (run / "reference.csv").open(newline="") as stream:
        observations = list(csv.DictReader(stream))
    if not observations or any(not {"x", "y"}.issubset(r) for r in observations):
        parser.error("Reference must be a nonempty x,y CSV")
    # Dakota calibration_data_file variance_type scalar takes VARIANCES, not SDs.
    (run / "calibration.dat").write_text(calibration_record(observations, args.sigma))
    settings = {"parameters": parameters, "backend": config.BACKEND,
                "concurrency": config.CONCURRENCY, "cpus_per_evaluation": config.CPUS_PER_EVALUATION,
                "memory": config.MEMORY, "time_limit": config.TIME_LIMIT,
                "partition": config.PARTITION, "job_timeout_seconds": config.JOB_TIMEOUT_SECONDS,
                "poll_seconds": config.POLL_SECONDS,
                "simulation_script": ("mcmc_simulate.py" if args.mode == "benchmark"
                                      else user["simulation_script"] or "simulate.py"),
                "simulation": user.get("simulation") if args.mode == "curve" else None,
                "python_executable": profile.get("python", sys.executable),
                "sigma": args.sigma, "seed": args.seed, "mode": args.mode, "mcmc_backend": args.backend,
                "samples": args.samples, "build_samples": args.build_samples,
                "likelihood_data_selection": {"mode": "full_curve" if args.mode == "benchmark" or args.max_ordinates is None else "explicit_thinning",
                                              "selected_ordinates": len(observations),
                                              "original_ordinates": len(original) if args.mode == "curve" else len(observations),
                                              "warning": "Thinning changes the likelihood; full curve is default when no max_ordinates is specified."}}
    write_scripts(run, settings, profile)
    (run / "config.json").write_text(json.dumps(settings, indent=2))
    write_provenance(run, settings, args.dakota, args.reference)
    (run / "dakota.in").write_text(input_text(run, parameters, len(observations),
                                                 args.samples, args.build_samples, args.seed, args.backend,
                                                 settings["python_executable"]))
    print(run, flush=True)
    if args.prepare_only:
        return
    if args.backend == "queso":
        # -check checks method instantiation, not just keyword parsing. The stock
        # Dakota binary can parse 'queso' but was built without HAVE_QUESO.
        preflight = subprocess.run([args.dakota, "-i", "dakota.in", "-check"], cwd=run,
                                   capture_output=True, text=True, check=False)
        (run / "dakota_check.log").write_text(preflight.stdout + preflight.stderr)
        if preflight.returncode:
            if "QUESO Bayesian calibration method unavailable" in preflight.stdout + preflight.stderr:
                raise SystemExit("This Dakota build does not include QUESO; see "
                                 f"{run / 'dakota_check.log'}. Use --backend dream or install a QUESO-enabled Dakota.")
            raise SystemExit(f"Dakota QUESO input check failed; see {run / 'dakota_check.log'}.")
    with (run / "dakota_console.log").open("w") as console:
        result = subprocess.run([args.dakota, "-i", "dakota.in", "-o", "dakota.out", "-e", "dakota.err"],
                                cwd=run, stdout=console, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise SystemExit(f"Dakota failed ({result.returncode}); see {run / 'dakota.err'} and dakota_console.log")
    subprocess.run([sys.executable, str(ROOT / "mcmc_analyze.py"), str(run)], check=True)
    if args.validate_samples and args.backend == "dream":
        subprocess.run([sys.executable, str(ROOT / "mcmc_validate.py"), str(run),
                        "--samples", str(args.validate_samples)], check=True)
    elif args.backend == "queso":
        print("QUESO: single exported chain cannot establish multi-chain convergence; "
              "DREAM-only GP log-density validation was not run. Intervals are exploratory.")


if __name__ == "__main__":
    main()