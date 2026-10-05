"""Make a unique run, generate Dakota 6.23 inputs, optionally execute Dakota."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config
from driver import curve
from provenance import write_provenance
from settings import load_settings
from simulate import simulate_curve

ROOT = Path(__file__).resolve().parent


def input_text(kind, run_dir):
    names = list(config.PARAMETERS)
    if not names or any(not name.isidentifier() for name in names):
        raise ValueError("Parameter names must be nonempty Python identifiers")
    initial = []
    for name, spec in config.PARAMETERS.items():
        low, x, high = (float(spec[k]) for k in ("lower", "initial", "upper"))
        if not low < high or not low <= x <= high:
            raise ValueError(f"Invalid bounds/initial point for {name}")
        initial.append((x - low) / (high - low))
    if config.BO_BATCH_SIZE > config.CONCURRENCY:
        raise ValueError("BO_BATCH_SIZE must not exceed CONCURRENCY")
    method = (
        f"asynch_pattern_search\n    max_function_evaluations = {config.OPT_EVALUATIONS}\n"
        "    variable_tolerance = 1.e-4\n    synchronization nonblocking"
        if kind == "opt" else
        f"efficient_global\n    gaussian_process dakota\n"
        f"    initial_samples = {config.BO_INITIAL_SAMPLES}\n"
        f"    batch_size = {config.BO_BATCH_SIZE}\n"
        f"    max_iterations = {config.BO_BATCHES * config.BO_BATCH_SIZE}\n    seed = {config.SEED}"
    )
    quoted_names = " ".join(f"'x_{name}'" for name in names)
    # fork receives argv without shell interpolation. shlex.quote protects paths containing spaces.
    driver = " ".join(map(shlex.quote, [sys.executable, str(ROOT / "driver.py")]))
    return f"""environment
  tabular_data
    tabular_data_file 'dakota_tabular.dat'

method
  {method}

variables
  continuous_design = {len(names)}
    initial_point {' '.join(f'{x:.17g}' for x in initial)}
    lower_bounds {' '.join('0' for _ in names)}
    upper_bounds {' '.join('1' for _ in names)}
    descriptors {quoted_names}

interface
  analysis_drivers = '{driver}'
  fork
    asynchronous evaluation_concurrency = {config.CONCURRENCY}
    parameters_file = 'params.in'
    results_file = 'results.out'
    work_directory named '{run_dir / 'params' / 'eval'}'
      directory_tag directory_save

responses
  objective_functions = 1
  no_gradients
  no_hessians
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["opt", "bo"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Defer plotting to another machine")
    parser.add_argument("--settings", type=Path, help="editable JSON settings template")
    parser.add_argument("--reference", type=Path, help="measured x,y CSV (if not supplied in settings)")
    parser.add_argument("--dakota", default=os.environ.get("DAKOTA", "dakota"))
    args = parser.parse_args()
    user = load_settings(args.settings, args.reference) if args.settings else {"mode": "demo", "reference": None, "simulation_script": None}
    if args.reference:
        user["reference"] = str(args.reference.resolve())
        user["mode"] = "measured"
    if user["mode"] == "measured":
        if not user["reference"] or not Path(user["reference"]).is_file():
            parser.error("Measured optimization needs an existing --reference x,y CSV")
        curve(Path(user["reference"]))  # validate before creating a run
    if config.BACKEND not in ("local", "slurm", "allocation"):
        raise ValueError("BACKEND must be local, slurm or allocation")
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = ROOT / "runs" / f"{stamp}_{args.kind}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix:03d}")
        suffix += 1
    run_dir.mkdir(parents=True)
    for name in ("params", "logs", "slurm", "results", "plots"):
        (run_dir / name).mkdir()
    settings = {
        "parameters": config.PARAMETERS, "backend": config.BACKEND,
        "mode": user["mode"], "reference_source": user["reference"],
        "simulation_script": user["simulation_script"] or "simulate.py",
        "simulation": user.get("simulation"), "python_executable": sys.executable,
        "diagnostic_sigma": user.get("mcmc", {}).get("sigma") if user["mode"] == "measured" else None,
        "concurrency": config.CONCURRENCY, "cpus_per_evaluation": config.CPUS_PER_EVALUATION,
        "memory": config.MEMORY, "time_limit": config.TIME_LIMIT,
        "partition": config.PARTITION, "job_timeout_seconds": config.JOB_TIMEOUT_SECONDS,
        "poll_seconds": config.POLL_SECONDS,
        "failure_penalty": config.FAILURE_PENALTY, "kind": args.kind,
        "seed": config.SEED, "opt_evaluations": config.OPT_EVALUATIONS,
        "bo_batches": config.BO_BATCHES, "bo_batch_size": config.BO_BATCH_SIZE,
        "bo_initial_samples": config.BO_INITIAL_SAMPLES,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    # Example reference data; replace with an actual measured reference.csv.
    if user["mode"] == "measured":
        (run_dir / "reference.csv").write_bytes(Path(user["reference"]).read_bytes())
    elif set(config.PARAMETERS) == {"p1", "p2", "p3", "p4"}:
        truth = {"p1": 1.7, "p2": 0.8, "p3": 7.0, "p4": 3.0}
        xs = [i / 10 for i in range(101)]
        with (run_dir / "reference.csv").open("w") as stream:
            stream.write("x,y\n")
            for x, y in zip(xs, simulate_curve(truth, xs)):
                stream.write(f"{x:.17g},{y:.17g}\n")
    else:
        raise ValueError("Replace the demo reference generation in run.py for new parameters")
    (run_dir / "dakota.in").write_text(input_text(args.kind, run_dir))
    write_provenance(run_dir, settings, args.dakota, user["reference"])
    (run_dir / "run_command.txt").write_text(f"{args.dakota} -i dakota.in -o dakota.out -e dakota.err\n")
    print(run_dir, flush=True)
    if args.prepare_only:
        return
    env = os.environ.copy()
    env["DAKOTA_RUN_DIR"] = str(run_dir)
    result = subprocess.run([args.dakota, "-i", "dakota.in", "-o", "dakota.out", "-e", "dakota.err"],
                            cwd=run_dir, env=env, check=False)
    if result.returncode:
        raise SystemExit(f"Dakota exited {result.returncode}; see {run_dir / 'dakota.err'}")
    subprocess.run([sys.executable, str(ROOT / "analyze.py"), str(run_dir)], check=True)
    if not args.no_plots:
        subprocess.run([sys.executable, str(ROOT / "plot_results.py"), str(run_dir)], check=True)


if __name__ == "__main__":
    main()