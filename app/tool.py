"""Beginner entry point: check, demo, opt, bo, mcmc, report, list."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import config
from dakota_checks import check_dakota
from project import init_project, validate_model
from resources import cluster_settings, write_scripts
from settings import load_settings

ROOT = Path(__file__).resolve().parent


def runtime(dakota=None):
    env = os.environ.copy()
    if dakota:
        candidate = shutil.which(dakota) or str(Path(dakota).expanduser().resolve())
        if not Path(candidate).is_file():
            raise FileNotFoundError(f"Dakota executable not found: {candidate}")
        return candidate, env
    executable = shutil.which("dakota")
    if executable:
        return executable, env
    raise FileNotFoundError("Dakota executable missing: install Dakota 6.23 or pass --dakota /path/to/dakota")


def launch(program, options, env, workdir=ROOT):
    runs = workdir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    old = set(runs.iterdir())
    subprocess.run([sys.executable, str(ROOT / program), *options], cwd=ROOT, env=env, check=True)
    made = [p for p in runs.iterdir() if p not in old and p.is_dir()]
    for run in sorted(made):
        if (run / "posterior_summary.json").exists() or (run / "best.json").exists():
            settings = json.loads((run / "config.json").read_text())
            sigma = settings.get("diagnostic_sigma")
            if (run / "best.json").is_file():
                best = json.loads((run / "best.json").read_text())
                command = [sys.executable, "-m", "fit_quality", "--reference", str(run / "reference.csv"),
                           "--prediction", str(run / best["curve_file"]), "--parameters", str(run / "best.json"),
                           "--parameter-definitions", str(run / "config.json"), "--parameter-count",
                           str(len(settings["parameters"])), "--output", str(run / "fit_quality")]
                if sigma is not None:
                    noise_path = run / "fit_noise.json"
                    noise_path.write_text(json.dumps({"sigma": sigma}) + "\n")
                    command += ["--noise-model", str(noise_path)]
                subprocess.run(command, cwd=ROOT, env=env, check=True)
            if (run / "best.json").is_file() and sigma is not None:
                subprocess.run([sys.executable, str(ROOT / "file_analysis.py"), "--run", str(run),
                                "--sigma", str(sigma)], cwd=ROOT, env=env, check=True)
            subprocess.run([sys.executable, str(ROOT / "report.py"), str(run)],
                           cwd=ROOT, env=env, check=True)
    return made


def main():
    parser = argparse.ArgumentParser(description="Dakota optimization/MCMC launcher and offline diagnostics")
    parser.add_argument("command", choices=("check", "doctor", "init", "validate-model", "resume", "demo", "opt", "bo", "mcmc", "replicate", "analyze", "report", "list", "submit"))
    parser.add_argument("path", nargs="?", type=Path, help="JSON template for opt/bo/mcmc; run directory for report")
    parser.add_argument("--dakota", help="Dakota 6.23 executable, otherwise auto-detect")
    parser.add_argument("--cluster", help="named cluster profile from clusters/<name>.json")
    parser.add_argument("--workdir", type=Path, default=ROOT, help="project root containing runs/")
    parser.add_argument("--mode", choices=("opt", "bo", "mcmc"), help="controller method for submit")
    parser.add_argument("--reference", type=Path, help="override measured curve in the template")
    parser.add_argument("--sigma", type=float, help="independently justified measurement noise SD for MCMC")
    parser.add_argument("--backend", choices=("dream", "queso"),
                        help="MCMC sampler (default dream); QUESO needs a Dakota build with QUESO support")
    parser.add_argument("--sampler", choices=("dream", "queso"), help="MCMC sampler; replaces --backend")
    parser.add_argument("--repeats", type=int, default=3, help="independent randomized starts/seeds for replicate command")
    parser.add_argument("--method", choices=("opt", "bo"), default="opt", help="optimizer to replicate")
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    if args.sampler:
        if args.backend:
            parser.error("Use --sampler or deprecated --backend, not both")
        args.backend = args.sampler
    if args.command == "init":
        if args.path is None:
            parser.error("init requires a new project directory")
        print(f"Created {init_project(args.path)}; replace reference.csv and edit model.py before validating")
        return
    if args.command == "validate-model":
        if args.path is None:
            parser.error("validate-model requires a JSON template")
        print(json.dumps(validate_model(args.path.resolve(), args.reference, args.cluster, args.workdir), indent=2))
        return
    if args.command == "list":
        for run in sorted((args.workdir / "runs").iterdir(), reverse=True):
            if (run / "config.json").is_file():
                outcome = run / "report.json"
                print(f"{run.name}: " + (json.loads(outcome.read_text())["headline"] if outcome.exists() else "no report yet"))
        return
    if args.command == "report":
        if args.path is None:
            parser.error("report requires a run directory")
        subprocess.run([sys.executable, str(ROOT / "report.py"), str(args.path.resolve())], check=True)
        return
    if args.command == "analyze":
        if args.path is None or args.sigma is None:
            parser.error("analyze requires a completed optimization run and --sigma OBSERVATION_SD")
        run = args.path.resolve()
        subprocess.run([sys.executable, str(ROOT / "file_analysis.py"), "--run", str(run),
                        "--sigma", str(args.sigma)], check=True)
        subprocess.run([sys.executable, str(ROOT / "report.py"), str(run)], check=True)
        return
    profile = cluster_settings(args.cluster) if args.cluster else {}
    env = os.environ.copy()
    if profile.get("ld_library_path"):
        env["LD_LIBRARY_PATH"] = ":".join(profile["ld_library_path"]) + ":" + env.get("LD_LIBRARY_PATH", "")
    if args.command in ("check", "doctor"):
        try:
            dakota, _ = runtime(args.dakota or profile.get("dakota"))
            if args.command == "doctor":
                result = check_dakota(dakota, env, profile.get("python", sys.executable))
                print(f"Dakota: {result['version'] or 'unable to start'} (exit {result['version_exit_code']})")
                print(f"QUESO: {result['queso']} ({result['queso_detail']})")
            else:
                subprocess.run([dakota, "-v"], env=env, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f"Dakota: unavailable ({error})")
        for module in ("numpy", "scipy", "matplotlib", "arviz", "corner", "sklearn"):
            try:
                __import__(module)
                print(f"Python {module}: available")
            except (ImportError, OSError) as error:
                print(f"Python {module}: missing ({error})")
        for command in ("sbatch", "squeue", "sacct", "scancel", "srun"):
            print(f"Slurm {command}: {shutil.which(command) or 'not available'}")
        if args.command == "doctor":
            print(f"Python interpreter: {profile.get('python', sys.executable)}")
            print("sacct accounting and compute-node visibility: not checked locally")
            if args.path:
                try:
                    user = load_settings(args.path)
                    print(f"Model: {user['simulation'] or user['simulation_script'] or 'built-in'}")
                    print(f"Reference: {user['reference'] or 'generated demo data'}")
                except (OSError, ValueError) as error:
                    print(f"Template: invalid ({error})")
        return
    dakota, _ = runtime(args.dakota or profile.get("dakota"))
    if args.command == "resume":
        if args.path is None:
            parser.error("resume requires an existing run directory")
        run = args.path.resolve()
        for name in ("dakota.in", "dakota.rst", "config.json"):
            if not (run / name).is_file():
                parser.error(f"Cannot resume: missing {run / name}")
        print("Dakota -read_restart reuses cached evaluations; method-state continuation is not guaranteed.")
        subprocess.run([dakota, "-i", "dakota.in", "-read_restart", "dakota.rst",
                        "-write_restart", "dakota.rst", "-o", "dakota.out", "-e", "dakota.err"],
                       cwd=run, env=env, check=True)
        return
    if args.command == "submit":
        if args.path is None or args.mode is None:
            parser.error("submit requires a template and --mode opt|bo|mcmc")
        template = args.path.resolve()
        load_settings(template)
        settings = {"concurrency": config.CONCURRENCY, "cpus_per_evaluation": config.CPUS_PER_EVALUATION,
                    "memory": config.MEMORY, "time_limit": config.TIME_LIMIT,
                    "partition": profile.get("partition", config.PARTITION),
                    "python_executable": profile.get("python", sys.executable)}
        destination = args.workdir / "runs" / "_submissions" / datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        write_scripts(destination, settings, profile, args.mode, template, dakota, args.workdir,
                  args.cluster)
        script = destination / "submit_allocation.sh"
        result = subprocess.run(["sbatch", "--parsable", str(script)], cwd=destination,
                                env=env, text=True, capture_output=True, check=True)
        (destination / "slurm_job_id.txt").write_text(result.stdout.strip() + "\n")
        print(f"Submitted {result.stdout.strip()}: {script}")
        return
    if args.command == "demo":
        if args.path is not None:
            parser.error("demo does not take a template path")
        template = ROOT / "templates/demo.json"
        launch("run.py", ["opt", "--settings", str(template), "--dakota", dakota], env)
        launch("run.py", ["bo", "--settings", str(template), "--dakota", dakota], env)
        launch("mcmc_run.py", ["--mode", "benchmark", "--settings", str(template),
                               "--dakota", dakota], env)
        launch("sbc_benchmark.py", ["--output", str(ROOT / "runs/sbc_benchmark")], env)
        print("Finished. Open each report.html in the new runs above; use list to find them.")
        return
    if args.path is None:
        parser.error(f"{args.command} requires a JSON template, e.g. templates/demo.json")
    template = args.path.resolve()
    if args.command == "replicate":
        if args.repeats < 2:
            parser.error("replicate needs >=2 independent randomized starts")
        import numpy as np
        from fit_quality.robustness import summarize
        source = json.loads(template.read_text())
        rng = np.random.default_rng(source.get("seed", 1729))
        if args.reference:
            source["reference"] = str(args.reference.resolve())
        if source.get("reference"):
            source["reference"] = str((template.parent / source["reference"]).resolve())
        if source.get("simulation_script"):
            source["simulation_script"] = str((template.parent / source["simulation_script"]).resolve())
        runs = []
        for index in range(args.repeats):
            config = json.loads(json.dumps(source))
            config["seed"] = int(rng.integers(1, 2**30))
            for spec in config["parameters"].values():
                spec["initial"] = float(rng.uniform(spec["lower"], spec["upper"]))
            with tempfile.TemporaryDirectory(prefix="dakota-replicate-") as temp:
                temporary = Path(temp) / "settings.json"
                temporary.write_text(json.dumps(config))
                runs.extend(launch("run.py", [args.method, "--settings", str(temporary), "--dakota", dakota], env))
        saved = ROOT / "runs" / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_robustness.json")
        saved.write_text(json.dumps(summarize(runs), indent=2) + "\n")
        print(f"Independent starts completed: {saved} (optimizer reproducibility, not identifiability)")
        return
    extra = ["--settings", str(template), "--dakota", dakota, "--workdir", str(args.workdir)]
    if args.cluster:
        extra += ["--cluster", args.cluster]
    if args.reference:
        extra += ["--reference", str(args.reference.resolve())]
    if args.command == "mcmc":
        if args.sigma is not None:
            extra += ["--sigma", str(args.sigma)]
        if args.backend is not None:
            extra += ["--backend", args.backend]
        launch("mcmc_run.py", ["--mode", "curve", *extra], env, args.workdir)
    else:
        if args.backend is not None:
            parser.error("--backend applies to mcmc; QUESO is a sampler, not an optimization objective")
        launch("run.py", [args.command, *extra], env, args.workdir)


if __name__ == "__main__":
    main()
