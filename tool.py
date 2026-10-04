"""Beginner entry point: check, demo, opt, bo, mcmc, report, list."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def runtime(dakota=None):
    env = os.environ.copy()
    deps = Path.home() / ".local/opt/dakota-6.23-deps/usr/lib/x86_64-linux-gnu"
    if deps.exists():
        paths = [deps, deps / "blas", deps / "lapack"]
        env["LD_LIBRARY_PATH"] = ":".join(map(str, paths)) + ":" + env.get("LD_LIBRARY_PATH", "")
    if dakota:
        candidate = Path(dakota).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Dakota executable not found: {candidate}")
        return str(candidate), env
    executable = shutil.which("dakota")
    if executable:
        return executable, env
    candidates = sorted((Path.home() / ".local/opt").glob("dakota-6.23.*/bin/dakota"))
    if candidates:
        return str(candidates[-1]), env
    raise FileNotFoundError("Dakota executable missing: install Dakota 6.23 or pass --dakota /path/to/dakota")


def launch(program, options, env):
    old = set((ROOT / "runs").iterdir())
    subprocess.run([sys.executable, str(ROOT / program), *options], cwd=ROOT, env=env, check=True)
    made = [p for p in (ROOT / "runs").iterdir() if p not in old and p.is_dir()]
    for run in sorted(made):
        if (run / "posterior_summary.json").exists() or (run / "best.json").exists():
            settings = json.loads((run / "config.json").read_text())
            sigma = settings.get("diagnostic_sigma")
            if (run / "best.json").is_file() and sigma is not None:
                subprocess.run([sys.executable, str(ROOT / "file_analysis.py"), "--run", str(run),
                                "--sigma", str(sigma)], cwd=ROOT, env=env, check=True)
            subprocess.run([sys.executable, str(ROOT / "report.py"), str(run)],
                           cwd=ROOT, env=env, check=True)
    return made


def main():
    parser = argparse.ArgumentParser(description="Dakota optimization/MCMC launcher and offline diagnostics")
    parser.add_argument("command", choices=("check", "demo", "opt", "bo", "mcmc", "analyze", "report", "list"))
    parser.add_argument("path", nargs="?", type=Path, help="JSON template for opt/bo/mcmc; run directory for report")
    parser.add_argument("--dakota", help="Dakota 6.23 executable, otherwise auto-detect")
    parser.add_argument("--reference", type=Path, help="override measured curve in the template")
    parser.add_argument("--sigma", type=float, help="independently justified measurement noise SD for MCMC")
    args = parser.parse_args()
    if args.command == "list":
        for run in sorted((ROOT / "runs").iterdir(), reverse=True):
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
    dakota, env = runtime(args.dakota)
    if args.command == "check":
        subprocess.run([dakota, "-v"], env=env, check=True)
        for module in ("numpy", "scipy", "matplotlib", "arviz", "corner", "sklearn"):
            try:
                __import__(module)
                print(f"Python {module}: available")
            except (ImportError, OSError) as error:
                print(f"Python {module}: missing ({error})")
        print("Slurm sbatch:", shutil.which("sbatch") or "not available (local demo still works)")
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
    extra = ["--settings", str(template), "--dakota", dakota]
    if args.reference:
        extra += ["--reference", str(args.reference.resolve())]
    if args.command == "mcmc":
        if args.sigma is not None:
            extra += ["--sigma", str(args.sigma)]
        launch("mcmc_run.py", ["--mode", "curve", *extra], env)
    else:
        launch("run.py", [args.command, *extra], env)


if __name__ == "__main__":
    main()
