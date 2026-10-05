"""Dakota standard parameters -> physical dict -> Slurm/local -> RMS -> results file."""

import csv
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_parameters(path, specs):
    """Dakota standard format: numeric value followed by a variable descriptor."""
    values = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[1].startswith("x_"):
            name = fields[1][2:]
            if name in specs:
                if name in values:
                    raise ValueError(f"Duplicate variable {name}")
                values[name] = float(fields[0])
    if set(values) != set(specs):
        raise ValueError(f"Expected normalized variables {list(specs)}, got {values}")
    result = {}
    for name, x in values.items():
        if not math.isfinite(x) or not -1e-8 <= x <= 1 + 1e-8:
            raise ValueError(f"Out-of-bounds normalized {name}={x}")
        spec = specs[name]
        result[name] = spec["lower"] + x * (spec["upper"] - spec["lower"])
    return result


def curve(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty curve: {path}")
    points = [(float(row["x"]), float(row["y"])) for row in rows]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError(f"Nonfinite curve: {path}")
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError(f"Unsorted/duplicate x values in {path}")
    return points


def rms(target, fitted):
    if len(target) != len(fitted):
        raise ValueError("Curve point counts differ; align/interpolate on reference grid")
    if any(not math.isclose(a[0], b[0], rel_tol=1e-8, abs_tol=1e-10)
           for a, b in zip(target, fitted)):
        raise ValueError("Curve x grids differ; align/interpolate on reference grid")
    value = math.sqrt(sum((a[1] - b[1]) ** 2 for a, b in zip(target, fitted)) / len(target))
    if not math.isfinite(value):
        raise ValueError("Nonfinite RMS")
    return value


def evaluate_loss(params, work, run, settings):
    """Slurm-backed evaluation: physical parameters -> simulated curve -> RMS."""
    (work / "physical_params.json").write_text(json.dumps(params, indent=2))
    shutil.copyfile(run / "reference.csv", work / "reference.csv")
    job_id = run_job(work, settings, settings.get("simulation_script", "simulate.py"))
    value = rms(curve(run / "reference.csv"), curve(work / "curve.csv"))
    destination = run / "results" / f"{work.name}.csv"
    shutil.copyfile(work / "curve.csv", destination)
    return value, job_id, str(destination.relative_to(run))


def run_job(work, settings, simulation_script="simulate.py"):
    backend = settings["backend"]
    python = settings.get("python_executable", sys.executable)
    simulation = settings.get("simulation")
    if simulation:
        paths = {"params": str((work / "physical_params.json").resolve()),
                 "curve": str((work / "curve.csv").resolve()),
                 "reference": str((work / "reference.csv").resolve()),
                 "work": str(work.resolve()), "python": python}
        tokens = shlex.split(simulation["command"])
        try:
            command = [token.format_map(paths) for token in tokens]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid simulation.command placeholder: {error}") from error
        if command[0] in ("python", "python3"):
            command[0] = python
        # The model source and executable are relative to the template, not the evaluation.
        for index, token in enumerate(command):
            candidate = Path(simulation["workdir"]) / token
            if token != command[0] and not Path(token).is_absolute() and candidate.is_file():
                command[index] = str(candidate.resolve())
        cwd = simulation["workdir"]
    else:
        command = [python, str(ROOT / simulation_script), str(work)]
        cwd = work
    if backend == "local":
        with (work / "simulation.stdout").open("w") as stdout, (work / "simulation.stderr").open("w") as stderr:
            subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr,
                           timeout=settings["job_timeout_seconds"], check=True)
        return "local"
    if backend == "allocation":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("allocation backend must run within a Slurm allocation")
        with (work / "simulation.stdout").open("w") as stdout, (work / "simulation.stderr").open("w") as stderr:
            subprocess.run(["srun", "--exclusive", "--nodes=1", "--ntasks=1",
                            f"--cpus-per-task={settings['cpus_per_evaluation']}",
                            "bash", str(ROOT / "run_slurm.sh"), *command], cwd=cwd,
                           stdout=stdout, stderr=stderr,
                           timeout=settings["job_timeout_seconds"], check=True)
        return f"{os.environ['SLURM_JOB_ID']} (allocation step)"
    if backend != "slurm":
        raise ValueError(f"Unknown backend {backend}")
    cmd = ["sbatch", "--parsable", f"--chdir={work}",
           f"--cpus-per-task={settings['cpus_per_evaluation']}",
           f"--mem={settings['memory']}", f"--time={settings['time_limit']}",
           f"--output={work / 'slurm-%j.out'}", f"--error={work / 'slurm-%j.err'}"]
    if settings["partition"]:
        cmd.append(f"--partition={settings['partition']}")
    cmd.extend([str(ROOT / "run_slurm.sh"), *command])
    submission = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    job_id = submission.stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError(f"Unrecognized sbatch response: {submission.stdout!r}")
    (work / "slurm_job_id.txt").write_text(job_id)
    deadline = time.monotonic() + settings["job_timeout_seconds"]
    while time.monotonic() < deadline:
        status = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True, check=True)
        if not status.stdout.strip():
            break
        time.sleep(settings["poll_seconds"])
    else:
        subprocess.run(["scancel", job_id], capture_output=True, check=False)
        raise TimeoutError(f"Slurm job {job_id} exceeded wall-clock timeout")
    # Accounting may lag behind squeue. Never interpret missing accounting as success.
    for _ in range(5):
        record = subprocess.run(["sacct", "-n", "-P", "-X", "-j", job_id,
                                 "--format=JobIDRaw,State,ExitCode"], capture_output=True, text=True, check=True)
        states = [s.split("|") for s in record.stdout.splitlines() if s.startswith(job_id + "|")]
        if states:
            if states[0][1] != "COMPLETED" or states[0][2] != "0:0":
                raise RuntimeError(f"Slurm job {job_id}: {states[0]}")
            return job_id
        time.sleep(settings["poll_seconds"])
    raise RuntimeError(f"No sacct record for job {job_id}; verify Slurm accounting")


def record_result(run, record, names):
    path = run / "evaluations.csv"
    fields = ["evaluation_id", "timestamp", "kind", "phase", *names,
              "objective", "status", "slurm_job_id", "runtime", "curve_file", "error"]
    with path.open("a+", newline="") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0, os.SEEK_END)
            new = stream.tell() == 0
            writer = csv.DictWriter(stream, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    # Dakota invokes: analysis_driver params.in results.out, inside the tagged work directory.
    if len(sys.argv) != 3:
        raise SystemExit("Usage: driver.py params.in results.out")
    work = Path.cwd()
    run = work.parent.parent
    settings = json.loads((run / "config.json").read_text())
    params_file, results_file = (Path(arg) for arg in sys.argv[1:])
    started = time.monotonic()
    job_id = ""
    params = {}
    value = settings["failure_penalty"]
    status = "failed"
    error = ""
    curve_file = ""
    try:
        params = parse_parameters(params_file, settings["parameters"])
        value, job_id, curve_file = evaluate_loss(params, work, run, settings)
        status = "ok"
    except Exception:
        error = traceback.format_exc()
        (run / "logs" / f"{work.name}.error.txt").write_text(error)
    finally:
        if not job_id and (work / "slurm_job_id.txt").exists():
            job_id = (work / "slurm_job_id.txt").read_text().strip()
        results_file.write_text(f"{value:.17g}\n")
        record = {"evaluation_id": work.name, "timestamp": datetime.now(timezone.utc).isoformat(),
                  "kind": settings["kind"],
                  "phase": ("initial" if settings["kind"] == "bo" and
                            work.name.rsplit(".", 1)[-1].isdigit() and
                            int(work.name.rsplit(".", 1)[-1]) <= settings["bo_initial_samples"]
                            else "acquisition" if settings["kind"] == "bo" else "search"),
                  "objective": value,
                  "status": status, "slurm_job_id": job_id, "runtime": round(time.monotonic() - started, 3),
                  "curve_file": curve_file, "error": error.splitlines()[-1] if error else ""}
        record.update(params)
        (work / "diagnostic.json").write_text(json.dumps(record, indent=2))
        record_result(run, record, settings["parameters"])


if __name__ == "__main__":
    main()