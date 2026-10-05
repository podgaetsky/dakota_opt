"""Scaffold a standalone project and smoke-test a configured forward model."""

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import config
from driver import curve, rms, run_job
from resources import cluster_settings
from settings import load_settings

ROOT = Path(__file__).resolve().parent

MODEL = '''"""Example forward model; replace y with your physical prediction."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--params", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
args = parser.parse_args()
parameters = json.loads(args.params.read_text())
with args.reference.open(newline="") as source, args.out.open("w", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(("x", "y"))
    for row in csv.DictReader(source):
        writer.writerow((row["x"], parameters["a"] * float(row["x"]) + parameters["b"]))
'''


def init_project(destination):
    """Create only new files, without modifying the package checkout."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    template = json.loads((ROOT / "templates/file_benchmark.json").read_text())
    template.pop("simulation_script")
    template["reference"] = "reference.csv"
    template["simulation"] = {"command": "{python} model.py --params {params} --out {curve} --reference {reference}",
                              "workdir": "."}
    template["execution"] = template.pop("backend")
    template["mcmc"]["sampler"] = template["mcmc"].pop("backend")
    output = {"template.json": json.dumps(template, indent=2) + "\n",
              "model.py": MODEL,
              "reference.csv": "x,y\n# Replace this placeholder with your strictly increasing measured grid.\n",
              "runs/.gitkeep": ""}
    for name in output:
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite {destination / name}")
    for name, text in output.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return destination


def validate_model(template, reference=None, cluster=None, workdir=None):
    """Run one midpoint simulation using the selected local/Slurm execution mode."""
    user = load_settings(template, reference)
    if user["mode"] != "measured":
        raise ValueError("validate-model needs a measured template with a real reference.csv")
    target = curve(Path(user["reference"]))
    profile = cluster_settings(cluster) if cluster else {}
    params = {name: (float(spec["lower"]) + float(spec["upper"])) / 2
              for name, spec in config.PARAMETERS.items()}
    settings = {"backend": config.BACKEND, "simulation": user["simulation"],
                "simulation_script": user["simulation_script"] or "simulate.py",
                "python_executable": profile.get("python", sys.executable),
                "cpus_per_evaluation": config.CPUS_PER_EVALUATION, "memory": config.MEMORY,
                "time_limit": config.TIME_LIMIT, "partition": profile.get("partition", config.PARTITION),
                "job_timeout_seconds": config.JOB_TIMEOUT_SECONDS, "poll_seconds": config.POLL_SECONDS}
    parent = Path(workdir or Path(template).resolve().parent) / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validate-model-", dir=parent) as directory:
        work = Path(directory)
        (work / "physical_params.json").write_text(json.dumps(params))
        (work / "reference.csv").write_bytes(Path(user["reference"]).read_bytes())
        start = time.monotonic()
        job_id = run_job(work, settings, settings["simulation_script"])
        elapsed = time.monotonic() - start
        fitted = curve(work / "curve.csv")
        loss = rms(target, fitted)
    if not math.isfinite(loss):
        raise ValueError("Simulation produced a nonfinite objective")
    # MCMC's build_samples and chain samples are intentionally not treated as exact runs.
    evaluations = config.OPT_EVALUATIONS
    estimated = evaluations * elapsed / config.CONCURRENCY
    return {"runtime_seconds": elapsed, "midpoint_rms": loss, "job_id": job_id,
            "opt_evaluations": evaluations, "estimated_opt_wall_seconds": estimated,
            "queue_latency": "included in observed runtime for Slurm; not separated"}
