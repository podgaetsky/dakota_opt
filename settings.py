"""Validate user-editable JSON templates without dynamic code execution."""

import json
from pathlib import Path

import config

CONFIG_KEYS = {
    "parameters": "PARAMETERS", "backend": "BACKEND", "concurrency": "CONCURRENCY",
    "cpus_per_evaluation": "CPUS_PER_EVALUATION", "memory": "MEMORY",
    "time_limit": "TIME_LIMIT", "partition": "PARTITION",
    "job_timeout_seconds": "JOB_TIMEOUT_SECONDS", "poll_seconds": "POLL_SECONDS",
    "failure_penalty": "FAILURE_PENALTY", "opt_evaluations": "OPT_EVALUATIONS",
    "bo_initial_samples": "BO_INITIAL_SAMPLES", "bo_batches": "BO_BATCHES",
    "bo_batch_size": "BO_BATCH_SIZE", "seed": "SEED",
}


def load_settings(path, reference_override=None):
    """Apply settings for the *current controller process*; runs freeze a snapshot."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    allowed = set(CONFIG_KEYS) | {"mode", "reference", "mcmc", "simulation_script"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown template keys: {sorted(unknown)}")
    if data.get("mode") not in ("demo", "measured"):
        raise ValueError("Template must set mode to demo or measured")
    if not isinstance(data.get("parameters"), dict) or not data["parameters"]:
        raise ValueError("Template needs a nonempty parameters dictionary")
    for name, spec in data["parameters"].items():
        if not name.isidentifier() or name.startswith("x_"):
            raise ValueError(f"Invalid parameter name {name!r}")
        if set(spec) != {"initial", "lower", "upper"}:
            raise ValueError(f"Parameter {name} needs initial, lower, upper")
        low, x, high = (float(spec[k]) for k in ("lower", "initial", "upper"))
        if not low < high or not low <= x <= high:
            raise ValueError(f"Invalid physical bounds for {name}")
    for key, attr in CONFIG_KEYS.items():
        if key in data:
            setattr(config, attr, data[key])
    if config.BACKEND not in {"local", "slurm", "allocation"}:
        raise ValueError("backend must be local, slurm or allocation")
    for key in ("concurrency", "cpus_per_evaluation", "opt_evaluations",
                "bo_initial_samples", "bo_batches", "bo_batch_size"):
        value = getattr(config, CONFIG_KEYS[key])
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.BO_BATCH_SIZE > config.CONCURRENCY:
        raise ValueError("bo_batch_size cannot exceed concurrency")
    mcmc = data.get("mcmc", {})
    if not isinstance(mcmc, dict) or set(mcmc) - {"samples", "build_samples", "sigma", "max_ordinates", "validate_samples", "backend"}:
        raise ValueError("Invalid mcmc settings; check template keys")
    if mcmc.get("backend", "dream") not in {"dream", "queso"}:
        raise ValueError("mcmc.backend must be dream or queso")
    script = data.get("simulation_script")
    if script is not None:
        if not isinstance(script, str) or not script.strip():
            raise ValueError("simulation_script must be a path to an existing Python file")
        script = Path(script).expanduser()
        if not script.is_absolute():
            script = path.parent / script
        script = script.resolve()
        if not script.is_file() or script.suffix != ".py":
            raise FileNotFoundError(f"Simulation Python script not found: {script}")
    reference = reference_override if reference_override is not None else data.get("reference")
    if data["mode"] == "measured":
        if not isinstance(reference, (str, Path)) or not str(reference).strip():
            raise ValueError("Measured template needs a reference path (x,y CSV)")
        reference = Path(reference)
        if not reference.is_absolute():
            reference = (Path.cwd() if reference_override is not None else path.parent) / reference
        if not reference.is_file():
            raise FileNotFoundError(f"Measured reference not found: {reference}. Replace the template placeholder.")
    else:
        if set(config.PARAMETERS) != {"p1", "p2", "p3", "p4"}:
            raise ValueError("Synthetic demo requires the four demo parameters; use measured mode for real data")
    return {"mode": data["mode"], "reference": str(reference) if reference else None,
            "simulation_script": str(script) if script else None, "mcmc": mcmc,
            "settings_path": str(path)}
